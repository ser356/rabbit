"""Desplazamiento de Cinna respecto al suelo con puntos seguidos por CoTracker3 (pre-registrado el 2026-10-27).

Sin entrenamiento y sin umbrales ajustados con etiquetas. Por cada tramo de 4 s (cada 2 s):
1. Se siguen puntos dentro de la máscara SAM2 (conejo) y fuera de ella (fondo), consultados en el centro del tramo.
2. Con los puntos de fondo cercanos se estima por RANSAC la homografía de cada fotograma al de consulta: descuenta
   el movimiento de la cámara, aunque siga al conejo.
3. La posición del conejo es la mediana de sus puntos llevados al fotograma de consulta; la cabeza o el acicalado,
   que mueven pocos puntos, apenas la cambian.
4. Para cada intervalo de 0,5 s se mide el diámetro de esa trayectoria en ±1 s, en unidades de √(área de la máscara),
   con un margen igual al doble del error de reproyección del fondo.
5. Locomoción si el límite inferior ≥ 0,5; bajo movimiento si el superior ≤ 0,25; si no, o sin puntos suficientes,
   desconocido.

Los umbrales son físicos y se fijan antes de ejecutar: medio tamaño corporal neto en 2 s es desplazamiento, un cuarto o
menos no lo es. Los 57 clips verificados se evalúan con la regla fija (no hay selección), pero los vídeos ya se
usaron en enfoques anteriores: no es un test con vídeos nuevos.
"""
import argparse
import gc
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .closure import evaluate_cases
from .motion import sha256
from .webclips import metadata, window_ends

REPO = Path(__file__).resolve().parents[2]
OUT, TRACKS = REPO / 'cierre' / 'pointtrack', REPO / 'runs' / 'pointtrack'
CHUNK, STRIDE, HALF, TARGET_FPS, SIDE = 4., 2., 1., 15., 512
RABBIT_POINTS, BACKGROUND_GRID, RING_POINTS = 64, (20, 15), 64
ERODE, DILATE, RING = .08, .15, 2.5  # en unidades de √área: interior del conejo, margen y anillo de suelo cercano
MIN_MASK, MIN_BACKGROUND, MIN_RABBIT, MIN_VALID = .001, 12, 5, .6
RANSAC_PX = 3.
LOCOMOTION, LOW = .5, .25  # diámetro neto en ±1 s, en √área; fijados antes de ejecutar
MODEL = dict(repo='facebookresearch/co-tracker', name='cotracker3_offline')


class CoTracker:
    """CoTracker3 offline vía torch.hub (se descarga en el primer uso)."""

    def __init__(self, device='mps', repo=MODEL['repo'], name=MODEL['name'], source='github'):
        import torch
        self.torch, self.device = torch, device
        self.model = torch.hub.load(repo, name, source=source).to(device).eval()
        self.info = dict(repo=repo, name=name, torch=torch.__version__, device=device)

    def __call__(self, frames, queries):
        """frames: uint8 (T, H, W, 3) RGB; queries: (N, 3) con (índice de fotograma, x, y) → (T, N, 2), (T, N)."""
        torch = self.torch
        video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)[None].float().to(self.device)
        q = torch.from_numpy(np.asarray(queries, np.float32))[None].to(self.device)
        with torch.no_grad():
            try:  # consultas en el centro del tramo: se sigue también hacia atrás
                tracks, visible = self.model(video, queries=q, backward_tracking=True)
            except TypeError:
                tracks, visible = self.model(video, queries=q)
        visible = visible[0]
        visible = visible if visible.dtype == torch.bool else visible > .5
        result = tracks[0].float().cpu().numpy(), visible.cpu().numpy().astype(bool)
        del video, q, tracks, visible
        self.release()
        return result

    def release(self):
        """Devuelve al sistema la caché de MPS: cada tramo tiene otra forma y, si no, la caché crece sin límite."""
        gc.collect()
        if self.device == 'mps':
            self.torch.mps.synchronize()
            self.torch.mps.empty_cache()
        elif str(self.device).startswith('cuda'):
            self.torch.cuda.empty_cache()

    def memory(self):
        if self.device == 'mps':
            return f', MPS {self.torch.mps.driver_allocated_memory() / 2 ** 30:.1f} GB'
        return ''


def chunk_starts(duration):
    starts = list(np.arange(0., max(duration - CHUNK, 0.) + 1e-6, STRIDE))
    if not starts or starts[-1] + CHUNK < duration - 1e-6:
        starts.append(max(0., duration - CHUNK))
    return [float(s) for s in starts]


def nearest_chunk(starts, centre):
    return int(np.argmin([abs(s + CHUNK / 2 - centre) for s in starts]))


def needed_chunks(starts, grid, clips):
    """Tramos necesarios para decidir los intervalos dentro de clips etiquetados (todos si `clips` es None)."""
    needed = set()
    for index, t in enumerate(grid):
        if not index:
            continue
        centre = t - .25
        if clips is None or any(c.start - 1e-6 <= centre <= c.end + 1e-6 for c in clips.itertuples()):
            needed.add(nearest_chunk(starts, centre))
    return sorted(needed)


def grid_points(region, count):
    ys, xs = np.nonzero(region)
    if not len(xs):
        return np.empty((0, 2))
    step = max(1, int(np.sqrt(len(xs) / count)))
    keep = (xs % step == 0) & (ys % step == 0)
    points = np.c_[xs[keep], ys[keep]].astype(float)
    if len(points) > count:
        points = points[np.linspace(0, len(points) - 1, count).round().astype(int)]
    return points


def sample_queries(mask):
    """Puntos del conejo (interior erosionado), de fondo (rejilla fuera de la máscara dilatada) y del anillo cercano."""
    mask = (mask > 0).astype(np.uint8)
    unit = np.sqrt(mask.sum())
    # Distancias euclídeas en lugar de dilatar con núcleos de cientos de píxeles (mismas regiones, mucho más rápido).
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    inner = (inside > ERODE * unit).astype(np.uint8)
    if inner.sum() < MIN_RABBIT:
        inner = mask
    outer = (outside <= DILATE * unit).astype(np.uint8)
    ring = ((outside > DILATE * unit) & (outside <= RING * unit)).astype(np.uint8)
    h, w = mask.shape
    gx, gy = np.meshgrid(np.linspace(.5, w - .5, BACKGROUND_GRID[0]), np.linspace(.5, h - .5, BACKGROUND_GRID[1]))
    background = np.c_[gx.ravel(), gy.ravel()]
    background = background[outer[background[:, 1].astype(int), background[:, 0].astype(int)] == 0]
    rabbit, near = grid_points(inner, RABBIT_POINTS), grid_points(ring, RING_POINTS)
    kinds = np.r_[np.zeros(len(rabbit)), np.ones(len(background)), np.full(len(near), 2)].astype(int)
    return np.r_[rabbit, background, near], kinds, float(unit)


def read_frames(video, indices):
    """Fotogramas pedidos, reducidos a lado largo SIDE, RGB."""
    wanted, frames, capture = set(indices), {}, cv2.VideoCapture(str(video))
    try:
        for frame in range(max(indices) + 1):
            ok, image = capture.read()
            if not ok:
                break
            if frame in wanted:
                scale = SIDE / max(image.shape[:2])
                small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
                frames[frame] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return frames


def track_case(video, masks, folder, tracker, clips=None):
    """Sigue puntos en los tramos necesarios y guarda uno por archivo; los ya guardados no se repiten."""
    fps, count = metadata(video)
    step = max(1, int(round(fps / TARGET_FPS)))
    starts = chunk_starts(count / fps)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plan = {k: list(range(int(round(starts[k] * fps)), min(count, int(round((starts[k] + CHUNK) * fps)) + 1), step))
            for k in needed_chunks(starts, window_ends(count, fps), clips)}
    plan = {k: v for k, v in plan.items() if not (folder / f'chunk_{k:04d}.npz').is_file()}
    if not plan:
        return 0
    frames = read_frames(video, sorted({f for v in plan.values() for f in v}))
    print(f'  {len(frames)} fotogramas en memoria ({sum(f.nbytes for f in frames.values()) / 2 ** 30:.2f} GB)', flush=True)
    done = 0
    for k, indices in plan.items():
        indices = [f for f in indices if f in frames]
        centre = int(np.argmin([abs(f / fps - (starts[k] + CHUNK / 2)) for f in indices]))
        query, mask = None, None
        for j in sorted(range(len(indices)), key=lambda j: abs(j - centre)):
            raw = cv2.imread(str(Path(masks) / 'masks' / f'{indices[j]:06d}.png'), cv2.IMREAD_GRAYSCALE)
            if raw is not None and np.count_nonzero(raw) >= MIN_MASK * raw.size:
                query = j
                mask = cv2.resize(raw, frames[indices[j]].shape[1::-1], interpolation=cv2.INTER_NEAREST)
                break
        if query is None:
            np.savez(folder / f'chunk_{k:04d}.npz', start=starts[k], frames=np.array(indices), query=-1)
            continue
        points, kinds, unit = sample_queries(mask)
        queries = np.c_[np.full(len(points), query), points]
        started = time.perf_counter()
        tracks, visible = tracker(np.stack([frames[f] for f in indices]), queries)
        elapsed = time.perf_counter() - started
        np.savez(folder / f'chunk_{k:04d}.npz', start=starts[k], frames=np.array(indices), query=query, fps=fps,
                 tracks=tracks, visible=visible, kinds=kinds, unit=unit)
        done += 1
        print(f'  [{done}/{len(plan)}] tramo {k} ({starts[k]:.0f}–{starts[k] + CHUNK:.0f} s): {int((kinds == 0).sum())} puntos conejo, '
              f'{int((kinds > 0).sum())} fondo, {len(indices)} fotogramas en {elapsed:.1f} s'
              f'{getattr(tracker, "memory", lambda: "")()}', flush=True)
    return len(plan)


def stabilised(chunk):
    """Por fotograma: desplazamiento del conejo en el fotograma de consulta, error del fondo y validez."""
    tracks, visible, kinds, q = chunk['tracks'], chunk['visible'], chunk['kinds'], int(chunk['query'])
    offsets = np.full((len(tracks), 2), np.nan)
    errors, shifts, background_count = np.full(len(tracks), np.nan), np.full(len(tracks), np.nan), np.zeros(len(tracks), int)
    for f in range(len(tracks)):
        both = visible[f] & visible[q]
        near = both & (kinds == 2)
        use = near if near.sum() >= MIN_BACKGROUND else both & (kinds > 0)
        rabbit = both & (kinds == 0)
        if use.sum() < MIN_BACKGROUND or rabbit.sum() < MIN_RABBIT:
            continue
        if f == q:
            H, inliers = np.eye(3), np.ones(use.sum(), bool)
        else:
            H, inliers = cv2.findHomography(tracks[f][use].astype(np.float32), tracks[q][use].astype(np.float32),
                                            cv2.RANSAC, RANSAC_PX)
            if H is None:
                continue
            inliers = inliers.ravel().astype(bool)
        if inliers.sum() < MIN_BACKGROUND:
            continue
        mapped = cv2.perspectiveTransform(tracks[f][use][inliers][None].astype(np.float64), H)[0]
        errors[f] = float(np.percentile(np.linalg.norm(mapped - tracks[q][use][inliers], axis=1), 95))
        rabbit_mapped = cv2.perspectiveTransform(tracks[f][rabbit][None].astype(np.float64), H)[0]
        offsets[f] = np.median(rabbit_mapped - tracks[q][rabbit], axis=0)
        shifts[f] = float(np.median(np.linalg.norm(tracks[f][use] - tracks[q][use], axis=1)))
        background_count[f] = int(inliers.sum())
    return offsets, errors, shifts, background_count


def decide(chunk, centre, cached=None):
    """Diámetro del desplazamiento estabilizado en [centre − 1, centre + 1] s, con margen de error."""
    empty = dict(movement='unknown', evidence_quality=0., span=np.nan, span_lower=np.nan, span_upper=np.nan,
                 valid_fraction=0., camera_shift=np.nan, background_inliers=0)
    if chunk is None or int(chunk['query']) < 0:
        return empty
    offsets, errors, shifts, background = cached if cached is not None else stabilised(chunk)
    times = np.asarray(chunk['frames']) / float(chunk['fps'])
    window = (times >= centre - HALF - 1e-6) & (times <= centre + HALF + 1e-6)
    valid = window & np.isfinite(offsets[:, 0])
    if not window.any():
        return empty
    fraction = valid.sum() / window.sum()
    unit = float(chunk['unit'])
    result = dict(empty, valid_fraction=float(fraction),
                  camera_shift=float(np.nanmax(shifts[window]) / unit) if np.isfinite(shifts[window]).any() else np.nan,
                  background_inliers=int(np.median(background[window])))
    if fraction < MIN_VALID or valid.sum() < 3:
        return result
    points = offsets[valid]
    span = float(np.linalg.norm(points[:, None] - points[None], axis=2).max() / unit)
    margin = float(2 * np.nanmax(errors[valid]) / unit)
    lower, upper = max(0., span - margin), span + margin
    movement = 'locomotion_candidate' if lower >= LOCOMOTION else 'low_motion_candidate' if upper <= LOW else 'unknown'
    quality = float(fraction) if movement != 'unknown' else 0.
    return dict(result, movement=movement, evidence_quality=quality, span=span, span_lower=lower, span_upper=upper)


def analyze(video, folder):
    fps, count = metadata(video)
    starts, grid = chunk_starts(count / fps), window_ends(count, fps)
    chunks, cache = {}, {}
    rows = []
    for index, t in enumerate(grid):
        row = dict(seconds=t, interval_seconds=t - grid[index - 1] if index else 0.)
        if not index:
            rows.append(dict(row, **decide(None, 0.)))
            continue
        k = nearest_chunk(starts, t - .25)
        if k not in chunks:
            path = Path(folder) / f'chunk_{k:04d}.npz'
            chunks[k] = dict(np.load(path)) if path.is_file() else None
            if chunks[k] is not None and int(chunks[k]['query']) >= 0:
                cache[k] = stabilised(chunks[k])
        rows.append(dict(row, chunk=k, **decide(chunks[k], t - .25, cache.get(k))))
    return rows


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    return float((positive[:, None] > negative[None]).mean() + .5 * (positive[:, None] == negative[None]).mean())


def clip_table(result, motion_dir):
    """Puntuación por clip sin umbral (máximo diámetro estabilizado) para AUC y diagnóstico; no decide nada."""
    rows = []
    for case, clips in result['clips'].items():
        frame = pd.read_csv(Path(motion_dir) / case / 'motion.csv')
        starts = frame.seconds - frame.interval_seconds
        for clip in clips['clips']:
            info = result['inputs'][case]
            labels = pd.read_csv(info['labels']['path'])
            row = labels[labels.clip_id == clip['clip_id']].iloc[0]
            inside = (starts >= row.start_seconds - 1e-6) & (frame.seconds <= row.start_seconds + row.duration_seconds + 1e-6) & \
                     (frame.interval_seconds > 0)
            part = frame[inside]
            rows.append(dict(case=case, clip_id=clip['clip_id'], label=clip['label'], prediction=clip['prediction'],
                             correct=clip['detection_correct'], events=len(clip['candidate_events']),
                             max_span=float(part.span.max()) if part.span.notna().any() else np.nan,
                             median_span=float(part.span.median()) if part.span.notna().any() else np.nan,
                             valid_fraction=float(part.valid_fraction.mean()) if len(part) else 0.,
                             camera_shift=float(part.camera_shift.median()) if part.camera_shift.notna().any() else np.nan,
                             known_fraction=clip['known_fraction']))
    return pd.DataFrame(rows)


def run(manifest, out=OUT, tracks=TRACKS, tracker=None, everything=False):
    manifest = Path(manifest).resolve()
    tracks = Path(tracks)
    checked = []

    def analyse(paths):
        folder = tracks / paths['video'].stem
        labels = pd.read_csv(paths['labels'])
        labels = labels[(labels.video_id == paths['video'].name) & labels.label.isin(['locomotion', 'resting'])]
        clips = None if everything else pd.DataFrame(dict(start=labels.start_seconds, end=labels.start_seconds +
                                                          labels.duration_seconds))
        if tracker is not None:
            print(paths['video'].name, flush=True)
            model = tracker()
            if not checked:
                checked.append(smoke(model))  # sin comprobación superada no se sigue ningún vídeo
            track_case(paths['video'], paths['masks'], folder, model, clips)
        return analyze(paths['video'], folder)

    info = dict(model=MODEL, chunk_seconds=CHUNK, stride_seconds=STRIDE, half_window_seconds=HALF, target_fps=TARGET_FPS,
                side=SIDE, locomotion_lower=LOCOMOTION, low_upper=LOW, min_valid=MIN_VALID, ransac_px=RANSAC_PX,
                points=dict(rabbit=RABBIT_POINTS, background_grid=BACKGROUND_GRID, ring=RING_POINTS),
                python=platform.python_version(), source_sha256=sha256(Path(__file__)))
    result = evaluate_cases(manifest, out, 'pointtrack', analyse, info)
    r = result['methods']['pointtrack']
    table = clip_table(result, out)
    table.to_csv(Path(out) / 'clips.csv', index=False)
    scored = table[table.max_span.notna()]
    within = {case: auc(g.max_span, g.label == 'locomotion') for case, g in scored.groupby('case') if g.label.nunique() == 2}
    result.update(original_closure_status=result['status'],
                  status='resolved_fixed_rule_on_seen_videos' if r['passed'] else 'not_resolved',
                  scope=('Regla física fijada antes de ejecutar, sin entrenamiento ni selección; los 57 clips verificados '
                         'se usaron en enfoques anteriores, así que no es un test con vídeos nuevos.'),
                  clip_auc_max_span=auc(scored.max_span, scored.label == 'locomotion'),
                  within_video_auc_max_span=within, clips_without_measure=int(table.max_span.isna().sum()),
                  clip_coverage=float((table.prediction != 'unknown').mean()))
    (Path(out) / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False, default=float))
    print(f"\nCoTracker (regla fija): locomoción {r['locomotion_hits']}/{r['locomotion_clips']}, reposo "
          f"{r['resting_correct']}/{r['resting_clips']}, reposos con alarma {r['resting_false_events']}, cobertura "
          f"{result['clip_coverage']:.2f}, AUC clip {result['clip_auc_max_span'] or float('nan'):.2f} → "
          f"{'PASA' if r['passed'] else 'NO PASA'}\nEstado: {result['status']}")
    return result


def render(manifest, out=OUT, tracks=TRACKS, videos=None, width=640):
    """Vídeos de diagnóstico: puntos del conejo (verde) y de fondo cercano (azul), diámetro y decisión."""
    manifest = Path(manifest).resolve()
    folder = Path(videos) if videos else REPO / 'runs' / 'pointtrack' / 'diagnostico'
    folder.mkdir(parents=True, exist_ok=True)
    colours = {'locomotion_candidate': (60, 180, 255), 'low_motion_candidate': (200, 160, 60), 'unknown': (150, 150, 150)}
    for case in json.loads(manifest.read_text())['cases']:
        video = (manifest.parent / Path(case['video']).expanduser()).resolve()
        labels = (manifest.parent / Path(case['labels']).expanduser()).resolve()
        motion = Path(out) / case['name'] / 'motion.csv'
        if not video.is_file() or not motion.is_file():
            continue
        frame_rows = pd.read_csv(motion)
        table = pd.read_csv(labels)
        table = table[(table.video_id == video.name) & table.label.isin(['locomotion', 'resting'])]
        chunks = {int(p.stem.split('_')[1]): dict(np.load(p)) for p in sorted((Path(tracks) / video.stem).glob('chunk_*.npz'))}
        chunks = {k: c for k, c in chunks.items() if int(c['query']) >= 0}
        if not chunks:
            continue
        fps, count = metadata(video)
        starts = chunk_starts(count / fps)
        by_frame = {}
        for k, c in chunks.items():
            for j, f in enumerate(c['frames']):
                if nearest_chunk(starts, f / fps) == k:
                    by_frame[int(f)] = (k, j)
        frames = read_frames(video, sorted(by_frame))
        scale = width / next(iter(frames.values())).shape[1]
        size = (width, int(round(next(iter(frames.values())).shape[0] * scale)))
        writer = cv2.VideoWriter(str(folder / f"{case['name']}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, size)
        for f in sorted(frames):
            k, j = by_frame[f]
            c, t = chunks[k], f / fps
            image = cv2.cvtColor(cv2.resize(frames[f], size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            for (x, y), kind, seen in zip(c['tracks'][j], c['kinds'], c['visible'][j]):
                if seen and kind != 1:
                    cv2.circle(image, (int(x * scale), int(y * scale)), 2, (0, 255, 0) if kind == 0 else (255, 120, 0), -1)
            row = frame_rows[(frame_rows.seconds - frame_rows.interval_seconds < t) & (frame_rows.seconds >= t)]
            truth = next((r.label for r in table.itertuples() if r.start_seconds <= t < r.start_seconds + r.duration_seconds), '-')
            movement = row.movement.iloc[0] if len(row) else 'unknown'
            span = row.span.iloc[0] if len(row) else np.nan
            cv2.rectangle(image, (0, 0), (width, 28), colours[movement], -1)
            cv2.putText(image, f't={t:5.1f}s verdad={truth} diametro={span:.2f} {movement}', (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
            writer.write(image)
        writer.release()
        print('diagnóstico', folder / f"{case['name']}.mp4", flush=True)


SMOKE_SIZE, SMOKE_TOLERANCE = 64, .1  # error máximo admitido: 0,1 tamaños (el umbral de reposo es 0,25)


def texture(rng, shape, sigma=3.):
    """Textura suave tipo imagen natural; el ruido blanco puro se pierde al reescalar dentro del modelo."""
    noise = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), sigma)
    return np.clip(128 + 60 * noise / noise.std(), 0, 255).astype(np.uint8)


def smoke(tracker):
    """Modelo real sobre una escena sintética: un parche de 64 px se desplaza 3 px por fotograma sobre fondo fijo.

    El error se expresa en tamaños del parche, la misma unidad que la regla (√área)."""
    rng = np.random.default_rng(0)
    background, patch = texture(rng, (288, 512, 3)), texture(rng, (SMOKE_SIZE, SMOKE_SIZE, 3), 2.)
    frames = []
    for f in range(30):
        image = background.copy()
        image[112:112 + SMOKE_SIZE, 100 + 3 * f:100 + SMOKE_SIZE + 3 * f] = patch
        frames.append(image)
    centre = 100 + 3 * 15 + SMOKE_SIZE / 2
    queries = np.array([[15, centre, 144.], [15, 40., 40.], [15, 460., 250.], [15, 300., 60.]])
    tracks, visible = tracker(np.stack(frames), queries)
    expected = np.c_[centre + 3 * (np.arange(30) - 15), np.full(30, 144.)]
    errors = np.linalg.norm(tracks[:, 0] - expected, axis=1)
    still = float(np.linalg.norm(tracks[:, 1:] - queries[None, 1:, 1:], axis=2).max())
    worst = float(max(errors.max(), still) / SMOKE_SIZE)
    print(f'CoTracker: error del parche máx. {errors.max():.1f} px / mediana {np.median(errors):.1f} px '
          f'(fotogramas 0, 15, 29: {errors[0]:.1f}, {errors[15]:.1f}, {errors[29]:.1f}); deriva del fondo {still:.1f} px; '
          f'peor error {worst:.3f} tamaños (máximo {SMOKE_TOLERANCE}); visibles {visible.mean():.2f}', flush=True)
    if worst > SMOKE_TOLERANCE:
        raise SystemExit('El modelo no sigue bien la escena sintética: no se ejecuta run.')
    return float(errors.max()), still


def main():
    parser = argparse.ArgumentParser(description='Desplazamiento respecto al suelo con CoTracker3; regla física fija')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('smoke').add_argument('--device', choices=['mps', 'cpu'], default='mps')
    for name in ('run', 'render'):
        command = sub.add_parser(name)
        command.add_argument('--manifest', required=True)
        command.add_argument('--out', default=str(OUT))
        command.add_argument('--tracks', default=str(TRACKS))
    sub.choices['run'].add_argument('--device', choices=['mps', 'cpu'], default='mps')
    sub.choices['run'].add_argument('--hub', help='Copia local del repositorio co-tracker (source=local)')
    sub.choices['run'].add_argument('--all', action='store_true', help='Seguir puntos en todo el vídeo, no solo en clips')
    sub.choices['run'].add_argument('--no-track', action='store_true', help='Solo analizar tramos ya guardados')
    args = parser.parse_args()
    if args.command == 'smoke':
        smoke(CoTracker(args.device))
    elif args.command == 'run':
        holder = {}

        def tracker():
            if 'model' not in holder:
                holder['model'] = CoTracker(args.device, args.hub or MODEL['repo'], source='local' if args.hub else 'github')
            return holder['model']
        run(args.manifest, args.out, args.tracks, None if args.no_track else tracker, args.all)
    else:
        render(args.manifest, args.out, args.tracks)


if __name__ == '__main__':
    main()
