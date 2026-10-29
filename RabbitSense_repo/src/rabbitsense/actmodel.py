"""Modelo propio de actos de Cinna (exploratorio, fijado el 2026-10-29 antes de ejecutar).

Clases: desplazarse, comer, tumbada y «otra» (agazapada, explorar, acicalarse, de pie). Se excluye «no visible».
La agrupación se decidió después de ver la verificación de las etiquetas (kappa 0,63): resultado exploratorio.

Unidad: ventanas de 2 s en tiempo real (`acts motion-sheets`, cierre/actos/ventanas_indice.csv).
Características, todas sin entrenar:
- cinemática: desplazamiento neto respecto al suelo en el bloque de 2 s (tramos de CoTracker) y movimiento interno de los puntos
  del conejo respecto a su mediana (masticar, acicalarse, cambios de postura);
- forma de la máscara SAM2: área, elongación, proporción alto/ancho y sus cambios;
- mapa de movimiento: energía dentro y fuera de la máscara (fuera ≈ cámara);
- V-JEPA 2 congelado (`facebook/vjepa2-vitl-fpc64-256`) sobre el recorte fijo del conejo, 16 fotogramas.
Clasificador: regresión logística multinomial con clases equilibradas, estandarización y C = 0,1 (fijos).
Validación: dejando fuera una sesión entera (7 sesiones); nada de la sesión evaluada entra en el ajuste.
Criterio exploratorio fijado: exactitud equilibrada ≥ 0,6 y sensibilidad ≥ 0,5 en cada clase (azar: 0,25).
"""
import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .acts import MOTION, image_box, motion_map, read_small, session_of, square
from .motion import sha256

REPO = Path(__file__).resolve().parents[2]
GROUPS = dict(locomotion='locomotion', eating='eating', lying='lying', loaf='other', exploring='other',
              grooming='other', rearing='other', binky='other')
CLASSES = ['locomotion', 'eating', 'lying', 'other']
MODEL_ID, FRAMES, C = 'facebook/vjepa2-vitl-fpc64-256', 16, .1
CRITERION = dict(balanced_accuracy=.6, min_recall=.5)
IMAGENET = ([.485, .456, .406], [.229, .224, .225])


class VJEPA:
    def __init__(self, device='mps', model_id=MODEL_ID):
        import torch
        import transformers
        self.torch, self.device = torch, device
        self.model = transformers.AutoModel.from_pretrained(model_id).to(device).eval()
        self.size = int(getattr(self.model.config, 'crop_size', 256))
        try:
            processor = transformers.AutoVideoProcessor.from_pretrained(model_id)
            mean, std = list(processor.image_mean), list(processor.image_std)
        except (OSError, ValueError, TypeError, AttributeError):
            mean, std = IMAGENET
        self.mean = torch.tensor(mean, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 1, 3, 1, 1)
        self.info = dict(model_id=model_id, torch=torch.__version__, transformers=transformers.__version__, device=device,
                         size=self.size, frames=FRAMES)

    def __call__(self, clip):
        """clip: uint8 (T, S, S, 3) BGR → vector medio de los tokens del codificador."""
        torch = self.torch
        rgb = np.ascontiguousarray(clip[..., ::-1])
        x = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).float()[None] / 255.
        with torch.no_grad():
            tokens = self.model.get_vision_features((x - self.mean) / self.std)
        return tokens.mean(dim=1)[0].float().cpu().numpy()


def mask_stats(mask):
    m = (mask > 0).astype(np.uint8) if mask is not None else None
    if m is None or m.sum() < .001 * m.size:
        return None
    moments = cv2.moments(m, binaryImage=True)
    cov = np.array([[moments['mu20'], moments['mu11']], [moments['mu11'], moments['mu02']]]) / moments['m00']
    eig = np.sort(np.linalg.eigvalsh(cov))
    x, y, w, h = cv2.boundingRect(m)
    return dict(area=float(m.sum() / m.size), elongation=float(np.sqrt(eig[1] / max(eig[0], 1e-9))), aspect=float(h / max(w, 1)))


def internal_motion(chunks, starts, times, start, end):
    """Movimiento de los puntos del conejo respecto a su mediana dentro de la ventana, en √área por fotograma."""
    from .pointtrack import nearest_chunk
    k = nearest_chunk(starts, (start + end) / 2)
    if k not in chunks or chunks[k][1] is None:
        return np.nan, np.nan
    chunk = chunks[k][0]
    inside = (np.asarray(chunk['times']) >= start) & (np.asarray(chunk['times']) < end)
    rabbit = np.asarray(chunk['kinds']) == 0
    tracks, visible = np.asarray(chunk['tracks']), np.asarray(chunk['visible'])
    rel = []
    for f in np.flatnonzero(inside):
        seen = rabbit & visible[f]
        if seen.sum() < 5:
            continue
        pts = tracks[f][seen]
        rel.append(np.where(seen[:, None], tracks[f] - np.median(pts, axis=0), np.nan))
    if len(rel) < 3:
        return np.nan, np.nan
    rel = np.array(rel)[:, rabbit]
    step = np.linalg.norm(np.diff(rel, axis=0), axis=2) / float(chunk['unit'])
    step = step[:, np.isfinite(step).any(axis=0)]  # puntos nunca visibles en dos fotogramas seguidos
    if not step.shape[1]:
        return np.nan, np.nan
    per_point = np.nanmedian(step, axis=0)
    per_point = per_point[np.isfinite(per_point)]
    return (float(np.median(per_point)), float(np.percentile(per_point, 90))) if len(per_point) else (np.nan, np.nan)


def union_inside(masks, frames, box, shape, size=MOTION['size']):
    """Unión de las máscaras de la ventana, recortada con la misma caja que los fotogramas: True = conejo."""
    valid = []
    for f in frames:
        m = cv2.imread(str(masks / 'masks' / f'{f:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if m is not None and np.count_nonzero(m) >= .001 * m.size:
            valid.append(m > 0)
    union = np.max(valid, axis=0).astype(np.uint8) * 255
    small = cv2.resize(union, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return square(np.repeat(small[..., None], 3, axis=2), box, size)[..., 0] > 127


def net_blocks(times, folder):
    """Desplazamiento neto por bloque de 2 s (inicio → tamaños corporales), calculado desde los tramos guardados."""
    from .activity import NET_STEP, step_rows
    if not Path(folder).is_dir():
        return {}
    return {round(r['seconds'] - r['interval_seconds'], 3): r['displacement'] for r in step_rows(times, folder, NET_STEP)
            if r['interval_seconds'] > 0 and r['measured']}


def cases_with_windows(manifest, windows):
    """(caso, vídeo, máscaras, ventanas) de cada caso del manifest que tiene ventanas."""
    manifest = Path(manifest).resolve()
    for case in json.loads(manifest.read_text())['cases']:
        mine = windows[windows.case == case['name']]
        if mine.empty:
            continue
        video = (manifest.parent / Path(case['video']).expanduser()).resolve()
        masks = (manifest.parent / Path(case['masks']).expanduser()).resolve() if 'masks' in case else None
        yield case['name'], video, masks, mine


def check_videos(manifest, windows):
    """Falla antes de cargar nada si algún vídeo con ventanas no existe o no se puede abrir."""
    windows = pd.read_csv(windows, dtype={'case': str}) if not isinstance(windows, pd.DataFrame) else windows
    missing = []
    for name, video, _, _ in cases_with_windows(manifest, windows):
        capture = cv2.VideoCapture(str(video))
        if not video.is_file() or not capture.isOpened():
            missing.append(str(video))
        capture.release()
    if missing:
        raise FileNotFoundError('No se pueden abrir estos vídeos:\n  ' + '\n  '.join(missing))


def window_features(manifest, windows, out, encoder=None, tracks_dir=None):
    """Características por ventana. Cada caso terminado se guarda en out/casos y se reutiliza si el código no cambia."""
    from . import pointtrack
    windows = pd.read_csv(windows, dtype={'case': str})
    check_videos(manifest, windows)
    tracks_dir = Path(tracks_dir or pointtrack.TRACKS)
    out = Path(out)
    cache = out / 'casos'
    cache.mkdir(parents=True, exist_ok=True)
    source = {name: sha256(Path(__file__).parent / f'{name}.py') for name in ('actmodel', 'acts', 'activity', 'pointtrack')}
    tables, deep = [], []
    for name, video, masks, mine in cases_with_windows(manifest, windows):
        stem = cache / name
        meta = dict(source=source, encoder=encoder.info if encoder is not None else None, windows=len(mine))
        stored = json.loads(stem.with_suffix('.json').read_text()) if stem.with_suffix('.json').is_file() else None
        if stored == meta:
            table = pd.read_csv(stem.with_suffix('.csv'), dtype={'case': str})
            vectors = np.load(stem.with_suffix('.npy')) if encoder is not None and len(table) else None
            print(f'{name}: {len(table)} ventanas (guardadas)', flush=True)
        else:
            table, vectors = case_features(name, video, masks, mine, encoder, tracks_dir)
            table.to_csv(stem.with_suffix('.csv'), index=False)
            if vectors is not None:
                np.save(stem.with_suffix('.npy'), vectors)
            stem.with_suffix('.json').write_text(json.dumps(meta, indent=2))
            print(f'{name}: {len(table)} ventanas', flush=True)
        tables.append(table)
        if vectors is not None:
            deep.append(vectors)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / 'caracteristicas.csv', index=False)
    if encoder is not None:
        np.save(out / 'vjepa.npy', np.concatenate(deep))
        (out / 'encoder.json').write_text(json.dumps(encoder.info, indent=2))
    return table


def case_features(name, video, masks, mine, encoder, tracks_dir):
    from . import activity, pointtrack
    rows, deep = [], []
    times = pointtrack.frame_times(video, tracks_dir / 'tiempos')
    folder = tracks_dir / video.stem
    chunks = activity.load_chunks(folder, times) if folder.is_dir() else {}
    starts = pointtrack.chunk_starts(pointtrack.duration(times))
    net = net_blocks(times, folder)
    members = {w.Index: [int(f) for f in np.flatnonzero((times >= w.start) & (times < w.end))] for w in mine.itertuples()}
    images = read_small(video, {f for frames in members.values() for f in frames})
    for w in mine.itertuples():
        frames = [f for f in members[w.Index] if f < len(images) and images[f] is not None]
        if len(frames) < 2:
            continue
        shape = images[frames[0]].shape
        box = image_box(masks, frames, shape)
        crops = [square(images[f], box, MOTION['size']) for f in frames]
        energy = motion_map(crops)
        inside = union_inside(masks, frames, box, shape) if box is not None else np.ones(energy.shape, bool)
        stats = [s for s in (mask_stats(cv2.imread(str(masks / 'masks' / f'{f:06d}.png'), cv2.IMREAD_GRAYSCALE))
                             for f in frames) if s] if masks else []
        row = dict(case=name, start=float(w.start), end=float(w.end), masked=box is not None,
                   motion_in=float(energy[inside].mean()) if inside.any() else np.nan,
                   motion_in_p99=float(np.percentile(energy[inside], 99)) if inside.any() else np.nan,
                   motion_out=float(energy[~inside].mean()) if (~inside).any() else np.nan)
        row['motion_ratio'] = row['motion_in'] / (row['motion_out'] + 1e-3) if np.isfinite(row['motion_out']) else np.nan
        for key in ('area', 'elongation', 'aspect'):
            values = np.array([s[key] for s in stats])
            row[f'{key}_mean'] = float(values.mean()) if stats else np.nan
            row[f'{key}_range'] = float(values.max() - values.min()) if stats else np.nan
        row['net_displacement'] = float(net.get(round(float(w.start), 3), np.nan))
        row['internal_median'], row['internal_p90'] = internal_motion(chunks, starts, times, w.start, w.end)
        rows.append(row)
        if encoder is not None:
            picks = [crops[int(round(j))] for j in np.linspace(0, len(crops) - 1, FRAMES)]
            clip = np.stack([cv2.resize(c, (encoder.size, encoder.size), interpolation=cv2.INTER_AREA) for c in picks])
            deep.append(encoder(clip))
    return pd.DataFrame(rows), (np.stack(deep) if deep else None)


HANDCRAFTED = ['net_displacement', 'internal_median', 'internal_p90', 'motion_in', 'motion_in_p99', 'motion_out',
               'motion_ratio', 'area_mean', 'area_range', 'elongation_mean', 'elongation_range', 'aspect_mean',
               'aspect_range']


def design(table, deep, kind):
    hand = table[HANDCRAFTED].to_numpy(float)
    missing = ~np.isfinite(hand)
    hand = np.c_[np.where(missing, 0., hand), missing[:, [0, 1]].astype(float)]  # ausencia ≠ quietud
    if kind == 'handcrafted':
        return hand
    if kind == 'deep':
        return deep
    return np.c_[hand, deep]


def evaluate(features_dir, labels, out, kinds=('handcrafted', 'deep', 'all')):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    features_dir, out = Path(features_dir), Path(out)
    deep = np.load(features_dir / 'vjepa.npy') if (features_dir / 'vjepa.npy').is_file() else None
    table = pd.read_csv(features_dir / 'caracteristicas.csv', dtype={'case': str})
    acts = pd.read_csv(labels, dtype={'case': str})[['case', 'start', 'act']]
    for frame in (table, acts):
        frame['start'] = frame.start.round(3)
    table = table.merge(acts, on=['case', 'start'], how='left', validate='one_to_one')
    keep = table.act.isin(GROUPS).to_numpy()
    y = table.act.map(GROUPS).to_numpy()
    groups = table.case.map(session_of).to_numpy()
    report, predictions = {}, table[['case', 'start', 'end', 'act']].copy()
    for kind in kinds:
        if kind != 'handcrafted' and deep is None:
            continue
        x = design(table, deep, kind)
        held = np.full(len(y), None, object)
        for g in sorted(set(groups[keep])):
            train, test = keep & (groups != g), keep & (groups == g)
            if len(set(y[train])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=5000))
            model.fit(x[train], y[train])
            held[test] = model.predict(x[test])
        ok = keep & (held != None)  # noqa: E711
        truth, guess = y[ok], held[ok].astype(str)
        recall = {c: float((guess[truth == c] == c).mean()) if (truth == c).any() else None for c in CLASSES}
        precision = {c: float((truth[guess == c] == c).mean()) if (guess == c).any() else None for c in CLASSES}
        balanced = float(np.mean([r for r in recall.values() if r is not None]))
        within = {}
        for case, idx in predictions[ok].groupby('case').groups.items():
            labels_here = set(y[idx])
            if len(labels_here) >= 2:
                within[case] = float((held[idx] == y[idx]).mean())
        report[kind] = dict(windows=int(ok.sum()), balanced_accuracy=balanced, recall=recall, precision=precision,
                            accuracy=float((guess == truth).mean()),
                            confusion=pd.crosstab(pd.Series(truth, name='verdad'), pd.Series(guess, name='modelo')).to_dict('index'),
                            within_video_accuracy=within,
                            passes=bool(balanced >= CRITERION['balanced_accuracy'] and
                                        all(r is not None and r >= CRITERION['min_recall'] for r in recall.values())))
        predictions[f'pred_{kind}'] = held
        print(f"{kind:<12} exactitud equilibrada {balanced:.2f} | " +
              ' '.join(f"{c} {recall[c] if recall[c] is not None else float('nan'):.2f}" for c in CLASSES) +
              f" | {'cumple' if report[kind]['passes'] else 'no cumple'}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / 'predicciones.csv', index=False)
    result = dict(results=report, classes=CLASSES, groups=GROUPS, C=C, criterion=CRITERION, validation='una sesión fuera',
                  sessions=sorted(set(groups[keep])), labels_sha256=sha256(labels), source_sha256=sha256(Path(__file__)),
                  python=platform.python_version(),
                  scope=('Exploratorio: etiquetas de Claude (kappa 0,63 frente a la persona), agrupación decidida tras la '
                         'verificación, mismos vídeos que el desarrollo. No es una validación.'))
    (out / 'evaluacion.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return result


def main():
    parser = argparse.ArgumentParser(description='Modelo propio de actos de Cinna (exploratorio)')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('features')
    command.add_argument('--manifest', required=True)
    command.add_argument('--windows', default=str(REPO / 'cierre' / 'actos' / 'ventanas_indice.csv'))
    command.add_argument('--out', default=str(REPO / 'runs' / 'actos_modelo'))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command.add_argument('--no-deep', action='store_true', help='Solo características sin red profunda')
    command = sub.add_parser('evaluate')
    command.add_argument('--features', default=str(REPO / 'runs' / 'actos_modelo'))
    command.add_argument('--labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_ventanas_claude_v2.csv'))
    command.add_argument('--out', default=str(REPO / 'cierre' / 'actos_modelo'))
    args = parser.parse_args()
    if args.command == 'features':
        check_videos(args.manifest, args.windows)
        window_features(args.manifest, args.windows, args.out, None if args.no_deep else VJEPA(args.device))
    else:
        evaluate(args.features, args.labels, args.out)


if __name__ == '__main__':
    main()
