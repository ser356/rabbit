"""Análisis automático de vídeos de Cinna, sin ninguna intervención (fijado el 2026-10-30).

1. Segmentación sin clics. OWLv2 («a photo of a rabbit») mira un fotograma por segundo. La primera detección fiable
   (puntuación ≥ 0,3 y un solo conejo) sustituye a la semilla manual de SAM2; las siguientes sirven de guías en el modo
   `guided` de `segment` (las anclas impares interiores se reservan para comprobar la máscara). Si el conejo aparece
   más tarde, el vídeo se recorta desde ahí y los fotogramas anteriores quedan sin máscara.
2. Desplazamiento respecto al suelo con CoTracker3 (`activity`: bloques netos de 2 s y pasos de 0,5 s).
3. Actos por ventana de 2 s con el modelo congelado entrenado solo con YouTube (`ytacts`): desplazarse, tumbado,
   otra (agazapado, explorar, acicalarse) y comer, que se muestra «sin validar» (sensibilidad 0,10 en Cinna). Si el
   detector no ve al conejo en ninguno de los 4 fotogramas de la ventana: «no visible».
4. Informe por vídeo: ventanas.csv, resumen.json e informe.html.

No estima estado de ánimo ni salud. `validate` compara, en los 17 vídeos ya analizados, las máscaras y el
desplazamiento automáticos con los obtenidos con semilla manual (criterio ADOPT_AUTO, fijado antes de ejecutar).
"""
import argparse
import html
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .acts import square_around
from .motion import sha256

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'runs' / 'analisis'
SAM_MODEL = REPO / 'weights' / 'sam2.1_s.pt'
AUTO = dict(anchor_step=1., score=.3, window=2.)
# Se adopta la segmentación automática si, frente a la de semilla manual en los 17 vídeos: la IoU mediana de las
# máscaras es ≥ 0,7; la fracción de fotogramas con máscara no baja más de 0,05; el desplazamiento neto de 2 s difiere
# en mediana ≤ 0,15 tamaños; y su AUC (desplazarse frente al resto, etiquetas de actos de Claude) es ≥ 0,9.
ADOPT_AUTO = dict(mask_iou_median=.7, mask_coverage_drop=.05, displacement_median_difference=.15, auc_min=.9)
NAMES = dict(locomotion='desplazarse', lying='tumbado', other='otra (agazapado, explorar, acicalarse)',
             eating='comer (sin validar)', not_visible='no visible')
VALIDATED = dict(locomotion=True, lying=True, other=True, eating=False)
COLOURS = dict(locomotion='#e8710a', lying='#1a73e8', other='#7cb342', eating='#a142f4', not_visible='#c7c7c7')


def video_info(video):
    capture = cv2.VideoCapture(str(video))
    try:
        info = (capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if not np.isfinite(info[0]) or info[0] <= 0 or info[1] <= 0:
        raise ValueError(f'Metadatos inválidos: {video}')
    return info


def usable(row):
    return row['box'] is not None and not row['multiple'] and row['score'] >= AUTO['score']


def detect_every(video, detector, step=AUTO['anchor_step']):
    """Detección del conejo en un fotograma cada `step` s; cajas (x, y, ancho, alto) en píxeles del vídeo original."""
    from .acts import read_small
    fps, count, width, height = video_info(video)
    every = max(1, int(round(fps * step)))
    images = read_small(video, set(range(0, count, every)))
    frames = [f for f in range(0, count, every) if f < len(images) and images[f] is not None]
    rows = []
    for f, found in zip(frames, detector([images[f] for f in frames])):
        scale = width / images[f].shape[1]
        best = found[0] if found else None
        rows.append(dict(frame=f, score=float(best[0]) if best else float('nan'), multiple=len(found) > 1,
                         box=[best[1] * scale, best[2] * scale, (best[3] - best[1]) * scale, (best[4] - best[2]) * scale]
                         if best else None))
    return rows, fps, width, height


def trim(video, start, path):
    """Copia el vídeo desde el fotograma `start` (mismo ritmo y tamaño) para que SAM2 empiece donde está el conejo."""
    capture = cv2.VideoCapture(str(video))
    fps, width, height = capture.get(cv2.CAP_PROP_FPS), int(capture.get(3)), int(capture.get(4))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    written, frame = 0, 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame >= start:
                writer.write(image)
                written += 1
            frame += 1
    finally:
        capture.release()
        writer.release()
    return written


def autosegment(video, out, detector, device='mps', model=SAM_MODEL, segment_fn=None):
    """Máscaras SAM2 sin clics; devuelve la carpeta con masks/000000.png… para todo el vídeo."""
    video, out = Path(video).resolve(), Path(out)
    record = out / 'auto.json'
    if record.is_file():
        stored = json.loads(record.read_text())
        if stored.get('video_sha256') == sha256(video) and Path(stored['masks']).is_dir():
            return Path(stored['masks'])
    out.mkdir(parents=True, exist_ok=True)
    rows, fps, width, height = detect_every(video, detector)
    good = [r for r in rows if usable(r)]
    _, count, _, _ = video_info(video)
    masks = out / 'mascaras'
    (masks / 'masks').mkdir(parents=True, exist_ok=True)
    result = dict(video=str(video), video_sha256=sha256(video), detector=getattr(detector, 'info', None), criterion=AUTO,
                  detections=len(rows), usable_detections=len(good), masks=str(masks))
    if not good:
        empty = np.zeros((height, width), np.uint8)
        for f in range(count):
            cv2.imwrite(str(masks / 'masks' / f'{f:06d}.png'), empty)
        record.write_text(json.dumps(dict(result, status='sin_conejo'), indent=2, ensure_ascii=False))
        return masks
    start = good[0]['frame']
    source = video if start == 0 else out / 'recorte.mp4'
    if start:
        trim(video, start, source)
    marks = [dict(time=(r['frame'] - start) / fps, box=r['box'] if usable(r) else None) for r in rows if r['frame'] >= start]
    (out / 'anclas.json').write_text(json.dumps(dict(width=width, height=height, marks=marks, source='owlv2_auto'),
                                                indent=1))
    if segment_fn is None:
        from .segment import segment as segment_fn
    summary = segment_fn(source, out / 'anclas.json', model, out / 'sam2', device, guided=True)
    empty = np.zeros((height, width), np.uint8)
    for f in range(count):
        path = out / 'sam2' / 'masks' / f'{f - start:06d}.png'
        if f >= start and path.is_file():
            shutil.copyfile(path, masks / 'masks' / f'{f:06d}.png')
        else:
            cv2.imwrite(str(masks / 'masks' / f'{f:06d}.png'), empty)
    if start:
        source.unlink(missing_ok=True)
    result.update(status='ok', start_frame=int(start), start_seconds=float(start / fps),
                  prompt_frames=summary.get('prompt_frames'), heldout_mean_iou=summary.get('heldout_mean_iou'),
                  nonempty_frames=summary.get('nonempty_frames'))
    record.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    return masks


def stream_windows(video, times, window=AUTO['window'], side=640):
    """Fotogramas reducidos agrupados por ventana de `window` s en tiempo real, sin cargar el vídeo entero."""
    capture, current, frames, images = cv2.VideoCapture(str(video)), None, [], {}
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok or index >= len(times):
                break
            k = int(times[index] // window)
            if current is not None and k != current:
                yield current, frames, images
                frames, images = [], {}
            current = k
            scale = side / max(image.shape[:2])
            images[index] = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                                       interpolation=cv2.INTER_AREA) if scale < 1 else image
            frames.append(index)
            index += 1
        if frames:
            yield current, frames, images
    finally:
        capture.release()


def act_windows(video, times, detector, encoder, frozen):
    """Acto de cada ventana de 2 s con el modelo congelado; «no visible» si no hay detección del conejo."""
    from .ytacts import embed, probabilities, probe_frames, window_detection
    rows, end = [], float(times[-1])
    for k, frames, images in stream_windows(video, times):
        start = k * AUTO['window']
        row = dict(start=round(start, 3), end=round(min(start + AUTO['window'], end), 3))
        if len(frames) < 2:
            continue
        summary = window_detection(detector([images[f] for f in probe_frames(frames)]))
        row.update(found=summary['found'], score_min=summary['score_min'])
        if not summary['found']:
            rows.append(dict(row, act='not_visible'))
            continue
        box = square_around(*summary['union'], images[frames[0]].shape)
        p = probabilities(frozen, embed(images, frames, box, encoder))[0]
        best = int(np.argmax(p))
        rows.append(dict(row, act=frozen['classes'][best], confidence=float(p[best]),
                         **{f'p_{c}': float(v) for c, v in zip(frozen['classes'], p)}))
    return pd.DataFrame(rows)


def bouts(acts, name):
    """Duraciones de los tramos seguidos de un mismo acto."""
    lengths, current = [], 0.
    for a, s, e in zip(acts.act, acts.start, acts.end):
        if a == name:
            current += e - s
        elif current:
            lengths.append(current)
            current = 0.
    return lengths + ([current] if current else [])


def report(video, out, acts, activity_dir, model_file=None):
    """Tabla por ventana, resumen e informe HTML."""
    out = Path(out)
    blocks = pd.read_csv(Path(activity_dir) / 'actividad_2s.csv')
    blocks = blocks[blocks.interval_seconds > 0].assign(start=lambda b: (b.seconds - b.interval_seconds).round(3))
    table = acts.merge(blocks[['start', 'displacement', 'displacement_lower', 'measured']], on='start', how='left')
    table['measured'] = table.measured.fillna(False).astype(bool)
    table['seconds'] = table.end - table.start
    activity = json.loads((Path(activity_dir) / 'resumen.json').read_text())
    visible = table[table.act != 'not_visible']
    seen = float(visible.seconds.sum())
    time = {c: dict(seconds=float(table[table.act == c].seconds.sum()),
                    share_of_visible=float(table[table.act == c].seconds.sum() / seen) if seen else None,
                    validated=VALIDATED[c]) for c in VALIDATED}
    lying, moving = bouts(table, 'lying'), bouts(table, 'locomotion')
    net = activity['net_2s']
    summary = dict(
        video=str(video), video_sha256=sha256(video), duration_seconds=float(table.end.max()) if len(table) else 0.,
        visible_seconds=seen, not_visible_seconds=float(table[table.act == 'not_visible'].seconds.sum()),
        time_by_act=time, longest_lying_bout_seconds=max(lying, default=0.), lying_bouts=len(lying),
        locomotion_bouts=len(moving),
        displacement=dict(body_units=net['displacement_body_units'], per_measured_minute=net['displacement_per_measured_minute'],
                          measured_seconds=net['measured_seconds']),
        model_sha256=sha256(model_file or REPO / 'benchmarks' / 'ytactos_modelo.json'),
        caveats=['Comer no está validado: en Cinna solo se reconoció 1 de cada 10 ventanas de comer.',
                 'Desplazarse, tumbado y otra: 0,72, 0,85 y 0,77 de aciertos en la única prueba con Cinna (etiquetas '
                 'de Claude, kappa 0,63 frente a la persona).',
                 'El desplazamiento está en tamaños corporales (√área de la máscara), no en centímetros.',
                 'El tiempo sin medida o «no visible» no es reposo.',
                 'No es un indicador de estado de ánimo ni de salud.'])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'ventanas.csv', index=False)
    (out / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    (out / 'informe.html').write_text(page(Path(video).name, table, summary), encoding='utf8')
    return summary


def page(title, table, summary, width=1000):
    end = max(summary['duration_seconds'], 1e-6)
    x = lambda t: width * t / end  # noqa: E731
    bars, acts = [], []
    for r in table.itertuples():
        acts.append(f'<rect x="{x(r.start):.1f}" y="0" width="{max(x(r.end) - x(r.start), 1):.1f}" height="28" '
                    f'fill="{COLOURS[r.act]}"><title>{r.start:.0f}–{r.end:.0f} s: {html.escape(NAMES[r.act])}</title></rect>')
        if r.measured:
            h = 60 * min(float(r.displacement), 2.) / 2
            bars.append(f'<rect x="{x(r.start):.1f}" y="{100 - h:.1f}" width="{max(x(r.end) - x(r.start) - 1, 1):.1f}" '
                        f'height="{h:.1f}" fill="#5f6368"><title>{r.displacement:.2f} tamaños</title></rect>')
        else:
            bars.append(f'<rect x="{x(r.start):.1f}" y="96" width="{max(x(r.end) - x(r.start) - 1, 1):.1f}" height="4" '
                        f'fill="#d93025"><title>sin medida</title></rect>')
    ticks = ''.join(f'<text x="{x(t):.1f}" y="118" font-size="11">{t:.0f}s</text>'
                    for t in np.arange(0, end, 10 if end <= 120 else 30))
    rows = ''.join(f'<tr><td><span class="dot" style="background:{COLOURS[c]}"></span>{html.escape(NAMES[c])}</td>'
                   f'<td>{v["seconds"]:.0f} s</td><td>{(v["share_of_visible"] or 0) * 100:.0f} %</td>'
                   f'<td>{"sí" if v["validated"] else "<b>no</b>"}</td></tr>' for c, v in summary['time_by_act'].items())
    d = summary['displacement']
    per = d['per_measured_minute']['estimate']
    caveats = ''.join(f'<li>{html.escape(c)}</li>' for c in summary['caveats'])
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{font-family:-apple-system,system-ui,sans-serif;max-width:1040px;margin:24px auto;padding:0 16px;color:#202124;background:#fff}}
table{{border-collapse:collapse}} td,th{{padding:4px 12px;border-bottom:1px solid #e0e0e0;text-align:left}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}} svg{{width:100%;height:auto}}
.muted{{color:#5f6368}}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="muted">Duración {summary['duration_seconds']:.0f} s · visible {summary['visible_seconds']:.0f} s ·
desplazamiento {d['body_units']['estimate']:.1f} tamaños corporales ({(per or 0):.1f} por minuto medido) ·
tumbado más largo {summary['longest_lying_bout_seconds']:.0f} s</p>
<svg viewBox="0 0 {width} 124" role="img" aria-label="Actos y desplazamiento por ventana de 2 s">{''.join(acts)}
<g>{''.join(bars)}</g><line x1="0" y1="100" x2="{width}" y2="100" stroke="#9aa0a6"/>{ticks}</svg>
<p class="muted">Arriba: acto de cada ventana de 2 s. Abajo: desplazamiento neto respecto al suelo (gris; tope en 2 tamaños;
rojo: sin medida).</p>
<table><tr><th>Acto</th><th>Tiempo</th><th>Del tiempo visible</th><th>Validado en Cinna</th></tr>{rows}</table>
<h2>Límites</h2><ul>{caveats}</ul></body></html>"""


class Stage:
    """Carga los modelos solo cuando hacen falta y los libera al cambiar de fase (memoria del Mac)."""

    def __init__(self, device):
        self.device, self.models = device, {}

    def get(self, name):
        if name not in self.models:
            if name == 'detector':
                from .ytacts import Detector
                self.models[name] = Detector(self.device)
            elif name == 'encoder':
                from .actmodel import VJEPA
                self.models[name] = VJEPA(self.device)
            elif name == 'tracker':
                from .pointtrack import CoTracker, smoke
                self.models[name] = CoTracker(self.device)
                smoke(self.models[name])
        return self.models[name]

    def drop(self, *names):
        import gc
        for name in names:
            self.models.pop(name, None)
        gc.collect()
        try:
            import torch
            if self.device == 'mps':
                torch.mps.empty_cache()
        except ImportError:
            pass


def analyze(videos, out=OUT, device='mps', stage=None, segment_fn=None, model_file=None):
    """Análisis completo de cada vídeo, por fases para no tener todos los modelos a la vez en memoria."""
    from . import activity, pointtrack
    stage = stage or Stage(device)
    out = Path(out)
    videos = [Path(v).resolve() for v in videos]
    masks = {v: autosegment(v, out / v.stem / 'segmentacion', stage.get('detector'), device, segment_fn=segment_fn)
             for v in videos}
    stage.drop('detector')
    for v in videos:
        times = pointtrack.frame_times(v, out / v.stem / 'tiempos')
        pointtrack.track_case(v, masks[v], out / v.stem / 'tramos', stage.get('tracker'), None, times)
        activity.measure(v, masks[v], out / v.stem / 'actividad', out / v.stem / 'tramos', None)
    stage.drop('tracker')
    model_file = Path(model_file or REPO / 'benchmarks' / 'ytactos_modelo.json')
    frozen = json.loads(model_file.read_text())
    summaries = {}
    for v in videos:
        times = pointtrack.frame_times(v, out / v.stem / 'tiempos')
        acts = act_windows(v, times, stage.get('detector'), stage.get('encoder'), frozen)
        summaries[v.stem] = report(v, out / v.stem, acts, out / v.stem / 'actividad', model_file)
        print(f'{v.name}: informe en {out / v.stem / "informe.html"}', flush=True)
    return summaries


def sheets(videos, out, detector):
    """Hojas de movimiento de cada ventana de 2 s, con la caja del detector y sin ninguna predicción de actos:
    las etiquetas se ponen antes de ver lo que dice el modelo."""
    from .acts import MOTION, sheet_line, square, write_sheets
    from .pointtrack import frame_times
    from .ytacts import probe_frames, window_detection
    out = Path(out)
    (out / 'hojas').mkdir(parents=True, exist_ok=True)
    rows = []
    for video in [Path(v).resolve() for v in videos]:
        times, lines = frame_times(video), []
        for k, frames, images in stream_windows(video, times):
            if len(frames) < 2:
                continue
            summary = window_detection(detector([images[f] for f in probe_frames(frames)]))
            box = square_around(*summary['union'], images[frames[0]].shape) if summary['union'] else None
            start = k * AUTO['window']
            line, _ = sheet_line([square(images[f], box, MOTION['size']) for f in frames], f'{start:.0f}s', box is not None)
            lines.append(line)
            rows.append(dict(video=video.stem, start=round(start, 3), end=round(min(start + AUTO['window'], times[-1]), 3),
                             sheet=f'{video.stem}_{(len(lines) - 1) // MOTION["rows"]:02d}',
                             row=(len(lines) - 1) % MOTION['rows'], found=summary['found']))
        write_sheets(lines, out / 'hojas', video.stem)
        print(f'{video.name}: {len(lines)} ventanas', flush=True)
    pd.DataFrame(rows).to_csv(out / 'ventanas.csv', index=False)
    return len(rows)


def score(analysis, labels, out):
    """Predicciones de `analyze` frente a etiquetas puestas antes de verlas (prueba de comer, fijada el 2026-10-31)."""
    from .acts import EVALUABLE
    from .actmodel import CLASSES, CRITERION, GROUPS
    from .ytacts import metrics
    labels_table = pd.read_csv(labels, dtype={'video': str})
    labels_table['start'] = labels_table.start.round(3)
    parts = []
    for video in sorted(set(labels_table.video)):
        predicted = pd.read_csv(Path(analysis) / video / 'ventanas.csv')
        predicted['start'] = predicted.start.round(3)
        parts.append(labels_table[labels_table.video == video].merge(
            predicted[['start', 'act']].rename(columns=dict(act='model')), on='start', how='inner'))
    table = pd.concat(parts, ignore_index=True)
    table = table[table.act.isin(GROUPS)].assign(truth=lambda t: t.act.map(GROUPS))
    result = metrics(table.truth.to_numpy(), table.model.to_numpy())
    eating = table[table.truth == 'eating']
    hits = int((eating.model == 'eating').sum())
    evaluable = len(eating) >= EVALUABLE['min_disjoint_windows']
    result.update(eating_windows=int(len(eating)), eating_hits=hits, eating_recall_interval=wilson(hits, len(eating)),
                  eating_videos=int(eating.video.nunique()), predicted_as={str(k): int(v) for k, v in
                                                                          eating.model.value_counts().items()},
                  classes_present=sorted(set(table.truth)), criterion=dict(min_recall=CRITERION['min_recall'],
                                                                           min_windows=EVALUABLE['min_disjoint_windows']),
                  verdict=('sin_veredicto' if not evaluable else
                           'comer_reconocido' if hits / len(eating) >= CRITERION['min_recall'] else 'comer_no_reconocido'),
                  scope=('Vídeos nuevos que el modelo no ha visto; etiquetas de Claude puestas antes de las predicciones. '
                         'Menos de 3 sesiones: no sustituye a la prueba con cámara fija.'),
                  labels_sha256=sha256(labels), classes=CLASSES)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'predicciones.csv', index=False)
    (out / 'prueba.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    low, high = result['eating_recall_interval']
    print(f"Comer: {hits}/{len(eating)} ventanas reconocidas ({low:.2f}–{high:.2f}) → {result['verdict']}", flush=True)
    return result


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((centre - half) / d), float((centre + half) / d))


def mask_agreement(manual, auto, every=5):
    """IoU mediana (fotogramas con ambas máscaras, uno de cada `every`) y fracción de fotogramas con máscara."""
    files = sorted((Path(manual) / 'masks').glob('*.png'))
    ious, have_manual, have_auto = [], 0, 0
    for path in files[::every]:
        a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(Path(auto) / 'masks' / path.name), cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        a = a > 0
        b = np.zeros_like(a) if b is None else cv2.resize(b, a.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        have_manual += a.mean() >= .001
        have_auto += b.mean() >= .001
        if a.mean() >= .001 and b.mean() >= .001:
            ious.append((a & b).sum() / (a | b).sum())
    n = max(len(files[::every]), 1)
    return dict(iou_median=float(np.median(ious)) if ious else None, frames=len(ious),
                coverage_manual=have_manual / n, coverage_auto=have_auto / n)


def validate(manifest, labels, out, device='mps', work=REPO / 'runs' / 'auto', stage=None, segment_fn=None):
    """Segmentación automática frente a semilla manual en los vídeos ya analizados (criterio ADOPT_AUTO)."""
    from . import activity, pointtrack
    from .actmodel import GROUPS, cases_with_windows
    manifest, out, work = Path(manifest).resolve(), Path(out), Path(work)
    acts = pd.read_csv(labels, dtype={'case': str})
    acts['start'] = acts.start.round(3)
    cases = list(cases_with_windows(manifest, acts))
    stage = stage or Stage(device)
    auto = {name: autosegment(video, work / name / 'segmentacion', stage.get('detector'), device, segment_fn=segment_fn)
            for name, video, _, _ in cases}
    stage.drop('detector')
    rows, agreement = [], {}
    for name, video, manual, _ in cases:
        times = pointtrack.frame_times(video, pointtrack.TRACKS / 'tiempos')
        pointtrack.track_case(video, auto[name], work / name / 'tramos', stage.get('tracker'), None, times)
        agreement[name] = mask_agreement(manual, auto[name])
        before = pd.DataFrame(activity.step_rows(times, pointtrack.TRACKS / video.stem, activity.NET_STEP))
        after = pd.DataFrame(activity.step_rows(times, work / name / 'tramos', activity.NET_STEP))
        for b, a in zip(before.itertuples(), after.itertuples()):
            if b.interval_seconds > 0:
                rows.append(dict(case=name, start=round(b.seconds - b.interval_seconds, 3), manual=b.displacement,
                                 manual_measured=b.measured, auto=a.displacement, auto_measured=a.measured))
        print(f'{name}: IoU {agreement[name]["iou_median"]}', flush=True)
    stage.drop('tracker')
    table = pd.DataFrame(rows).merge(acts[['case', 'start', 'act']], on=['case', 'start'], how='inner')
    both = table[table.manual_measured & table.auto_measured]
    difference = (both.auto - both.manual).abs()
    scored = table[table.auto_measured & table.act.isin(GROUPS)]
    auc = pointtrack.auc(scored.auto, scored.act == 'locomotion')
    ious = [a['iou_median'] for a in agreement.values() if a['iou_median'] is not None]
    coverage_drop = float(np.mean([a['coverage_manual'] - a['coverage_auto'] for a in agreement.values()]))
    result = dict(criterion=ADOPT_AUTO, mask_iou_median=float(np.median(ious)) if ious else None,
                  mask_coverage_drop=coverage_drop, displacement_blocks=int(len(both)),
                  displacement_median_difference=float(difference.median()) if len(both) else None, auc=auc,
                  coverage=dict(manual=float(table.manual_measured.mean()), auto=float(table.auto_measured.mean())),
                  per_video=agreement, labels_sha256=sha256(labels), source_sha256=sha256(Path(__file__)))
    result['adopt_auto'] = bool(result['mask_iou_median'] is not None and len(both) and
                                result['mask_iou_median'] >= ADOPT_AUTO['mask_iou_median'] and
                                coverage_drop <= ADOPT_AUTO['mask_coverage_drop'] and
                                result['displacement_median_difference'] <= ADOPT_AUTO['displacement_median_difference'] and
                                auc is not None and auc >= ADOPT_AUTO['auc_min'])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'bloques.csv', index=False)
    (out / 'validacion.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    print(f"IoU mediana {result['mask_iou_median']} | caída de cobertura {coverage_drop:.3f} | diferencia de "
          f"desplazamiento {result['displacement_median_difference']} | AUC {auc} → "
          f"{'se adopta' if result['adopt_auto'] else 'no se adopta'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Análisis automático de vídeos de Cinna (sin intervención)')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('analyze', help='Informe de uno o varios vídeos')
    command.add_argument('videos', nargs='+')
    command.add_argument('--out', default=str(OUT))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command = sub.add_parser('validate', help='Automático frente a semilla manual (ADOPT_AUTO)')
    command.add_argument('--manifest', required=True)
    command.add_argument('--labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_ventanas_claude_v2.csv'))
    command.add_argument('--out', default=str(REPO / 'cierre' / 'auto'))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command = sub.add_parser('sheets', help='Hojas de movimiento para etiquetar, sin predicciones')
    command.add_argument('videos', nargs='+')
    command.add_argument('--out', default=str(REPO / 'runs' / 'hojas_nuevas'))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command = sub.add_parser('score', help='Predicciones de analyze frente a etiquetas previas')
    command.add_argument('--analysis', default=str(OUT))
    command.add_argument('--labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_comiendo_claude.csv'))
    command.add_argument('--out', default=str(REPO / 'cierre' / 'comiendo'))
    args = parser.parse_args()
    if args.command == 'sheets':
        from .ytacts import Detector
        sheets(args.videos, args.out, Detector(args.device))
    elif args.command == 'score':
        score(args.analysis, args.labels, args.out)
    elif args.command == 'analyze':
        analyze(args.videos, args.out, args.device)
    else:
        validate(args.manifest, args.labels, args.out, args.device)


if __name__ == '__main__':
    main()
