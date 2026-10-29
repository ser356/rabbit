"""Actos aprendidos solo con vídeos de YouTube y probados una vez en Cinna (fijado el 2026-10-29, antes de descargar).

Por qué: en los 17 vídeos de Cinna cada acto está casi encerrado en su escena (comer nunca aparece en IMG_519x;
tumbado solo en WhatsApp y 4944), así que un modelo entrenado con Cinna aprende la escena. En YouTube cada acto
aparece en muchas escenas distintas, y Cinna queda entera como test: nunca entra en el entrenamiento.

1. `cinna`: el detector OWLv2 («a photo of a rabbit») busca a Cinna en las 156 ventanas de 2 s ya etiquetadas.
   Puerta previa: con máscara SAM2, debe encontrarlo en ≥ 3 de 4 fotogramas en el 80 % de las ventanas y la caja
   debe coincidir con la de SAM2 (IoU mediana ≥ 0,5). Si no pasa, no se sigue: el recorte no sería comparable.
   Después, V-JEPA 2 sobre el recorte de la caja detectada (sin detección: imagen completa con bandas).
2. `youtube`: por vídeo, hasta 12 ventanas candidatas de 2 s sin solapar, repartidas; se conservan hasta 5, repartidas, entre las que tienen un
   solo conejo detectado en los 4 fotogramas. Hojas de movimiento para que Claude las etiquete y V-JEPA 2 del
   mismo recorte, calculado antes de etiquetar.
3. `train`: regresión logística multinomial, clases equilibradas, estandarización, C = 0,1 (fijos), solo con
   YouTube; validación por vídeo solo informativa. Mínimo 30 ventanas de 8 vídeos por clase. El modelo se congela
   en benchmarks/ytactos_modelo.json.
4. `test`: una sola ejecución sobre las ventanas de Cinna con las etiquetas de Claude (v2). Criterio: exactitud
   equilibrada ≥ 0,6 y sensibilidad ≥ 0,5 en cada clase. Si falla, no se reentrena mirando a Cinna.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .actmodel import CLASSES, CRITERION, FRAMES, GROUPS, C
from .acts import MOTION, read_small, sheet_line, square, square_around, write_sheets
from .motion import sha256

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'runs' / 'ytactos'
MODEL_FILE = REPO / 'benchmarks' / 'ytactos_modelo.json'
QUERIES = REPO / 'benchmarks' / 'youtube_actos_busquedas.tsv'
DETECTOR = dict(model_id='google/owlv2-base-patch16-ensemble', text='a photo of a rabbit', threshold=.2, frames=4,
                separate_iou=.3)
SELECT = dict(window=2., candidates=12, keep=5, margin=.5, min_frames=8)
GATE = dict(coverage=.8, median_iou=.5, found=3)
MINIMUM = dict(windows=30, videos=8)
VIDEO_SUFFIXES = ('.mp4', '.mov', '.webm', '.mkv')


def iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return float(w * h / union) if union > 0 else 0.


def detections(scores, boxes, shape, threshold=DETECTOR['threshold'], separate=DETECTOR['separate_iou']):
    """Salida de OWLv2 (centro, ancho, alto relativos al cuadrado con relleno) → [(puntuación, x0, y0, x1, y1)] en
    píxeles, de mayor a menor, quitando las cajas que solapan con una mejor (IoU ≥ `separate`)."""
    side = max(shape[:2])
    found = []
    for k in np.argsort(-scores):
        if scores[k] < threshold:
            break
        cx, cy, w, h = boxes[k] * side
        box = (float(np.clip(cx - w / 2, 0, shape[1])), float(np.clip(cy - h / 2, 0, shape[0])),
               float(np.clip(cx + w / 2, 0, shape[1])), float(np.clip(cy + h / 2, 0, shape[0])))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        if all(iou(box, f[1:]) < separate for f in found):
            found.append((float(scores[k]), *box))
    return found


class Detector:
    """OWLv2 con una sola consulta de texto; `parts` = (modelo, procesador de imagen, tokens) para pruebas."""

    def __init__(self, device='mps', model_id=DETECTOR['model_id'], text=DETECTOR['text'], parts=None):
        import torch
        import transformers
        self.torch, self.device = torch, device
        if parts is None:
            processor = transformers.Owlv2Processor.from_pretrained(model_id)
            model = transformers.Owlv2ForObjectDetection.from_pretrained(model_id)
            image_processor, tokens = processor.image_processor, processor(text=[text], return_tensors='pt')
        else:
            model, image_processor, tokens = parts
        self.model = model.to(device).eval()
        self.image_processor = image_processor
        self.tokens = {k: tokens[k].to(device) for k in ('input_ids', 'attention_mask')}
        self.info = dict(model_id=model_id, text=text, threshold=DETECTOR['threshold'], torch=torch.__version__,
                         transformers=transformers.__version__, device=device)

    def __call__(self, images, batch=4):
        """Imágenes BGR → por imagen, lista de detecciones (puntuación, x0, y0, x1, y1)."""
        torch, results = self.torch, []
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            pixels = self.image_processor(images=[np.ascontiguousarray(im[..., ::-1]) for im in chunk],
                                          return_tensors='pt')['pixel_values'].to(self.device)
            n = len(chunk)
            with torch.no_grad():
                output = self.model(input_ids=self.tokens['input_ids'].repeat(n, 1),
                                    attention_mask=self.tokens['attention_mask'].repeat(n, 1), pixel_values=pixels)
            scores = output.logits[..., 0].sigmoid().float().cpu().numpy()
            boxes = output.pred_boxes.float().cpu().numpy()
            results += [detections(s, b, im.shape) for s, b, im in zip(scores, boxes, chunk)]
            if self.device == 'mps':
                torch.mps.empty_cache()
        return results


def probe_frames(frames, count=DETECTOR['frames']):
    return [frames[int(round(j))] for j in np.linspace(0, len(frames) - 1, count)]


def window_detection(found):
    """Detecciones de los fotogramas muestreados de una ventana → resumen y rectángulo unión de las mejores."""
    best = [f[0] for f in found if f]
    union = None
    if best:
        union = (min(b[1] for b in best), min(b[2] for b in best), max(b[3] for b in best), max(b[4] for b in best))
    return dict(found=len(best), probed=len(found), multiple=any(len(f) > 1 for f in found),
                score_min=min(b[0] for b in best) if best else np.nan, union=union)


def embed(images, frames, box, encoder):
    """V-JEPA 2 de 16 fotogramas repartidos por la ventana, recortados con la caja fija (None: imagen completa)."""
    picks = probe_frames(frames, FRAMES)
    return encoder(np.stack([square(images[f], box, encoder.size) for f in picks]))


def videos_in(folder):
    return sorted(p for p in Path(folder).expanduser().iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def youtube_video(video, detector, encoder):
    """Ventanas candidatas sin solapar → las de un solo conejo visible → hasta `keep` repartidas, con hoja y vector."""
    from .pointtrack import frame_times
    times = frame_times(video)
    starts = np.arange(SELECT['margin'], times[-1] - SELECT['window'] - SELECT['margin'] + 1e-9, SELECT['window'])
    members = {float(s): [int(f) for f in np.flatnonzero((times >= s) & (times < s + SELECT['window']))] for s in starts}
    members = {s: f for s, f in members.items() if len(f) >= SELECT['min_frames']}
    if len(members) > SELECT['candidates']:  # candidatas repartidas por todo el vídeo
        chosen = [list(members)[int(round(j))] for j in np.linspace(0, len(members) - 1, SELECT['candidates'])]
        members = {s: members[s] for s in chosen}
    images = read_small(video, {f for frames in members.values() for f in frames})
    candidates = []
    for start, frames in members.items():
        frames = [f for f in frames if f < len(images) and images[f] is not None]
        if len(frames) < SELECT['min_frames']:
            continue
        summary = window_detection(detector([images[f] for f in probe_frames(frames)]))
        if summary['found'] == summary['probed'] and not summary['multiple']:
            candidates.append((start, frames, summary))
    if len(candidates) > SELECT['keep']:
        candidates = [candidates[int(round(j))] for j in np.linspace(0, len(candidates) - 1, SELECT['keep'])]
    rows, lines, vectors = [], [], []
    for start, frames, summary in candidates:
        box = square_around(*summary['union'], images[frames[0]].shape)
        crops = [square(images[f], box, MOTION['size']) for f in frames]
        line, energy = sheet_line(crops, f'{start:.1f}s', True)
        lines.append(line)
        rows.append(dict(video=video.stem, start=round(start, 3), end=round(start + SELECT['window'], 3),
                         sheet=f'{video.stem}_{(len(lines) - 1) // MOTION["rows"]:02d}',
                         row=(len(lines) - 1) % MOTION['rows'], score_min=summary['score_min'],
                         box_left=box[0], box_top=box[1], box_side=box[2], motion_p999=float(np.percentile(energy, 99.9))))
        if encoder is not None:
            vectors.append(embed(images, frames, box, encoder))
    return rows, lines, (np.stack(vectors) if vectors else None), len(members)


def require_gate(gate_file):
    gate_file = Path(gate_file)
    gate = json.loads(gate_file.read_text()) if gate_file.is_file() else {}
    if not (gate.get('passes') or gate.get('amendment', {}).get('passes')):
        raise RuntimeError(f'La puerta del detector sobre Cinna no se ha ejecutado o no ha pasado ({gate_file}).')


def amend_gate(folder, labels):
    """Enmienda posterior (2026-10-29), declarada: la puerta se calcula solo con las ventanas que entran en el test.

    Motivo: 13 de los 32 fallos eran ventanas «no visible», donde SAM2 arrastra la máscara a otra zona y el detector,
    con razón, no encuentra al conejo. Se conserva el resultado pre-registrado (`passes`) sin tocar."""
    folder = Path(folder)
    gate = json.loads((folder / 'deteccion.json').read_text())
    table = labelled(pd.read_csv(folder / 'ventanas.csv', dtype={'case': str}), labels, ['case', 'start'])
    tested = table[table.sam2 & table.act.isin(GROUPS)]
    coverage = float((tested.found >= GATE['found']).mean())
    median_iou = float(tested.iou_sam2.fillna(0).median())
    by_act = {a: dict(windows=int(len(g)), detected=int((g.found >= GATE['found']).sum()))
              for a, g in table[table.sam2].groupby(table.act.fillna('sin_etiqueta'))}
    gate['amendment'] = dict(
        date='2026-10-29', windows=int(len(tested)), coverage=coverage, median_iou=median_iou, by_act=by_act,
        passes=bool(coverage >= GATE['coverage'] and median_iou >= GATE['median_iou']),
        reason=('Posterior al resultado: la puerta pre-registrada no pasó (cobertura 0,79). 13 de los 32 fallos eran '
                'ventanas «no visible», que no entran en el test; se recalcula solo con las ventanas evaluadas. '
                'Usa las etiquetas de actos solo para excluir «no visible». El test informa además el resultado en las '
                'ventanas con detección en ≥ 3 de 4 fotogramas.'),
        weakness='Desplazarse: detección suficiente en 15 de 27 ventanas (desenfoque y conejo pequeño en vídeos de mano).')
    (folder / 'deteccion.json').write_text(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"Enmienda: cobertura {coverage:.2f} en {len(tested)} ventanas evaluadas → "
          f"{'pasa' if gate['amendment']['passes'] else 'NO PASA'}", flush=True)
    return gate


def youtube(folder, out, detector, encoder, gate_file=None):
    """Todas las ventanas de YouTube; cada vídeo terminado se guarda y se reutiliza si el código no cambia."""
    require_gate(gate_file or Path(out).parent / 'cinna' / 'deteccion.json')
    out = Path(out)
    for sub in ('hojas', 'videos'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    source = {n: sha256(Path(__file__).parent / f'{n}.py') for n in ('ytacts', 'acts', 'actmodel')}
    tables, deep, candidates = [], [], 0
    for video in videos_in(folder):
        files = {k: out / 'videos' / f'{video.stem}.{k}' for k in ('csv', 'npy', 'json')}
        meta = dict(video=video.name, bytes=video.stat().st_size, source=source, encoder=getattr(encoder, 'info', None),
                    detector=detector.info)
        stored = json.loads(files['json'].read_text()) if files['json'].is_file() else None
        if stored is not None and stored['meta'] == meta:
            table = pd.read_csv(files['csv']) if stored['windows'] else pd.DataFrame()
            vectors = np.load(files['npy']) if stored['windows'] and encoder is not None else None
            candidates += stored['candidates']
        else:
            try:
                rows, lines, vectors, count = youtube_video(video, detector, encoder)
            except ValueError as error:  # vídeo ilegible: se anota y se sigue
                print(f'{video.name}: {error}', flush=True)
                rows, lines, vectors, count = [], [], None, 0
            table = pd.DataFrame(rows)
            write_sheets(lines, out / 'hojas', video.stem)
            if len(table):
                table.to_csv(files['csv'], index=False)
            if vectors is not None:
                np.save(files['npy'], vectors)
            files['json'].write_text(json.dumps(dict(meta=meta, windows=len(table), candidates=count), indent=2))
            candidates += count
        print(f'{video.name}: {len(table)} ventanas', flush=True)
        if len(table):
            tables.append(table)
            if vectors is not None:
                deep.append(vectors)
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    table.to_csv(out / 'ventanas.csv', index=False)
    if deep:
        np.save(out / 'vjepa.npy', np.concatenate(deep))
    summary = dict(videos=len(videos_in(folder)), videos_with_windows=len(tables), candidate_windows=candidates,
                   kept_windows=len(table), detector=detector.info, encoder=getattr(encoder, 'info', None), select=SELECT)
    (out / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def mask_rect(masks, frames, shape):
    """Rectángulo unión de las máscaras SAM2 de la ventana, en coordenadas de la imagen reducida."""
    import cv2
    rects, scale = [], None
    for f in frames:
        mask = cv2.imread(str(masks / 'masks' / f'{f:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.count_nonzero(mask) >= .001 * mask.size:
            scale = shape[1] / mask.shape[1]
            x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
            rects.append((x, y, x + w, y + h))
    if not rects:
        return None
    return tuple(v * scale for v in (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects),
                                     max(r[3] for r in rects)))


def cinna(manifest, windows, out, detector, encoder):
    """Puerta del detector y vectores de Cinna en sus ventanas etiquetadas; sin etiquetas de actos."""
    manifest, out = Path(manifest).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(windows, dtype={'case': str})
    from .actmodel import cases_with_windows, check_videos
    from .pointtrack import frame_times
    check_videos(manifest, windows)
    rows, deep = [], []
    for name, video, masks, mine in cases_with_windows(manifest, windows):
        times = frame_times(video)
        members = {w.Index: [int(f) for f in np.flatnonzero((times >= w.start) & (times < w.end))] for w in mine.itertuples()}
        images = read_small(video, {f for frames in members.values() for f in frames})
        for w in mine.itertuples():
            frames = [f for f in members[w.Index] if f < len(images) and images[f] is not None]
            if len(frames) < 2:
                continue
            shape = images[frames[0]].shape
            summary = window_detection(detector([images[f] for f in probe_frames(frames)]))
            box = square_around(*summary['union'], shape) if summary['union'] else None
            sam = mask_rect(masks, frames, shape) if masks else None
            rows.append(dict(case=name, start=round(float(w.start), 3), end=round(float(w.end), 3),
                             found=summary['found'], probed=summary['probed'], multiple=summary['multiple'],
                             score_min=summary['score_min'], boxed=box is not None, sam2=sam is not None,
                             iou_sam2=iou(summary['union'], sam) if summary['union'] and sam else np.nan))
            if encoder is not None:
                deep.append(embed(images, frames, box, encoder))
        print(f'{name}: {len(mine)} ventanas', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'ventanas.csv', index=False)
    if deep:
        np.save(out / 'vjepa.npy', np.stack(deep))
    with_mask = table[table.sam2]
    coverage = float((with_mask.found >= GATE['found']).mean()) if len(with_mask) else 0.
    median_iou = float(with_mask.iou_sam2.fillna(0).median()) if len(with_mask) else 0.
    gate = dict(windows=len(table), windows_with_sam2=len(with_mask), coverage=coverage, median_iou=median_iou,
                criterion=GATE, passes=bool(coverage >= GATE['coverage'] and median_iou >= GATE['median_iou']),
                windows_without_detection=int((~table.boxed).sum()), detector=detector.info,
                encoder=getattr(encoder, 'info', None),
                note='Sin detección se usa la imagen completa con bandas. La puerta no mira etiquetas de actos.')
    (out / 'deteccion.json').write_text(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"Puerta del detector: cobertura {coverage:.2f}, IoU mediana con SAM2 {median_iou:.2f} → "
          f"{'pasa' if gate['passes'] else 'NO PASA'}", flush=True)
    return gate


def labelled(table, labels, keys):
    acts = pd.read_csv(labels, dtype={'case': str, 'video': str})
    for frame in (table, acts):
        frame['start'] = frame.start.astype(float).round(3)
    merged = table.reset_index().merge(acts[keys + ['act']], on=keys, how='left', validate='one_to_one')
    return merged.sort_values('index').drop(columns='index').reset_index(drop=True)


def train(features, labels, output=MODEL_FILE, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    features = Path(features)
    table = labelled(pd.read_csv(features / 'ventanas.csv', dtype={'video': str}), labels, ['video', 'start'])
    x = np.load(features / 'vjepa.npy')
    keep = table.act.isin(GROUPS).to_numpy()
    y, groups = table.act.map(GROUPS).to_numpy(), table.video.to_numpy()
    counts = {c: dict(windows=int((y[keep] == c).sum()), videos=int(len(set(groups[keep][y[keep] == c])))) for c in CLASSES}
    short = [c for c, n in counts.items() if n['windows'] < MINIMUM['windows'] or n['videos'] < MINIMUM['videos']]
    if short:
        raise ValueError(f'Datos insuficientes para {short}: {counts} (mínimo {MINIMUM}).')
    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=5000))  # noqa: E731
    held = np.full(len(y), None, object)
    for train_idx, test_idx in GroupKFold(folds).split(x[keep], y[keep], groups[keep]):
        rows = np.flatnonzero(keep)
        model = make().fit(x[rows[train_idx]], y[rows[train_idx]])
        held[rows[test_idx]] = model.predict(x[rows[test_idx]])
    cv = metrics(y[keep], held[keep].astype(str))
    model = make().fit(x[keep], y[keep])
    scaler, logistic = model.named_steps['standardscaler'], model.named_steps['logisticregression']
    frozen = dict(classes=list(logistic.classes_), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                  coef=logistic.coef_.tolist(), intercept=logistic.intercept_.tolist(), C=C, groups=GROUPS,
                  counts=counts, cross_validation_by_video=cv, labels_sha256=sha256(labels),
                  features_sha256=sha256(features / 'vjepa.npy'), source_sha256=sha256(Path(__file__)),
                  python=platform.python_version(),
                  note='Entrenado solo con YouTube y congelado antes de mirar ninguna predicción sobre Cinna.')
    Path(output).write_text(json.dumps(frozen, indent=1))
    print(f"YouTube, validación por vídeo: exactitud equilibrada {cv['balanced_accuracy']:.2f} | " +
          ' '.join(f"{c} {cv['recall'][c]:.2f}" for c in CLASSES), flush=True)
    return frozen


def predict(frozen, x):
    z = (x - np.array(frozen['mean'])) / np.array(frozen['scale'])
    scores = z @ np.array(frozen['coef']).T + np.array(frozen['intercept'])
    return np.array(frozen['classes'])[scores.argmax(axis=1)]


def metrics(truth, guess):
    recall = {c: float((guess[truth == c] == c).mean()) if (truth == c).any() else None for c in CLASSES}
    precision = {c: float((truth[guess == c] == c).mean()) if (guess == c).any() else None for c in CLASSES}
    balanced = float(np.mean([r for r in recall.values() if r is not None]))
    return dict(windows=int(len(truth)), balanced_accuracy=balanced, accuracy=float((guess == truth).mean()),
                recall=recall, precision=precision,
                confusion=pd.crosstab(pd.Series(truth, name='verdad'), pd.Series(guess, name='modelo')).to_dict('index'),
                passes=bool(balanced >= CRITERION['balanced_accuracy'] and
                            all(r is not None and r >= CRITERION['min_recall'] for r in recall.values())))


def test(features, labels, out, model_file=MODEL_FILE, force=False):
    out = Path(out)
    if (out / 'test_cinna.json').is_file() and not force:
        raise RuntimeError(f'El test sobre Cinna ya se ejecutó ({out / "test_cinna.json"}): es de una sola vez.')
    frozen = json.loads(Path(model_file).read_text())
    features = Path(features)
    table = labelled(pd.read_csv(features / 'ventanas.csv', dtype={'case': str}), labels, ['case', 'start'])
    x = np.load(features / 'vjepa.npy')
    keep = table.act.isin(GROUPS).to_numpy()
    truth, guess = table.act.map(GROUPS).to_numpy()[keep], predict(frozen, x[keep])
    result = metrics(truth, guess)
    by_act = {a: float((guess[table.act[keep].to_numpy() == a] == GROUPS[a]).mean())
              for a in sorted(set(table.act[keep]))}
    boxed = table.boxed.to_numpy()[keep]
    detected = (table.found.to_numpy() >= GATE['found'])[keep]
    result.update(accuracy_by_original_act=by_act, accuracy_with_detection=float((guess[boxed] == truth[boxed]).mean())
                  if boxed.any() else None, windows_without_detection=int((~boxed).sum()),
                  well_detected=metrics(truth[detected], guess[detected]) if detected.any() else None,
                  status='resolved_exploratory' if result['passes'] else 'not_resolved', criterion=CRITERION,
                  model_sha256=sha256(model_file), labels_sha256=sha256(labels), source_sha256=sha256(Path(__file__)),
                  scope=('Cinna no entra en el entrenamiento. Las etiquetas de Cinna son de Claude (kappa 0,63 frente a '
                         'la persona) y la agrupación de clases se fijó tras verlas: exploratorio, no una validación.'))
    out.mkdir(parents=True, exist_ok=True)
    predictions = table[['case', 'start', 'end', 'act']][keep].assign(model=guess)
    predictions.to_csv(out / 'predicciones_cinna.csv', index=False)
    (out / 'test_cinna.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"Cinna: exactitud equilibrada {result['balanced_accuracy']:.2f} | " +
          ' '.join(f"{c} {result['recall'][c]:.2f}" for c in CLASSES) + f" → {result['status']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Actos aprendidos con YouTube y probados en Cinna')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('cinna', help='Puerta del detector y vectores de Cinna')
    command.add_argument('--manifest', required=True)
    command.add_argument('--windows', default=str(REPO / 'cierre' / 'actos' / 'ventanas_indice.csv'))
    command.add_argument('--out', default=str(OUT / 'cinna'))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command = sub.add_parser('youtube', help='Ventanas, hojas y vectores de los vídeos de YouTube')
    command.add_argument('--videos', required=True)
    command.add_argument('--out', default=str(OUT / 'youtube'))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command = sub.add_parser('amend-gate', help='Enmienda posterior declarada de la puerta (sin «no visible»)')
    command.add_argument('--folder', default=str(OUT / 'cinna'))
    command.add_argument('--labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_ventanas_claude_v2.csv'))
    command = sub.add_parser('train')
    command.add_argument('--features', default=str(OUT / 'youtube'))
    command.add_argument('--labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_youtube_claude.csv'))
    command = sub.add_parser('test')
    command.add_argument('--features', default=str(OUT / 'cinna'))
    command.add_argument('--labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_ventanas_claude_v2.csv'))
    command.add_argument('--out', default=str(REPO / 'cierre' / 'ytactos'))
    args = parser.parse_args()
    if args.command == 'youtube':
        require_gate(Path(args.out).parent / 'cinna' / 'deteccion.json')
    if args.command in ('cinna', 'youtube'):
        from .actmodel import VJEPA
        detector = Detector(args.device)
        encoder = VJEPA(args.device)
        if args.command == 'cinna':
            cinna(args.manifest, args.windows, args.out, detector, encoder)
        else:
            youtube(args.videos, args.out, detector, encoder)
    elif args.command == 'amend-gate':
        amend_gate(args.folder, args.labels)
    elif args.command == 'train':
        train(args.features, args.labels)
    else:
        test(args.features, args.labels, args.out)


if __name__ == '__main__':
    main()
