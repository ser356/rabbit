"""Segmentación solo donde se usa: un fotograma por tramo de CoTracker (fijada el 2026-10-31, antes de ejecutar).

En la cadena automática, las máscaras solo sirven para el fotograma de consulta de cada tramo de 4 s (cada 2 s):
colocar los puntos del conejo y del fondo y medir su tamaño (√área). Segmentar con SAM2 los otros ~59 de cada 60
fotogramas cuesta minutos y no se usa. Aquí, en el fotograma central de cada tramo (o, si ahí no se ve al conejo, en
el más cercano de hasta 3 fotogramas de seguimiento a cada lado):
1. OWLv2 busca al conejo (puntuación ≥ 0,3 y uno solo, como en `auto`);
2. SAM2 de imagen segmenta con esa caja como indicación: el mismo `sam2.1_s.pt` en PyTorch (ultralytics) o el SAM2
   de Core ML que publica Apple (`apple/coreml-sam2.1-small`);
3. se guarda masks/<fotograma>.png solo para ese fotograma; `track_case` busca la máscara más cercana al centro, así
   que encuentra esta.

Cada fotograma se segmenta desde cero: si SAM2 «se pierde» en un fotograma, no arrastra el error al siguiente.

`validate` compara, en los 17 vídeos, el desplazamiento con estas máscaras frente al de semilla manual (criterio
ADOPT_QUERY, fijado antes de ejecutar). Sustituye a `auto validate`. Si se adopta, `analyze` la usa por defecto.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .motion import sha256

REPO = Path(__file__).resolve().parents[2]
SAM_COREML = REPO / 'weights' / 'coreml' / 'sam2'
RECORD = REPO / 'cierre' / 'segmentacion_rapida' / 'validacion.json'
QUERY = dict(neighbours=3, side=1024, score=.3)
# Se adopta si, frente a la semilla manual en los 17 vídeos: la IoU mediana de las máscaras en los fotogramas de
# consulta es ≥ 0,7; el desplazamiento neto de 2 s difiere en mediana ≤ 0,15 tamaños; su AUC (desplazarse frente al
# resto, etiquetas de Claude) es ≥ 0,9; y la cobertura de bloques medidos no baja más de 0,05.
ADOPT_QUERY = dict(mask_iou_median=.7, displacement_median_difference=.15, auc_min=.9, coverage_drop=.05)


class TorchSAM:
    """SAM2 de imagen con una caja (x0, y0, x1, y1): el mismo modelo que la segmentación de vídeo, vía ultralytics."""

    def __init__(self, model=REPO / 'weights' / 'sam2.1_s.pt', device='mps'):
        from ultralytics import SAM
        self.model, self.device = SAM(str(model)), device
        self.info = dict(backend='torch', model=str(Path(model).name), device=device)

    def __call__(self, image, box):
        result = self.model(image, bboxes=[list(map(float, box))], device=self.device, verbose=False)[0]
        if result.masks is None or not len(result.masks.data):
            return np.zeros(image.shape[:2], np.uint8)
        mask = result.masks.data[0].cpu().numpy().astype(np.uint8) * 255
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        return mask


class CoreMLSAM:
    """SAM2 de Core ML de Apple (codificador de imagen, de indicaciones y decodificador de máscaras).

    Los nombres de entradas y salidas se leen de los propios modelos; la caja se da como dos esquinas con las etiquetas
    2 y 3, como en SAM2, en coordenadas de la imagen cuadrada de entrada."""

    def __init__(self, folder=SAM_COREML, compute='all', models=None):
        if models is None:
            from .coreml import load
            found = {kind: sorted(Path(folder).rglob(f'*{kind}*.mlpackage')) for kind in
                     ('ImageEncoder', 'PromptEncoder', 'MaskDecoder')}
            missing = [k for k, v in found.items() if not v]
            if missing:
                raise FileNotFoundError(f'Faltan modelos Core ML de SAM2 en {folder}: {missing}')
            models = {k: load(v[0], compute) for k, v in found.items()}
        self.encoder, self.prompt, self.decoder = models['ImageEncoder'], models['PromptEncoder'], models['MaskDecoder']
        spec = self.encoder.get_spec().description.input[0]
        self.image_input = spec.name
        self.size = int(spec.type.imageType.width) if spec.type.WhichOneof('Type') == 'imageType' else QUERY['side']
        self.image_is_pixels = spec.type.WhichOneof('Type') == 'imageType'
        names = [i.name for i in self.prompt.get_spec().description.input]
        self.points_input = next(n for n in names if 'point' in n.lower() or 'coord' in n.lower())
        self.labels_input = next(n for n in names if n != self.points_input)
        self.decoder_inputs = [i.name for i in self.decoder.get_spec().description.input]
        self.info = dict(backend='coreml', compute=compute, size=self.size)

    def __call__(self, image, box):
        h, w = image.shape[:2]
        square = cv2.resize(np.ascontiguousarray(image[..., ::-1]), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if self.image_is_pixels:
            from PIL import Image
            encoded = self.encoder.predict({self.image_input: Image.fromarray(square)})
        else:
            encoded = self.encoder.predict({self.image_input: square.transpose(2, 0, 1)[None].astype(np.float32)})
        x0, y0, x1, y1 = box
        sx, sy = self.size / w, self.size / h
        points = np.array([[[x0 * sx, y0 * sy], [x1 * sx, y1 * sy]]], np.float32)
        prompts = self.prompt.predict({self.points_input: points, self.labels_input: np.array([[2, 3]], np.float32)})
        available = {**encoded, **prompts}
        missing = [n for n in self.decoder_inputs if n not in available]
        if missing:
            raise KeyError(f'El decodificador de SAM2 espera {missing}; hay {sorted(available)}')
        output = self.decoder.predict({n: available[n] for n in self.decoder_inputs})
        arrays = {k: np.asarray(v) for k, v in output.items()}
        masks = next(v for v in arrays.values() if v.ndim == 4)
        scores = next((v for v in arrays.values() if v.ndim == 2), np.zeros((1, masks.shape[1])))
        best = masks[0, int(np.argmax(scores[0]))].astype(np.float32)
        return (cv2.resize(best, (w, h), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8) * 255


def query_frames(video, times):
    """Por tramo de CoTracker: candidatos a fotograma de consulta, del central hacia fuera (como track_case)."""
    from . import pointtrack
    from .auto import video_info
    fps = video_info(video)[0]
    step = max(1, int(round(fps / pointtrack.TARGET_FPS)))
    plans = []
    for k, start in enumerate(pointtrack.chunk_starts(pointtrack.duration(times))):
        indices = pointtrack.chunk_frames(times, start, step)
        if not indices:
            continue
        centre = int(np.argmin([abs(times[f] - (start + pointtrack.CHUNK / 2)) for f in indices]))
        order = sorted(range(len(indices)), key=lambda j: abs(j - centre))[:2 * QUERY['neighbours'] + 1]
        plans.append(dict(chunk=k, candidates=[indices[j] for j in order]))
    return plans


def segment_queries(video, out, detector, segmenter, times=None):
    """Máscaras solo en los fotogramas de consulta; devuelve la carpeta con masks/ (el resto de fotogramas, sin archivo)."""
    from .acts import read_small
    from .auto import usable, video_info
    from .pointtrack import MIN_MASK, frame_times
    video, out = Path(video).resolve(), Path(out)
    record = out / 'consultas.json'
    if record.is_file():
        stored = json.loads(record.read_text())
        if stored.get('video_sha256') == sha256(video) and stored.get('segmenter') == segmenter.info:
            return out
    times = frame_times(video) if times is None else times
    _, _, width, height = video_info(video)
    plans = query_frames(video, times)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    for old in (out / 'masks').glob('*.png'):
        old.unlink()
    images = read_small(video, {f for p in plans for f in p['candidates']}, QUERY['side'])
    rows, started = [], time.perf_counter()
    for plan in plans:
        row = dict(chunk=plan['chunk'], frame=None, score=np.nan, mask_fraction=0.)
        for f in plan['candidates']:
            if f >= len(images) or images[f] is None:
                continue
            image = images[f]
            found = detector([image])[0]
            best = found[0] if found else None
            candidate = dict(box=[best[1], best[2], best[3] - best[1], best[4] - best[2]] if best else None,
                             score=best[0] if best else np.nan, multiple=len(found) > 1)
            if not usable(candidate):
                continue
            mask = segmenter(image, best[1:])
            if np.count_nonzero(mask) < MIN_MASK * mask.size:
                continue
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / 'masks' / f'{f:06d}.png'), mask)
            row.update(frame=int(f), score=float(best[0]), mask_fraction=float(np.count_nonzero(mask) / mask.size))
            break
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'consultas.csv', index=False)
    record.write_text(json.dumps(dict(video=str(video), video_sha256=sha256(video), segmenter=segmenter.info,
                                      detector=getattr(detector, 'info', None), criterion=QUERY, chunks=len(rows),
                                      with_mask=sum(r['frame'] is not None for r in rows),
                                      seconds=time.perf_counter() - started), indent=2, default=float))
    return out


def make_segmenter(kind, device='mps'):
    return CoreMLSAM() if kind == 'coreml' else TorchSAM(device=device)


def adopted(record=RECORD):
    record = Path(record)
    return record.is_file() and json.loads(record.read_text()).get('adopt_query', False)


def validate(manifest, labels, out, device='mps', segmenter='torch', work=REPO / 'runs' / 'consultas', detector=None,
             sam=None, tracker=None):
    """Máscaras de consulta frente a la semilla manual en los vídeos ya analizados (ADOPT_QUERY)."""
    from . import activity, pointtrack
    from .actmodel import GROUPS, cases_with_windows
    from .auto import mask_agreement
    manifest, out, work = Path(manifest).resolve(), Path(out), Path(work)
    acts = pd.read_csv(labels, dtype={'case': str})
    acts['start'] = acts.start.round(3)
    if detector is None:
        from .ytacts import Detector
        detector = Detector(device)
    sam = sam or make_segmenter(segmenter, device)
    cases = list(cases_with_windows(manifest, acts))
    masks, timing = {}, {}
    for name, video, _, _ in cases:
        times = pointtrack.frame_times(video, pointtrack.TRACKS / 'tiempos')
        masks[name] = segment_queries(video, work / name / 'segmentacion', detector, sam, times)
        timing[name] = json.loads((masks[name] / 'consultas.json').read_text())['seconds']
        print(f'{name}: máscaras de consulta en {timing[name]:.0f} s', flush=True)
    detector = sam = None
    if tracker is None:
        tracker = pointtrack.CoTracker(device)
        pointtrack.smoke(tracker)
    rows, agreement = [], {}
    for name, video, manual, _ in cases:
        times = pointtrack.frame_times(video, pointtrack.TRACKS / 'tiempos')
        pointtrack.track_case(video, masks[name], work / name / 'tramos', tracker, None, times)
        agreement[name] = mask_agreement(manual, masks[name], every=1, only_auto=True)
        before = pd.DataFrame(activity.step_rows(times, pointtrack.TRACKS / video.stem, activity.NET_STEP))
        after = pd.DataFrame(activity.step_rows(times, work / name / 'tramos', activity.NET_STEP))
        for b, a in zip(before.itertuples(), after.itertuples()):
            if b.interval_seconds > 0:
                rows.append(dict(case=name, start=round(b.seconds - b.interval_seconds, 3), manual=b.displacement,
                                 manual_measured=b.measured, query=a.displacement, query_measured=a.measured))
    table = pd.DataFrame(rows).merge(acts[['case', 'start', 'act']], on=['case', 'start'], how='inner')
    both = table[table.manual_measured & table.query_measured]
    difference = (both['query'] - both.manual).abs()
    scored = table[table.query_measured & table.act.isin(GROUPS)]
    auc = pointtrack.auc(scored['query'], scored.act == 'locomotion')
    ious = [a['iou_median'] for a in agreement.values() if a['iou_median'] is not None]
    coverage = dict(manual=float(table.manual_measured.mean()), query=float(table.query_measured.mean()))
    result = dict(criterion=ADOPT_QUERY, mask_iou_median=float(np.median(ious)) if ious else None,
                  displacement_blocks=int(len(both)),
                  displacement_median_difference=float(difference.median()) if len(both) else None, auc=auc,
                  coverage=coverage, segmentation_seconds=timing, per_video=agreement,
                  segmenter=segmenter, labels_sha256=sha256(labels), source_sha256=sha256(Path(__file__)))
    result['adopt_query'] = bool(result['mask_iou_median'] is not None and len(both) and
                                 result['mask_iou_median'] >= ADOPT_QUERY['mask_iou_median'] and
                                 result['displacement_median_difference'] <= ADOPT_QUERY['displacement_median_difference'] and
                                 auc is not None and auc >= ADOPT_QUERY['auc_min'] and
                                 coverage['manual'] - coverage['query'] <= ADOPT_QUERY['coverage_drop'])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'bloques.csv', index=False)
    (out / 'validacion.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    print(f"IoU mediana {result['mask_iou_median']} | diferencia de desplazamiento "
          f"{result['displacement_median_difference']} | AUC {auc} | cobertura {coverage['manual']:.2f} → "
          f"{coverage['query']:.2f} | segmentación {sum(timing.values()):.0f} s en total → "
          f"{'se adopta' if result['adopt_query'] else 'no se adopta'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Segmentación solo en los fotogramas de consulta de CoTracker')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('download-coreml', help='Descarga el SAM2 de Core ML de Apple')
    command.add_argument('--repo', default='apple/coreml-sam2.1-small')
    command = sub.add_parser('validate', help='Frente a la semilla manual (ADOPT_QUERY)')
    command.add_argument('--manifest', required=True)
    command.add_argument('--labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_ventanas_claude_v2.csv'))
    command.add_argument('--out', default=str(RECORD.parent))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command.add_argument('--sam', choices=['torch', 'coreml'], default='torch')
    args = parser.parse_args()
    if args.command == 'download-coreml':
        from huggingface_hub import snapshot_download
        print(snapshot_download(args.repo, local_dir=str(SAM_COREML)))
    else:
        validate(args.manifest, args.labels, args.out, args.device, args.sam)


if __name__ == '__main__':
    main()
