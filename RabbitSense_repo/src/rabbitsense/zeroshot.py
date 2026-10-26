"""Locomoción por clasificación vídeo-texto zero-shot (X-CLIP), sin entrenamiento ni ajuste con los vídeos evaluados.

Cada intervalo de 0,5 s se clasifica con una ventana de 8 fotogramas recortados alrededor de la guía. Las frases,
el recorte y la banda de decisión están fijados en este archivo antes de ejecutar; el veredicto usa los mismos
criterios que el resto de métodos (closure.CRITERIA).
"""
import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .closure import verdict
from .evaluate import clip_metrics
from .motion import sha256, timeline
from .tracking import load_anchors

MODEL_ID = 'microsoft/xclip-base-patch32'
PROMPTS = {'locomotion': ['a rabbit hopping across the floor', 'a rabbit walking across the room',
                          'a rabbit running'],
           'stationary': ['a rabbit sitting still', 'a rabbit lying down resting',
                          'a rabbit sitting and looking around', 'a rabbit grooming itself']}
FRAMES, SPACING, LEAD = 8, .25, 1.25  # ventana de 1,75 s: desde t-1,25 hasta t+0,5
BAND = (.4, .6)  # p(locomoción) <= 0,4: bajo movimiento; >= 0,6: locomoción; entre ambas: desconocido
CROP_SCALE, SIZE = 2., 224


def crop(image, box, scale=CROP_SCALE, size=SIZE):
    """Cuadrado de lado scale·max(ancho, alto) centrado en la guía; el borde se replica fuera de la imagen."""
    x, y, w, h = box
    side = max(2, int(round(scale * max(w, h))))
    x0, y0 = int(round(x + w / 2 - side / 2)), int(round(y + h / 2 - side / 2))
    pad = max(0, -x0, -y0, x0 + side - image.shape[1], y0 + side - image.shape[0])
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else image
    patch = padded[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return cv2.cvtColor(cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def sample_crops(video, anchors_path):
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f'No se puede abrir el vídeo: {video}')
        fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Metadatos de vídeo inválidos.')
        anchors = load_anchors(anchors_path, int(capture.get(3)), int(capture.get(4)))
        wanted = {}
        for k in range(int((count - 1) / fps / SPACING) + 1):
            wanted.setdefault(int(round(k * SPACING * fps)), k)
        crops = {}
        for frame in range(count):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f'Error decodificando fotograma {frame}/{count}.')
            if frame in wanted:
                exact, box = anchors.at_frame(frame, fps)
                box = box if exact else anchors.interpolate(frame / fps)
                crops[wanted[frame]] = None if box is None else crop(image, box)
    finally:
        capture.release()
    return crops, fps, count


def classify(crops, fps, count, scorer, step=.5, batch=8):
    stride = max(1, round(fps * step))
    grid = [f for f in range(count) if f % stride == 0 or f == count - 1]
    rows, pending = [], []
    for index, frame in enumerate(grid):
        t = frame / fps
        row = dict(seconds=t, interval_seconds=t - grid[index - 1] / fps if index else 0., movement='unknown',
                   reason='first_sample' if index == 0 else 'window_outside_video', p_locomotion=None,
                   evidence_quality=0.)
        rows.append(row)
        if index == 0:
            continue
        first = int(round((t - LEAD) / SPACING))
        keys = range(first, first + FRAMES)
        if first < 0 or any(k not in crops for k in keys):
            continue
        if any(crops[k] is None for k in keys):
            row['reason'] = 'guide_missing'
            continue
        pending.append((row, np.stack([crops[k] for k in keys])))
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        for (row, _), p in zip(chunk, scorer([clip for _, clip in chunk])):
            p = float(p)
            movement, reason = (('locomotion_candidate', 'zero_shot_locomotion') if p >= BAND[1] else
                                ('low_motion_candidate', 'zero_shot_stationary') if p <= BAND[0] else
                                ('unknown', 'zero_shot_ambiguous'))
            row.update(movement=movement, reason=reason, p_locomotion=p, evidence_quality=abs(2 * p - 1))
    return rows


class XClipScorer:
    """p(locomoción) = suma del softmax sobre las frases de locomoción frente a todas las frases."""

    def __init__(self, model_id=MODEL_ID, device='cpu'):
        import torch
        import transformers
        from transformers import XCLIPModel, XCLIPProcessor
        self.torch, self.device = torch, device
        self.processor = XCLIPProcessor.from_pretrained(model_id)
        self.model = XCLIPModel.from_pretrained(model_id).to(device).eval()
        self.texts = PROMPTS['locomotion'] + PROMPTS['stationary']
        self.info = dict(model_id=model_id, revision=getattr(self.model.config, '_commit_hash', None),
                         torch=torch.__version__, transformers=transformers.__version__, device=device)

    def __call__(self, clips):
        inputs = self.processor(text=self.texts, videos=[list(clip) for clip in clips], return_tensors='pt',
                                padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits_per_video
        probabilities = logits.softmax(dim=-1).cpu().numpy()
        return probabilities[:, :len(PROMPTS['locomotion'])].sum(axis=1)


def evaluate(manifest, out, scorer):
    manifest, out = Path(manifest).resolve(), Path(out)
    suite, inputs = {}, {}
    for case in json.loads(manifest.read_text())['cases']:
        paths = {key: (manifest.parent / Path(case[key]).expanduser()).resolve() for key in ('video', 'anchors', 'labels')}
        destination = out / case['name']
        destination.mkdir(parents=True, exist_ok=True)
        rows = classify(*sample_crops(paths['video'], paths['anchors']), scorer)
        frame = pd.DataFrame(rows)
        frame.to_csv(destination / 'motion.csv', index=False)
        (destination / 'timeline.json').write_text(json.dumps(timeline(rows), indent=2))
        clips = clip_metrics(frame, pd.read_csv(paths['labels']), paths['video'].name)
        suite[case['name']] = dict(motion=dict(xclip=dict(legacy_clips=clips)))
        inputs[case['name']] = {key: dict(path=str(path), sha256=sha256(path)) for key, path in paths.items()}
        print(case['name'], frame.movement.value_counts().to_dict(), flush=True)
    result = verdict(suite)
    result.update(inputs=inputs, manifest_sha256=sha256(manifest),
                  source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                  method=dict(prompts=PROMPTS, frames=FRAMES, spacing_seconds=SPACING, lead_seconds=LEAD, band=BAND,
                              crop_scale=CROP_SCALE, size=SIZE, scorer=getattr(scorer, 'info', None),
                              python=platform.python_version(), opencv=cv2.__version__),
                  clips={name: case['motion']['xclip']['legacy_clips'] for name, case in suite.items()})
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False))
    return result


def run(manifest, labels, out, scorer, informative=None):
    from .handheld import relabel, sufficiency
    report = sufficiency(labels)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['sufficient']:
        raise SystemExit('NO SE EJECUTA: clips verificados insuficientes.')
    out = Path(out)
    result = evaluate(manifest, out / 'decisive', scorer)
    result['test_set'] = 'independent_handheld_IMG_5190-5193'
    (out / 'decisive' / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False))
    r = result['methods']['xclip']
    print(f"\nX-CLIP test independiente: locomoción {r['locomotion_hits']}/{r['locomotion_clips']}, "
          f"reposo {r['resting_correct']}/{r['resting_clips']}, episodios falsos {r['resting_false_events']} → "
          f"{'PASA' if r['passed'] else 'NO PASA'}\nEstado: {result['status']}")
    if informative:
        relabel(out / 'informative' / 'verdict.json', evaluate(informative, out / 'informative', scorer),
                'informative_not_decisive', report, 'development_videos')
    return result


def main():
    parser = argparse.ArgumentParser(description='Locomoción zero-shot con X-CLIP; sin entrenamiento')
    for name in ('manifest', 'labels', 'out'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--informative', help='Manifest de desarrollo; resultado siempre informativo')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    run(args.manifest, args.labels, args.out, XClipScorer(device=args.device), args.informative)


if __name__ == '__main__':
    main()
