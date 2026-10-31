"""Resultados de referencia («golden») para los tests de RabbitSenseKit (Swift), sacados del código Python validado.

- `pure`: partes sin modelos (clasificador congelado, detecciones, cajas, recortes, preprocesado de OWLv2,
  desplazamiento con cámara fija). Se generan en cualquier máquina.
- `models` (Mac): fotogramas reales de ventanas de 2 s y lo que dan con ellos OWLv2 y V-JEPA 2 en PyTorch, para
  comprobar que la versión Swift + Core ML da lo mismo.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / 'swift' / 'RabbitSenseKit'
FIXTURES = PACKAGE / 'Tests' / 'RabbitSenseKitTests' / 'Fixtures'
MODEL_FILE = REPO / 'benchmarks' / 'ytactos_modelo.json'


def detection_list(found):
    return [dict(score=f[0], x0=f[1], y0=f[2], x1=f[3], y1=f[4]) for f in found]


def pure(out=FIXTURES, seed=0):
    from transformers import Owlv2ImageProcessor
    from .acts import square, square_around
    from .fixedcam import net_displacement
    from .ytacts import detections, probabilities, probe_frames, window_detection
    rng = np.random.default_rng(seed)
    frozen = json.loads(MODEL_FILE.read_text())
    cases = dict(model_sha=None)
    x = rng.normal(0, 1, (6, len(frozen['mean']))) * np.array(frozen['scale']) + np.array(frozen['mean'])
    cases['classifier'] = [dict(x=v.tolist(), p=p.tolist()) for v, p in zip(x, probabilities(frozen, x))]
    cases['detections'] = []
    for k in range(8):
        n = 40
        scores = rng.uniform(0, .6, n)
        boxes = np.c_[rng.uniform(.1, .9, (n, 2)), rng.uniform(.02, .4, (n, 2))]
        boxes[1] = boxes[0] + [.01, 0, 0, 0]  # un duplicado casi idéntico: debe eliminarse
        shape = (int(rng.integers(200, 700)), int(rng.integers(200, 700)), 3)
        cases['detections'].append(dict(scores=scores.tolist(), boxes=boxes.ravel().tolist(), height=shape[0],
                                        width=shape[1], found=detection_list(detections(scores, boxes, shape))))
    cases['windows'] = []
    for k in range(6):
        found = []
        for _ in range(4):
            m = int(rng.integers(0, 3))
            found.append([(float(rng.uniform(.2, .9)), *sorted(rng.uniform(0, 300, 2)), *sorted(rng.uniform(0, 200, 2)))
                          for _ in range(m)])
            found[-1] = [(f[0], f[1], f[3], f[2], f[4]) for f in found[-1]]
        summary = window_detection(found)
        cases['windows'].append(dict(found=[detection_list(f) for f in found], count=summary['found'],
                                     probed=summary['probed'], multiple=summary['multiple'],
                                     union=list(summary['union']) if summary['union'] else None))
    cases['square_around'] = []
    for _ in range(10):
        x0, x1 = sorted(rng.uniform(0, 640, 2))
        y0, y1 = sorted(rng.uniform(0, 360, 2))
        shape = (360, 640, 3)
        cases['square_around'].append(dict(box=[x0, y0, x1, y1], width=640, height=360,
                                           square=list(square_around(x0, y0, x1, y1, shape))))
    cases['probe'] = [dict(count=c, n=n, indices=[int(i) for i in probe_frames(list(range(c)), n)])
                      for c, n in ((60, 4), (61, 16), (7, 4), (120, 16), (24, 16), (17, 4))]
    cases['crops'] = []
    for height, width, box, size in ((30, 40, (5, 3, 20), 16), (30, 40, None, 16), (48, 64, (10, 4, 12), 32),
                                     (20, 26, (0, 0, 20), 64)):
        image = rng.integers(0, 256, (height, width, 3)).astype(np.uint8)
        crop = square(image, box, size)
        cases['crops'].append(dict(width=width, height=height, pixels=image.ravel().tolist(),
                                   box=list(box) if box else None, size=size, out=crop.ravel().tolist()))
    cases['owl'] = []
    for height, width, size in ((30, 50, 24), (36, 20, 24), (12, 16, 32)):
        rgb = rng.integers(0, 256, (height, width, 3)).astype(np.uint8)
        processor = Owlv2ImageProcessor(size={'height': size, 'width': size})
        pixels = processor(images=[rgb], return_tensors='np')['pixel_values'][0]
        cases['owl'].append(dict(width=width, height=height, size=size, rgb=rgb.ravel().tolist(),
                                 pixels=pixels.astype(np.float64).ravel().tolist()))
    cases['fixedcam'] = []
    for k in range(8):
        times = np.sort(rng.uniform(0, 4, 12))
        xs, ys = rng.uniform(0, 500, 12), rng.uniform(0, 300, 12)
        areas = rng.uniform(1000, 5000, 12)
        xs[rng.random(12) < .2] = np.nan
        start, end = sorted(rng.uniform(0, 4, 2))
        value = net_displacement(times, xs, ys, areas, start, end)
        cases['fixedcam'].append(dict(times=times.tolist(), xs=[None if np.isnan(v) else v for v in xs], ys=ys.tolist(),
                                      areas=areas.tolist(), start=start, end=end, displacement=value))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pure.json').write_text(json.dumps(cases))
    resources = PACKAGE / 'Sources' / 'RabbitSenseKit' / 'Resources'
    resources.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL_FILE, resources / 'ytactos_modelo.json')
    print(f'{out / "pure.json"} y modelo congelado copiado a {resources}', flush=True)


def models(videos, out=FIXTURES / 'models', windows=3, device='mps'):
    """Ventanas reales: los fotogramas que usa el análisis (PNG, lado largo 640) y lo que dan con ellos OWLv2, V-JEPA 2
    y SAM2 de imagen en PyTorch. Python lee los mismos PNG que leerá Swift, para comparar con la misma entrada."""
    from .acts import square, square_around
    from .actmodel import VJEPA
    from .auto import stream_windows
    from .pointtrack import frame_times
    from .queryseg import TorchSAM
    from .ytacts import Detector, probabilities, probe_frames, window_detection
    frozen = json.loads(MODEL_FILE.read_text())
    detector, encoder, sam = Detector(device), VJEPA(device), TorchSAM(device=device)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    records = []
    for video in [Path(v).resolve() for v in videos]:
        times = frame_times(video)
        chosen = 0
        for k, frames, images in stream_windows(video, times):
            if chosen >= windows or len(frames) < 16:
                continue
            name = f'{video.stem}_{k:04d}'
            folder = out / name
            folder.mkdir()
            probe, picks = probe_frames(frames), probe_frames(frames, 16)
            for f in sorted(set(probe) | set(picks)):
                cv2.imwrite(str(folder / f'{f:06d}.png'), images[f])
            loaded = {f: cv2.imread(str(folder / f'{f:06d}.png')) for f in sorted(set(probe) | set(picks))}
            found = detector([loaded[f] for f in probe])
            summary = window_detection(found)
            record = dict(name=name, count=len(frames), times=[float(times[f]) for f in frames],
                          probe=[int(f) for f in probe], picks=[int(f) for f in picks],
                          found=[detection_list(f) for f in found],
                          union=list(summary['union']) if summary['union'] else None, masks=[])
            for f, boxes in zip(probe, found):
                if boxes:
                    mask = sam(loaded[f], boxes[0][1:])
                    cv2.imwrite(str(folder / f'mask_{f:06d}.png'), mask)
                    record['masks'].append(int(f))
            if summary['union']:
                box = square_around(*summary['union'], loaded[probe[0]].shape)
                clip = np.stack([square(loaded[f], box, encoder.size) for f in picks])
                vector = encoder(clip)
                p = probabilities(frozen, vector)[0]
                record.update(square=list(box), embedding=vector.astype(float).tolist(), probabilities=p.tolist(),
                              act=frozen['classes'][int(np.argmax(p))])
            records.append(record)
            chosen += 1
        print(f'{video.name}: {chosen} ventanas', flush=True)
    (out / 'models.json').write_text(json.dumps(dict(windows=records, detector=detector.info, encoder=encoder.info,
                                                     sam=sam.info)))


def main():
    parser = argparse.ArgumentParser(description='Resultados de referencia para los tests de Swift')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('pure')
    command = sub.add_parser('models')
    command.add_argument('videos', nargs='+')
    command.add_argument('--windows', type=int, default=3)
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    args = parser.parse_args()
    if args.command == 'pure':
        pure()
    else:
        models(args.videos, windows=args.windows, device=args.device)


if __name__ == '__main__':
    main()
