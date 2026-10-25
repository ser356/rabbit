import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import cv2
import numpy as np

from .evaluate import box_metrics
from .motion import sha256
from .tracking import load_anchors


def main():
    parser = argparse.ArgumentParser(description='Auditoría zero-shot contra anclas originales, sin entrenar ni exportar guías')
    for name in ('suite', 'model', 'clip', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args()
    manifest, weights, clip_weights = [p.resolve(strict=True) for p in (args.suite, args.model, args.clip)]
    from ultralytics import YOLOWorld
    from ultralytics.nn.text_model import CLIP

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        'protocol': 'Highest-score detection, confidence >= 0.25, imgsz=640, class-agnostic NMS. '
                    'No anchor used for candidate selection. All visible original anchors evaluated. '
                    'Prompts explored on IMG_5177; same-video development diagnostic, not independent test. '
                    'Species scores are not calibrated identity probabilities. No behavioral labels generated.',
        'device': args.device, 'python': platform.python_version(),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'ultralytics', 'clip', 'opencv-python')},
        'inputs': {name: {'path': str(path), 'sha256': sha256(path)}
                   for name, path in [('manifest', manifest), ('model', weights), ('clip', clip_weights)]},
        'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'experiments': {}}
    for experiment, names in [('rabbit', ['rabbit']), ('proposals', ['rabbit', 'bunny', 'animal', 'dog', 'cat'])]:
        model = YOLOWorld(str(weights))
        model.model.clip_model = CLIP(str(clip_weights), device='cpu')
        model.model.set_classes(names, cache_clip_model=True)
        model.model.names = dict(enumerate(names))
        cases = {}
        for case in json.loads(manifest.read_text())['cases']:
            video, anchors_path = [(manifest.parent / case[key]).resolve(strict=True) for key in ('video', 'anchors')]
            capture = cv2.VideoCapture(str(video))
            rows, images = [], []
            try:
                if not capture.isOpened():
                    raise ValueError(f'No se puede abrir {video}')
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError('FPS inválidos.')
                anchors = load_anchors(anchors_path, int(capture.get(3)), int(capture.get(4)))
                for seconds, truth in zip(anchors.times, anchors.boxes):
                    if truth is None:
                        continue
                    frame = round(float(seconds) * fps)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError(f'Error decodificando fotograma {frame}.')
                    result = model.predict(image, device=args.device, conf=.25, imgsz=640,
                                           agnostic_nms=True, verbose=False)[0]
                    detections = [{'xyxy': b.xyxy[0].cpu().tolist(), 'score': float(b.conf.item()),
                                   'class': names[int(b.cls.item())]} for b in result.boxes]
                    detections.sort(key=lambda b: b['score'], reverse=True)
                    prediction = None
                    if detections:
                        xyxy = np.array(detections[0]['xyxy'])
                        prediction = np.r_[xyxy[:2], xyxy[2:] - xyxy[:2]]
                    rows.append({'seconds': float(seconds), 'frame_seconds': frame / fps,
                                 'detections': detections, **box_metrics(prediction, truth)})
                    view = cv2.resize(result.plot(), (203, 360))
                    cv2.putText(view, f'{seconds:.2f}s IoU={rows[-1]["iou"]:.2f}', (4, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
                    images.append(view)
            finally:
                capture.release()
            if not rows:
                raise ValueError(f'Sin anclas visibles: {anchors_path}')
            cases[case['name']] = {
                'video_sha256': sha256(video), 'anchors_sha256': sha256(anchors_path),
                'n': len(rows), 'available': sum(r['available'] for r in rows),
                'mean_iou_all': float(np.mean([r['iou'] for r in rows])), 'rows': rows}
            images.extend([np.zeros_like(images[0])] * (-len(images) % 4))
            montage = np.vstack([np.hstack(images[i:i + 4]) for i in range(0, len(images), 4)])
            destination = args.out / f'{experiment}_{case["name"]}.jpg'
            if not cv2.imwrite(str(destination), montage):
                raise RuntimeError(f'No se puede escribir {destination}')
            print(experiment, case['name'], {k: v for k, v in cases[case['name']].items()
                                           if k in ('n', 'available', 'mean_iou_all')}, flush=True)
        report['experiments'][experiment] = {'prompts': names, 'cases': cases}
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()