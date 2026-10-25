import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .evaluate import box_metrics, clip_metrics
from .body import BodyMotion
from .motion import aggregate, frame_evidence, sha256, timeline
from .tracking import correspondences, load_anchors


def segment(video, anchors_path, model_path, out, device, guided=False):
    from ultralytics.models.sam import SAM2VideoPredictor
    import torch
    import ultralytics

    video, anchors_path, model_path = [Path(p).resolve(strict=True) for p in (video, anchors_path, model_path)]
    out = Path(out)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').unlink(missing_ok=True)
    sources = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f'No se puede abrir {video}')
        fps, count = capture.get(5), int(capture.get(7))
        width, height = int(capture.get(3)), int(capture.get(4))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError('Metadatos de vídeo inválidos.')
    anchors = load_anchors(anchors_path, width, height)
    exact, initial = anchors.at_frame(0, fps)
    if not exact or initial is None:
        raise ValueError('Se requiere una ancla original visible en el primer fotograma.')
    references = {round(t * fps): box for t, box in zip(anchors.times, anchors.boxes) if box is not None}
    visible = list(references)
    heldout = set(visible[1:-1:2]) if guided else set(visible[1:])
    prompts = {frame: box for frame, box in references.items() if frame not in heldout and
               (frame == 0 or (box[2] < width and box[3] < height))}

    class PromptedVideoPredictor(SAM2VideoPredictor):
        frame_index = -1

        def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
            self.frame_index += 1
            if self.frame_index in prompts and self.frame_index > 0:
                box = prompts[self.frame_index]
                self.inference_state['im'] = im
                coordinates, kinds, _ = self._prepare_prompts(
                    im.shape[2:], self.src_shape, bboxes=[np.r_[box[:2], box[:2] + box[2:]].tolist()])
                self.add_new_prompts(obj_id=0, points=coordinates, labels=kinds, frame_idx=self.dataset.frame)
            return super().inference(im, bboxes, points, labels, masks)

    predictor_type = PromptedVideoPredictor if guided else SAM2VideoPredictor
    predictor = predictor_type(overrides=dict(model=str(model_path), device=device, imgsz=1024,
                                                 task='segment', mode='predict', save=False, verbose=False))
    prompt = np.r_[initial[:2], initial[:2] + initial[2:]].tolist()
    stream = predictor(source=str(video), bboxes=[prompt], stream=True)
    writer = cv2.VideoWriter(str(out / 'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('No se puede crear el vídeo segmentado.')
    rows, checks = [], []
    try:
        for frame, result in enumerate(stream):
            mask = np.zeros((height, width), np.uint8)
            if result.masks is not None and len(result.masks.data):
                if len(result.masks.data) != 1:
                    raise RuntimeError('La segmentación de un objeto devolvió varias identidades.')
                mask = result.masks.data[0].cpu().numpy().astype(np.uint8) * 255
            if mask.shape != (height, width):
                raise RuntimeError('La máscara no conserva las dimensiones originales.')
            box = np.array(cv2.boundingRect(mask), dtype=float) if mask.any() else None
            rows.append(dict(frame=frame, seconds=frame / fps, mask_pixels=int(np.count_nonzero(mask)),
                             **dict(zip(('x', 'y', 'width', 'height'), box if box is not None else [None] * 4))))
            if frame in references:
                checks.append(dict(frame=frame, seconds=frame / fps, used_as_prompt=frame in prompts,
                                   withheld=frame in heldout,
                                   **box_metrics(box, references[frame])))
            if not cv2.imwrite(str(out / 'masks' / f'{frame:06d}.png'), mask):
                raise RuntimeError(f'No se puede escribir la máscara {frame}.')
            image = result.orig_img.copy()
            selected = mask > 0
            image[selected] = (image[selected] * .55 + np.array([40, 230, 60]) * .45).astype(np.uint8)
            cv2.putText(image, f'SAM2 guided={guided} frame={frame} mask={rows[-1]["mask_pixels"]}',
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
            if frame % 100 == 0:
                print(f'SEGMENT {frame + 1}/{count}', flush=True)
    finally:
        stream.close()
        writer.release()
    if len(rows) != count:
        raise RuntimeError(f'Vídeo incompleto: {len(rows)}/{count} fotogramas.')
    pd.DataFrame(rows).to_csv(out / 'tracks.csv', index=False)
    held = [r for r in checks if r['withheld']]
    summary = dict(protocol=('Original odd interior visible anchors withheld; all others prompt SAM2. ' if guided else
                            'Only original first-frame box prompts SAM2. All later original boxes excluded from inference. ') +
                            'Corrections spanning the full image width or height are excluded as unlocalized. '
                            'Box IoU is not mask accuracy or behavioral validation. SAM video scores are not identity confidence.',
                   prompt_frames=sorted(prompts),
                   frames=count, fps=fps, device=device, nonempty_frames=sum(r['mask_pixels'] > 0 for r in rows),
                   heldout_n=len(held), heldout_mean_iou=float(np.mean([r['iou'] for r in held])) if held else None,
                   anchors=checks, environment=dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
                                                    opencv=cv2.__version__),
                   inputs={name: dict(path=str(path), sha256=sha256(path))
                           for name, path in [('video', video), ('anchors', anchors_path), ('model', model_path)]},
                   source_sha256=sources)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print({k: summary[k] for k in ('frames', 'nonempty_frames', 'heldout_n', 'heldout_mean_iou')}, flush=True)
    return summary


def measure_masks(directory):
    directory = Path(directory)
    segmentation = json.loads((directory / 'summary.json').read_text())
    (directory / 'motion_summary.json').unlink(missing_ok=True)
    video = segmentation['inputs']['video']['path']
    if sha256(video) != segmentation['inputs']['video']['sha256']:
        raise ValueError('El vídeo no coincide con las máscaras.')
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(video)
    tracker, rows, last = BodyMotion(), [], 0
    flow_rows, evidence = [], []
    previous, previous_mask, previous_box, previous_center = None, None, None, None
    fps, count = segmentation['fps'], segmentation['frames']
    writer = cv2.VideoWriter(str(directory / 'motion_overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (int(capture.get(3)), int(capture.get(4))))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError('No se puede crear la revisión de movimiento.')
    images = []
    try:
        for frame in range(count):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f'No se puede decodificar el fotograma {frame}.')
            mask = cv2.imread(str(directory / 'masks' / f'{frame:06d}.png'), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != image.shape[:2] or not np.isin(mask, [0, 255]).all():
                raise ValueError(f'Máscara inválida: {frame}.')
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            foreground = ((distance > max(2., .2 * distance.max())) * 255).astype(np.uint8)
            background = 255 - cv2.dilate(mask, np.ones((21, 21), np.uint8))
            masks = foreground, background
            box = np.array(cv2.boundingRect(mask), dtype=float) if mask.any() else None
            moments = cv2.moments(mask, True)
            center = np.array([moments['m10'], moments['m01']]) / moments['m00'] if box is not None else None
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if frame == 0:
                tracker.reset(gray, box, masks)
            else:
                tracker.update(gray, box, masks)
                item = dict(reason='mask_missing')
                if box is not None and previous_box is not None:
                    a, b, errors, total = correspondences(previous, gray, previous_mask, 250)
                    xy = np.clip(np.rint(b).astype(int), [0, 0], [gray.shape[1] - 1, gray.shape[0] - 1])
                    inside = foreground[xy[:, 1], xy[:, 0]] > 0
                    a, b, errors = a[inside], b[inside], errors[inside]
                    if len(a) >= 6:
                        item = frame_evidence(previous, gray, box, dict(a=a, b=b,
                            fb_error=float(np.quantile(errors, .9)), old_box=previous_box,
                            rabbit_points=len(a), quality=len(a) / total, centers=(previous_center, center)))
                if frame in segmentation.get('prompt_frames', [0]):
                    item = dict(reason='annotation_correction_boundary')
                evidence.append(item)
            selected = mask > 0
            image[selected] = (image[selected] * .55 + np.array([40, 230, 60]) * .45).astype(np.uint8)
            images.append(image)
            if frame % max(1, round(fps * .5)) == 0 or frame == count - 1:
                rows.append(dict(seconds=frame / fps, **tracker.measure((frame - last) / fps, .2)))
                flow_rows.append(dict(seconds=frame / fps, **aggregate(evidence, (frame - last) / fps, .2)))
                for buffered in images:
                    cv2.rectangle(buffered, (0, 0), (buffered.shape[1], 60), (15, 15, 15), -1)
                    for line, text in enumerate([flow_rows[-1]['movement'], flow_rows[-1]['reason']]):
                        cv2.putText(buffered, text, (8, 22 + line * 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    .43, (240, 240, 240), 1, cv2.LINE_AA)
                    writer.write(buffered)
                images = []
                evidence = []
                tracker.reset(gray, box, masks)
                last = frame
            previous, previous_mask, previous_box, previous_center = gray, foreground, box, center
    finally:
        capture.release()
        writer.release()
    pd.DataFrame(rows).to_csv(directory / 'motion.csv', index=False)
    pd.DataFrame(flow_rows).to_csv(directory / 'flow_motion.csv', index=False)
    (directory / 'flow_timeline.json').write_text(json.dumps(timeline(flow_rows), indent=2, allow_nan=False))
    (directory / 'timeline.json').write_text(json.dumps(timeline(rows), indent=2, allow_nan=False))
    report = dict(protocol='SAM2 mask interior persistent flow; local background outside dilated mask; '
                           'same spatial camera validation and body fit as BodyMotion. '
                           'Candidate translation, not validated behavior or calibrated confidence.',
                  flow_protocol='Forward/backward LK inside consecutive eroded SAM2 masks; existing local camera '
                                'affine and temporal noise/coverage gates. Positive flow additionally requires mask '
                                'centroid path above the same threshold after subtracting camera residual only; '
                                'that diagnostic does NOT bound segmentation or parallax error. Masked flow can include articulation or '
                                'parallax, and mask identity is not validated automatically. Not rigid torso speed.',
                  segmentation_sha256=sha256(directory / 'summary.json'),
                  source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                  reasons=pd.Series([r['reason'] for r in rows]).value_counts().to_dict(),
                  flow_movements=pd.Series([r['movement'] for r in flow_rows]).value_counts().to_dict(),
                  movements=pd.Series([r['movement'] for r in rows]).value_counts().to_dict())
    (directory / 'motion_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('MASKED FLOW:', report['flow_movements'], 'RIGID BODY:', report['movements'], flush=True)
    return rows


def assess_masks(manifest, baseline, directories, output):
    manifest, baseline = Path(manifest).resolve(), Path(baseline).resolve()
    historical = json.loads(baseline.read_text())
    report = dict(protocol='Existing original held-out boxes, same timestamps as historical tracking benchmark. '
                           'SAM2 prompts differ by run and are listed explicitly; this is not independent generalization. '
                           'Legacy behavioral labels describe clips, not frame-level events. '
                           'Box overlap does not validate mask identity or segmentation accuracy.', cases={},
                  manifest_sha256=sha256(manifest), baseline_sha256=sha256(baseline))
    for case in json.loads(manifest.read_text())['cases']:
        name = case['name']
        directory = Path(directories[name]).resolve()
        summary = json.loads((directory / 'summary.json').read_text())
        motion_summary = json.loads((directory / 'motion_summary.json').read_text())
        if motion_summary['segmentation_sha256'] != sha256(directory / 'summary.json'):
            raise ValueError(f'Movimiento y segmentación de ejecuciones distintas: {name}.')
        for key in ('video', 'anchors'):
            if summary['inputs'][key]['sha256'] != historical[name]['inputs'][key]['sha256']:
                raise ValueError(f'Entradas distintas al baseline: {name}/{key}.')
        target = {round(r['seconds'] * summary['fps']) for r in historical[name]['tracking_holdout']['anchors']}
        reserved = [r for r in summary['anchors'] if r['frame'] in target]
        if len(reserved) != len(target) or any(r['used_as_prompt'] for r in reserved):
            raise ValueError(f'Comparación contaminada o incompleta: {name}.')
        labels_path = (manifest.parent / case['labels']).resolve()
        if sha256(labels_path) != historical[name]['inputs']['labels']['sha256']:
            raise ValueError(f'Etiquetas distintas al baseline: {name}.')
        labels = pd.read_csv(labels_path)
        metrics = {}
        for method, file in [('masked_flow', 'flow_motion.csv'), ('rigid_body', 'motion.csv')]:
            frame = pd.read_csv(directory / file)
            metrics[method] = dict(raw_counts=frame.movement.value_counts().to_dict(),
                                   clips=clip_metrics(frame, labels, Path(summary['inputs']['video']['path']).name))
        report['cases'][name] = dict(directory=str(directory), frames=summary['frames'],
            nonempty_frames=summary['nonempty_frames'], prompt_frames=summary.get('prompt_frames', [0]),
            heldout_n=len(reserved), sam_mean_iou=float(np.mean([r['iou'] for r in reserved])),
            historical_tracking=historical[name]['tracking_holdout']['summary'], metrics=metrics,
            artifact_sha256={file: sha256(directory / file) for file in
                ['summary.json', 'tracks.csv', 'motion.csv', 'flow_motion.csv', 'motion_summary.json', 'motion_overlay.mp4']})
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description='Segmentación temporal con una caja original inicial; sin pseudoetiquetas')
    for name in ('video', 'anchors', 'model', 'out'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--guided', action='store_true', help='Corregir con anclas pares; reservar impares y excluir cajas de imagen completa')
    args = parser.parse_args()
    segment(args.video, args.anchors, args.model, args.out, args.device, args.guided)
    measure_masks(args.out)


if __name__ == '__main__':
    main()