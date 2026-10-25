import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .bridge import refine_video
from .tracking import (GuidedTracker, correspondences, load_anchors, rectangle_mask,
                       transform_points)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def camera_motion(prev, cur, box0, box1):
    mask = 255 - rectangle_mask(prev.shape, box0, -.2)
    mask &= 255 - rectangle_mask(prev.shape, box1, -.2)
    x, y, w, h = box0
    yy, xx = np.ogrid[:prev.shape[0], :prev.shape[1]]
    radius = max(100., 1.5 * max(w, h))
    mask[(xx - x - w / 2) ** 2 + (yy - y - h / 2) ** 2 > radius ** 2] = 0
    a, b, fb, total = correspondences(prev, cur, mask)
    if len(a) < 16:
        return None
    validation = np.arange(len(a)) % 4 == 0
    matrix, inliers = cv2.estimateAffine2D(a[~validation], b[~validation],
                                          method=cv2.RANSAC, ransacReprojThreshold=2.)
    if matrix is None or inliers is None or inliers.sum() < 10 or inliers.mean() < .6:
        return None
    singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
    if singular.min() < .85 or singular.max() > 1.18 or np.linalg.det(matrix[:, :2]) <= 0:
        return None
    residual = np.linalg.norm(b[validation] - transform_points(matrix, a[validation]), axis=1)
    error = float(np.quantile(residual, .75))
    if error > 3.:
        return None
    return {'matrix': matrix, 'error': error, 'points': len(a),
            'quality': float(inliers.mean()), 'fb_error': float(np.quantile(fb, .9))}


def frame_evidence(prev, cur, box, rabbit):
    if box is None or 'a' not in rabbit:
        return {'reason': rabbit['reason']}
    camera = camera_motion(prev, cur, rabbit['old_box'], box)
    if camera is None:
        return {'reason': 'camera_reference_unreliable'}
    residuals = rabbit['b'] - transform_points(camera['matrix'], rabbit['a'])
    median = np.median(residuals, axis=0)
    scatter = float(np.median(np.linalg.norm(residuals - median, axis=1)))
    width = float((box[2] + rabbit['old_box'][2]) / 2)
    result = {'reason': 'measured', 'vector': median / width,
              'uncertainty': (camera['error'] + rabbit['fb_error'] + scatter) / width,
              'quality': min(camera['quality'], rabbit['quality']),
              'camera_error_px': camera['error'], 'rabbit_points': rabbit['rabbit_points']}
    if 'centers' in rabbit:
        first, second = rabbit['centers']
        result.update(centroid_vector=(second - transform_points(camera['matrix'], first[None])[0]) / width,
                      centroid_camera_error=camera['error'] / width)
    return result


def runs(evidence, key):
    """Consecutive measured frames; any gap starts a new run instead of being bridged."""
    groups, current = [], []
    for item in evidence:
        if item['reason'] == 'measured' and key in item:
            current.append(item)
        elif current:
            groups.append(current)
            current = []
    return groups + [current] if current else groups


def trajectory_span(steps, errors):
    """Diameter of the cumulative path and its bounds. Per-frame errors add in quadrature,
    so noise grows with the square root of the frame count instead of linearly."""
    positions = np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
    variance = np.r_[0., np.cumsum(np.square(errors))]
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    sigma = np.sqrt(np.abs(variance[:, None] - variance[None, :]))
    return (float(distances.max()), float(np.maximum(0., distances - sigma).max()),
            float((distances + sigma).max()))


def span_over_runs(groups, vector, error):
    spans = [trajectory_span(np.array([e[vector] for e in g]), np.array([e[error] for e in g])) for g in groups]
    return tuple(max(values) for values in zip(*spans))


def aggregate(evidence, duration, threshold, minimum_coverage=.8):
    good = [e for e in evidence if e['reason'] == 'measured']
    coverage = len(good) / max(1, len(evidence))
    row = {'movement': 'unknown', 'reason': 'insufficient_visual_evidence',
           'valid_fraction': coverage, 'evidence_quality': 0., 'speed_bodywidths_s': None,
           'speed_lower': None, 'speed_upper': None, 'camera_error_px': None,
           'rabbit_points': 0, 'interval_seconds': duration}
    if not good or duration <= 0:
        return row
    span, lower, upper = span_over_runs(runs(evidence, 'vector'), 'vector', 'uncertainty')
    speed, lower = span / duration, lower / duration
    upper = upper / duration if coverage == 1. else None
    row.update(speed_bodywidths_s=speed, speed_lower=lower, speed_upper=upper,
               evidence_quality=float(coverage * np.mean([e['quality'] for e in good])),
               camera_error_px=float(np.mean([e['camera_error_px'] for e in good])),
               rabbit_points=int(np.median([e['rabbit_points'] for e in good])))
    if coverage >= minimum_coverage:
        if lower >= threshold:
            row.update(movement='locomotion_candidate', reason='visual_motion_above_noise')
        elif upper is not None and upper < threshold:
            row.update(movement='low_motion_candidate', reason='visual_motion_below_threshold')
        else:
            row['reason'] = 'incomplete_interval' if upper is None else 'motion_near_uncertainty_band'
    centered = [e for e in good if 'centroid_vector' in e]
    if centered:
        center_span, center_lower, _ = span_over_runs(runs(evidence, 'centroid_vector'),
                                                      'centroid_vector', 'centroid_camera_error')
        center_adjusted = center_lower / duration
        center_coverage = len(centered) / max(1, len(evidence))
        row.update(centroid_speed_bodywidths_s=center_span / duration,
                   centroid_speed_camera_adjusted=center_adjusted, centroid_valid_fraction=center_coverage)
        if row['movement'] == 'locomotion_candidate' and (center_adjusted < threshold or center_coverage < minimum_coverage):
            row.update(movement='unknown', reason='surface_motion_without_centroid_translation')
    return row


def timeline(rows):
    intervals = []
    for row in rows:
        duration = row['interval_seconds']
        if duration <= 0:
            continue
        start, end = row['seconds'] - duration, row['seconds']
        if intervals and intervals[-1]['movement'] == row['movement']:
            item = intervals[-1]
            old_duration = item['end_seconds'] - item['start_seconds']
            item['evidence_quality'] = ((item['evidence_quality'] * old_duration +
                                         row['evidence_quality'] * duration) / (old_duration + duration))
            item['end_seconds'] = end
        else:
            intervals.append({'start_seconds': start, 'end_seconds': end,
                              'movement': row['movement'], 'evidence_quality': row['evidence_quality']})
    return intervals


def render_overlay(video, path, tracks, rows, fps, size):
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    try:
        if not cap.isOpened() or not writer.isOpened():
            raise RuntimeError('No se puede abrir el vídeo de diagnóstico.')
        times = np.array([row['seconds'] for row in rows])
        for frame, track in enumerate(tracks.itertuples()):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Error renderizando fotograma {frame}.')
            if np.isfinite(track.x):
                x, y, w, h = map(int, [track.x, track.y, track.width, track.height])
                cv2.rectangle(image, (x, y), (x + w, y + h), (160, 160, 160), 1)
                x, y = int(track.refined_x), int(track.refined_y)
                cv2.rectangle(image, (x, y), (x + w, y + h), (60, 210, 80), 2)
            index = min(int(np.searchsorted(times, frame / fps)), len(rows) - 1)
            row = rows[index]
            lines = [f'{frame / fps:.2f}s {track.refinement}', row['movement'],
                     f'Q={row["evidence_quality"]:.2f} (not probability)']
            cv2.rectangle(image, (0, 0), (size[0], 84), (15, 15, 15), -1)
            for line, text in enumerate(lines):
                cv2.putText(image, text, (10, 23 + 25 * line), cv2.FONT_HERSHEY_SIMPLEX,
                            .46, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        cap.release()
        writer.release()


def analyze(video, anchors_path, out, step=.5, threshold=.2, video_out=False, anchors=None):
    if not np.isfinite(step) or step <= 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError('step debe ser positivo y threshold no negativo; ambos finitos.')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'No se puede abrir el vídeo: {video}')
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(3)), int(cap.get(4))
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Metadatos de vídeo inválidos.')
        anchors = anchors if anchors is not None else load_anchors(anchors_path, width, height)
        if anchors.times[-1] > count / fps:
            raise ValueError('Hay anclas posteriores al final del vídeo.')
        stride = max(1, round(fps * step))
        tracker = GuidedTracker()
        prev = None
        rows, tracks, evidence = [], [], []
        reasons = Counter()
        last_sample = 0
        for frame in range(count):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Error decodificando fotograma {frame}/{count}.')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            exact, anchor = anchors.at_frame(frame, fps)
            guide = anchor if exact else anchors.interpolate(frame / fps)
            box, diagnostic = tracker.update(image, prev, guide)
            if prev is not None:
                item = frame_evidence(prev, gray, box, diagnostic)
                evidence.append(item)
                reasons[item['reason']] += 1
            values = [None] * 4 if box is None else box.tolist()
            displacement = (np.median(diagnostic['b'] - diagnostic['a'], axis=0)
                            if 'a' in diagnostic else [None, None])
            tracks.append({'frame': frame, 'seconds': frame / fps,
                           'source': 'lost' if box is None else 'manual' if exact else 'interpolated_guide',
                           'visual_dx': displacement[0], 'visual_dy': displacement[1],
                           **dict(zip(['x', 'y', 'width', 'height'], values))})
            if frame % stride == 0 or frame == count - 1:
                row = aggregate(evidence, (frame - last_sample) / fps, threshold)
                row.update(seconds=frame / fps, tracking=tracks[-1]['source'])
                rows.append(row)
                evidence = []
                last_sample = frame
            prev = gray
            if frame % 100 == 0 or frame == count - 1:
                print(f'FLOW {frame + 1}/{count}', flush=True)
    finally:
        cap.release()
    pd.DataFrame(rows).to_csv(out / 'motion.csv', index=False)
    refined = pd.DataFrame(tracks)
    bridge = refine_video(video, anchors)
    for column in ('refined_x', 'refined_y', 'refinement', 'bridge_points'):
        refined[column] = bridge[column]
    refined.to_csv(out / 'tracks.csv', index=False)
    if video_out:
        render_overlay(video, out / 'overlay.mp4', refined, rows, fps, (width, height))
    (out / 'timeline.json').write_text(json.dumps(timeline(rows), indent=2))
    summary = {'algorithm': 'sequential_visual_flow', 'samples': len(rows), 'frames': count,
               'fps': fps, 'width': width, 'height': height,
               'movement': dict(Counter(r['movement'] for r in rows)),
               'tracking': dict(Counter(t['source'] for t in tracks)),
               'refinement': refined.refinement.value_counts().to_dict(),
               'frame_reasons': dict(reasons), 'sample_reasons': dict(Counter(r['reason'] for r in rows)),
               'parameters': {'step': step, 'threshold': threshold, 'seed': 0, 'opencv_threads': 1},
               'motion_measure': 'Diameter of the cumulative camera-compensated residual path per full interval; '
                                 'per-frame errors add in quadrature; gaps split the path and add no distance. '
                                 'Upper bound unavailable with missing evidence; body articulation is not excluded.',
               'environment': {'python': platform.python_version(), 'opencv': cv2.__version__,
                               'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform()},
               'inputs': {'video': str(Path(video).resolve()), 'video_sha256': sha256(video),
                          'anchors': str(Path(anchors_path).resolve()), 'anchors_sha256': sha256(anchors_path)},
               'source_sha256': {path.name: sha256(path) for path in Path(__file__).parent.glob('*.py')},
               'warnings': ['Manual/interpolated guides with experimental visual center correction; not autonomous detection.',
                            'Box-core features can include background; no semantic segmentation is claimed.',
                            'Evidence quality and uncertainty band are not calibrated probabilities.',
                            'Residual motion is not validated locomotion or a welfare indicator.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def run(args):
    if not args.anchors:
        raise ValueError('El análisis visual requiere --anchors con el JSON original.')
    print(json.dumps(analyze(args.video, args.anchors, args.out, args.step,
                             args.threshold, args.video_out), indent=2))