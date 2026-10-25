import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from . import v36, v37
from .body import analyze_body
from .bridge import refine_video
from .motion import analyze, sha256, timeline
from .tracking import Anchors, export_guides, load_anchors


def box_metrics(prediction, truth):
    if prediction is None or not np.isfinite(prediction).all():
        return {'iou': 0., 'center_error_bodywidths': None, 'available': False}
    left = np.maximum(prediction[:2], truth[:2])
    right = np.minimum(prediction[:2] + prediction[2:], truth[:2] + truth[2:])
    area = float(np.prod(np.maximum(0., right - left)))
    union = float(np.prod(prediction[2:]) + np.prod(truth[2:]) - area)
    center = np.linalg.norm(prediction[:2] + prediction[2:] / 2 - truth[:2] - truth[2:] / 2)
    return {'iou': area / union if union else 0.,
            'center_error_bodywidths': float(center / truth[2]), 'available': True}


def heldout_tracking(video, anchors_path):
    cap = cv2.VideoCapture(str(video))
    rows = []
    try:
        if not cap.isOpened():
            raise ValueError(f'No se puede abrir {video}')
        anchors = load_anchors(anchors_path, int(cap.get(3)), int(cap.get(4)))
        visible = [i for i, box in enumerate(anchors.boxes) if box is not None]
        held = set(visible[1:-1:2])
        retained = [i for i in range(len(anchors.times)) if i not in held]
        training = Anchors(anchors.times[retained], [anchors.boxes[i] for i in retained])
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('FPS inválidos para evaluación.')
        refined = refine_video(video, training)
        for i in sorted(held):
            seconds, truth = anchors.times[i], anchors.boxes[i]
            guide = training.interpolate(seconds)
            prediction = refined.iloc[round(seconds * fps)][['refined_x', 'refined_y', 'width', 'height']].to_numpy(dtype=float)
            for name, box in [('interpolation', guide), ('visual_bridge', prediction)]:
                rows.append({'seconds': float(seconds), 'frame_seconds': round(seconds * fps) / fps,
                             'algorithm': name, **box_metrics(box, truth)})
    finally:
        cap.release()
    result = {}
    for name in ('interpolation', 'visual_bridge'):
        subset = [r for r in rows if r['algorithm'] == name]
        available = [r['center_error_bodywidths'] for r in subset if r['available']]
        result[name] = {'n': len(subset), 'available': len(available),
                        'mean_iou_all': float(np.mean([r['iou'] for r in subset])) if subset else None,
                        'median_center_error_available': float(np.median(available)) if available else None}
    return {'protocol': 'Odd interior original anchors withheld before any interpolation; endpoints retained. '
                         'No derived CSV rows used. Single-video development diagnostic, not independent test.',
            'summary': result, 'anchors': rows}


def clip_metrics(frame, labels, video_name):
    rows = []
    times = frame.seconds.to_numpy()
    if len(times) < 2:
        return {'clips': [], 'n': 0, 'accuracy_including_abstentions': None}
    starts = np.r_[times[0], times[:-1]]
    mapping = {'locomotion': 'locomotion_candidate', 'resting': 'low_motion_candidate'}
    selected = labels[(labels.video_id == video_name) &
                      labels.reviewed.astype(str).str.lower().isin(['si', 'sí', 'yes', 'true']) &
                      labels.label.isin(mapping)]
    for clip in selected.itertuples():
        end = clip.start_seconds + clip.duration_seconds
        inside = (starts >= clip.start_seconds - 1e-6) & (times <= end + 1e-6) & (times > starts)
        subset = frame.loc[inside]
        durations = times[inside] - starts[inside]
        votes = {label: float(durations[subset.movement.to_numpy() == label].sum())
                 for label in ('locomotion_candidate', 'low_motion_candidate', 'unknown')}
        total = sum(votes.values())
        predicted = max(votes, key=votes.get) if total else 'unknown'
        if total == 0 or votes[predicted] <= total / 2:
            predicted = 'unknown'
        events = []
        for start, stop, movement in zip(starts[inside], times[inside], subset.movement):
            if movement != 'locomotion_candidate':
                continue
            if events and abs(events[-1]['end_seconds'] - start) < 1e-6:
                events[-1]['end_seconds'] = float(stop)
                events[-1]['duration_seconds'] = float(stop - events[-1]['start_seconds'])
            else:
                events.append({'start_seconds': float(start), 'end_seconds': float(stop),
                               'duration_seconds': float(stop - start)})
        rows.append({'clip_id': clip.clip_id, 'label': clip.label, 'prediction': predicted,
                     'candidate_events': events,
                     'correct': predicted == mapping[clip.label], 'votes_seconds': votes,
                     'evaluated_seconds': total, 'clip_seconds': clip.duration_seconds,
                     'known_fraction': 1 - votes['unknown'] / total if total else 0.})
    recalls = [np.mean([r['correct'] for r in rows if r['label'] == label])
               for label in mapping if any(r['label'] == label for r in rows)]
    known = [r for r in rows if r['prediction'] != 'unknown']
    return {'clips': rows, 'n': len(rows),
            'accuracy_including_abstentions': float(np.mean([r['correct'] for r in rows])) if rows else None,
            'accuracy_on_predicted': float(np.mean([r['correct'] for r in known])) if known else None,
            'clip_coverage': len(known) / len(rows) if rows else None,
            'balanced_accuracy': float(np.mean(recalls)) if len(recalls) == 2 else None,
            'warning': 'Reviewed legacy clip labels; not frame-level truth. Exploring is not a negative class. '
                       'Candidate events describe predictions, not event precision or recall. '
                       'A single-class subset cannot establish specificity or balanced accuracy.'}


def compare(video, tracks, anchors, labels, out, video_out=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    frames = {}
    for name, module in [('v36', v36), ('v37', v37)]:
        cv2.setRNGSeed(0)
        module.run(SimpleNamespace(video=str(video), tracks=str(tracks), out=str(out / name),
                                   step=.5, threshold=.2, max_jump=1.5, max_scale_change=1.8,
                                   video_out=False))
        frames[name] = pd.read_csv(out / name / f'motion_{name}.csv')
    summary = analyze(video, anchors, out / 'flow', video_out=video_out)
    frames['flow'] = pd.read_csv(out / 'flow' / 'motion.csv')
    body_out = out / 'body'
    body_out.mkdir(parents=True, exist_ok=True)
    body_rows = analyze_body(video, load_anchors(anchors, summary['width'], summary['height']), .5, .2,
                             body_out / 'overlay.mp4' if video_out else None)
    frames['body'] = pd.DataFrame(body_rows)
    frames['body'].to_csv(body_out / 'motion.csv', index=False)
    (body_out / 'timeline.json').write_text(json.dumps(timeline(body_rows), indent=2))
    annotations = pd.read_csv(labels)
    reference = frames['v36'].seconds.to_numpy()
    report = {'inputs': {key: {'path': str(Path(path).resolve()), 'sha256': sha256(path)}
                         for key, path in [('video', video), ('tracks', tracks),
                                           ('anchors', anchors), ('labels', labels)]},
              'environment': summary['environment'], 'parameters': summary['parameters'],
              'source_sha256': summary['source_sha256'],
              'body_measure': 'Persistent guide-core points; similarity center translation after spatially validated '
                              'background affine. Speed is trajectory diameter / interval, not distance traveled. '
                              'Low motion describes stable center, not rest. No semantic floor or body identity guarantee. '
                              'Error bands and evidence quality are uncalibrated diagnostics.',
              'body_reasons': frames['body'].reason.value_counts().to_dict(),
              'sampling_note': 'Counts include the zero-duration first sample for compatibility. '
                               'Time coverage excludes it. Anchors use nearest decoded frame.',
              'tracking_holdout': heldout_tracking(video, anchors), 'motion': {}}
    for name, frame in frames.items():
        indices = np.array([np.argmin(abs(frame.seconds.to_numpy() - t)) for t in reference])
        if len(np.unique(indices)) != len(indices):
            raise ValueError('El remapeo temporal reutiliza muestras.')
        common = frame.iloc[indices].copy()
        if np.max(abs(common.seconds.to_numpy() - reference)) > .001:
            raise ValueError('Las rejillas temporales no coinciden.')
        common['seconds'] = reference
        last = common[(common.seconds >= 30) & (common.seconds <= 41)]
        durations = np.diff(reference, prepend=reference[0])
        report['motion'][name] = {'samples_common': len(common),
                                  'counts': common.movement.value_counts().to_dict(),
                                  'known_fraction': float((common.movement != 'unknown').mean()),
                                  'known_time_fraction': float(durations[common.movement.to_numpy() != 'unknown'].sum()
                                                               / durations.sum()) if durations.sum() else None,
                                  'seconds_30_41': last.movement.value_counts().to_dict(),
                                  'legacy_clips': clip_metrics(common, annotations, Path(video).name)}
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({name: value['counts'] for name, value in report['motion'].items()}, indent=2))
    print('Tracking holdout:', report['tracking_holdout']['summary'])
    return report


def main():
    parser = argparse.ArgumentParser(description='Comparación reproducible, sin entrenamiento ni pseudoetiquetas')
    for name in ('video', 'tracks', 'anchors', 'labels'):
        parser.add_argument(f'--{name}')
    parser.add_argument('--out', required=True)
    parser.add_argument('--suite', help='Manifest JSON con vídeos y anotaciones existentes')
    parser.add_argument('--video-out', action='store_true')
    args = parser.parse_args()
    if args.suite:
        manifest = Path(args.suite).resolve()
        cases = json.loads(manifest.read_text())['cases']
        reports = {}
        for case in cases:
            paths = {key: (manifest.parent / case[key]).resolve()
                     for key in ('video', 'anchors', 'labels')}
            destination = Path(args.out) / case['name']
            destination.mkdir(parents=True, exist_ok=True)
            if 'tracks' in case:
                tracks = (manifest.parent / case['tracks']).resolve()
            else:
                tracks = destination / 'guides.csv'
                cap = cv2.VideoCapture(str(paths['video']))
                try:
                    if not cap.isOpened():
                        raise ValueError(f'Vídeo ausente: {paths["video"]}')
                    anchors = load_anchors(paths['anchors'], int(cap.get(3)), int(cap.get(4)))
                    export_guides(anchors, cap.get(cv2.CAP_PROP_FPS),
                                  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), tracks)
                finally:
                    cap.release()
            reports[case['name']] = compare(paths['video'], tracks, paths['anchors'], paths['labels'],
                                             destination, args.video_out)
        Path(args.out, 'suite.json').write_text(json.dumps(reports, indent=2))
    else:
        if not all((args.video, args.tracks, args.anchors, args.labels)):
            parser.error('Indique --suite o bien --video, --tracks, --anchors y --labels.')
        compare(args.video, args.tracks, args.anchors, args.labels, args.out, args.video_out)


if __name__ == '__main__':
    main()