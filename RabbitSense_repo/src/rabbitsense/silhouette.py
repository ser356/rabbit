"""Locomoción respecto al suelo con máscaras SAM2.

Dos señales por par de fotogramas consecutivos:
1. Suelo: homografía RANSAC con puntos del anillo que rodea la máscara (no la caja ni los muebles lejanos).
   El paso del conejo es el centroide actual menos el centroide anterior proyectado por esa homografía,
   normalizado por sqrt(área de la máscara). Se agrega con el mismo diámetro de trayectoria que `flow`.
2. Silueta: elongación y área de la máscara, que cambian al saltar aunque la cámara se mueva.

La regla se elige entre CANDIDATES con los vídeos de desarrollo (validación cruzada por vídeo) y se congela en
benchmarks/silueta_regla.json antes del test. Test mixto pre-registrado: especificidad en vídeos nuevos; sensibilidad
en vídeos nuevos si tienen suficientes clips de locomoción, si no, la de la validación cruzada.
"""
import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .motion import aggregate, sha256
from .tracking import correspondences

RING = (.15, 1.)  # suelo entre 0,15 y 1,0 × sqrt(área) desde el borde de la máscara
MIN_POINTS, MIN_QUADRANTS, RANSAC_PX, MAX_ERROR_PX, MASK_JITTER_PX = 20, 3, 2., 3., 1.
THRESHOLD, STEP, MAX_SIDE, MIN_MASK_FRACTION = .2, .5, 960, .001
MINIMUM_COVERAGE = .8
DEFAULT_RULE = dict(kind='floor', threshold=.2)
# Lista cerrada, fijada antes de ver resultados de desarrollo:
# floor: solo suelo; floor_shape: además la silueta debe cambiar para declarar locomoción;
# floor_or_shape: si el suelo no decide, un cambio grande de silueta con máscara completa declara locomoción.
CANDIDATES = ([dict(kind='floor', threshold=t) for t in (.15, .2, .3, .4)] +
              [dict(kind=k, threshold=t, shape=e) for k, shapes in (('floor_shape', (.05, .1, .2)),
                                                                    ('floor_or_shape', (.1, .2, .3)))
               for t in (.15, .2, .3, .4) for e in shapes])
RULE_FILE = Path(__file__).resolve().parents[2] / 'benchmarks' / 'silueta_regla.json'


def project(matrix, points):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def shape(mask):
    moments = cv2.moments(mask, True)
    if moments['m00'] < MIN_MASK_FRACTION * mask.size:
        return None
    covariance = np.array([[moments['mu20'], moments['mu11']], [moments['mu11'], moments['mu02']]]) / moments['m00']
    small, large = np.maximum(np.linalg.eigvalsh(covariance), 1e-9)
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    return dict(area=moments['m00'], center=np.array([moments['m10'], moments['m01']]) / moments['m00'],
                elongation=float(np.sqrt(large / small)), distance=distance)


def floor_step(prev, gray, first, second):
    size = float(np.sqrt((first['area'] + second['area']) / 2))
    near, far = max(2., RING[0] * size), max(4., RING[1] * size)
    ring = (((first['distance'] > near) & (first['distance'] <= far)) * 255).astype(np.uint8)
    a, b, _, _ = correspondences(prev, gray, ring, 400)
    if len(b):
        xy = np.clip(np.rint(b).astype(int), 0, [gray.shape[1] - 1, gray.shape[0] - 1])
        outside = second['distance'][xy[:, 1], xy[:, 0]] > near
        a, b = a[outside], b[outside]
    quadrants = len(set((2 * (a[:, 1] > first['center'][1]) + (a[:, 0] > first['center'][0])).tolist()))
    if len(a) < MIN_POINTS or quadrants < MIN_QUADRANTS:
        return dict(reason='floor_reference_insufficient')
    held = np.arange(len(a)) % 4 == 0
    matrix, inliers = cv2.findHomography(a[~held], b[~held], cv2.RANSAC, RANSAC_PX)
    if matrix is None or inliers is None or inliers.mean() < .5:
        return dict(reason='floor_homography_unreliable')
    error = float(np.quantile(np.linalg.norm(b[held] - project(matrix, a[held]), axis=1), .75))
    c = first['center']
    jacobian = np.array([(project(matrix, np.array([c + d]))[0] - project(matrix, np.array([c - d]))[0]) / 2
                         for d in np.eye(2)]).T
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if error > MAX_ERROR_PX or singular.min() < .7 or singular.max() > 1.4:
        return dict(reason='floor_homography_unreliable')
    return dict(reason='measured', vector=(second['center'] - project(matrix, c[None])[0]) / size,
                uncertainty=(error + MASK_JITTER_PX) / size, quality=float(inliers.mean()),
                camera_error_px=error, rabbit_points=int(len(a)))


def value(row, key):
    item = row.get(key)
    return None if item is None or (isinstance(item, float) and np.isnan(item)) else float(item)


def decide(row, rule):
    if rule['kind'] not in ('floor', 'floor_shape', 'floor_or_shape'):
        raise ValueError(f'Regla desconocida: {rule}')
    lower, upper, shape_change = value(row, 'speed_lower'), value(row, 'speed_upper'), value(row, 'elongation_range')
    movement = 'unknown'
    if lower is not None and (value(row, 'valid_fraction') or 0.) >= MINIMUM_COVERAGE:
        if lower >= rule['threshold']:
            movement = 'locomotion_candidate'
        elif upper is not None and upper < rule['threshold']:
            movement = 'low_motion_candidate'
    if rule['kind'] == 'floor_shape' and movement == 'locomotion_candidate' and (
            shape_change is None or shape_change < rule['shape']):
        movement = 'unknown'
    if (rule['kind'] == 'floor_or_shape' and movement == 'unknown' and shape_change is not None
            and shape_change >= rule['shape'] and value(row, 'mask_fraction') == 1.):
        movement = 'locomotion_candidate'
    return movement


def redecide(frame, rule):
    frame = frame.copy()
    rows = frame.to_dict('records')
    frame['movement'] = ['unknown' if i == 0 else decide(r, rule) for i, r in enumerate(rows)]
    return frame


def objective(counts):
    rates = [counts[a] / counts[b] for a, b in (('locomotion_hits', 'locomotion_clips'),
                                                ('resting_correct', 'resting_clips')) if counts[b]]
    return (min(rates) if rates else 0., -counts['resting_false_events'])


def select(detections, names):
    """Mejor candidata sobre `names`: max(min(sensibilidad, especificidad)), luego menos episodios falsos,
    luego la primera candidata de la lista (la más simple)."""
    from .closure import tally
    scores = [objective(tally({n: dict(detection=d[n]) for n in names})) for d in detections]
    best = max(range(len(detections)), key=lambda i: (scores[i], -i))
    return best, scores


def develop(manifest, dev_out, output=RULE_FILE, candidates=CANDIDATES, apply=None):
    from .closure import CRITERIA, tally
    from .evaluate import clip_metrics
    apply = apply or redecide
    manifest = Path(manifest).resolve()
    cases = json.loads(manifest.read_text())['cases']
    detections, hashes = [dict() for _ in candidates], {}
    for case in cases:
        motion = Path(dev_out) / case['name'] / 'motion.csv'
        frame = pd.read_csv(motion)
        labels = pd.read_csv((manifest.parent / Path(case['labels']).expanduser()).resolve())
        video = Path(case['video']).name
        hashes[case['name']] = sha256(motion)
        for i, rule in enumerate(candidates):
            detections[i][case['name']] = clip_metrics(apply(frame, rule), labels, video)['detection']
    names = [c['name'] for c in cases]
    held_out = {}
    for name in names:
        best, _ = select(detections, [n for n in names if n != name])
        held_out[name] = dict(rule=candidates[best], detection=detections[best][name])
    cross_validated = tally({n: dict(detection=h['detection']) for n, h in held_out.items()})
    best, scores = select(detections, names)
    report = dict(rule=candidates[best], criteria=CRITERIA, cross_validated=cross_validated, held_out=held_out,
                  training_scores=[dict(rule=r, objective=list(sc)) for r, sc in zip(candidates, scores)],
                  protocol='Leave-one-video-out over development videos; selection maximizes min(sensitivity, '
                           'specificity) of clip detection, then fewer false episodes, then candidate order.',
                  motion_sha256=hashes, manifest_sha256=sha256(manifest),
                  source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report


def mixed_verdict(test, cross_validated, criteria):
    """Especificidad siempre en vídeos nuevos. Sensibilidad en vídeos nuevos si hay suficientes clips; si no, la
    estimada por validación cruzada en desarrollo."""
    if test['resting_clips'] < criteria['minimum_resting_clips']:
        return dict(status='descriptive_only_insufficient_sample', passed=False)
    independent = test['locomotion_clips'] >= criteria['minimum_locomotion_clips']
    source = test if independent else cross_validated
    sensitivity = source['locomotion_hits'] / source['locomotion_clips'] if source['locomotion_clips'] else 0.
    specificity = test['resting_correct'] / test['resting_clips']
    checks = dict(specificity_new_videos=specificity >= criteria['specificity'],
                  enough_locomotion_clips=source['locomotion_clips'] >= criteria['minimum_locomotion_clips'],
                  sensitivity=sensitivity >= criteria['sensitivity'],
                  hit_in_every_locomotion_video=all(d['locomotion_hits'] >= criteria['hit_per_locomotion_video']
                                                    for d in source['per_video'].values() if d['locomotion_clips']))
    passed = all(checks.values())
    status = ('not_resolved' if not passed else 'resolved_independent_test' if independent else
              'resolved_specificity_independent_sensitivity_cross_validated')
    return dict(status=status, passed=passed, checks=checks, specificity_new_videos=specificity,
                sensitivity=sensitivity, sensitivity_source='new_videos' if independent else 'cross_validation')


def analyze(video, masks, rule=DEFAULT_RULE):
    masks = Path(masks)
    summary = json.loads((masks / 'summary.json').read_text())
    if sha256(video) != summary['inputs']['video']['sha256']:
        raise ValueError(f'Las máscaras de {masks} no corresponden a {video}.')
    capture = cv2.VideoCapture(str(video))
    fps, count = summary['fps'], summary['frames']
    stride = max(1, round(fps * STEP))
    rows, evidence, shapes = [], [], []
    previous, last = None, 0
    try:
        for frame in range(count):
            ok, image = capture.read()
            mask = cv2.imread(str(masks / 'masks' / f'{frame:06d}.png'), cv2.IMREAD_GRAYSCALE)
            if not ok or mask is None or mask.shape != image.shape[:2]:
                raise ValueError(f'Vídeo y máscara no coinciden en el fotograma {frame}.')
            scale = min(1., MAX_SIDE / max(mask.shape))
            size = (round(mask.shape[1] * scale), round(mask.shape[0] * scale))
            gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
            current = shape(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))
            if frame:
                evidence.append(dict(reason='mask_missing') if current is None or previous[1] is None else
                                floor_step(previous[0], gray, previous[1], current))
            shapes.append(current)
            if frame % stride == 0 or frame == count - 1:
                row = aggregate(evidence, (frame - last) / fps, THRESHOLD)
                present = [s for s in shapes if s is not None]
                row.update(seconds=frame / fps, mask_fraction=len(present) / len(shapes),
                           elongation_range=float(np.ptp([s['elongation'] for s in present])) if present else None,
                           log_area_range=float(np.ptp(np.log([s['area'] for s in present]))) if present else None)
                row['movement'] = decide(row, rule) if frame else 'unknown'
                rows.append(row)
                evidence, shapes, last = [], [current], frame
            previous = gray, current
    finally:
        capture.release()
    return rows


def run(manifest, out, labels=None, decisive=False, rule_file=RULE_FILE):
    from .closure import CRITERIA, evaluate_cases, tally
    from .handheld import relabel, sufficiency
    if decisive and not Path(rule_file).is_file():
        raise SystemExit('NO SE EJECUTA: la regla no está congelada. Primero desarrollo, después test.')
    frozen = json.loads(Path(rule_file).read_text()) if decisive else None
    rule = frozen['rule'] if decisive else DEFAULT_RULE
    cv2.setNumThreads(1)
    info = dict(rule=rule, ring=RING, min_points=MIN_POINTS, min_quadrants=MIN_QUADRANTS, ransac_px=RANSAC_PX,
                max_error_px=MAX_ERROR_PX, mask_jitter_px=MASK_JITTER_PX, step=STEP, max_side=MAX_SIDE,
                python=platform.python_version(), opencv=cv2.__version__)
    result = evaluate_cases(manifest, out, 'silhouette', lambda paths: analyze(paths['video'], paths['masks'], rule), info)
    if not decisive:
        return relabel(Path(out) / 'verdict.json', result, 'development_not_decisive', None, 'development_labels')
    test = tally(result['clips'])
    mixed = mixed_verdict(test, frozen['cross_validated'], CRITERIA)
    result.update(status=mixed['status'], passing_methods=['silhouette'] if mixed['passed'] else [], mixed=mixed,
                  test_counts=test, rule_file_sha256=sha256(rule_file), sufficiency=sufficiency(labels),
                  original_closure_status=result['status'])
    (Path(out) / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False))
    cv = frozen['cross_validated']
    print(f"\nVídeos nuevos: reposo {test['resting_correct']}/{test['resting_clips']}, episodios falsos "
          f"{test['resting_false_events']}, locomoción {test['locomotion_hits']}/{test['locomotion_clips']}"
          f"\nValidación cruzada (desarrollo): locomoción {cv['locomotion_hits']}/{cv['locomotion_clips']}"
          f"\nEstado: {mixed['status']}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Locomoción respecto al suelo con máscaras SAM2')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('run', help='Desarrollo (por defecto) o test decisivo con la regla congelada')
    command.add_argument('--manifest', required=True)
    command.add_argument('--out', required=True)
    command.add_argument('--labels', help='Etiquetas de los vídeos de test (solo con --decisive)')
    command.add_argument('--decisive', action='store_true')
    command = sub.add_parser('develop', help='Validación cruzada y congelación de la regla')
    command.add_argument('--manifest', required=True)
    command.add_argument('--dev-out', required=True)
    command.add_argument('--output', default=str(RULE_FILE))
    args = parser.parse_args()
    if args.command == 'develop':
        report = develop(args.manifest, args.dev_out, args.output)
        print(json.dumps(dict(rule=report['rule'], cross_validated=report['cross_validated']), indent=2))
    elif args.decisive and not args.labels:
        parser.error('--decisive requiere --labels')
    else:
        run(args.manifest, args.out, args.labels, args.decisive)


if __name__ == '__main__':
    main()
