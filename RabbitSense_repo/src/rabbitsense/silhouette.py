"""Locomoción respecto al suelo con máscaras SAM2.

Dos señales por par de fotogramas consecutivos:
1. Suelo: homografía RANSAC con puntos del anillo que rodea la máscara (no la caja ni los muebles lejanos).
   El paso del conejo es el centroide actual menos el centroide anterior proyectado por esa homografía,
   normalizado por sqrt(área de la máscara). Se agrega con el mismo diámetro de trayectoria que `flow`.
2. Silueta: elongación y área de la máscara, que cambian al saltar aunque la cámara se mueva.

La regla de decisión se fija con vídeos de desarrollo y se congela (FROZEN_RULE) antes del test con vídeos nuevos.
"""
import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np

from .motion import aggregate, sha256
from .tracking import correspondences

RING = (.15, 1.)  # suelo entre 0,15 y 1,0 × sqrt(área) desde el borde de la máscara
MIN_POINTS, MIN_QUADRANTS, RANSAC_PX, MAX_ERROR_PX, MASK_JITTER_PX = 20, 3, 2., 3., 1.
THRESHOLD, STEP, MAX_SIDE, MIN_MASK_FRACTION = .2, .5, 960, .001
DEFAULT_RULE = dict(kind='floor')  # decisión de `aggregate` sobre el desplazamiento respecto al suelo
FROZEN_RULE = None  # se fija tras el desarrollo; el test decisivo se niega a ejecutarse mientras sea None


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


def decide(row, rule):
    if rule['kind'] == 'floor':
        return row['movement']
    raise ValueError(f'Regla desconocida: {rule}')


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


def run(manifest, out, labels=None, decisive=False):
    from .closure import evaluate_cases
    from .handheld import relabel, sufficiency
    rule = FROZEN_RULE if decisive else DEFAULT_RULE
    if decisive and rule is None:
        raise SystemExit('NO SE EJECUTA: la regla no está congelada (FROZEN_RULE). Primero desarrollo, después test.')
    cv2.setNumThreads(1)
    info = dict(rule=rule, ring=RING, min_points=MIN_POINTS, min_quadrants=MIN_QUADRANTS, ransac_px=RANSAC_PX,
                max_error_px=MAX_ERROR_PX, mask_jitter_px=MASK_JITTER_PX, threshold=THRESHOLD, step=STEP,
                max_side=MAX_SIDE, python=platform.python_version(), opencv=cv2.__version__)
    result = evaluate_cases(manifest, out, 'silhouette', lambda paths: analyze(paths['video'], paths['masks'], rule), info)
    if decisive:
        report = sufficiency(labels)
        if not report['sufficient']:
            return relabel(Path(out) / 'verdict.json', result, 'descriptive_only_insufficient_sample', report,
                           'verified_continuous_video_only')
        r = result['methods']['silhouette']
        print(f"\nSilueta, test con vídeos nuevos: locomoción {r['locomotion_hits']}/{r['locomotion_clips']}, "
              f"reposo {r['resting_correct']}/{r['resting_clips']}, episodios falsos {r['resting_false_events']} → "
              f"{'PASA' if r['passed'] else 'NO PASA'}\nEstado: {result['status']}")
        return result
    return relabel(Path(out) / 'verdict.json', result, 'development_not_decisive', None, 'development_labels')


def main():
    parser = argparse.ArgumentParser(description='Locomoción respecto al suelo con máscaras SAM2')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--labels', help='Etiquetas de los vídeos de test (solo con --decisive)')
    parser.add_argument('--decisive', action='store_true', help='Test con vídeos nuevos y regla congelada')
    args = parser.parse_args()
    if args.decisive and not args.labels:
        parser.error('--decisive requiere --labels')
    run(args.manifest, args.out, args.labels, args.decisive)


if __name__ == '__main__':
    main()
