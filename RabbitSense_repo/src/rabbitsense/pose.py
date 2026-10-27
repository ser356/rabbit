"""Locomoción por la marcha: puntos del cuerpo con DeepLabCut SuperAnimal-Quadruped (preentrenado, sin ajuste).

Al desplazarse, las patas se mueven respecto al cuerpo; en reposo no, aunque la cámara se mueva. Las posiciones de
patas y rodillas se expresan en el eje del cuerpo (cuello-cola) y en longitudes corporales; por intervalo de 0,5 s
se mide su excursión (diámetro de la trayectoria). La regla se elige entre CANDIDATES con validación cruzada por
vídeo y se congela en benchmarks/pose_regla.json antes del test, con el mismo protocolo mixto que `silhouette`.

`keypoints` necesita torch y deeplabcut (entorno de entrenamiento); el resto solo numpy/pandas/OpenCV.
"""
import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .motion import sha256

SUPER_ANIMAL, MODEL, DETECTOR = 'superanimal_quadruped', 'hrnet_w32', 'fasterrcnn_resnet50_fpn_v2'
BODYPARTS = ['nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left', 'right_eye', 'right_earbase',
             'right_earend', 'right_antler_base', 'right_antler_end', 'left_eye', 'left_earbase', 'left_earend',
             'left_antler_base', 'left_antler_end', 'neck_base', 'neck_end', 'throat_base', 'throat_end', 'back_base',
             'back_end', 'back_middle', 'tail_base', 'tail_end', 'front_left_thai', 'front_left_knee',
             'front_left_paw', 'front_right_thai', 'front_right_knee', 'front_right_paw', 'back_left_paw',
             'back_left_thai', 'back_right_thai', 'back_left_knee', 'back_right_knee', 'back_right_paw',
             'belly_bottom', 'body_middle_right', 'body_middle_left']
TORSO = ['neck_base', 'back_base', 'back_middle', 'back_end', 'tail_base', 'belly_bottom', 'body_middle_right',
         'body_middle_left']
LIMBS = {'all': ['front_left_knee', 'front_left_paw', 'front_right_knee', 'front_right_paw', 'back_left_knee',
                 'back_left_paw', 'back_right_knee', 'back_right_paw'],
         'hind': ['back_left_knee', 'back_left_paw', 'back_right_knee', 'back_right_paw']}
SCORE, PAD, STEP = .4, .1, .5
MIN_POSITIONS, MIN_LIMBS, MIN_COVERAGE = 3, 2, .5
# Lista cerrada, fijada antes de ver resultados. pose: solo marcha; pose_or_floor: locomoción si la marcha o el suelo
# (regla congelada de silhouette) la indican; bajo movimiento si la marcha lo indica y el suelo no dice locomoción,
# o si la marcha no decide y el suelo indica bajo movimiento.
CANDIDATES = [dict(kind=kind, limbs=limbs, high=high, low=low) for kind in ('pose', 'pose_or_floor')
              for limbs in ('all', 'hind') for high in (.15, .2, .3, .4, .5) for low in (.05, .1) if low < high]
RULE_FILE = Path(__file__).resolve().parents[2] / 'benchmarks' / 'pose_regla.json'
FLOOR_RULE_FILE = Path(__file__).resolve().parents[2] / 'benchmarks' / 'silueta_regla.json'


class SuperAnimalRunner:
    """Puntos del cuerpo dentro de cajas dadas (modo top-down), sin ejecutar el detector de DeepLabCut."""

    def __init__(self, device='cpu', snapshot=None, batch_size=8):
        import deeplabcut
        import torch
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
        from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
        from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path
        config = PoseConfig.build_for_superanimal_inference(super_animal=SUPER_ANIMAL, model_name=MODEL,
                                                            detector_name=DETECTOR, max_individuals=1, device=device)
        if list(config['metadata']['bodyparts']) != BODYPARTS:
            raise RuntimeError('Los puntos del modelo no coinciden con BODYPARTS.')
        snapshot = Path(snapshot) if snapshot else get_super_animal_snapshot_path(SUPER_ANIMAL, MODEL)
        self.runner = get_pose_inference_runner(config, snapshot_path=snapshot, batch_size=batch_size, device=device,
                                                max_individuals=1)
        self.info = dict(super_animal=SUPER_ANIMAL, model=MODEL, snapshot_sha256=sha256(snapshot), device=device,
                         deeplabcut=deeplabcut.__version__, torch=torch.__version__)

    def __call__(self, images, boxes):
        predictions = self.runner.inference([(image, {'bboxes': np.array([box], float)})
                                             for image, box in zip(images, boxes)])
        return np.stack([p['bodyparts'][0] for p in predictions])


def padded_box(mask, pad=PAD):
    x, y, w, h = cv2.boundingRect(mask)
    x0, y0 = max(0., x - pad * w), max(0., y - pad * h)
    x1, y1 = min(mask.shape[1], x + w + pad * w), min(mask.shape[0], y + h + pad * h)
    return [x0, y0, x1 - x0, y1 - y0]


def keypoints(video, masks, output, runner, batch=16, preview=None):
    masks = Path(masks)
    summary = json.loads((masks / 'summary.json').read_text())
    if sha256(video) != summary['inputs']['video']['sha256']:
        raise ValueError(f'Las máscaras de {masks} no corresponden a {video}.')
    capture = cv2.VideoCapture(str(video))
    rows, pending, shown = [], [], []
    every = max(1, summary['frames'] // 12)

    def flush():
        if pending:
            points = runner([p[1] for p in pending], [p[2] for p in pending])
            for (frame, image, box), values in zip(pending, points):
                rows.append([frame, *box, *values.ravel()])
                if preview is not None and frame % every == 0 and len(shown) < 12:
                    shown.append(draw(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), box, values))
            pending.clear()

    try:
        for frame in range(summary['frames']):
            ok, image = capture.read()
            mask = cv2.imread(str(masks / 'masks' / f'{frame:06d}.png'), cv2.IMREAD_GRAYSCALE)
            if not ok or mask is None or mask.shape != image.shape[:2]:
                raise ValueError(f'Vídeo y máscara no coinciden en el fotograma {frame}.')
            if np.count_nonzero(mask) < .001 * mask.size:
                rows.append([frame, *[np.nan] * (4 + 3 * len(BODYPARTS))])
                continue
            pending.append((frame, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), padded_box(mask)))
            if len(pending) == batch:
                flush()
        flush()
    finally:
        capture.release()
    columns = ['frame', 'box_x', 'box_y', 'box_w', 'box_h'] + [f'{p}_{k}' for p in BODYPARTS for k in ('x', 'y', 'score')]
    frame = pd.DataFrame(sorted(rows, key=lambda r: r[0]), columns=columns)
    frame.insert(1, 'seconds', frame.frame / summary['fps'])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format='%.2f')
    if preview is not None and shown:
        size = (240, 320)
        tiles = [cv2.resize(t, size[::-1]) for t in shown] + [np.zeros((*size, 3), np.uint8)] * (-len(shown) % 4)
        cv2.imwrite(str(preview), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    return frame


def draw(image, box, values):
    image = image.copy()
    x, y, w, h = map(int, box)
    cv2.rectangle(image, (x, y), (x + w, y + h), (160, 160, 160), 2)
    colors = {name: (60, 220, 60) if name in LIMBS['all'] else (220, 120, 40) if name in TORSO else (200, 200, 200)
              for name in BODYPARTS}
    for name, (px, py, score) in zip(BODYPARTS, values):
        if score >= SCORE:
            cv2.circle(image, (int(px), int(py)), max(3, image.shape[1] // 150), colors[name], -1)
    return image


def body_frame(points):
    """Centro, rotación y longitud corporal de un fotograma; None si faltan cuello o cola fiables."""
    score = lambda name: points[f'{name}_score']
    xy = lambda name: np.array([points[f'{name}_x'], points[f'{name}_y']], float)
    torso = [xy(n) for n in TORSO if score(n) >= SCORE]
    head = next((n for n in ('neck_base', 'back_base') if score(n) >= SCORE), None)
    tail = next((n for n in ('tail_base', 'back_end') if score(n) >= SCORE), None)
    if len(torso) < 2 or head is None or tail is None:
        return None
    axis = xy(head) - xy(tail)
    length = float(np.linalg.norm(axis))
    if length < 1:
        return None
    u = axis / length
    return np.mean(torso, axis=0), np.array([u, [-u[1], u[0]]]), length


def interval_features(frames, limbs):
    """Mediana entre patas/rodillas fiables del diámetro de su trayectoria en el eje del cuerpo (longitudes)."""
    frames_ok = [(f, body_frame(f)) for f in frames]
    valid = [(f, b) for f, b in frames_ok if b is not None]
    spans = []
    for limb in LIMBS[limbs]:
        positions = np.array([b[1] @ (np.array([f[f'{limb}_x'], f[f'{limb}_y']], float) - b[0]) / b[2]
                              for f, b in valid if f[f'{limb}_score'] >= SCORE])
        if len(positions) >= MIN_POSITIONS:
            spans.append(float(np.linalg.norm(positions[:, None] - positions[None], axis=2).max()))
    return dict(limb_excursion=float(np.median(spans)) if len(spans) >= MIN_LIMBS else None, limbs_used=len(spans),
                pose_coverage=len(valid) / len(frames) if frames else 0.)


def features(points, floor=None, floor_rule=None, step=STEP):
    """Filas en la rejilla de 0,5 s (mismos instantes que flow/silhouette) con características de marcha y suelo."""
    from .silhouette import decide as floor_decide
    records = points.to_dict('records')
    fps = 1 / np.median(np.diff(points.seconds)) if len(points) > 1 else 1.
    stride = max(1, round(fps * step))
    count = len(records)
    grid = [f for f in range(count) if f % stride == 0 or f == count - 1]
    floor_rows = {}
    if floor is not None:
        floor_rows = {round(r['seconds'], 3): r for r in floor.to_dict('records')}
    rows = []
    for index, frame in enumerate(grid):
        start = grid[index - 1] if index else frame
        window = [r for r in records[start:frame + 1] if np.isfinite(r['box_w'])]
        row = dict(seconds=records[frame]['seconds'],
                   interval_seconds=records[frame]['seconds'] - records[start]['seconds'],
                   box_fraction=len(window) / (frame - start + 1))
        for limbs in LIMBS:
            values = interval_features(window, limbs)
            row.update({f'{key}_{limbs}': value for key, value in values.items()})
        match = floor_rows.get(round(row['seconds'], 3))
        row['floor_movement'] = floor_decide(match, floor_rule) if match is not None and floor_rule else 'unknown'
        rows.append(row)
    return rows


def decide(row, rule):
    if rule['kind'] not in ('pose', 'pose_or_floor'):
        raise ValueError(f'Regla desconocida: {rule}')
    excursion = row.get(f'limb_excursion_{rule["limbs"]}')
    excursion = None if excursion is None or (isinstance(excursion, float) and np.isnan(excursion)) else excursion
    movement = 'unknown'
    if excursion is not None and (row.get(f'pose_coverage_{rule["limbs"]}') or 0.) >= MIN_COVERAGE:
        movement = ('locomotion_candidate' if excursion >= rule['high'] else
                    'low_motion_candidate' if excursion <= rule['low'] else 'unknown')
    if rule['kind'] == 'pose_or_floor':
        floor = row.get('floor_movement', 'unknown')
        if floor == 'locomotion_candidate':
            movement = 'locomotion_candidate'
        elif movement == 'unknown' and floor == 'low_motion_candidate':
            movement = 'low_motion_candidate'
    return movement


def redecide(frame, rule):
    frame = frame.copy()
    frame['movement'] = ['unknown' if i == 0 else decide(r, rule) for i, r in enumerate(frame.to_dict('records'))]
    return frame


def analyze_case(paths, keypoint_dir, rule, floor_rule):
    points = pd.read_csv(Path(keypoint_dir) / f"{paths['video'].stem}.csv")
    floor = pd.read_csv(paths['floor']) if 'floor' in paths else None
    rows = features(points, floor, floor_rule)
    for index, row in enumerate(rows):
        row['movement'] = 'unknown' if index == 0 else decide(row, rule)
        row['evidence_quality'] = row[f'pose_coverage_{rule["limbs"]}']  # cobertura, no probabilidad
    return rows


def run(manifest, keypoint_dir, out, labels=None, decisive=False, rule_file=RULE_FILE):
    from .closure import CRITERIA, evaluate_cases, tally
    from .handheld import relabel, sufficiency
    from .silhouette import mixed_verdict
    if decisive and not Path(rule_file).is_file():
        raise SystemExit('NO SE EJECUTA: la regla de pose no está congelada. Primero desarrollo, después test.')
    floor_rule = json.loads(FLOOR_RULE_FILE.read_text())['rule']
    frozen = json.loads(Path(rule_file).read_text()) if decisive else None
    rule = frozen['rule'] if decisive else CANDIDATES[0]
    info = dict(rule=rule, floor_rule=floor_rule, score=SCORE, pad=PAD, step=STEP, min_positions=MIN_POSITIONS,
                min_limbs=MIN_LIMBS, min_coverage=MIN_COVERAGE, python=platform.python_version())
    result = evaluate_cases(manifest, out, 'pose', lambda paths: analyze_case(paths, keypoint_dir, rule, floor_rule),
                            info)
    if not decisive:
        return relabel(Path(out) / 'verdict.json', result, 'development_not_decisive', None, 'development_labels')
    test = tally(result['clips'])
    mixed = mixed_verdict(test, frozen['cross_validated'], CRITERIA)
    result.update(status=mixed['status'], passing_methods=['pose'] if mixed['passed'] else [], mixed=mixed,
                  test_counts=test, rule_file_sha256=sha256(rule_file), sufficiency=sufficiency(labels),
                  original_closure_status=result['status'])
    (Path(out) / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False))
    print(f"\nVídeos nuevos: reposo {test['resting_correct']}/{test['resting_clips']}, episodios falsos "
          f"{test['resting_false_events']}, locomoción {test['locomotion_hits']}/{test['locomotion_clips']}\n"
          f"Estado: {mixed['status']}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Locomoción por la marcha con DeepLabCut SuperAnimal-Quadruped')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('keypoints', help='Puntos del cuerpo (entorno con torch y deeplabcut)')
    command.add_argument('--manifest', required=True)
    command.add_argument('--out', required=True)
    command.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    command = sub.add_parser('run', help='Desarrollo (por defecto) o test decisivo con la regla congelada')
    for name in ('manifest', 'keypoints', 'out'):
        command.add_argument(f'--{name}', required=True)
    command.add_argument('--labels')
    command.add_argument('--decisive', action='store_true')
    command = sub.add_parser('develop', help='Validación cruzada y congelación de la regla')
    command.add_argument('--manifest', required=True)
    command.add_argument('--dev-out', required=True)
    command.add_argument('--output', default=str(RULE_FILE))
    args = parser.parse_args()
    if args.command == 'keypoints':
        manifest = Path(args.manifest).resolve()
        runner = SuperAnimalRunner(args.device)
        for case in json.loads(manifest.read_text())['cases']:
            video = (manifest.parent / Path(case['video']).expanduser()).resolve()
            masks = (manifest.parent / Path(case['masks']).expanduser()).resolve()
            out = Path(args.out)
            keypoints(video, masks, out / f'{video.stem}.csv', runner, preview=out / f'{video.stem}.jpg')
            print('PUNTOS', case['name'], flush=True)
        (Path(args.out) / 'runner.json').write_text(json.dumps(runner.info, indent=2))
    elif args.command == 'develop':
        from .silhouette import develop
        report = develop(args.manifest, args.dev_out, args.output, CANDIDATES, redecide)
        print(json.dumps(dict(rule=report['rule'], cross_validated=report['cross_validated']), indent=2))
    elif args.decisive and not args.labels:
        parser.error('--decisive requiere --labels')
    else:
        run(args.manifest, args.keypoints, args.out, args.labels, args.decisive)


if __name__ == '__main__':
    main()
