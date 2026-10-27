import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import pose
from rabbitsense.motion import sha256

BASE = {'neck_base': (40, 0), 'back_middle': (0, 0), 'tail_base': (-40, 0), 'body_middle_left': (0, 15),
        'front_left_paw': (30, 20), 'front_right_paw': (30, -20), 'back_left_paw': (-30, 20),
        'back_right_paw': (-30, -20), 'back_left_knee': (-20, 18), 'back_right_knee': (-20, -18)}


def record(frame, fps=10., shift=(0, 0), angle=0., scale=1., hop=0.):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    row = dict(frame=frame, seconds=frame / fps, box_x=0., box_y=0., box_w=100., box_h=60.)
    for name in pose.BODYPARTS:
        point = np.array(BASE.get(name, (0, 0)), float)
        if name.startswith('back_') and name.endswith(('paw', 'knee')):
            point[0] += hop * 40 * np.sin(frame)
        x, y = rotation @ point * scale + np.array([200, 150]) + shift
        row.update({f'{name}_x': x, f'{name}_y': y, f'{name}_score': .9 if name in BASE else .1})
    return row


def test_limb_excursion_ignores_camera_translation_rotation_and_zoom():
    still = [record(f, shift=(5 * f, -3 * f), angle=.1 * f, scale=1 + .02 * f) for f in range(6)]
    hopping = [record(f, hop=1.) for f in range(6)]
    assert pose.interval_features(still, 'all')['limb_excursion'] == pytest.approx(0, abs=1e-6)
    moving = pose.interval_features(hopping, 'hind')
    assert moving['limb_excursion'] > .5 and moving['limbs_used'] == 4 and moving['pose_coverage'] == 1
    assert pose.interval_features(hopping, 'all')['limbs_used'] == 6


def test_missing_body_axis_or_limbs_abstain():
    rows = [record(f) for f in range(6)]
    for row in rows:
        row['neck_base_score'] = row['back_base_score'] = .1
    assert pose.interval_features(rows, 'all') == dict(limb_excursion=None, limbs_used=0, pose_coverage=0.)


def test_features_grid_floor_merge_and_rules():
    points = pd.DataFrame([record(f, hop=1. if f > 10 else 0.) for f in range(21)])
    floor = pd.DataFrame(dict(seconds=[0., .5, 1., 1.5, 2.], speed_lower=[None, .0, .5, .0, .0],
                              speed_upper=[None, .1, .9, .1, .1], valid_fraction=[0., 1., 1., 1., 0.],
                              elongation_range=[0.] * 5, mask_fraction=[1.] * 5))
    rows = pose.features(points, floor, dict(kind='floor', threshold=.3))
    assert [r['seconds'] for r in rows] == [0., .5, 1., 1.5, 2.]
    assert [r['floor_movement'] for r in rows] == ['unknown', 'low_motion_candidate', 'locomotion_candidate',
                                                   'low_motion_candidate', 'unknown']
    only_pose = dict(kind='pose', limbs='hind', high=.3, low=.05)
    combined = dict(only_pose, kind='pose_or_floor')
    assert [pose.decide(r, only_pose) for r in rows[1:]] == ['low_motion_candidate', 'low_motion_candidate',
                                                             'locomotion_candidate', 'locomotion_candidate']
    assert [pose.decide(r, combined) for r in rows[1:]] == ['low_motion_candidate', 'locomotion_candidate',
                                                            'locomotion_candidate', 'locomotion_candidate']
    assert len(pose.CANDIDATES) == 40


def test_keypoints_uses_mask_boxes_and_skips_empty_masks(tmp_path):
    video = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (160, 120))
    (tmp_path / 'masks').mkdir()
    for frame in range(5):
        writer.write(np.full((120, 160, 3), 90, np.uint8))
        mask = np.zeros((120, 160), np.uint8)
        if frame != 2:
            mask[40:80, 50:110] = 255
        cv2.imwrite(str(tmp_path / 'masks' / f'{frame:06d}.png'), mask)
    writer.release()
    (tmp_path / 'summary.json').write_text(json.dumps(dict(fps=10., frames=5, inputs=dict(video=dict(sha256=sha256(video))))))
    boxes = []

    def runner(images, batch_boxes):
        boxes.extend(batch_boxes)
        assert all(image.shape == (120, 160, 3) for image in images)
        return np.array([[[b[0] + 1, b[1] + 1, .8]] * len(pose.BODYPARTS) for b in batch_boxes])

    frame = pose.keypoints(video, tmp_path, tmp_path / 'out' / 'clip.csv', runner, batch=3, preview=tmp_path / 'p.jpg')
    assert frame.frame.tolist() == [0, 1, 2, 3, 4] and frame.seconds.tolist() == [0., .1, .2, .3, .4]
    assert np.isnan(frame.loc[2, 'box_w']) and len(boxes) == 4
    assert boxes[0] == pytest.approx([44., 36., 72., 48.])
    assert frame.loc[0, 'nose_x'] == pytest.approx(45.)
    assert cv2.imread(str(tmp_path / 'p.jpg')) is not None


def test_superanimal_runner_respects_given_boxes(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('deeplabcut')
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.models import PoseModel
    config = PoseConfig.build_for_superanimal_inference(super_animal=pose.SUPER_ANIMAL, model_name=pose.MODEL,
                                                        detector_name=pose.DETECTOR, max_individuals=1, device='cpu')
    torch.save({'model': PoseModel.build(config['model']).state_dict()}, tmp_path / 'random.pt')
    runner = pose.SuperAnimalRunner('cpu', snapshot=tmp_path / 'random.pt', batch_size=2)
    image = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    points = runner([image, image], [[100., 80., 60., 40.], [10., 10., 50., 50.]])
    assert points.shape == (2, len(pose.BODYPARTS), 3)
    assert 90 <= points[0, :, 0].min() and points[0, :, 0].max() <= 170
    assert points[1, :, 0].max() <= 70


def test_decisive_pose_run_refuses_until_rule_is_frozen(tmp_path):
    with pytest.raises(SystemExit, match='congelada'):
        pose.run(tmp_path / 'm.json', tmp_path, tmp_path / 'out', tmp_path / 'l.csv', decisive=True,
                 rule_file=tmp_path / 'missing.json')


def test_development_run_end_to_end(tmp_path):
    (tmp_path / 'kp').mkdir()
    pd.DataFrame([record(f, hop=1. if f < 40 else 0.) for f in range(81)]).to_csv(tmp_path / 'kp' / 'clip.csv', index=False)
    (tmp_path / 'clip.mp4').write_bytes(b'x')
    (tmp_path / 'labels.csv').write_text('clip_id,video_id,reviewed,label,start_seconds,duration_seconds\n'
                                         'a,clip.mp4,si,locomotion,0,4\nb,clip.mp4,si,resting,4,4\n')
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='clip', video='clip.mp4', labels='labels.csv')])))
    result = pose.run(tmp_path / 'm.json', tmp_path / 'kp', tmp_path / 'out')
    counts = result['methods']['pose']
    assert result['status'] == 'development_not_decisive'
    assert (counts['locomotion_hits'], counts['resting_correct'], counts['resting_false_events']) == (1, 1, 0)
    assert {'limb_excursion_hind', 'pose_coverage_all', 'floor_movement'} <= set(
        pd.read_csv(tmp_path / 'out' / 'clip' / 'motion.csv').columns)
