import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense.evaluate import box_metrics, clip_metrics, heldout_tracking
from rabbitsense.bridge import refine_video
from rabbitsense.motion import aggregate, analyze, camera_motion, frame_evidence, timeline
from rabbitsense.tracking import Anchors, GuidedTracker, load_anchors, transform_points


def scene(camera=(0, 0), animal=(0, 0)):
    rng = np.random.default_rng(17)
    background = rng.integers(25, 100, (240, 320), dtype=np.uint8)
    patch = rng.integers(150, 230, (70, 80), dtype=np.uint8)
    transform = np.array([[1., 0., camera[0]], [0., 1., camera[1]]])
    image = cv2.warpAffine(background, transform, (320, 240), borderMode=cv2.BORDER_REFLECT)
    x, y = 110 + camera[0] + animal[0], 85 + camera[1] + animal[1]
    image[y:y + 70, x:x + 80] = patch
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), np.array([x, y, 80, 70], dtype=float)


@pytest.mark.parametrize('camera,animal', [((4, 2), (0, 0)), ((4, 2), (3, 0)), ((-3, 3), (0, -2))])
def test_visual_motion_separates_camera(camera, animal):
    cv2.setRNGSeed(0)
    first, box0 = scene()
    second, box1 = scene(camera, animal)
    gray0, gray1 = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (first, second)]
    tracker = GuidedTracker()
    tracker.update(first, None, box0)
    box, diagnostic = tracker.update(second, gray0, box1)
    evidence = frame_evidence(gray0, gray1, box, diagnostic)
    assert evidence['reason'] == 'measured'
    np.testing.assert_allclose(evidence['vector'] * 80, animal, atol=.25)
    result = aggregate([evidence] * 15, .5, .2)
    assert result['movement'] == ('locomotion_candidate' if any(animal) else 'low_motion_candidate')


def test_camera_affine_and_blank_rejection():
    image, box = scene()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    matrix = np.array([[1.01, .02, -3.], [-.01, 1., 2.]])
    cur = cv2.warpAffine(gray, matrix, gray.shape[::-1])
    camera = camera_motion(gray, cur, box, box)
    assert camera is not None
    point = np.array([[150., 120.]])
    np.testing.assert_allclose(transform_points(camera['matrix'], point), transform_points(matrix, point), atol=.3)
    assert camera_motion(np.zeros_like(gray), np.zeros_like(gray), box, box) is None


def test_unknown_and_uncertainty_not_forced_to_motion():
    assert aggregate([], 0., .2)['movement'] == 'unknown'
    item = dict(reason='measured', vector=np.array([.1, 0.]), uncertainty=.1,
                quality=1., camera_error_px=1., rabbit_points=10, linear=np.eye(2))
    assert aggregate([item], .5, .2)['movement'] == 'unknown'
    assert aggregate([item] + [{'reason': 'lost'}] * 3, .5, .2)['valid_fraction'] == .25


def test_anchor_validation_and_units(tmp_path):
    path = tmp_path / 'anchors.json'
    path.write_text(json.dumps({'reference_width': 100, 'reference_height': 100,
                                'anchors': [{'time': 0, 'bbox': [10, 20, 40, 60]}]}))
    data = load_anchors(path, 200, 300)
    np.testing.assert_array_equal(data.boxes[0], [20, 60, 60, 120])
    assert data.interpolate(1) is None
    path.write_text(json.dumps({'width': 100, 'height': 100,
                                'marks': [{'time': 0, 'box': [0, 0, 20, 20]}] * 2}))
    with pytest.raises(ValueError, match='únicos'):
        load_anchors(path, 100, 100)


def test_clip_metrics_do_not_relabel_exploration():
    frame = pd.DataFrame({'seconds': [0., .5, 1., 1.5, 2.],
                           'movement': ['unknown'] + ['locomotion_candidate'] * 4})
    labels = pd.DataFrame([dict(clip_id='a', video_id='test.mp4', reviewed='si', label='locomotion',
                                start_seconds=.5, duration_seconds=1.),
                           dict(clip_id='b', video_id='test.mp4', reviewed='si', label='exploring_sniffing',
                                start_seconds=0., duration_seconds=2.)])
    result = clip_metrics(frame, labels, 'test.mp4')
    assert result['n'] == 1
    assert result['clips'][0]['evaluated_seconds'] == 1.
    assert result['accuracy_including_abstentions'] == 1.
    assert result['balanced_accuracy'] is None


def test_pipeline_full_duration_and_missing_guides(tmp_path):
    image, box = scene()
    video = tmp_path / 'test.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 240))
    assert writer.isOpened()
    for _ in range(13):
        writer.write(image)
    writer.release()
    anchors = tmp_path / 'anchors.json'
    anchors.write_text(json.dumps({'width': 320, 'height': 240,
                                   'marks': [{'time': 0., 'box': box.tolist()},
                                             {'time': .5, 'box': box.tolist()},
                                             {'time': 1., 'box': box.tolist()},
                                             {'time': 1.2, 'box': None}]}))
    output = tmp_path / 'result'
    result = analyze(video, anchors, output, video_out=True)
    assert result['frames'] == 13
    assert not (output / 'body_motion.csv').exists()
    assert 'body_movement' not in result
    frame = pd.read_csv(output / 'motion.csv')
    assert frame.iloc[-1].seconds == 1.2
    assert frame.iloc[-1].movement == 'unknown'
    intervals = json.loads((output / 'timeline.json').read_text())
    assert intervals[-1]['end_seconds'] == 1.2
    overlay = cv2.VideoCapture(str(output / 'overlay.mp4'))
    assert overlay.get(cv2.CAP_PROP_FRAME_COUNT) == 13
    assert overlay.get(cv2.CAP_PROP_FPS) == 10.
    overlay.release()
    holdout = heldout_tracking(video, anchors)
    assert holdout['summary']['interpolation']['n'] == 1
    assert holdout['summary']['interpolation']['mean_iou_all'] == 1.
    with pytest.raises(ValueError, match='step'):
        analyze(video, anchors, output, step=float('nan'))


def test_absent_boxes_penalize_iou():
    assert box_metrics(None, np.array([0, 0, 10, 10]))['iou'] == 0.


def test_timeline_duration_weighted_quality():
    rows = [dict(seconds=.5, interval_seconds=.5, movement='unknown', evidence_quality=.2),
            dict(seconds=.7, interval_seconds=.2, movement='unknown', evidence_quality=.9)]
    result = timeline(rows)
    assert len(result) == 1
    assert result[0]['evidence_quality'] == pytest.approx((.1 + .18) / .7)


def test_visual_bridge_preserves_anchors_and_original_guides(tmp_path):
    video = tmp_path / 'bridge.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 240))
    for offset in (0, 3, 8, 9, 10):
        image, _ = scene(animal=(offset, 0))
        writer.write(image)
    writer.release()
    _, first = scene()
    _, last = scene(animal=(10, 0))
    anchors = Anchors(np.array([0., .4]), [first, last])
    result = refine_video(video, anchors)
    np.testing.assert_allclose(result.refined_x, [110., 113., 118., 119., 120.], atol=.2)
    np.testing.assert_allclose(result.x, [110., 112.5, 115., 117.5, 120.])
    assert (result.refinement == 'persistent_visual_bridge').all()
    assert (result.bridge_points >= 6).all()