import json

import cv2
import numpy as np
import pytest

from rabbitsense.body import BodyMotion, analyze_body
from rabbitsense.motion import timeline
from rabbitsense.tracking import Anchors, transform_points
from test_motion import scene


@pytest.mark.parametrize('translation,angle,scale', [(0., 0., 1.), (0., 30., 1.), (0., 0., 1.2),
                                                    (30., 0., 1.), (30., 30., 1.)])
def test_body_decomposition_with_moving_camera(translation, angle, scale):
    cv2.setRNGSeed(0)
    image, box = scene()
    tracker = BodyMotion()
    tracker.reset(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), box)
    original = tracker.points.copy()
    center = tuple(box[:2] + box[2:] / 2)
    for t in np.linspace(0., 1., 11)[1:]:
        body = cv2.getRotationMatrix2D(center, angle * t, 1. + (scale - 1.) * t)
        body[:, 2] += [translation * np.sin(np.pi * t), 0.]
        camera = cv2.getRotationMatrix2D((160., 120.), -10. * t, 1. + .1 * t)
        camera[:, 2] += [15. * t, -8. * t]
        current = original.copy()
        current[tracker.foreground] = transform_points(body, original[tracker.foreground])
        tracker.points = transform_points(camera, current).astype(np.float32)
        tracker.paths.append(tracker.points.copy())
        tracker.errors.append(np.zeros(len(current)))
    row = tracker.measure(.5, .2)
    assert row['movement'] == ('locomotion_candidate' if translation else 'low_motion_candidate')
    assert row['translation_span_bodywidths'] == pytest.approx(translation / 80., abs=.001)
    assert row['body_rotation_degrees'] == pytest.approx(angle, abs=.01)
    assert row['body_scale_ratio'] == pytest.approx(scale, abs=.001)


def test_image_tracking_preserves_ids_and_rejects_missing_guide():
    cv2.setRNGSeed(0)
    tracker = BodyMotion()
    image, box = scene()
    tracker.reset(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), box)
    count = len(tracker.points)
    for t in range(1, 11):
        image, box = scene(camera=(t, 0), animal=(2 * t, 0))
        tracker.update(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), box)
        assert len(tracker.points) == count
    row = tracker.measure(.5, .2)
    assert row['movement'] == 'locomotion_candidate'
    assert row['translation_span_bodywidths'] == pytest.approx(.25, abs=.02)
    tracker.update(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    assert tracker.measure(.5, .2)['movement'] == 'unknown'


def test_background_plane_disagreement_abstains():
    cv2.setRNGSeed(0)
    image, box = scene()
    tracker = BodyMotion()
    tracker.reset(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), box)
    original = tracker.points.copy()
    for t in range(1, 4):
        current = original.copy()
        current[(~tracker.foreground) & (original[:, 0] > 150), 1] += 10 * t
        tracker.points = current
        tracker.paths.append(current.copy())
        tracker.errors.append(np.zeros(len(current)))
    result = tracker.measure(.5, .2)
    assert result['movement'] == 'unknown'
    assert result['reason'] == 'camera_spatial_validation_failed'


def test_local_features_do_not_establish_whole_body_motion():
    image, box = scene()
    tracker = BodyMotion()
    tracker.reset(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), box)
    fg = tracker.foreground
    tracker.points[fg] = np.random.default_rng(2).uniform([140., 110.], [150., 120.], (fg.sum(), 2))
    tracker.paths = [tracker.points.copy(), tracker.points.copy()]
    tracker.errors = [np.zeros(len(tracker.points))]
    assert tracker.measure(.5, .2)['reason'] == 'body_support_too_local'


def test_dead_points_are_not_reacquired():
    image, box = scene()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tracker = BodyMotion()
    tracker.reset(gray, box)
    tracker.points[:20] = np.nan
    for _ in range(3):
        tracker.update(gray, box)
        assert np.isnan(tracker.points[:20]).all()


def test_mask_points_exclude_floor_inside_loose_box():
    image, box = scene()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = np.zeros_like(gray)
    cv2.ellipse(foreground, (150, 120), (35, 25), 0, 0, 360, 255, -1)
    background = 255 - cv2.dilate(foreground, np.ones((21, 21), np.uint8))
    tracker = BodyMotion()
    tracker.reset(gray, np.array([70., 40., 180., 160.]), (foreground, background))
    xy = np.rint(tracker.points).astype(int)
    assert tracker.foreground.sum() >= 12
    assert (~tracker.foreground).sum() >= 24
    assert (foreground[xy[tracker.foreground, 1], xy[tracker.foreground, 0]] > 0).all()
    assert (background[xy[~tracker.foreground, 1], xy[~tracker.foreground, 0]] > 0).all()
    empty = np.zeros_like(gray)
    tracker.update(gray, box, (empty, background))
    assert np.isnan(tracker.points[tracker.foreground]).all()
    assert tracker.measure(.5, .2)['movement'] == 'unknown'


def test_body_video_covers_partial_interval_without_inventing_guides(tmp_path):
    image, box = scene()
    video = tmp_path / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 240))
    assert writer.isOpened()
    for _ in range(13):
        writer.write(image)
    writer.release()
    anchors = Anchors(np.array([0., 1., 1.2]), [box, box, None])
    overlay = tmp_path / 'overlay.mp4'
    rows = analyze_body(video, anchors, .5, .2, overlay)
    assert [r['seconds'] for r in rows] == [0., .5, 1., 1.2]
    assert rows[-1]['interval_seconds'] == .2
    assert rows[-1]['movement'] == 'unknown'
    assert rows[-1]['speed_upper'] is None
    assert rows[1]['movement'] == 'low_motion_candidate'
    assert timeline(rows)[-1]['end_seconds'] == 1.2
    json.dumps(rows, allow_nan=False)
    capture = cv2.VideoCapture(str(overlay))
    assert capture.get(cv2.CAP_PROP_FPS) == 10.
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 13


@pytest.mark.parametrize('step,threshold', [(0., .2), (float('nan'), .2), (.5, 0.), (.5, float('inf'))])
def test_body_rejects_invalid_parameters(step, threshold):
    with pytest.raises(ValueError, match='step y threshold'):
        analyze_body('unused.mp4', Anchors(np.array([]), []), step, threshold)