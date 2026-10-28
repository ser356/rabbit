import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import activity, pointtrack
from test_pointtrack import RABBIT, FakeTracker, fake_frames, make_case  # noqa: F401 (fixture)

UNIT = np.sqrt(RABBIT[0] * RABBIT[1])


def test_continuous_displacement_matches_the_synthetic_scene(tmp_path, fake_frames):
    """pan: avanza 40 px/s de 0 a 4 s con cámara fija; quieto de 4 a 10 s mientras la cámara barre y se acicala."""
    video, masks = make_case(tmp_path, 'pan', 'pan')
    summary = activity.measure(video, masks, tmp_path / 'out', tmp_path / 'tracks' / 'pan', FakeTracker('pan', groom=True))
    rows = pd.read_csv(tmp_path / 'out' / 'actividad.csv')
    moving = rows[(rows.seconds > .5) & (rows.seconds <= 4.)]
    still = rows[(rows.seconds > 4.5) & (rows.seconds <= 9.5)]
    assert moving.displacement.to_numpy() == pytest.approx(np.full(len(moving), 20 / UNIT), rel=.05)
    assert still.displacement.max() < .05 and rows[rows.interval_seconds > 0].measured.all()
    expected = 4 * 40 / UNIT
    assert summary['displacement_body_units']['estimate'] == pytest.approx(expected, rel=.05)
    assert summary['displacement_body_units']['lower'] <= summary['displacement_body_units']['estimate']
    assert summary['displacement_per_measured_minute']['estimate'] == pytest.approx(expected / (summary['measured_seconds'] / 60))
    assert summary['unmeasured_seconds'] == pytest.approx(0, abs=1e-6)
    assert summary['rule_seconds']['clear_displacement'] > 2 and summary['rule_seconds']['low_motion'] > 3
    assert set(rows.columns) >= {'displacement_lower', 'displacement_upper', 'movement', 'span'}
    image = cv2.imread(str(tmp_path / 'out' / 'actividad.png'))
    assert image is not None and image.shape[1] == 1200
    stored = json.loads((tmp_path / 'out' / 'resumen.json').read_text())
    assert stored['video_sha256'] and 'no son centímetros' in stored['units']


def test_camera_following_the_rabbit_still_counts_as_displacement(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'follow', 'follow')
    summary = activity.measure(video, masks, tmp_path / 'out', tmp_path / 'tracks' / 'follow', FakeTracker('follow'))
    assert summary['displacement_body_units']['estimate'] == pytest.approx(4 * 40 / UNIT, rel=.05)


def test_without_mask_nothing_is_measured_and_nothing_is_rest(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'blind', 'pan', masked=False)
    summary = activity.measure(video, masks, tmp_path / 'out', tmp_path / 'tracks' / 'blind', FakeTracker('pan'))
    assert summary['measured_seconds'] == 0 and summary['displacement_per_measured_minute']['estimate'] is None
    assert summary['rule_seconds']['low_motion'] == 0
    assert summary['unmeasured_seconds'] == pytest.approx(pointtrack.frame_times(video)[-1])
