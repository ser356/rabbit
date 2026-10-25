import numpy as np
import pandas as pd
import pytest

from rabbitsense.evaluate import clip_metrics
from rabbitsense.motion import aggregate


@pytest.mark.parametrize('reverse', [False, True])
def test_path_does_not_cancel_when_rabbit_returns(reverse):
    evidence = [dict(reason='measured', vector=np.array([.04 * (-1 if reverse and i >= 5 else 1), 0.]),
                     uncertainty=.001, quality=1., camera_error_px=.01,
                     rabbit_points=30, linear=np.eye(2)) for i in range(10)]
    result = aggregate(evidence, .5, .2)
    assert result['movement'] == 'locomotion_candidate'
    assert result['speed_bodywidths_s'] == pytest.approx(.8)


def test_noise_is_not_accumulated_as_certain_locomotion():
    evidence = [dict(reason='measured', vector=np.array([.01 * (-1) ** i, 0.]),
                     uncertainty=.015, quality=1., camera_error_px=1.,
                     rabbit_points=30, linear=np.eye(2)) for i in range(10)]
    result = aggregate(evidence, .5, .2)
    assert result['movement'] != 'locomotion_candidate'


@pytest.mark.parametrize('displacement', [0., .012])
def test_missing_frame_does_not_extrapolate_evidence(displacement):
    evidence = [dict(reason='measured', vector=np.array([displacement, 0.]), uncertainty=.001,
                     quality=1., camera_error_px=.01, rabbit_points=30,
                     linear=np.eye(2)) for _ in range(9)]
    result = aggregate(evidence + [{'reason': 'tracking_lost'}], .5, .2)
    assert result['movement'] == 'unknown'
    assert result['speed_upper'] is None
    assert result['reason'] == 'incomplete_interval'
    assert result['speed_bodywidths_s'] == pytest.approx(9 * displacement / .5)


def test_clip_event_does_not_replace_majority_metric():
    frame = pd.DataFrame({'seconds': [0., .5, 1., 1.5, 2.],
                          'movement': ['unknown', 'low_motion_candidate', 'locomotion_candidate',
                                       'low_motion_candidate', 'low_motion_candidate']})
    labels = pd.DataFrame([dict(clip_id='a', video_id='video.mp4', reviewed='si',
                               label='locomotion', start_seconds=0., duration_seconds=2.)])
    result = clip_metrics(frame, labels, 'video.mp4')
    assert result['accuracy_including_abstentions'] == 0.
    assert result['clips'][0]['prediction'] == 'low_motion_candidate'
    assert result['clips'][0]['candidate_events'] == [dict(start_seconds=.5, end_seconds=1., duration_seconds=.5)]