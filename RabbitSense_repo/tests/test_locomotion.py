import numpy as np
import pandas as pd
import pytest

from rabbitsense.closure import verdict
from rabbitsense.evaluate import clip_metrics
from rabbitsense.motion import aggregate


@pytest.mark.parametrize('reverse', [False, True])
def test_path_does_not_cancel_when_rabbit_returns(reverse):
    evidence = [dict(reason='measured', vector=np.array([.04 * (-1 if reverse and i >= 5 else 1), 0.]),
                     uncertainty=.001, quality=1., camera_error_px=.01,
                     rabbit_points=30, linear=np.eye(2)) for i in range(10)]
    result = aggregate(evidence, .5, .2)
    assert result['movement'] == 'locomotion_candidate'
    assert result['speed_bodywidths_s'] == pytest.approx(.4 if reverse else .8)


def test_noise_is_not_accumulated_as_certain_locomotion():
    evidence = [dict(reason='measured', vector=np.array([.01 * (-1) ** i, 0.]),
                     uncertainty=.015, quality=1., camera_error_px=1.,
                     rabbit_points=30, linear=np.eye(2)) for i in range(10)]
    result = aggregate(evidence, .5, .2)
    assert result['movement'] != 'locomotion_candidate'


@pytest.mark.parametrize('displacement', [0., .011])
def test_missing_frame_does_not_extrapolate_evidence(displacement):
    evidence = [dict(reason='measured', vector=np.array([displacement, 0.]), uncertainty=.001,
                     quality=1., camera_error_px=.01, rabbit_points=30,
                     linear=np.eye(2)) for _ in range(9)]
    result = aggregate(evidence + [{'reason': 'tracking_lost'}], .5, .2)
    assert result['movement'] == 'unknown'
    assert result['speed_upper'] is None
    assert result['reason'] == 'incomplete_interval'
    assert result['speed_bodywidths_s'] == pytest.approx(9 * displacement / .5)


def measured(vector, uncertainty):
    return dict(reason='measured', vector=np.asarray(vector, dtype=float), uncertainty=uncertainty,
                quality=1., camera_error_px=.5, rabbit_points=30)


@pytest.mark.parametrize('fps', [12, 30, 60])
def test_stationary_noise_does_not_grow_into_locomotion_with_frame_rate(fps):
    rng = np.random.default_rng(fps)
    for _ in range(50):
        evidence = [measured(rng.normal(0, .002, 2), .013) for _ in range(round(fps * .5))]
        result = aggregate(evidence, .5, .2)
        assert result['movement'] != 'locomotion_candidate'
        assert result['speed_bodywidths_s'] < .2


@pytest.mark.parametrize('fps', [12, 30])
def test_measured_rest_is_low_motion_at_benchmark_frame_rates(fps):
    evidence = [measured([.002 * (-1) ** i, 0.], .013) for i in range(round(fps * .5))]
    assert aggregate(evidence, .5, .2)['movement'] == 'low_motion_candidate'


@pytest.mark.parametrize('fps', [12, 30, 60])
def test_translation_detected_independently_of_frame_rate(fps):
    evidence = [measured([.5 / fps, 0.], .013) for _ in range(round(fps * .5))]
    result = aggregate(evidence, .5, .2)
    assert result['movement'] == 'locomotion_candidate'
    assert result['speed_bodywidths_s'] == pytest.approx(.5)


def test_gap_splits_path_and_is_not_bridged():
    step = measured([.03, 0.], .001)
    result = aggregate([step] * 4 + [{'reason': 'tracking_lost'}] + [step] * 4, .5, .2)
    assert result['speed_bodywidths_s'] == pytest.approx(4 * .03 / .5)
    assert result['speed_upper'] is None


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

def test_detection_requires_evidence_and_never_rewards_abstention():
    frame = pd.DataFrame({'seconds': [0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4.],
                          'movement': ['unknown', 'low_motion_candidate', 'locomotion_candidate',
                                       'low_motion_candidate', 'low_motion_candidate',
                                       'unknown', 'unknown', 'unknown', 'unknown']})
    labels = pd.DataFrame([dict(clip_id=c, video_id='video.mp4', reviewed='si', label=label,
                                start_seconds=start, duration_seconds=2.)
                           for c, label, start in [('a', 'locomotion', 0.), ('b', 'resting', 0.), ('c', 'resting', 2.)]])
    result = clip_metrics(frame, labels, 'video.mp4')
    assert [r['detection_correct'] for r in result['clips']] == [True, False, False]
    assert result['detection'] == dict(locomotion_hits=1, locomotion_clips=1, resting_correct=0,
                                       resting_clips=2, resting_false_events=1)


def detection(hits, loco, correct, rest, false=0):
    return dict(legacy_clips=dict(detection=dict(locomotion_hits=hits, locomotion_clips=loco, resting_correct=correct,
                                                 resting_clips=rest, resting_false_events=false)))


def test_closure_verdict_pools_videos_and_demands_every_locomotion_video():
    suite = {'5177': dict(motion=dict(flow=detection(3, 3, 0, 0), v36=detection(3, 3, 0, 0))),
             'locomotion': dict(motion=dict(flow=detection(1, 2, 0, 0), v36=detection(0, 2, 0, 0))),
             'resting': dict(motion=dict(flow=detection(0, 0, 4, 4), v36=detection(0, 0, 4, 4))),
             'resting_972F': dict(motion=dict(flow=detection(0, 0, 3, 3), v36=detection(0, 0, 3, 3))),
             'resting_4944': dict(motion=dict(flow=detection(0, 0, 1, 3, 2), v36=detection(0, 0, 3, 3)))}
    result = verdict(suite)
    assert result['methods']['flow']['checks']['specificity']
    assert result['methods']['flow']['passed']
    assert not result['methods']['v36']['checks']['hit_in_every_locomotion_video']
    assert result['passing_methods'] == ['flow']
    assert result['status'] == 'closed_resolved_on_development_benchmark'
    suite['resting_4944']['motion']['flow'] = detection(0, 0, 0, 3, 3)
    assert verdict(suite)['status'] == 'closed_not_resolvable_with_existing_data'


def test_closure_verdict_rejects_too_few_clips():
    suite = {'a': dict(motion=dict(flow=detection(2, 2, 2, 2)))}
    result = verdict(suite)['methods']['flow']
    assert result['sensitivity'] == result['specificity'] == 1.
    assert not result['passed']
