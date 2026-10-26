import json

import cv2
import numpy as np
import pytest

from rabbitsense import silhouette
from rabbitsense.motion import sha256


def scene(tmp_path, rabbit_speed, camera_speed, frames=25, fps=12.):
    rng = np.random.default_rng(5)
    world = cv2.GaussianBlur(rng.integers(0, 255, (400, 700), dtype=np.uint8), (3, 3), 0)
    fur = cv2.GaussianBlur(rng.integers(120, 255, (60, 60), dtype=np.uint8), (3, 3), 0)
    video = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), fps, (320, 240))
    (tmp_path / 'masks').mkdir()
    for frame in range(frames):
        canvas = world.copy()
        rx, cx = 200 + rabbit_speed * frame, 60 + camera_speed * frame
        canvas[150:210, rx:rx + 60] = fur
        image = canvas[60:300, cx:cx + 320]
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        mask = np.zeros((240, 320), np.uint8)
        mask[90:150, rx - cx:rx - cx + 60] = 255
        cv2.imwrite(str(tmp_path / 'masks' / f'{frame:06d}.png'), mask)
    writer.release()
    (tmp_path / 'summary.json').write_text(json.dumps(dict(fps=fps, frames=frames,
                                                           inputs=dict(video=dict(sha256=sha256(video))))))
    return video


@pytest.mark.parametrize('rabbit,camera,expected', [(0, 3, 'low_motion_candidate'), (3, 3, 'locomotion_candidate'),
                                                    (0, 0, 'low_motion_candidate'), (3, 0, 'locomotion_candidate')])
def test_floor_reference_separates_rabbit_from_camera(tmp_path, rabbit, camera, expected):
    cv2.setRNGSeed(0)
    rows = silhouette.analyze(scene(tmp_path, rabbit, camera), tmp_path)
    assert [r['seconds'] for r in rows] == pytest.approx([0., .5, 1., 1.5, 2.])
    assert rows[0]['movement'] == 'unknown'
    assert [r['movement'] for r in rows[1:]] == [expected] * 4
    if rabbit:
        assert np.median([r['speed_bodywidths_s'] for r in rows[1:]]) == pytest.approx(3 * 12 / 60, rel=.15)
    assert all(r['mask_fraction'] == 1 and r['elongation_range'] == pytest.approx(0, abs=.01) for r in rows[1:])


def test_missing_masks_and_wrong_video_abstain_or_fail(tmp_path):
    video = scene(tmp_path, 3, 3)
    cv2.imwrite(str(tmp_path / 'masks' / '000008.png'), np.zeros((240, 320), np.uint8))
    rows = silhouette.analyze(video, tmp_path)
    assert rows[2]['movement'] == 'unknown' and rows[2]['mask_fraction'] < 1
    summary = json.loads((tmp_path / 'summary.json').read_text())
    summary['inputs']['video']['sha256'] = '0' * 64
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='no corresponden'):
        silhouette.analyze(video, tmp_path)


def test_decisive_run_refuses_until_rule_is_frozen(tmp_path):
    with pytest.raises(SystemExit, match='congelada'):
        silhouette.run(tmp_path / 'm.json', tmp_path / 'out', tmp_path / 'labels.csv', decisive=True,
                       rule_file=tmp_path / 'missing.json')


def row(lower, upper, shape=0., coverage=1., mask=1.):
    return dict(speed_lower=lower, speed_upper=upper, elongation_range=shape, valid_fraction=coverage, mask_fraction=mask)


def test_candidate_rules():
    floor, gated, fallback = (dict(kind='floor', threshold=.2), dict(kind='floor_shape', threshold=.2, shape=.1),
                              dict(kind='floor_or_shape', threshold=.2, shape=.2))
    assert silhouette.decide(row(.3, .5), floor) == 'locomotion_candidate'
    assert silhouette.decide(row(.0, .1), floor) == 'low_motion_candidate'
    assert silhouette.decide(row(.1, .3), floor) == 'unknown'
    assert silhouette.decide(row(.3, .5, coverage=.5), floor) == 'unknown'
    assert silhouette.decide(row(.3, .5, shape=.05), gated) == 'unknown'
    assert silhouette.decide(row(.3, .5, shape=.15), gated) == 'locomotion_candidate'
    assert silhouette.decide(row(float('nan'), float('nan'), shape=.3), fallback) == 'locomotion_candidate'
    assert silhouette.decide(row(float('nan'), float('nan'), shape=.3, mask=.9), fallback) == 'unknown'
    assert silhouette.decide(row(.0, .1, shape=.3), fallback) == 'low_motion_candidate'
    assert len(silhouette.CANDIDATES) == 28 and silhouette.CANDIDATES[1] == silhouette.DEFAULT_RULE


def development_case(tmp_path, name, lower, label):
    import pandas as pd
    rows = [dict(seconds=0., movement='unknown', **row(None, None))]
    rows += [dict(seconds=.5 * k, movement='unknown', **row(lower, lower + .05)) for k in range(1, 9)]
    (tmp_path / 'dev' / name).mkdir(parents=True)
    pd.DataFrame(rows).to_csv(tmp_path / 'dev' / name / 'motion.csv', index=False)
    (tmp_path / f'{name}.csv').write_text('clip_id,video_id,reviewed,label,start_seconds,duration_seconds\n'
                                          f'c,{name}.MOV,si,{label},0,4\n')
    return dict(name=name, video=f'{name}.MOV', labels=f'{name}.csv')


def test_develop_cross_validates_by_video_and_freezes_rule(tmp_path):
    cases = [development_case(tmp_path, 'a', .35, 'locomotion'), development_case(tmp_path, 'b', .25, 'locomotion'),
             development_case(tmp_path, 'c', .02, 'resting'), development_case(tmp_path, 'd', .12, 'resting')]
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=cases)))
    report = silhouette.develop(tmp_path / 'm.json', tmp_path / 'dev', tmp_path / 'rule.json')
    assert report['rule'] == dict(kind='floor', threshold=.2)
    assert report['held_out']['b']['rule']['threshold'] in (.15, .2)
    assert report['held_out']['d']['rule']['threshold'] <= .3
    assert report['cross_validated']['locomotion_clips'] == 2 and report['cross_validated']['resting_clips'] == 2
    assert json.loads((tmp_path / 'rule.json').read_text())['rule'] == report['rule']


def counts(hits, loco, correct, rest, per_video=None):
    return dict(locomotion_hits=hits, locomotion_clips=loco, resting_correct=correct, resting_clips=rest,
                resting_false_events=0, per_video=per_video or dict(v=dict(locomotion_hits=hits, locomotion_clips=loco)))


def test_mixed_verdict_uses_cross_validation_only_without_new_locomotion():
    from rabbitsense.closure import CRITERIA
    cv = counts(5, 6, 8, 10)
    assert silhouette.mixed_verdict(counts(0, 1, 9, 11), cv, CRITERIA)['status'] == \
        'resolved_specificity_independent_sensitivity_cross_validated'
    assert silhouette.mixed_verdict(counts(0, 1, 7, 11), cv, CRITERIA)['status'] == 'not_resolved'
    assert silhouette.mixed_verdict(counts(0, 0, 9, 9), cv, CRITERIA)['status'] == 'descriptive_only_insufficient_sample'
    independent = silhouette.mixed_verdict(counts(3, 5, 9, 11), cv, CRITERIA)
    assert independent['sensitivity_source'] == 'new_videos' and independent['status'] == 'not_resolved'
    assert silhouette.mixed_verdict(counts(5, 5, 9, 11), counts(0, 6, 0, 10), CRITERIA)['status'] == \
        'resolved_independent_test'


def test_development_run_writes_features_and_is_never_a_verdict(tmp_path):
    video = scene(tmp_path, 3, 3, frames=49)
    (tmp_path / 'labels.csv').write_text('clip_id,video_id,reviewed,label,start_seconds,duration_seconds\n'
                                         'a,clip.avi,si,locomotion,0,4\n')
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='clip', video=video.name, masks='.',
                                                                 labels='labels.csv')])))
    result = silhouette.run(tmp_path / 'm.json', tmp_path / 'out')
    assert result['status'] == 'development_not_decisive' and result['passing_methods'] == []
    (tmp_path / 'rule.json').write_text(json.dumps(dict(rule=dict(kind='floor', threshold=.2),
                                                        cross_validated=counts(5, 5, 10, 10))))
    labels = tmp_path / 'test_labels.csv'
    labels.write_text('clip_id,video_id,reviewed,label,start_seconds,duration_seconds\n'
                      'a,clip.avi,si,locomotion,0,4\n')
    decisive = silhouette.run(tmp_path / 'm.json', tmp_path / 'test', labels, decisive=True,
                              rule_file=tmp_path / 'rule.json')
    assert decisive['status'] == 'descriptive_only_insufficient_sample' and decisive['passing_methods'] == []
    assert decisive['test_counts']['locomotion_hits'] == 1
    assert result['methods']['silhouette']['locomotion_hits'] == 1
    assert result['method']['rule'] == silhouette.DEFAULT_RULE
    import pandas as pd
    columns = pd.read_csv(tmp_path / 'out' / 'clip' / 'motion.csv').columns
    assert {'speed_lower', 'speed_upper', 'valid_fraction', 'elongation_range', 'log_area_range'} <= set(columns)
