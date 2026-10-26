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
        silhouette.run(tmp_path / 'm.json', tmp_path / 'out', tmp_path / 'labels.csv', decisive=True)


def test_development_run_writes_features_and_is_never_a_verdict(tmp_path):
    video = scene(tmp_path, 3, 3, frames=49)
    (tmp_path / 'labels.csv').write_text('clip_id,video_id,reviewed,label,start_seconds,duration_seconds\n'
                                         'a,clip.avi,si,locomotion,0,4\n')
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='clip', video=video.name, masks='.',
                                                                 labels='labels.csv')])))
    result = silhouette.run(tmp_path / 'm.json', tmp_path / 'out')
    assert result['status'] == 'development_not_decisive' and result['passing_methods'] == []
    assert result['methods']['silhouette']['locomotion_hits'] == 1
    assert result['method']['rule'] == silhouette.DEFAULT_RULE
    import pandas as pd
    columns = pd.read_csv(tmp_path / 'out' / 'clip' / 'motion.csv').columns
    assert {'speed_lower', 'speed_upper', 'valid_fraction', 'elongation_range', 'log_area_range'} <= set(columns)
