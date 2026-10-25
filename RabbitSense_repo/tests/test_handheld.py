import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import handheld
from rabbitsense.tracking import load_anchors


def labels_csv(path, rows):
    pd.DataFrame(rows, columns=['video', 'start_s', 'end_s', 'label', 'review_status', 'notes']).to_csv(path, index=False)
    return path


def test_masks_become_guides_and_empty_masks_stay_missing(tmp_path):
    (tmp_path / 'masks').mkdir()
    for frame in range(4):
        mask = np.zeros((60, 80), np.uint8)
        if frame != 2:
            mask[10 + frame:30 + frame, 20:50] = 255
        cv2.imwrite(str(tmp_path / 'masks' / f'{frame:06d}.png'), mask)
    (tmp_path / 'summary.json').write_text(json.dumps(dict(fps=10., frames=4)))
    assert handheld.masks_to_anchors(tmp_path, tmp_path / 'anchors.json') == (3, 4)
    anchors = load_anchors(tmp_path / 'anchors.json', 160, 120)
    assert anchors.boxes[2] is None
    np.testing.assert_allclose(anchors.boxes[1], [40, 22, 60, 40])
    assert anchors.at_frame(3, 10.)[0]


def test_only_continuous_video_verification_of_long_clips_counts(tmp_path):
    rows = [['IMG_1', 0, 4, 'locomotion', handheld.VERIFIED, ''],
            ['IMG_1', 4, 8, 'resting', 'revisado_fotogramas_0_5s_pendiente_video_continuo', 'provisional'],
            ['IMG_1', 8, 10, 'resting', handheld.VERIFIED, 'corto'],
            ['IMG_1', 10, 14, 'resting', handheld.VERIFIED, None]]
    report = handheld.prepare(labels_csv(tmp_path / 'in.csv', rows), tmp_path / 'out.csv')
    result = pd.read_csv(tmp_path / 'out.csv')
    assert result.reviewed.tolist() == ['si', 'no', 'no', 'si']
    assert result.video_id.unique().tolist() == ['IMG_1.MOV']
    assert report['locomotion'] == dict(verified=1, pending_candidates=0, required=5, reachable_if_all_pending_confirmed=False)
    assert report['resting']['verified'] == 1 and report['resting']['pending_candidates'] == 1
    assert not report['sufficient']


def test_run_refuses_without_enough_verified_clips(tmp_path, monkeypatch):
    rows = [['IMG_1', 4 * i, 4 * i + 4, 'locomotion' if i < 5 else 'resting', handheld.VERIFIED, ''] for i in range(14)]
    handheld.prepare(labels_csv(tmp_path / 'in.csv', rows), tmp_path / 'labels.csv')
    calls = []
    monkeypatch.setattr('rabbitsense.closure.close', lambda *args: calls.append(args) or {})
    assert handheld.run('manifest.json', tmp_path / 'labels.csv', tmp_path / 'out') is None
    assert calls == []
    rows.append(['IMG_1', 56, 60, 'resting', handheld.VERIFIED, ''])
    handheld.prepare(labels_csv(tmp_path / 'in.csv', rows), tmp_path / 'labels.csv')
    assert handheld.run('manifest.json', tmp_path / 'labels.csv', tmp_path / 'out') == {}
    assert len(calls) == 1


def test_seed_and_review_pages_reference_original_videos(tmp_path):
    videos = tmp_path / 'videos'
    videos.mkdir()
    writer = cv2.VideoWriter(str(videos / 'IMG_1.mov'), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 48))
    for _ in range(3):
        writer.write(np.full((48, 64, 3), 120, np.uint8))
    writer.release()
    page = handheld.seeds(videos, ['IMG_1'], tmp_path / 'out')
    assert '"name": "IMG_1"' in page.read_text()
    labels = labels_csv(tmp_path / 'labels.csv', [['IMG_1', 0, 4, 'resting', 'pendiente', None],
                                                 ['IMG_1', 4, 5, 'unknown', 'pendiente', 'corto']])
    text = handheld.review(videos, labels, tmp_path / 'out').read_text()
    assert (videos / 'IMG_1.mov').resolve().as_uri() in text
    assert '"evaluable": false' in text and handheld.VERIFIED in text and '__' not in text.replace('__proto__', '')
    with pytest.raises(FileNotFoundError):
        handheld.video_path(videos, 'IMG_2')
