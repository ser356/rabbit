import json
from pathlib import Path

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


def fake_close(calls):
    def close(manifest, out, announce=True):
        calls.append(json.loads(Path(manifest).read_text()))
        Path(out).mkdir(parents=True, exist_ok=True)
        return dict(status='closed_resolved_on_development_benchmark', passing_methods=['flow'],
                    methods=dict(flow=dict(passed=True)))
    return close


def manifest(tmp_path, labels):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(cases=[dict(name='IMG_1', video='~/IMG_1.MOV', anchors='a.json', labels=labels.name)])))
    return path


def test_run_refuses_without_enough_verified_clips(tmp_path, monkeypatch):
    rows = [['IMG_1', 4 * i, 4 * i + 4, 'locomotion' if i < 5 else 'resting', handheld.VERIFIED, ''] for i in range(14)]
    labels = tmp_path / 'labels.csv'
    handheld.prepare(labels_csv(tmp_path / 'in.csv', rows), labels)
    calls = []
    monkeypatch.setattr('rabbitsense.closure.close', fake_close(calls))
    assert handheld.run(manifest(tmp_path, labels), labels, tmp_path / 'out') is None
    assert calls == []
    with pytest.raises(ValueError, match='descriptivo'):
        handheld.run(manifest(tmp_path, labels), labels, tmp_path / 'out', include_unverified=True)
    rows.append(['IMG_1', 56, 60, 'resting', handheld.VERIFIED, ''])
    handheld.prepare(labels_csv(tmp_path / 'in.csv', rows), labels)
    assert handheld.run(manifest(tmp_path, labels), labels, tmp_path / 'out')['passing_methods'] == ['flow']
    case = calls[0]['cases'][0]
    assert case['labels'] == str(labels.resolve()) and case['video'] == str(Path('~/IMG_1.MOV').expanduser())


def test_descriptive_run_is_never_a_verdict_and_marks_unverified_labels(tmp_path, monkeypatch):
    rows = [['IMG_1', 0, 4, 'locomotion', 'pendiente', ''], ['IMG_1', 4, 8, 'resting', handheld.VERIFIED, ''],
            ['IMG_1', 8, 10, 'resting', 'pendiente', '']]
    labels = tmp_path / 'labels.csv'
    handheld.prepare(labels_csv(tmp_path / 'in.csv', rows), labels)
    old = tmp_path / 'old.csv'
    old.write_text('x')
    combined = tmp_path / 'combined.json'
    combined.write_text(json.dumps(dict(cases=[dict(name='old', video='v', anchors='a', labels='old.csv'),
                                               dict(name='IMG_1', video='v', anchors='a', labels='labels.csv')])))
    calls = []
    monkeypatch.setattr('rabbitsense.closure.close', fake_close(calls))
    result = handheld.run(manifest(tmp_path, labels), labels, tmp_path / 'out', descriptive=True,
                          include_unverified=True, informative=combined)
    assert result['status'] == 'descriptive_only_insufficient_sample' and result['passing_methods'] == []
    assert 'passed' not in result['methods']['flow'] and result['label_source'] == 'includes_unverified_proposals'
    used = pd.read_csv(tmp_path / 'out' / 'labels_used.csv')
    assert used.reviewed.tolist() == ['si', 'si', 'no']
    assert calls[0]['cases'][0]['labels'] == str((tmp_path / 'out' / 'labels_used.csv').resolve())
    assert [c['labels'] for c in calls[1]['cases']] == [str(old.resolve()), str((tmp_path / 'out' / 'labels_used.csv').resolve())]
    informative = json.loads((tmp_path / 'out' / 'informative' / 'verdict.json').read_text())
    assert informative['status'] == 'informative_not_decisive'
    assert pd.read_csv(labels).reviewed.tolist() == ['no', 'si', 'no']


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


def test_descriptive_run_end_to_end_with_real_closure(tmp_path):
    rng = np.random.default_rng(3)
    background = rng.integers(25, 100, (240, 320), dtype=np.uint8)
    patch = rng.integers(150, 230, (70, 80), dtype=np.uint8)
    writer = cv2.VideoWriter(str(tmp_path / 'IMG_1.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 12., (320, 240))
    marks = []
    for frame in range(73):
        x = 40 + 3 * min(frame, 36)
        image = background.copy()
        image[85:155, x:x + 80] = patch
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        marks.append(dict(time=frame / 12, box=[x, 85, 80, 70]))
    writer.release()
    (tmp_path / 'a.json').write_text(json.dumps(dict(width=320, height=240, marks=marks)))
    labels = tmp_path / 'labels.csv'
    handheld.prepare(labels_csv(tmp_path / 'in.csv', [['IMG_1', 0, 3, 'locomotion', 'pendiente', ''],
                                                      ['IMG_1', 3, 6, 'resting', 'pendiente', '']]), labels)
    labels.write_text(labels.read_text().replace('IMG_1.MOV', 'IMG_1.avi'))
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='IMG_1', video='IMG_1.avi', anchors='a.json',
                                                                 labels='labels.csv')])))
    assert handheld.run(tmp_path / 'm.json', labels, tmp_path / 'out') is None
    result = handheld.run(tmp_path / 'm.json', labels, tmp_path / 'out', descriptive=True, include_unverified=True)
    flow = result['methods']['flow']
    assert (flow['locomotion_hits'], flow['resting_correct'], flow['resting_false_events']) == (1, 1, 0)
    assert result['status'] == 'descriptive_only_insufficient_sample'
    saved = json.loads((tmp_path / 'out' / 'decisive' / 'verdict.json').read_text())
    assert saved['status'] == result['status'] and saved['sufficiency']['sufficient'] is False
