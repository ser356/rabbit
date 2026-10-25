import json
import sys

import cv2
import numpy as np

from rabbitsense import closure


def test_closure_runs_suite_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    background = rng.integers(25, 100, (240, 320), dtype=np.uint8)
    patch = rng.integers(150, 230, (70, 80), dtype=np.uint8)
    video = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 12., (320, 240))
    marks, masks = [], tmp_path / 'sam2' / 'masks'
    masks.mkdir(parents=True)
    for frame in range(49):
        x = 40 + 4 * min(frame, 24)
        image = background.copy()
        image[85:155, x:x + 80] = patch
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        mask = np.zeros_like(image)
        mask[85:155, x:x + 80] = 255
        cv2.imwrite(str(masks / f'{frame:06d}.png'), mask)
        if frame % 6 == 0:
            marks.append(dict(time=frame / 12, box=[x, 85, 80, 70]))
    writer.release()
    (tmp_path / 'anchors.json').write_text(json.dumps(dict(width=320, height=240, marks=marks)))
    (tmp_path / 'labels.csv').write_text('clip_id,video_id,reviewed,label,start_seconds,duration_seconds\n'
                                         'a,clip.avi,si,locomotion,0,2\nb,clip.avi,si,resting,2,2\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(cases=[dict(name='synthetic', video='clip.avi',
                                                    anchors='anchors.json', labels='labels.csv')])))
    held = [round(m['time'] * 12) for m in marks][1:]
    (tmp_path / 'sam2' / 'tracks.csv').write_text('frame\n')
    (tmp_path / 'sam2' / 'summary.json').write_text(json.dumps(dict(
        fps=12., frames=49, nonempty_frames=49, prompt_frames=[0],
        anchors=[dict(frame=f, used_as_prompt=False, iou=1.) for f in held],
        inputs={k: dict(path=str(tmp_path / f), sha256=closure.sha256(tmp_path / f))
                for k, f in [('video', 'clip.avi'), ('anchors', 'anchors.json')]})))
    before = closure.sha256(tmp_path / 'sam2' / 'summary.json')
    monkeypatch.setattr(sys, 'argv', ['closure', '--suite', str(manifest), '--out', str(tmp_path / 'out'),
                                      '--segmentation', f'synthetic={tmp_path / "sam2"}'])
    closure.main()
    result = json.loads((tmp_path / 'out' / 'verdict.json').read_text())
    for method in ('flow', 'sam2_masked_flow'):
        counts = result['methods'][method]
        assert (counts['locomotion_hits'], counts['resting_correct'], counts['resting_false_events']) == (1, 1, 0)
        assert not counts['passed']
    assert result['status'] == 'closed_not_resolvable_with_existing_data'
    assert closure.sha256(tmp_path / 'sam2' / 'summary.json') == before
    assert not (tmp_path / 'sam2' / 'motion_summary.json').exists()
