import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from rabbitsense.motion import aggregate, sha256
from rabbitsense.segment import assess_masks, measure_masks, segment
from test_motion import scene


@pytest.mark.parametrize('guided,broad', [(False, False), (True, False), (True, True)])
def test_segment_reserved_anchors_and_mask_motion(tmp_path, monkeypatch, guided, broad):
    image, box = scene()
    video = tmp_path / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 240))
    assert writer.isOpened()
    for _ in range(13):
        writer.write(image)
    writer.release()
    mask = np.zeros((240, 320), np.uint8)
    x, y, w, h = map(int, box)
    mask[y:y+h, x:x+w] = 1
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: mask))
    calls = []
    corrections = []

    class Predictor:
        def __init__(self, overrides):
            assert overrides['save'] is False
            self.src_shape = image.shape[:2]
            self.inference_state = {}

        def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
            return None

        def _prepare_prompts(self, dst_shape, src_shape, bboxes):
            return bboxes, [2, 3], None

        def add_new_prompts(self, **kwargs):
            corrections.append(kwargs)

        def __call__(self, **kwargs):
            calls.append(kwargs)
            for frame in range(13):
                self.dataset = SimpleNamespace(frame=frame + 1)
                self.inference(SimpleNamespace(shape=(1, 3, 1024, 1024)))
                yield SimpleNamespace(orig_img=image, masks=SimpleNamespace(data=[tensor]))

    for name, module in [('ultralytics.models.sam', SimpleNamespace(SAM2VideoPredictor=Predictor)),
                         ('torch', SimpleNamespace(__version__='test')),
                         ('ultralytics', SimpleNamespace(__version__='test'))]:
        monkeypatch.setitem(sys.modules, name, module)
    anchors = tmp_path / 'anchors.json'
    anchors.write_text(json.dumps(dict(width=320, height=240,
                                      marks=[dict(time=0., box=box.tolist()),
                                             dict(time=.5, box=[0., 0., 20., 20.]),
                                             dict(time=1., box=[0., 0., 320., 240.] if broad else box.tolist())])))
    weights = tmp_path / 'weights.pt'
    weights.write_bytes(b'test fixture')
    out = tmp_path / 'output'
    summary = segment(video, anchors, weights, out, 'cpu', guided)
    assert calls == [dict(source=str(video), bboxes=[[x, y, x+w, y+h]], stream=True)]
    assert corrections == ([dict(obj_id=0, points=[[x, y, x+w, y+h]], labels=[2, 3], frame_idx=11)] if guided and not broad else [])
    assert summary['prompt_frames'] == ([0, 10] if guided and not broad else [0])
    assert summary['heldout_n'] == (1 if guided else 2)
    assert summary['heldout_mean_iou'] == (0 if guided else .5)
    assert summary['nonempty_frames'] == 13
    assert len(list((out / 'masks').glob('*.png'))) == 13
    capture = cv2.VideoCapture(str(out / 'overlay.mp4'))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 13
    rows = measure_masks(out)
    assert [r['seconds'] for r in rows] == [0., .5, 1., 1.2]
    assert rows[1]['movement'] == 'low_motion_candidate'
    flow = json.loads((out / 'flow_timeline.json').read_text())
    assert flow[-1]['end_seconds'] == 1.2
    if guided and not broad:
        assert any(r['movement'] == 'unknown' and r['end_seconds'] == 1. for r in flow)
    capture = cv2.VideoCapture(str(out / 'motion_overlay.mp4'))
    assert capture.get(5) == 10.
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 13
    labels = tmp_path / 'labels.csv'
    labels.write_text('video_id,reviewed,label,start_seconds,duration_seconds,clip_id\ninput.avi,si,resting,0,1.2,test\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(cases=[dict(name='test', labels=labels.name)])))
    historical = dict(test=dict(inputs={**summary['inputs'], 'labels': dict(sha256=sha256(labels))},
                               tracking_holdout=dict(summary={}, anchors=[dict(seconds=r['seconds'])
                                   for r in summary['anchors'] if r['withheld']])))
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps(historical))
    result = assess_masks(manifest, baseline, dict(test=out), tmp_path / 'comparison.json')
    assert result['cases']['test']['heldout_n'] == summary['heldout_n']
    assert result['cases']['test']['metrics']['masked_flow']['clips']['n'] == 1
    historical['test']['tracking_holdout']['anchors'].append(dict(seconds=0.))
    baseline.write_text(json.dumps(historical))
    with pytest.raises(ValueError, match='contaminada'):
        assess_masks(manifest, baseline, dict(test=out), tmp_path / 'comparison.json')
    (out / 'masks' / '000005.png').unlink()
    with pytest.raises(ValueError, match='Máscara inválida: 5'):
        measure_masks(out)
    assert not (out / 'motion_summary.json').exists()


@pytest.mark.parametrize('center,expected', [(0., 'unknown'), (.15, 'locomotion_candidate')])
def test_surface_motion_requires_mask_centroid_translation(center, expected):
    item = dict(reason='measured', vector=np.array([.1, 0.]), uncertainty=.01, quality=1.,
                camera_error_px=.1, rabbit_points=20)
    assert aggregate([item] * 5, .5, .2)['movement'] == 'locomotion_candidate'
    item.update(centroid_vector=np.array([center, 0.]), centroid_camera_error=.01)
    row = aggregate([item] * 5, .5, .2)
    assert row['movement'] == expected
    assert row['centroid_valid_fraction'] == 1.
    if center == 0.:
        assert row['reason'] == 'surface_motion_without_centroid_translation'
    item['vector'] = np.array([0., 0.])
    assert aggregate([item] * 5, .5, .2)['movement'] == 'low_motion_candidate'