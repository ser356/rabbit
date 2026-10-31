import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import auto, pointtrack, queryseg
from test_actmodel import draw_video
from test_auto import FakeStage, frozen_model, late_video
from test_pointtrack import RABBIT, FakeTracker, fake_frames, make_case  # noqa: F401 (fixture)
from test_ytacts import Bright


class BoxSAM:
    """SAM2 falso: dentro de la caja, los píxeles claros del pelaje sintético."""

    info = dict(backend='fake')

    def __init__(self):
        self.calls = 0

    def __call__(self, image, box):
        self.calls += 1
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        mask = np.zeros(image.shape[:2], np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask[y0:y1, x0:x1] = (gray[y0:y1, x0:x1] > 170) * 255
        return mask


def test_masks_only_at_the_frames_cotracker_uses_as_queries(tmp_path, fake_frames):
    video, _ = make_case(tmp_path, 'pan', 'pan')
    draw_video(video, 'pan')
    times = pointtrack.frame_times(video)
    sam = BoxSAM()
    folder = queryseg.segment_queries(video, tmp_path / 'seg', Bright(), sam, times)
    written = sorted(int(p.stem) for p in (folder / 'masks').glob('*.png'))
    assert len(written) == 4 == sam.calls  # un fotograma por tramo de 4 s (10 s de vídeo)
    pointtrack.track_case(video, folder, tmp_path / 'tracks', FakeTracker('pan'), None, times)
    queries = []
    for path in sorted((tmp_path / 'tracks').glob('chunk_*.npz')):
        chunk = np.load(path)
        queries.append(int(chunk['frames'][int(chunk['query'])]))
    assert queries == written
    table = pd.read_csv(folder / 'consultas.csv')
    assert table.frame.notna().all() and (table.mask_fraction > 0).all()
    again = BoxSAM()
    queryseg.segment_queries(video, tmp_path / 'seg', Bright(), again, times)
    assert again.calls == 0  # ya hechas: se reutilizan


def test_neighbour_frames_are_used_when_the_centre_has_no_rabbit(tmp_path):
    late_video(tmp_path / 'late.mp4', appear=58, frames=150)  # el centro del primer tramo (2 s) aún no lo tiene
    times = pointtrack.frame_times(tmp_path / 'late.mp4')
    folder = queryseg.segment_queries(tmp_path / 'late.mp4', tmp_path / 'seg', Bright(), BoxSAM(), times)
    table = pd.read_csv(folder / 'consultas.csv')
    assert table.frame.iloc[0] > 58 and table.frame.iloc[0] - 60 <= 2 * queryseg.QUERY['neighbours']


class FakeModel:
    def __init__(self, inputs, predict, image=None):
        import coremltools as ct
        self.spec = ct.proto.Model_pb2.Model()
        for name in inputs:
            field = self.spec.description.input.add()
            field.name = name
            if name == image:
                field.type.imageType.width = field.type.imageType.height = 64
            else:
                field.type.multiArrayType.SetInParent()
        self.predict = predict

    def get_spec(self):
        return self.spec


def test_coreml_sam_glue_scales_the_box_to_the_square_input():
    pytest.importorskip('coremltools')
    seen = {}

    def decode(inputs):
        x0, y0, x1, y1 = inputs['sparse'].ravel()
        masks = np.full((1, 3, 64, 64), -1., np.float32)
        masks[0, 1, int(y0):int(y1), int(x0):int(x1)] = 1.
        return {'masks': masks, 'scores': np.array([[.1, .9, .2]], np.float32)}

    def prompt(inputs):
        seen['labels'] = inputs['labels']
        return {'sparse': inputs['points'], 'dense': np.zeros((1, 2), np.float32)}
    models = dict(ImageEncoder=FakeModel(['image'], lambda i: {'embedding': np.ones((1, 4), np.float32)}, image='image'),
                  PromptEncoder=FakeModel(['points', 'labels'], prompt),
                  MaskDecoder=FakeModel(['embedding', 'sparse', 'dense'], decode))
    sam = queryseg.CoreMLSAM(models=models)
    image = np.zeros((100, 200, 3), np.uint8)
    mask = sam(image, (50, 20, 150, 80))
    ys, xs = np.nonzero(mask)
    assert mask.shape == (100, 200) and seen['labels'].tolist() == [[2, 3]]
    assert xs.min() == pytest.approx(50, abs=3) and xs.max() == pytest.approx(150, abs=3)
    assert ys.min() == pytest.approx(20, abs=3) and ys.max() == pytest.approx(80, abs=3)


def test_validate_compares_query_masks_with_manual_seed(tmp_path, fake_frames, monkeypatch):
    monkeypatch.setattr(pointtrack, 'TRACKS', tmp_path / 'tracks')
    video, masks = make_case(tmp_path, 'pan', 'pan')
    draw_video(video, 'pan')
    times = pointtrack.frame_times(video, tmp_path / 'tracks' / 'tiempos')
    pointtrack.track_case(video, masks, tmp_path / 'tracks' / 'pan', FakeTracker('pan'), None, times)
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='pan', video=video.name, masks=masks.name)])))
    pd.DataFrame(dict(case='pan', start=[0., 2., 4., 6., 8.], act=['locomotion', 'locomotion', 'loaf', 'loaf', 'eating'],
                      end=[2., 4., 6., 8., 10.])).to_csv(tmp_path / 'acts.csv', index=False)
    result = queryseg.validate(tmp_path / 'm.json', tmp_path / 'acts.csv', tmp_path / 'out', 'cpu', 'fake',
                               tmp_path / 'work', Bright(), BoxSAM(), FakeTracker('pan'))
    assert result['mask_iou_median'] > .8 and result['per_video']['pan']['frames'] == 4
    assert result['displacement_median_difference'] < .05 and result['auc'] == 1. and result['adopt_query']
    assert queryseg.adopted(tmp_path / 'out' / 'validacion.json')


def test_analyze_with_query_segmentation(tmp_path, fake_frames):
    video, _ = make_case(tmp_path, 'pan', 'pan')
    draw_video(video, 'pan')
    auto.analyze([video], tmp_path / 'out', 'cpu', FakeStage(FakeTracker('pan')), model_file=frozen_model(tmp_path / 'm.json'),
                 segmentation='query', sam=BoxSAM())
    table = pd.read_csv(tmp_path / 'out' / 'pan' / 'ventanas.csv')
    assert table.measured.all() and table.displacement[0] == pytest.approx(80 / np.sqrt(RABBIT[0] * RABBIT[1]), rel=.1)
    assert len(list((tmp_path / 'out' / 'pan' / 'segmentacion_consultas' / 'masks').glob('*.png'))) == 4
