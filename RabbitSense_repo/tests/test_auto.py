import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import auto, pointtrack
from rabbitsense.tracking import load_anchors
from test_actmodel import Stub, draw_video
from test_pointtrack import FPS, H, RABBIT, W, FakeTracker, fake_frames, make_case  # noqa: F401 (fixture)
from test_ytacts import Bright


class FakeSAM:
    """Como segment(guided=True): comprueba las anclas y segmenta umbralizando el pelaje claro del vídeo sintético."""

    def __init__(self):
        self.calls = []

    def __call__(self, source, anchors, model, out, device, guided=False):
        fps, count, width, height = auto.video_info(source)
        loaded = load_anchors(anchors, width, height)
        exact, box = loaded.at_frame(0, fps)
        assert exact and box is not None and guided
        self.calls.append(source)
        (out / 'masks').mkdir(parents=True, exist_ok=True)
        capture, j = cv2.VideoCapture(str(source)), 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            mask = ((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 170) * 255).astype(np.uint8)
            cv2.imwrite(str(out / 'masks' / f'{j:06d}.png'), mask)
            j += 1
        capture.release()
        return dict(prompt_frames=[0], heldout_mean_iou=.9, nonempty_frames=j)


def late_video(path, appear=60, frames=150):
    rng = np.random.default_rng(2)
    floor = cv2.GaussianBlur(rng.integers(0, 255, (H, W)).astype(np.uint8), (0, 0), 3)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for f in range(frames):
        image = floor.copy()
        if f >= appear:
            x = 100 + (f - appear)
            image[120:120 + RABBIT[1], x:x + RABBIT[0]] = 230
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()


def test_segmentation_starts_where_the_rabbit_first_appears(tmp_path):
    late_video(tmp_path / 'late.mp4')
    sam, detector = FakeSAM(), Bright()
    masks = auto.autosegment(tmp_path / 'late.mp4', tmp_path / 'seg', detector, 'cpu', segment_fn=sam)
    record = json.loads((tmp_path / 'seg' / 'auto.json').read_text())
    assert record['status'] == 'ok' and record['start_frame'] == 60 and not (tmp_path / 'seg' / 'recorte.mp4').exists()
    files = sorted((masks / 'masks').glob('*.png'))
    assert len(files) == 150
    assert not cv2.imread(str(files[59]), cv2.IMREAD_GRAYSCALE).any()
    for f in (60, 100, 149):  # la máscara del recorte j va al fotograma 60 + j
        ys, xs = np.nonzero(cv2.imread(str(files[f]), cv2.IMREAD_GRAYSCALE))
        assert xs.min() == pytest.approx(100 + (f - 60), abs=2) and ys.min() == pytest.approx(120, abs=2)
    calls = detector.calls
    again = Bright()
    assert auto.autosegment(tmp_path / 'late.mp4', tmp_path / 'seg', again, 'cpu', segment_fn=sam) == masks
    assert calls and again.calls == 0 and len(sam.calls) == 1  # ya segmentado: no se repite


def test_no_rabbit_gives_empty_masks_and_no_segmentation(tmp_path):
    late_video(tmp_path / 'empty.mp4', appear=10 ** 6, frames=40)
    sam = FakeSAM()
    masks = auto.autosegment(tmp_path / 'empty.mp4', tmp_path / 'seg', Bright(), 'cpu', segment_fn=sam)
    assert not sam.calls and json.loads((tmp_path / 'seg' / 'auto.json').read_text())['status'] == 'sin_conejo'
    assert len(list((masks / 'masks').glob('*.png'))) == 40


class FakeStage:
    def __init__(self, tracker):
        self.models = dict(detector=Bright(), encoder=Stub(), tracker=tracker)

    def get(self, name):
        return self.models[name]

    def drop(self, *names):
        pass


def frozen_model(path):
    """Modelo congelado de juguete sobre los 3 valores de Stub (media, desviación, 1)."""
    path.write_text(json.dumps(dict(classes=['eating', 'locomotion', 'lying', 'other'], mean=[0., 0., 0.],
                                    scale=[1., 1., 1.], coef=[[0, 0, 0], [0, .1, 0], [0, 0, 0], [0, 0, 0]],
                                    intercept=[0., 0., 0., .5])))
    return path


def test_analyze_writes_windows_summary_and_report(tmp_path, fake_frames):
    video, _ = make_case(tmp_path, 'pan', 'pan')
    draw_video(video, 'pan')
    summaries = auto.analyze([video], tmp_path / 'out', 'cpu', FakeStage(FakeTracker('pan')), FakeSAM(),
                             frozen_model(tmp_path / 'm.json'))
    folder = tmp_path / 'out' / 'pan'
    table = pd.read_csv(folder / 'ventanas.csv')
    assert list(table.start) == [0., 2., 4., 6., 8.] and table.measured.all()
    assert table.displacement[0] == pytest.approx(80 / np.sqrt(RABBIT[0] * RABBIT[1]), rel=.1)
    assert set(table.act) <= {'eating', 'locomotion', 'lying', 'other'} and table.confidence.between(0, 1).all()
    summary = summaries['pan']
    assert summary['time_by_act']['eating']['validated'] is False and summary['visible_seconds'] > 9
    assert sum(v['seconds'] for v in summary['time_by_act'].values()) == pytest.approx(summary['visible_seconds'])
    page = (folder / 'informe.html').read_text()
    assert 'comer (sin validar)' in page and '<svg' in page and 'estado de ánimo' in page


def test_windows_without_rabbit_are_not_visible(tmp_path):
    late_video(tmp_path / 'late.mp4')
    times = pointtrack.frame_times(tmp_path / 'late.mp4')
    acts = auto.act_windows(tmp_path / 'late.mp4', times, Bright(), Stub(), json.loads(frozen_model(tmp_path / 'm.json').read_text()))
    assert list(acts.act[:1]) == ['not_visible'] and (acts.act[acts.start >= 2] != 'not_visible').all()


def test_bouts_and_mask_agreement(tmp_path):
    acts = pd.DataFrame(dict(act=['lying', 'lying', 'other', 'lying'], start=[0., 2., 4., 6.], end=[2., 4., 6., 7.]))
    assert auto.bouts(acts, 'lying') == [4., 1.]
    for name, shift in (('a', 0), ('b', 10)):
        (tmp_path / name / 'masks').mkdir(parents=True)
        for f in range(10):
            mask = np.zeros((50, 80), np.uint8)
            if f < 8 or name == 'a':
                mask[10:30, 20 + shift:60 + shift] = 255
            cv2.imwrite(str(tmp_path / name / 'masks' / f'{f:06d}.png'), mask)
    result = auto.mask_agreement(tmp_path / 'a', tmp_path / 'b', every=1)
    assert result['iou_median'] == pytest.approx(30 / 50) and result['coverage_auto'] == .8 and result['coverage_manual'] == 1


def test_validate_compares_automatic_with_manual_masks(tmp_path, fake_frames, monkeypatch):
    monkeypatch.setattr(pointtrack, 'TRACKS', tmp_path / 'tracks')
    video, masks = make_case(tmp_path, 'pan', 'pan')
    draw_video(video, 'pan')
    times = pointtrack.frame_times(video, tmp_path / 'tracks' / 'tiempos')
    pointtrack.track_case(video, masks, tmp_path / 'tracks' / 'pan', FakeTracker('pan'), None, times)
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='pan', video=video.name, masks=masks.name)])))
    pd.DataFrame(dict(case='pan', start=[0., 2., 4., 6., 8.], act=['locomotion', 'locomotion', 'loaf', 'loaf', 'eating'],
                      end=[2., 4., 6., 8., 10.])).to_csv(tmp_path / 'acts.csv', index=False)
    result = auto.validate(tmp_path / 'm.json', tmp_path / 'acts.csv', tmp_path / 'out', 'cpu', tmp_path / 'work',
                           FakeStage(FakeTracker('pan')), FakeSAM())
    assert result['mask_iou_median'] > .8 and result['displacement_median_difference'] < .05
    assert result['auc'] == 1. and result['adopt_auto']
    assert (tmp_path / 'out' / 'validacion.json').is_file()


def test_sheets_have_no_predictions_and_score_uses_prior_labels(tmp_path):
    late_video(tmp_path / 'late.mp4')
    assert auto.sheets([tmp_path / 'late.mp4'], tmp_path / 'hojas', Bright()) == 3
    table = pd.read_csv(tmp_path / 'hojas' / 'ventanas.csv')
    assert list(table.start) == [0., 2., 4.] and list(table.found) == [0, 4, 4] and 'act' not in table
    assert cv2.imread(str(tmp_path / 'hojas' / 'hojas' / 'late_00.jpg')).shape[0] == 3 * 200
    (tmp_path / 'analisis' / 'late').mkdir(parents=True)
    pd.DataFrame(dict(start=[0., 2., 4.], act=['not_visible', 'eating', 'other'])).to_csv(
        tmp_path / 'analisis' / 'late' / 'ventanas.csv', index=False)
    pd.DataFrame(dict(video='late', start=[0., 2., 4.], act=['not_visible', 'eating', 'eating'])).to_csv(
        tmp_path / 'labels.csv', index=False)
    result = auto.score(tmp_path / 'analisis', tmp_path / 'labels.csv', tmp_path / 'out')
    assert result['eating_windows'] == 2 and result['eating_hits'] == 1 and result['verdict'] == 'sin_veredicto'
    assert result['predicted_as'] == {'eating': 1, 'other': 1}
    low, high = result['eating_recall_interval']
    assert low < .5 < high
