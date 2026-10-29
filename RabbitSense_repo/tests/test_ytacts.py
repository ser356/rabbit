import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import actmodel, ytacts
from test_actmodel import Stub, draw_video
from test_pointtrack import FPS, H, RABBIT, W, fake_frames, make_case  # noqa: F401 (fixture)


class Bright:
    """Detector falso: cada zona clara (pelaje sintético) es un conejo, con su caja exacta."""

    def __init__(self, shift=0):
        self.calls, self.shift, self.info = 0, shift, dict(model_id='bright')

    def __call__(self, images):
        self.calls += len(images)
        found = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            count, _, stats, _ = cv2.connectedComponentsWithStats((gray > 170).astype(np.uint8))
            boxes = [(.9, x + self.shift, y, x + w + self.shift, y + h) for x, y, w, h, area in stats[1:] if area > 200]
            found.append(sorted(boxes, reverse=True))
        return found


def test_detections_scale_to_the_padded_square_and_drop_duplicates():
    scores = np.array([.9, .8, .1, .7])
    boxes = np.array([[.25, .25, .1, .1], [.26, .25, .1, .1], [.5, .5, .2, .2], [.75, .4, .1, .2]])
    found = ytacts.detections(scores, boxes, (300, 400, 3), threshold=.2)
    assert len(found) == 2  # la segunda solapa con la primera; la tercera no llega al umbral
    assert found[0] == pytest.approx((.9, 80., 80., 120., 120.))  # cuadrado de 400 px, no 400×300
    assert found[1][1:] == pytest.approx((280., 120., 320., 200.))


def write_video(path, rabbits, seconds=12):
    rng = np.random.default_rng(1)
    floor = cv2.GaussianBlur(rng.integers(0, 255, (H, W)).astype(np.uint8), (0, 0), 3)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for f in range(int(seconds * FPS)):
        image = floor.copy()
        for k in range(rabbits):
            x = 60 + 200 * k + f // 4
            image[100:100 + RABBIT[1], x:x + RABBIT[0]] = 220
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()


def gate(tmp_path, passes=True):
    folder = tmp_path / 'out' / 'cinna'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'deteccion.json').write_text(json.dumps(dict(passes=passes)))


def test_youtube_keeps_single_rabbit_windows_with_sheets_and_vectors(tmp_path):
    videos = tmp_path / 'videos'
    videos.mkdir()
    write_video(videos / 'one.mp4', 1)
    write_video(videos / 'two.mp4', 2)
    with pytest.raises(RuntimeError, match='puerta'):
        ytacts.youtube(videos, tmp_path / 'out' / 'youtube', Bright(), Stub())
    gate(tmp_path)
    detector = Bright()
    summary = ytacts.youtube(videos, tmp_path / 'out' / 'youtube', detector, Stub())
    table = pd.read_csv(tmp_path / 'out' / 'youtube' / 'ventanas.csv')
    assert set(table.video) == {'one'} and len(table) == ytacts.SELECT['keep']  # dos conejos: ninguna ventana
    assert (table.start.diff().dropna() >= ytacts.SELECT['window']).all()  # sin solapar
    assert np.load(tmp_path / 'out' / 'youtube' / 'vjepa.npy').shape == (5, 3)
    assert summary['candidate_windows'] == 2 * 5 and summary['kept_windows'] == 5
    sheet = cv2.imread(str(tmp_path / 'out' / 'youtube' / 'hojas' / 'one_00.jpg'))
    assert sheet.shape[0] == 5 * actmodel.MOTION['size'] and list(table.row) == [0, 1, 2, 3, 4]
    assert set(table.sheet) == {'one_00'}
    calls = detector.calls
    again = Bright()
    ytacts.youtube(videos, tmp_path / 'out' / 'youtube', again, Stub())
    assert calls > 0 and again.calls == 0  # vídeos terminados: se reutilizan


def test_cinna_gate_compares_detections_with_sam2(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'pan', 'pan')
    draw_video(video, 'pan')
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='pan', video=video.name, masks=masks.name)])))
    pd.DataFrame(dict(case='pan', start=np.arange(0., 10., 2.), end=np.arange(2., 12., 2.))).to_csv(tmp_path / 'w.csv',
                                                                                                   index=False)
    good = ytacts.cinna(tmp_path / 'm.json', tmp_path / 'w.csv', tmp_path / 'ok', Bright(), Stub())
    assert good['passes'] and good['coverage'] == 1 and good['median_iou'] > .8
    assert np.load(tmp_path / 'ok' / 'vjepa.npy').shape == (5, 3)
    bad = ytacts.cinna(tmp_path / 'm.json', tmp_path / 'w.csv', tmp_path / 'bad', Bright(shift=50), Stub())
    assert not bad['passes'] and bad['median_iou'] < .5


def features(folder, rows, rng, key):
    acts = ['locomotion', 'eating', 'lying', 'loaf', 'grooming', 'not_visible']
    x, table, labels = [], [], []
    for k in range(rows):
        act = acts[k % len(acts)]
        vector = rng.normal(0, .5, 6)
        if act in actmodel.GROUPS:
            vector[actmodel.CLASSES.index(actmodel.GROUPS[act])] += 3
        block = k // len(acts)  # cada vídeo recibe bloques con todos los actos
        name, start = f'v{block % 12}', .5 + 2 * ((block // 12) * len(acts) + k % len(acts))
        table.append({key: name, 'start': start, 'end': start + 2, 'boxed': k % 9 != 0, 'found': 4 if k % 9 else 0})
        labels.append({key: name, 'start': start, 'act': act})
        x.append(vector)
    folder.mkdir(parents=True)
    pd.DataFrame(table).to_csv(folder / 'ventanas.csv', index=False)
    np.save(folder / 'vjepa.npy', np.array(x))
    pd.DataFrame(labels).to_csv(folder / 'labels.csv', index=False)
    return folder, folder / 'labels.csv'


def test_train_on_youtube_freezes_and_tests_cinna_once(tmp_path):
    rng = np.random.default_rng(0)
    yt, yt_labels = features(tmp_path / 'yt', 360, rng, 'video')
    frozen = ytacts.train(yt, yt_labels, tmp_path / 'modelo.json')
    assert frozen['cross_validation_by_video']['balanced_accuracy'] > .9
    assert frozen['counts']['other'] == dict(windows=120, videos=12)
    cinna, cinna_labels = features(tmp_path / 'cinna', 120, rng, 'case')
    result = ytacts.test(cinna, cinna_labels, tmp_path / 'cierre', tmp_path / 'modelo.json')
    assert result['passes'] and result['status'] == 'resolved_exploratory' and result['windows'] == 100
    assert set(result['accuracy_by_original_act']) == {'locomotion', 'eating', 'lying', 'loaf', 'grooming'}
    assert result['well_detected']['windows'] < result['windows'] and result['well_detected']['passes']
    with pytest.raises(RuntimeError, match='una sola vez'):
        ytacts.test(cinna, cinna_labels, tmp_path / 'cierre', tmp_path / 'modelo.json')


def test_training_refuses_classes_with_too_few_videos(tmp_path):
    yt, yt_labels = features(tmp_path / 'yt', 60, np.random.default_rng(0), 'video')
    with pytest.raises(ValueError, match='insuficientes'):
        ytacts.train(yt, yt_labels, tmp_path / 'modelo.json')


def test_owlv2_wrapper_runs_a_tiny_random_model():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    config = transformers.Owlv2Config(
        text_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2, vocab_size=99,
                         max_position_embeddings=16),
        vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
                           image_size=64, patch_size=16), projection_dim=32)
    torch.manual_seed(0)
    model = transformers.Owlv2ForObjectDetection(config)
    processor = transformers.Owlv2ImageProcessor(size={'height': 64, 'width': 64})
    tokens = dict(input_ids=torch.randint(0, 99, (1, 16)), attention_mask=torch.ones(1, 16, dtype=torch.long))
    detector = ytacts.Detector('cpu', 'tiny', 'a rabbit', parts=(model, processor, tokens))
    images = [np.random.default_rng(k).integers(0, 255, (48, 80, 3)).astype(np.uint8) for k in range(3)]
    found = detector(images, batch=2)
    assert len(found) == 3
    for boxes in found:
        for score, x0, y0, x1, y1 in boxes:
            assert score >= ytacts.DETECTOR['threshold'] and 0 <= x0 < x1 <= 80 and 0 <= y0 < y1 <= 48


def test_amended_gate_excludes_not_visible_and_keeps_the_original_verdict(tmp_path):
    folder = tmp_path / 'cinna'
    folder.mkdir()
    rows = [dict(case='a', start=2. * k, end=2. * k + 2, sam2=True, found=4 if k < 15 else 0,
                 iou_sam2=.9 if k < 15 else np.nan) for k in range(20)]
    pd.DataFrame(rows).to_csv(folder / 'ventanas.csv', index=False)
    labels = pd.DataFrame(dict(case='a', start=[2. * k for k in range(20)],
                               act=['eating'] * 15 + ['loaf'] + ['not_visible'] * 4))
    labels.to_csv(tmp_path / 'labels.csv', index=False)
    (folder / 'deteccion.json').write_text(json.dumps(dict(passes=False, coverage=.75)))
    with pytest.raises(RuntimeError):
        ytacts.require_gate(folder / 'deteccion.json')
    gate = ytacts.amend_gate(folder, tmp_path / 'labels.csv')
    assert gate['passes'] is False and gate['amendment']['windows'] == 16
    assert gate['amendment']['coverage'] == pytest.approx(15 / 16) and gate['amendment']['passes']
    assert gate['amendment']['by_act']['not_visible'] == dict(windows=4, detected=0)
    ytacts.require_gate(folder / 'deteccion.json')
