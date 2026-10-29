import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import actmodel, pointtrack
from test_pointtrack import FPS, FRAMES, H, RABBIT, W, FakeTracker, fake_frames, make_case, scenario  # noqa: F401 (fixture)

UNIT = np.sqrt(RABBIT[0] * RABBIT[1])


def draw_video(video, kind):
    """Fondo con textura que se desplaza con la cámara y conejo claro con textura propia (escena `scenario`)."""
    rng = np.random.default_rng(0)
    floor = cv2.GaussianBlur(rng.integers(0, 255, (H, 3 * W)).astype(np.uint8), (0, 0), 3)
    fur = cv2.GaussianBlur(rng.integers(150, 255, (RABBIT[1], RABBIT[0])).astype(np.uint8), (0, 0), 1.5)
    rabbit_x, background = scenario(kind)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for f in range(FRAMES):
        offset = int(round(-background[f]))
        image = floor[:, offset:offset + W].copy()
        x = int(round(rabbit_x[f]))
        image[120:120 + RABBIT[1], x:x + RABBIT[0]] = fur
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()


class Stub:
    size, info = 32, dict(model_id='stub')

    def __init__(self):
        self.shapes = []

    def __call__(self, clip):
        self.shapes.append(clip.shape)
        return np.array([clip.mean(), clip.std(), 1.])


def features(tmp_path, name, groom, masked=True, encoder=None):
    video, masks = make_case(tmp_path, name, 'pan', masked=masked)
    draw_video(video, 'pan')
    times = pointtrack.frame_times(video, tmp_path / 'tracks' / 'tiempos')
    pointtrack.track_case(video, masks, tmp_path / 'tracks' / video.stem, FakeTracker('pan', groom=groom), None, times)
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name=name, video=video.name, masks=masks.name)])))
    pd.DataFrame(dict(case=name, start=np.arange(0., 10., 2.), end=np.arange(2., 12., 2.))).to_csv(tmp_path / 'w.csv',
                                                                                                    index=False)
    table = actmodel.window_features(tmp_path / 'm.json', tmp_path / 'w.csv', tmp_path / name, encoder, tmp_path / 'tracks')
    return table.set_index('start')


def test_window_features_measure_displacement_motion_and_shape(tmp_path, fake_frames):
    encoder = Stub()
    table = features(tmp_path, 'groom', True, encoder=encoder)
    assert table.loc[0., 'net_displacement'] == pytest.approx(80 / UNIT, rel=.05)
    assert table.loc[2., 'net_displacement'] == pytest.approx(80 / UNIT, rel=.05)
    assert table.loc[6., 'net_displacement'] < .05  # la cámara barre, el conejo no se mueve respecto al suelo
    assert table.loc[0., 'motion_in'] > 5 * table.loc[4., 'motion_in']  # se desplaza frente a quieto
    assert table.loc[6., 'motion_out'] > 5 * table.loc[4., 'motion_out']  # el barrido se ve fuera de la máscara
    assert table.area_mean.to_numpy() == pytest.approx(np.full(5, RABBIT[0] * RABBIT[1] / (W * H)), rel=.05)
    assert table.aspect_mean.to_numpy() == pytest.approx(np.full(5, RABBIT[1] / RABBIT[0]), rel=.05)
    assert table.masked.all() and set(encoder.shapes) == {(actmodel.FRAMES, 32, 32, 3)}
    assert np.load(tmp_path / 'groom' / 'vjepa.npy').shape == (5, 3)
    still = features(tmp_path / 'b', 'still', False)
    assert table.internal_p90.min() > 5 * max(still.internal_p90.max(), 1e-3)  # acicalarse mueve puntos del conejo


def test_window_without_mask_or_tracks_is_missing_not_still(tmp_path, fake_frames):
    table = features(tmp_path, 'blind', False, masked=False)
    assert not table.masked.any() and table.area_mean.isna().all()
    assert table.net_displacement.isna().all() and table.internal_median.isna().all()
    assert np.isfinite(table.motion_in).all()


def synthetic(tmp_path, seed=0, flip=None):
    """Cuatro sesiones (IMG_5190 e IMG_5191 son la misma) con las cuatro clases; cada clase marca una característica."""
    rng = np.random.default_rng(seed)
    acts = ['locomotion', 'eating', 'lying', 'loaf', 'exploring', 'not_visible']
    signal = dict(locomotion='net_displacement', eating='motion_in', lying='area_mean', loaf='elongation_mean',
                  exploring='elongation_mean')
    rows, labels = [], []
    for case in ('5177', 'IMG_5190', 'IMG_5191', 'WhatsApp Video 1', 'resting'):
        for k in range(24):
            act = acts[k % len(acts)]
            row = dict(case=case, start=2. * k, end=2. * k + 2, masked=True,
                       **{c: rng.normal(0, .3) for c in actmodel.HANDCRAFTED})
            if act in signal:
                row[signal[act]] += 3
            rows.append(row)
            labels.append(dict(case=case, start=2. * k, act=act if case != flip else acts[(k + 1) % len(acts)],
                               end=2. * k + 2))
    folder = tmp_path / 'features'
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / 'caracteristicas.csv', index=False)
    np.save(folder / 'vjepa.npy', rng.normal(size=(len(rows), 8)))
    pd.DataFrame(labels).to_csv(tmp_path / 'labels.csv', index=False)
    return folder, tmp_path / 'labels.csv'


def test_evaluation_leaves_whole_sessions_out_and_applies_the_criterion(tmp_path):
    folder, labels = synthetic(tmp_path)
    result = actmodel.evaluate(folder, labels, tmp_path / 'out')
    assert result['sessions'] == ['5177', 'mano_IMG_519x', 'resting', 'whatsapp']
    hand, deep = result['results']['handcrafted'], result['results']['deep']
    assert hand['windows'] == 100 and hand['passes'] and hand['balanced_accuracy'] > .9
    assert not deep['passes'] and deep['balanced_accuracy'] < .5  # ruido: cerca del azar (0,25)
    assert set(result['results']) == {'handcrafted', 'deep', 'all'}
    stored = json.loads((tmp_path / 'out' / 'evaluacion.json').read_text())
    assert stored['criterion'] == actmodel.CRITERION and 'No es una validación' in stored['scope']
    table = pd.read_csv(tmp_path / 'out' / 'predicciones.csv', dtype={'case': str})
    assert table[table.act == 'not_visible'].pred_handcrafted.isna().all()


def test_labels_of_the_held_out_session_do_not_reach_its_predictions(tmp_path):
    folder, labels = synthetic(tmp_path / 'a')
    clean = actmodel.evaluate(folder, labels, tmp_path / 'a' / 'out')
    folder, labels = synthetic(tmp_path / 'b', flip='IMG_5191')
    flipped = actmodel.evaluate(folder, labels, tmp_path / 'b' / 'out')
    a = pd.read_csv(tmp_path / 'a' / 'out' / 'predicciones.csv', dtype={'case': str})
    b = pd.read_csv(tmp_path / 'b' / 'out' / 'predicciones.csv', dtype={'case': str})
    same = a.case.isin(['IMG_5190', 'IMG_5191']) & a.act.ne('not_visible') & b.act.ne('not_visible')
    assert same.sum() > 30
    assert (a[same].pred_all == b[same].pred_all).all()  # misma sesión fuera: sus etiquetas no entran en su modelo
    assert clean['results']['all']['balanced_accuracy'] != flipped['results']['all']['balanced_accuracy']


def test_vjepa_wrapper_runs_a_tiny_random_model(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'VJEPA2Config'):
        pytest.skip('transformers sin V-JEPA 2')
    config = transformers.VJEPA2Config(crop_size=32, frames_per_clip=16, patch_size=16, tubelet_size=2, hidden_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, mlp_ratio=2, pred_hidden_size=16,
                                       pred_num_hidden_layers=1, pred_num_attention_heads=2)
    torch.manual_seed(0)
    transformers.VJEPA2Model(config).save_pretrained(tmp_path / 'tiny')
    encoder = actmodel.VJEPA('cpu', str(tmp_path / 'tiny'))
    assert encoder.size == 32 and encoder.info['frames'] == actmodel.FRAMES
    clip = np.random.default_rng(0).integers(0, 255, (actmodel.FRAMES, 32, 32, 3)).astype(np.uint8)
    vector = encoder(clip)
    assert vector.shape == (32,) and np.isfinite(vector).all()
    assert not np.allclose(vector, encoder(clip[:, :, :, ::-1].copy()))  # el orden de canales importa
