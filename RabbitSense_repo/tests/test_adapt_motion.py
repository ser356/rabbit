import json
import warnings

import numpy as np
import pandas as pd
import pytest

from rabbitsense import adapt, adapt_motion
from test_adapt import make_manifest, tiny_model


def test_static_counterfactual_and_moving_window(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    encoder = adapt.Encoder('cpu', tiny_model(tmp_path))
    frames = np.full((2, 8, 224, 224, 3), 80, np.uint8)
    frames[1, :4, :, :100] = 220
    real, still = adapt_motion.encode_contrast(encoder, frames)
    assert np.allclose(real[0], still[0], atol=1e-6)
    assert np.linalg.norm(real[1] - still[1]) > 1e-4
    assert np.allclose(np.linalg.norm(real, axis=1), 1.)


def test_weights_balance_domains_videos_and_classes():
    y = np.array([1., 1., 1., 0., 0., 1., 0.])
    groups = np.array(['a', 'a', 'a', 'a', 'a', 'b', 'c'])
    w = adapt_motion.group_weights(y, groups)
    assert w[y == 1].sum() == pytest.approx(.5)
    assert w[(y == 1) & (groups == 'a')].sum() == pytest.approx(w[(y == 1) & (groups == 'b')].sum())
    assert w.sum() == pytest.approx(1.)
    with pytest.raises(ValueError, match='Falta una clase'):
        adapt_motion.group_weights(np.ones(3), np.array(['a'] * 3))


def test_containment_and_smoothing_respect_time_boundaries():
    data = dict(seconds=np.array([1.5, 2., 4., 8.]), start=np.array([.25, .75, 2.75, 6.75]),
                stop=np.array([2., 2.5, 4.5, 8.5]))
    clips = pd.DataFrame(dict(start=[0., 6.], end=[4., 10.], label=['locomotion', 'resting']))
    assert np.allclose(adapt_motion.labels_inside(data, clips), [1., 1., np.nan, 0.], equal_nan=True)
    with pytest.raises(ValueError, match='solapados'):
        adapt_motion.labels_inside(data, pd.concat([clips, clips]))
    t, p = np.array([1., 1.5, 2., 5., 5.5]), np.array([0., 1., 0., 1., 1.])
    result = adapt_motion.smooth_predictions({'x': (t, p)}, 3)['x'][1]
    assert np.allclose(result, [.5, 1 / 3, .5, 1., 1.])
    assert np.array_equal(adapt_motion.smooth_predictions({'x': (t, p)}, 1)['x'][1], p)


def test_nested_sessions_and_held_labels_do_not_affect_predictions(tmp_path, monkeypatch):
    pytest.importorskip('sklearn')
    pytest.importorskip('torch')
    manifest = make_manifest(tmp_path, names=('a', 'b', 'IMG_5190', 'IMG_5191'))
    cases = adapt.load_cases(manifest)
    data = {}
    for case in cases:
        ends = np.arange(1.5, 9.5, .5)
        x = (ends < 4.).astype(float)[:, None]
        data[case['name']] = dict(seconds=ends, appearance=x, physical=x, still=np.zeros_like(x),
                                 start=ends - 1.25, stop=ends + .5, fps=10., count=100)
    external = [('youtube', dict(appearance=np.array([[0.], [1.]]), still=np.zeros((2, 1)), bag=np.array([0, 1])),
                 np.array([0., 1.]))]
    configs = tuple(('appearance', source, 1., loss) for source in ('youtube', 'combined') for loss in ('window', 'mil'))
    configs += tuple(('physical', 'cinna', 1., loss) for loss in ('window', 'mil', 'rbf_window', 'rbf_mil'))
    seen = []
    original = adapt_motion.fit_model

    def spy(training, outside, config):
        seen.append({name for name, _, _ in training})
        return original(training, outside, config)

    monkeypatch.setattr(adapt_motion, 'fit_model', spy)
    report = adapt_motion.nested(cases, data, external, tmp_path / 'first',
                                 configs=configs, smoothing=(1, 3))
    first = adapt.load_predictions(tmp_path / 'first', 'selected_nested__session')
    for names in seen:
        assert not ({'IMG_5190', 'IMG_5191'} & names) or {'IMG_5190', 'IMG_5191'} <= names
        assert len(names) < len(cases)
    assert set(report['choices']) == {'a', 'b', 'mano_IMG_5190-5193'}
    for held, choice in report['choices'].items():
        assert held not in choice['training_sessions']
    assert report['result']['verdict']['locomotion_clips'] == 4
    labels = pd.read_csv(tmp_path / 'labels.csv')
    mask = labels.video_id.isin(['IMG_5190.mp4', 'IMG_5191.mp4']) & labels.label.isin(adapt.CLASSES)
    labels.loc[mask, 'label'] = labels.loc[mask, 'label'].map({'locomotion': 'resting', 'resting': 'locomotion'})
    labels.to_csv(tmp_path / 'labels.csv', index=False)
    adapt_motion.nested(adapt.load_cases(manifest), data, external, tmp_path / 'second',
                         configs=configs, smoothing=(1, 3))
    second = adapt.load_predictions(tmp_path / 'second', 'selected_nested__session')
    for name in ('IMG_5190', 'IMG_5191'):
        assert np.array_equal(first[name][1], second[name][1])
    saved = json.loads((tmp_path / 'first/report.json').read_text())
    assert saved['band'] == list(adapt.BAND)


def test_mil_does_not_force_pauses_positive():
    pytest.importorskip('sklearn')
    pytest.importorskip('torch')
    x = np.array([[0.], [0.], [1.], [0.], [0.], [0.]])
    data = dict(appearance=x, still=np.zeros_like(x), bag=np.array([0, 0, 0, 1, 1, 1]))
    labels = np.array([1., 1., 1., 0., 0., 0.])
    entries = [('a', data, labels), ('b', data, labels)]
    model = adapt_motion.fit_model(entries, entries, ('appearance', 'combined', 1., 'mil'))
    p = adapt.predict(model, x)
    assert p[2] >= .6 and np.max(p[[0, 1, 3, 4, 5]]) <= .4


@pytest.mark.parametrize('loss', ['window', 'mil'])
def test_constant_fractional_features_have_zero_weight(loss):
    pytest.importorskip('sklearn')
    pytest.importorskip('torch')
    x = np.c_[np.array([0., 0., 1., 0., 0., 0.]), np.full(6, np.log(2.)), np.zeros(6)]
    d = dict(physical=x, bag=np.array([0, 0, 0, 1, 1, 1]))
    y = np.array([1., 1., 1., 0., 0., 0.])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        m = adapt_motion.fit_model([('a', d, y)], [], ('physical', 'cinna', 1., loss))
    assert m['weights'][2:] == [0., 0.]
    changed = x.copy()
    changed[:, 1:] = 400.
    assert np.array_equal(adapt.predict(m, x), adapt.predict(m, changed))


def test_nonlinear_model_serialization_and_interactions():
    pytest.importorskip('sklearn')
    x = np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (20, 1))
    y = (x[:, 0] != x[:, 1]).astype(float)
    entries = [('a', dict(physical=x), y)]
    model = adapt_motion.fit_model(entries, [], ('physical', 'cinna', 1., 'rbf_window'))
    p = adapt_motion.predict_model(json.loads(json.dumps(model)), x)
    assert np.min(p[y == 1]) >= .6 and np.max(p[y == 0]) <= .4