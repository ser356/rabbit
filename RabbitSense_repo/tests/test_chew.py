import numpy as np
import pandas as pd
import pytest

from rabbitsense import chew


def scene(frequency, fps=30., seconds=2., amplitude=3., noise=1., seed=0):
    """Fondo con textura y un «hocico» de 24 px que sube y baja a `frequency` Hz (0: quieto)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(60, 200, (256, 256)).astype(np.float32)
    times = np.arange(int(fps * seconds)) / fps
    frames = []
    for t in times:
        image = base + rng.normal(0, noise, base.shape)
        shift = int(round(amplitude * np.sin(2 * np.pi * frequency * t))) if frequency else 0
        image[120 + shift:144 + shift, 120:144] = 250
        frames.append(np.clip(image, 0, 255).astype(np.uint8))
    return np.stack(frames), times


def test_chewing_rhythm_scores_higher_than_stillness_or_slow_motion():
    chewing = chew.chew_score(*scene(3.))[0]
    still = chew.chew_score(*scene(0.))[0]
    slow = chew.chew_score(*scene(.7))[0]
    fast = chew.chew_score(*scene(9.))[0]
    assert chewing > .6 and chewing > 2 * max(still, slow, fast)


def test_frame_rate_comes_from_real_times_and_short_windows_are_skipped():
    frames, times = scene(3., fps=60.)
    score, fps = chew.chew_score(frames, times)
    assert fps == pytest.approx(60.) and score > .6
    assert np.isnan(chew.chew_score(frames[:5], times[:5])[0])


def test_moving_rabbit_cannot_be_chewing():
    out = chew.effective([.8, .8, .8], [.1, 1.2, np.nan])
    assert list(out) == [.8, 0., .8]


def test_evaluation_fixes_threshold_on_development_and_tests_once(tmp_path):
    rng = np.random.default_rng(0)
    acts = ['eating'] * 10 + ['loaf'] * 20 + ['locomotion'] * 10
    dev = pd.DataFrame(dict(case='c', start=np.arange(40) * 2., found=4,
                            score=[.7 + .1 * rng.random() if a == 'eating' else .2 + .3 * rng.random() for a in acts]))
    dev.to_csv(tmp_path / 'desarrollo.csv', index=False)
    pd.DataFrame(dict(case='c', start=np.arange(40) * 2., act=acts)).to_csv(tmp_path / 'dev_labels.csv', index=False)
    pd.DataFrame(dict(case='c', start=np.arange(40) * 2., net_displacement=[1. if a == 'locomotion' else .1 for a in acts])
                 ).to_csv(tmp_path / 'features.csv', index=False)
    test = pd.DataFrame(dict(video='v', start=np.arange(6) * 2., found=4, score=[.75, .8, .3, .72, .9, .76]))
    test.to_csv(tmp_path / 'prueba.csv', index=False)
    (tmp_path / 'analisis' / 'v').mkdir(parents=True)
    pd.DataFrame(dict(start=np.arange(6) * 2., displacement=[.1, .1, .1, 2., .1, .1])).to_csv(
        tmp_path / 'analisis' / 'v' / 'ventanas.csv', index=False)
    pd.DataFrame(dict(video='v', start=np.arange(6) * 2., act='eating')).to_csv(tmp_path / 'test_labels.csv', index=False)
    result = chew.evaluate(tmp_path, tmp_path / 'dev_labels.csv', tmp_path / 'features.csv', tmp_path / 'test_labels.csv',
                           tmp_path / 'analisis', tmp_path / 'out')
    assert result['development']['auc'] == 1. and result['development']['false_alarms_by_act']['locomotion'] == 0
    assert result['test']['hits'] == 4 and result['adopted']  # la ventana con desplazamiento 2 no cuenta
