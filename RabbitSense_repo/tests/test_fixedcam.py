import numpy as np
import pytest

from rabbitsense import fixedcam, swiftgolden


def test_net_displacement_interpolates_and_uses_median_size():
    times = [0., .5, 1., 1.5, 2.]
    xs, ys = [0., 10., 20., 30., 40.], [0.] * 5
    areas = [100., 100., 400., 100., 100.]
    assert fixedcam.net_displacement(times, xs, ys, areas, .25, 1.75) == pytest.approx(30 / 10)
    assert fixedcam.position(times, xs, ys, .75) == pytest.approx((15., 0.))


def test_missing_positions_give_none_not_rest():
    times, xs, ys, areas = [0., 1., 2.], [np.nan, 5., np.nan], [0., 0., 0.], [100.] * 3
    assert fixedcam.net_displacement(times, xs, ys, areas, 0., 2.) is None
    assert fixedcam.position(times, xs, ys, 1.1) == (5., 0.)


def test_pure_golden_is_reproducible(tmp_path, monkeypatch):
    pytest.importorskip('transformers')
    monkeypatch.setattr(swiftgolden, 'PACKAGE', tmp_path / 'pkg')
    swiftgolden.pure(tmp_path / 'a')
    swiftgolden.pure(tmp_path / 'b')
    assert (tmp_path / 'a' / 'pure.json').read_text() == (tmp_path / 'b' / 'pure.json').read_text()
    assert (tmp_path / 'pkg' / 'Sources' / 'RabbitSenseKit' / 'Resources' / 'ytactos_modelo.json').is_file()
