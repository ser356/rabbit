import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import activity, pointtrack
from test_pointtrack import FPS, RABBIT, FakeTracker, fake_frames, make_case  # noqa: F401 (fixture)

UNIT = np.sqrt(RABBIT[0] * RABBIT[1])


def test_continuous_displacement_matches_the_synthetic_scene(tmp_path, fake_frames):
    """pan: avanza 40 px/s de 0 a 4 s con cámara fija; quieto de 4 a 10 s mientras la cámara barre y se acicala."""
    video, masks = make_case(tmp_path, 'pan', 'pan')
    summary = activity.measure(video, masks, tmp_path / 'out', tmp_path / 'tracks' / 'pan', FakeTracker('pan', groom=True))
    rows = pd.read_csv(tmp_path / 'out' / 'actividad.csv')
    moving = rows[(rows.seconds > .5) & (rows.seconds <= 4.)]
    still = rows[(rows.seconds > 4.5) & (rows.seconds <= 9.5)]
    assert moving.displacement.to_numpy() == pytest.approx(np.full(len(moving), 20 / UNIT), rel=.05)
    assert still.displacement.max() < .05 and rows[rows.interval_seconds > 0].measured.all()
    expected = 4 * 40 / UNIT
    assert summary['displacement_body_units']['estimate'] == pytest.approx(expected, rel=.05)
    assert summary['displacement_body_units']['lower'] <= summary['displacement_body_units']['estimate']
    assert summary['displacement_per_measured_minute']['estimate'] == pytest.approx(expected / (summary['measured_seconds'] / 60))
    assert summary['unmeasured_seconds'] == pytest.approx(0, abs=1e-6)
    assert summary['rule_seconds']['clear_displacement'] > 2 and summary['rule_seconds']['low_motion'] > 3
    assert set(rows.columns) >= {'displacement_lower', 'displacement_upper', 'movement', 'span'}
    image = cv2.imread(str(tmp_path / 'out' / 'actividad.png'))
    assert image is not None and image.shape[1] == 1200
    stored = json.loads((tmp_path / 'out' / 'resumen.json').read_text())
    assert stored['video_sha256'] and 'no son centímetros' in stored['units']


def test_camera_following_the_rabbit_still_counts_as_displacement(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'follow', 'follow')
    summary = activity.measure(video, masks, tmp_path / 'out', tmp_path / 'tracks' / 'follow', FakeTracker('follow'))
    assert summary['displacement_body_units']['estimate'] == pytest.approx(4 * 40 / UNIT, rel=.05)


def test_without_mask_nothing_is_measured_and_nothing_is_rest(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'blind', 'pan', masked=False)
    summary = activity.measure(video, masks, tmp_path / 'out', tmp_path / 'tracks' / 'blind', FakeTracker('pan'))
    assert summary['measured_seconds'] == 0 and summary['displacement_per_measured_minute']['estimate'] is None
    assert summary['rule_seconds']['low_motion'] == 0
    assert summary['unmeasured_seconds'] == pytest.approx(pointtrack.frame_times(video)[-1])


class Jittery(FakeTracker):
    """Como FakeTracker, pero los puntos del conejo tiemblan juntos ±3 px por fotograma (seguimiento ruidoso)."""

    def __call__(self, frames, queries):
        tracks, visible = super().__call__(frames, queries)
        index = frames[:, 0, 0, 0].astype(int) + 256 * frames[:, 0, 0, 1].astype(int)
        q = int(queries[0, 0])
        left = self.rabbit_x[index[q]]
        x0, y0 = queries[:, 1], queries[:, 2]
        inside = (x0 >= left - 1) & (x0 <= left + RABBIT[0] + 1) & (y0 >= 119) & (y0 <= 121 + RABBIT[1])
        noise = np.random.default_rng(0).uniform(-3, 3, (10_000, 2))
        tracks[:, inside] += noise[index][:, None, :]
        return tracks, visible


def test_two_second_blocks_keep_straight_motion_and_cut_accumulated_jitter(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'pan', 'pan')
    summary = activity.measure(video, masks, tmp_path / 'clean', tmp_path / 'tracks' / 'pan', FakeTracker('pan'))
    assert summary['net_2s']['displacement_body_units']['estimate'] == pytest.approx(4 * 40 / UNIT, rel=.05)
    blocks = pd.read_csv(tmp_path / 'clean' / 'actividad_2s.csv')
    assert list(blocks.seconds[:3]) == [0., 2., 4.] and blocks[blocks.interval_seconds > 0].measured.all()
    video, masks = make_case(tmp_path, 'jit', 'pan')
    noisy = activity.measure(video, masks, tmp_path / 'noisy', tmp_path / 'tracks' / 'jit', Jittery('pan'))
    fine = pd.read_csv(tmp_path / 'noisy' / 'actividad.csv')
    net = pd.read_csv(tmp_path / 'noisy' / 'actividad_2s.csv')
    rest_fine = fine[(fine.seconds > 6) & (fine.interval_seconds > 0)].displacement.sum()
    rest_net = net[(net.seconds > 6) & (net.interval_seconds > 0)].displacement.sum()
    assert rest_fine > 2.5 * rest_net  # pasos cortos acumulan el temblor; bloques netos no
    assert noisy['net_2s']['displacement_body_units']['estimate'] < noisy['displacement_body_units']['estimate']


def test_evaluation_applies_the_preregistered_criterion(tmp_path, fake_frames):
    cases, rows = [], []
    for name, tracker in (('IMG_5199', Jittery('pan')), ('fixed', FakeTracker('pan'))):
        video, masks = make_case(tmp_path, name, 'pan')
        activity.measure(video, masks, tmp_path / 'out' / name, tmp_path / 'tracks' / name, tracker)
        cases.append(dict(name=name, video=video.name, labels='labels.csv', masks=masks.name))
        rows += [dict(clip_id=f'{name}_l', video_id=video.name, start_seconds=0., duration_seconds=4., label='locomotion',
                      reviewed='si'),
                 dict(clip_id=f'{name}_r', video_id=video.name, start_seconds=6., duration_seconds=4., label='resting',
                      reviewed='si')]
    pd.DataFrame(rows).to_csv(tmp_path / 'labels.csv', index=False)
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=cases)))
    result = activity.evaluate(tmp_path / 'out', tmp_path / 'm.json')
    assert result['rate_05']['auc'] == result['rate_2s']['auc'] == 1.
    assert result['handheld_rest_median_drop'] > activity.ADOPT['handheld_rest_median_drop'] and result['adopt_2s']
    table = pd.read_csv(tmp_path / 'out' / 'evaluacion_clips.csv')
    assert len(table) == 4 and table.handheld.sum() == 2


class LosesRabbit(FakeTracker):
    """Como FakeTracker, pero CoTracker da por ocultos los puntos del conejo mientras se desplaza (0–4 s)."""

    def __call__(self, frames, queries):
        tracks, visible = super().__call__(frames, queries)
        index = frames[:, 0, 0, 0].astype(int) + 256 * frames[:, 0, 0, 1].astype(int)
        q = int(queries[0, 0])
        left = self.rabbit_x[index[q]]
        x0, y0 = queries[:, 1], queries[:, 2]
        inside = (x0 >= left - 1) & (x0 <= left + RABBIT[0] + 1) & (y0 >= 119) & (y0 <= 121 + RABBIT[1])
        moving = index / FPS < 4.2
        visible[np.ix_(moving & (np.arange(len(index)) != q), inside)] = False
        return tracks, visible


def test_mask_centre_covers_blocks_where_rabbit_points_are_lost(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'pan', 'pan')
    times = pointtrack.frame_times(video, tmp_path / 'tracks' / 'tiempos')
    folder = tmp_path / 'tracks' / 'pan'
    pointtrack.track_case(video, masks, folder, LosesRabbit('pan'), None, times)
    plain = pd.DataFrame(activity.step_rows(times, folder, 2.))
    moving = plain[(plain.interval_seconds > 0) & (plain.seconds <= 4.)]
    assert not moving.measured.any() and set(moving.reason) == {'pocos_puntos_conejo'}
    backed = pd.DataFrame(activity.step_rows(times, folder, 2., masks))
    moving = backed[(backed.interval_seconds > 0) & (backed.seconds <= 4.)]
    assert moving.measured.all() and set(moving.source) == {'mascara'}
    assert moving.displacement.to_numpy() == pytest.approx(np.full(len(moving), 80 / UNIT), rel=.05)
    still = backed[(backed.interval_seconds > 0) & (backed.seconds > 6.)]
    assert set(still.source) == {'puntos'} and still.displacement.max() < .05
    only_mask = pd.DataFrame(activity.step_rows(times, folder, 2., masks, 'mask'))
    later = only_mask[(only_mask.interval_seconds > 0) & (only_mask.seconds > 6.)]
    assert later.measured.all() and later.displacement.max() < .05  # la máscara coincide con los puntos


def test_mask_that_changes_size_is_not_used(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'pan', 'pan')
    for f in range(0, 26):  # 0–0,9 s, antes del fotograma de consulta: la máscara se arrastra a una zona mucho mayor
        mask = cv2.imread(str(masks / 'masks' / f'{f:06d}.png'), cv2.IMREAD_GRAYSCALE)
        mask[20:100, 300:500] = 255
        cv2.imwrite(str(masks / 'masks' / f'{f:06d}.png'), mask)
    times = pointtrack.frame_times(video, tmp_path / 'tracks' / 'tiempos')
    folder = tmp_path / 'tracks' / 'pan'
    pointtrack.track_case(video, masks, folder, LosesRabbit('pan'), None, times)
    rows = pd.DataFrame(activity.step_rows(times, folder, 2., masks))
    first = rows[(rows.interval_seconds > 0) & (rows.seconds <= 2.)]
    assert not first.measured.any()


def test_coverage_reports_gain_reasons_and_criterion(tmp_path, fake_frames, monkeypatch):
    monkeypatch.setattr(pointtrack, 'TRACKS', tmp_path / 'tracks')
    video, masks = make_case(tmp_path, 'pan', 'pan')
    times = pointtrack.frame_times(video, tmp_path / 'tracks' / 'tiempos')
    pointtrack.track_case(video, masks, tmp_path / 'tracks' / 'pan', LosesRabbit('pan'), None, times)
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name='pan', video=video.name, masks=masks.name)])))
    pd.DataFrame(dict(case='pan', start=[0., 2., 4., 6., 8.], act=['locomotion', 'locomotion', 'loaf', 'loaf', 'eating'],
                      end=[2., 4., 6., 8., 10.])).to_csv(tmp_path / 'acts.csv', index=False)
    result = activity.coverage(tmp_path / 'm.json', tmp_path / 'acts.csv', tmp_path / 'out')
    assert result['coverage']['sin']['locomotion'] == 0 and result['coverage']['con']['locomotion'] == 1
    assert result['unmeasured_reasons']['sin'] == {"(True, 'pocos_puntos_conejo')": 2}
    assert result['consistency']['median_abs_difference'] < .05 and result['auc']['con'] == 1.
    assert result['adopt_mask'] and result['mask_blocks_added'] == 2
    assert len(pd.read_csv(tmp_path / 'out' / 'bloques.csv')) == 5
