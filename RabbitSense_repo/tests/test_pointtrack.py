import json

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import pointtrack

W, H, FPS, FRAMES, RABBIT = 512, 288, 30., 300, (60, 40)


def scenario(kind):
    """Posición del conejo y desplazamiento de la cámara (en píxeles de imagen) por fotograma.

    follow: el conejo avanza 0–4 s y la cámara lo sigue (en imagen no se mueve); quieto 6–10 s.
    pan: el conejo avanza 0–4 s con cámara fija; quieto 6–10 s mientras la cámara barre y se acicala.
    """
    t = np.arange(FRAMES) / FPS
    world = 150 + 40 * np.clip(t, 0, 4)
    if kind == 'follow':
        camera = 40 * np.clip(t, 0, 4)
    else:
        camera = 40 * np.clip(t - 6, 0, 4)
    return world - camera, -camera  # x del conejo en imagen; desplazamiento del fondo en imagen


class FakeTracker:
    """Devuelve las trayectorias exactas de la escena sintética (el índice de fotograma va codificado en el píxel)."""

    def __init__(self, kind, groom=False):
        self.rabbit_x, self.background = scenario(kind)
        self.groom = groom
        self.calls = 0

    def __call__(self, frames, queries):
        self.calls += 1
        index = frames[:, 0, 0, 0].astype(int) + 256 * frames[:, 0, 0, 1].astype(int)
        q = int(queries[0, 0])
        x0, y0 = queries[:, 1], queries[:, 2]
        left = self.rabbit_x[index[q]]
        inside = (x0 >= left - 1) & (x0 <= left + RABBIT[0] + 1) & (y0 >= 120 - 1) & (y0 <= 120 + RABBIT[1] + 1)
        tracks = np.zeros((len(frames), len(queries), 2))
        for j, f in enumerate(index):
            dx = np.where(inside, self.rabbit_x[f] - left, self.background[f] - self.background[index[q]])
            if self.groom:
                wiggle = (np.arange(len(queries)) % 5 == 0) & inside
                dx = dx + wiggle * 15 * np.sin(f / 3)
            tracks[j] = np.c_[x0 + dx, y0]
        visible = (tracks[..., 0] >= 0) & (tracks[..., 0] < W)
        return tracks, visible


def make_case(tmp_path, name, kind, masked=True):
    video = tmp_path / f'{name}.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for _ in range(FRAMES):
        writer.write(np.zeros((H, W, 3), np.uint8))
    writer.release()
    masks = tmp_path / f'masks_{name}' / 'masks'
    masks.mkdir(parents=True)
    rabbit_x, _ = scenario(kind)
    for f in range(FRAMES):
        mask = np.zeros((H, W), np.uint8)
        if masked:
            x = int(round(rabbit_x[f]))
            mask[120:120 + RABBIT[1], x:x + RABBIT[0]] = 255
        cv2.imwrite(str(masks / f'{f:06d}.png'), mask)
    return video, masks.parent


@pytest.fixture
def fake_frames(monkeypatch):
    def read(video, indices):
        frames = {}
        for f in indices:
            image = np.zeros((H, W, 3), np.uint8)
            image[0, 0, 0], image[0, 0, 1] = f % 256, f // 256
            frames[f] = image
        return frames
    monkeypatch.setattr(pointtrack, 'read_frames', read)


def write_manifest(tmp_path, cases):
    rows = []
    for name, video, _ in cases:
        rows += [dict(clip_id=f'{name}_l', video_id=video.name, start_seconds=0., duration_seconds=4., label='locomotion',
                      reviewed='si'),
                 dict(clip_id=f'{name}_r', video_id=video.name, start_seconds=6., duration_seconds=4., label='resting',
                      reviewed='si')]
    pd.DataFrame(rows).to_csv(tmp_path / 'labels.csv', index=False)
    (tmp_path / 'm.json').write_text(json.dumps(dict(cases=[dict(name=n, video=v.name, labels='labels.csv', masks=m.name)
                                                            for n, v, m in cases])))
    return tmp_path / 'm.json'


def test_chunk_plan_covers_every_window():
    starts = pointtrack.chunk_starts(10.)
    assert starts == [0., 2., 4., 6.]
    assert pointtrack.chunk_starts(3.) == [0.]
    for centre in np.arange(.25, 10., .5):
        k = pointtrack.nearest_chunk(starts, centre)
        assert starts[k] - 1e-6 <= max(centre - 1, 0) and min(centre + 1, 10.) <= starts[k] + pointtrack.CHUNK + 1e-6
    clips = pd.DataFrame(dict(start=[0.], end=[4.]))
    assert pointtrack.needed_chunks(starts, [0, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4.], clips) == [0, 1]


def test_queries_split_rabbit_background_and_ring():
    mask = np.zeros((H, W), np.uint8)
    mask[100:160, 200:280] = 1
    points, kinds, unit = pointtrack.sample_queries(mask)
    assert unit == pytest.approx(np.sqrt(60 * 80))
    rabbit, background = points[kinds == 0], points[kinds > 0]
    assert 10 <= len(rabbit) <= pointtrack.RABBIT_POINTS and (kinds == 2).sum() > 10
    assert ((rabbit[:, 0] >= 200) & (rabbit[:, 0] < 280) & (rabbit[:, 1] >= 100) & (rabbit[:, 1] < 160)).all()
    assert not ((background[:, 0] >= 200) & (background[:, 0] < 280) & (background[:, 1] >= 100) &
                (background[:, 1] < 160)).any()


@pytest.mark.parametrize('kind', ['follow', 'pan'])
def test_displacement_relative_to_floor_not_image(tmp_path, fake_frames, kind):
    video, masks = make_case(tmp_path, kind, kind)
    tracker = FakeTracker(kind, groom=kind == 'pan')
    folder = tmp_path / 'tracks'
    clips = pd.DataFrame(dict(start=[0., 6.], end=[4., 10.]))
    assert pointtrack.track_case(video, masks, folder, tracker, clips) == 4
    assert pointtrack.track_case(video, masks, folder, tracker, clips) == 0  # guardados: no se repite
    rows = pd.DataFrame(pointtrack.analyze(video, folder))
    moving = rows[(rows.seconds > .5) & (rows.seconds <= 3.5)]
    resting = rows[(rows.seconds > 7.) & (rows.seconds <= 9.)]
    assert (moving.movement == 'locomotion_candidate').all(), moving[['seconds', 'span', 'span_lower']]
    assert (resting.movement == 'low_motion_candidate').all(), resting[['seconds', 'span', 'span_upper']]
    if kind == 'pan':
        assert resting.camera_shift.max() > 1.  # la cámara se mueve más de un tamaño y aun así es reposo
    else:
        assert moving.camera_shift.max() > 1.


def test_missing_mask_or_points_give_unknown(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'nomask', 'pan', masked=False)
    folder = tmp_path / 'tracks'
    pointtrack.track_case(video, masks, folder, FakeTracker('pan'), pd.DataFrame(dict(start=[0.], end=[4.])))
    rows = pd.DataFrame(pointtrack.analyze(video, folder))
    assert (rows.movement == 'unknown').all() and (rows.evidence_quality == 0).all()
    assert pointtrack.decide(None, 1.)['movement'] == 'unknown'


def test_run_applies_closure_criteria_and_renders(tmp_path, fake_frames, monkeypatch):
    cases = []
    for name, kind in (('a', 'follow'), ('b', 'pan')):
        video, masks = make_case(tmp_path, name, kind)
        cases.append((name, video, masks))
    manifest = write_manifest(tmp_path, cases)
    trackers = {'a': FakeTracker('follow'), 'b': FakeTracker('pan', groom=True)}
    current = {}
    original = pointtrack.track_case

    def track(video, masks, folder, tracker, clips=None):
        return original(video, masks, folder, trackers[video.stem], clips)
    monkeypatch.setattr(pointtrack, 'track_case', track)
    smoked = []
    monkeypatch.setattr(pointtrack, 'smoke', lambda model: smoked.append(model) or (0., 0.))
    result = pointtrack.run(manifest, tmp_path / 'out', tmp_path / 'tracks', tracker=lambda: current)
    assert smoked == [current]  # una sola comprobación antes del primer vídeo
    r = result['methods']['pointtrack']
    assert (r['locomotion_hits'], r['resting_correct'], r['resting_false_events']) == (2, 2, 0)
    assert result['status'] == 'not_resolved'  # menos clips que el mínimo de CRITERIA
    assert result['clip_auc_max_span'] == 1. and result['clip_coverage'] == 1.
    table = pd.read_csv(tmp_path / 'out' / 'clips.csv')
    assert len(table) == 4 and table.correct.all()
    again = pointtrack.run(manifest, tmp_path / 'out2', tmp_path / 'tracks', tracker=None)
    assert again['methods']['pointtrack']['locomotion_hits'] == 2
    pointtrack.render(manifest, tmp_path / 'out', tmp_path / 'tracks', videos=tmp_path / 'diag')
    capture = cv2.VideoCapture(str(tmp_path / 'diag' / 'a.mp4'))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) > 30
    capture.release()


def test_smoke_check_accepts_exact_tracker_and_rejects_frozen_points():
    centre = 100 + 45 + pointtrack.SMOKE_SIZE / 2

    def exact(frames, queries):
        tracks = np.repeat(queries[None, :, 1:], len(frames), axis=0).astype(float)
        tracks[:, 0, 0] = centre + 3 * (np.arange(len(frames)) - 15)
        return tracks, np.ones(tracks.shape[:2], bool)
    assert pointtrack.smoke(exact) == (0., 0.)

    def frozen(frames, queries):
        return np.repeat(queries[None, :, 1:], len(frames), axis=0).astype(float), np.ones((len(frames), len(queries)), bool)
    with pytest.raises(SystemExit):
        pointtrack.smoke(frozen)


def test_run_refuses_to_track_when_smoke_fails(tmp_path, fake_frames):
    video, masks = make_case(tmp_path, 'a', 'follow')
    manifest = write_manifest(tmp_path, [('a', video, masks)])

    def frozen(frames, queries):
        return np.repeat(queries[None, :, 1:], len(frames), axis=0).astype(float), np.ones((len(frames), len(queries)), bool)
    with pytest.raises(SystemExit):
        pointtrack.run(manifest, tmp_path / 'out', tmp_path / 'tracks', tracker=lambda: frozen)
    assert not (tmp_path / 'tracks' / 'a').exists()


def test_queries_are_fast_for_a_large_rabbit():
    mask = np.zeros((288, 512), np.uint8)
    mask[40:260, 80:420] = 1  # √área ≈ 270 px: el anillo llega a más de 600 px
    start = __import__('time').perf_counter()
    points, kinds, unit = pointtrack.sample_queries(mask)
    assert __import__('time').perf_counter() - start < 1. and (kinds == 0).sum() > 10
