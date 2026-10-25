from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from rabbitsense import v36, v37


@pytest.mark.parametrize('module', [v36, v37])
def test_legacy_overlay_uses_sample_rate(tmp_path, module):
    video = tmp_path / 'static.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 240))
    assert writer.isOpened()
    image = np.random.default_rng(3).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for _ in range(6):
        writer.write(image)
    writer.release()
    tracks = tmp_path / 'tracks.csv'
    pd.DataFrame([dict(seconds=t, status='manual', x=110, y=80, width=80, height=80)
                  for t in (0., .5)]).to_csv(tracks, index=False)
    module.run(SimpleNamespace(video=str(video), tracks=str(tracks), out=str(tmp_path),
                               step=.5, threshold=.2, max_jump=1.5, max_scale_change=1.8,
                               video_out=True))
    name = module.__name__.rsplit('.', 1)[-1]
    cap = cv2.VideoCapture(str(tmp_path / f'motion_{name}.mp4'))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 2
    assert cap.get(cv2.CAP_PROP_FPS) == 2.
    cap.release()