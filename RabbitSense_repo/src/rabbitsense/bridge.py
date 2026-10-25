import cv2
import numpy as np
import pandas as pd

from .tracking import rectangle_mask


def refine_video(video, anchors):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'No se puede abrir {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Metadatos de vídeo inválidos para el puente visual.')
        rows, history = [], []
        previous, points, first = None, None, None
        for frame in range(count):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Error en puente visual, fotograma {frame}.')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            guide = anchors.interpolate(frame / fps)
            exact, anchor = anchors.at_frame(frame, fps)
            if exact:
                guide = anchor
            values = [None] * 4 if guide is None else guide.tolist()
            rows.append(dict(zip(['x', 'y', 'width', 'height'], values)))
            rows[-1].update(refined_x=values[0], refined_y=values[1],
                            refinement='guide_only', bridge_points=0)
            if points is not None and previous is not None:
                alive = np.isfinite(points).all(axis=1)
                ids = np.flatnonzero(alive)
                current = np.full_like(points, np.nan)
                if len(ids):
                    old = points[ids].reshape(-1, 1, 2)
                    nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, old, None,
                                                            winSize=(21, 21), maxLevel=3)
                    if nxt is not None:
                        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, previous, nxt, None,
                                                                  winSize=(21, 21), maxLevel=3)
                        if back is not None:
                            b = nxt.reshape(-1, 2)
                            good = ((status.ravel() == 1) & (reverse.ravel() == 1) &
                                    (np.linalg.norm(old.reshape(-1, 2) - back.reshape(-1, 2), axis=1) <= 1.) &
                                    (b[:, 0] >= 0) & (b[:, 0] < gray.shape[1]) &
                                    (b[:, 1] >= 0) & (b[:, 1] < gray.shape[0]))
                            current[ids[good]] = b[good]
                points = current
                history.append(points.copy())
            if exact:
                if anchor is not None and points is not None and first is not None:
                    x, y, w, h = anchor
                    good = (np.isfinite(points).all(axis=1) &
                            (points[:, 0] >= x + .12 * w) & (points[:, 0] <= x + .88 * w) &
                            (points[:, 1] >= y + .12 * h) & (points[:, 1] <= y + .88 * h))
                    if good.sum() >= 6:
                        paths = np.stack(history)[:, good]
                        displacement = np.median(paths - paths[0], axis=1)
                        correction = displacement - np.linspace(0, 1, len(paths))[:, None] * displacement[-1]
                        for index, delta in enumerate(correction, first):
                            rows[index]['refined_x'] += float(delta[0])
                            rows[index]['refined_y'] += float(delta[1])
                            rows[index]['refinement'] = 'persistent_visual_bridge'
                            rows[index]['bridge_points'] = int(good.sum())
                points, first, history = None, None, []
                if anchor is not None:
                    found = cv2.goodFeaturesToTrack(gray, 250, .01, 4,
                                                    mask=rectangle_mask(gray.shape, anchor, .2), blockSize=5)
                    if found is not None:
                        points = found.reshape(-1, 2)
                        first = frame
                        history = [points.copy()]
            previous = gray
    finally:
        cap.release()
    return pd.DataFrame(rows)