import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class Anchors:
    times: np.ndarray
    boxes: list

    def at_frame(self, frame, fps):
        ids = np.flatnonzero(np.rint(self.times * fps).astype(int) == frame)
        return (True, self.boxes[ids[-1]]) if len(ids) else (False, None)

    def interpolate(self, seconds):
        i = int(np.searchsorted(self.times, seconds))
        if i < len(self.times) and abs(self.times[i] - seconds) < 1e-6:
            return self.boxes[i]
        if i == 0 or i == len(self.times):
            return None
        a, b = self.boxes[i - 1:i + 1]
        if a is None or b is None:
            return None
        weight = (seconds - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        return a * (1 - weight) + b * weight


def load_anchors(path, width, height):
    path = Path(path)
    if path.suffix.lower() != '.json':
        raise ValueError('Use el JSON original de anclas; el CSV interpolado no identifica anclas únicas.')
    data = json.loads(path.read_text())
    if 'marks' in data:
        rw, rh = data['width'], data['height']
        entries = [(m['time'], m['box']) for m in data['marks']]
    else:
        rw, rh = data['reference_width'], data['reference_height']
        entries = [(m['time'], None if m['bbox'] is None else
                    [m['bbox'][0], m['bbox'][1], m['bbox'][2] - m['bbox'][0],
                     m['bbox'][3] - m['bbox'][1]]) for m in data['anchors']]
    if rw <= 0 or rh <= 0 or not entries:
        raise ValueError('Dimensiones o anclas vacías inválidas.')
    entries.sort(key=lambda item: item[0])
    times = np.array([item[0] for item in entries], dtype=float)
    if not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Tiempos de anclas deben ser finitos, únicos y crecientes.')
    scale = np.array([width / rw, height / rh] * 2)
    boxes = []
    for _, box in entries:
        if box is None:
            boxes.append(None)
            continue
        b = np.asarray(box, dtype=float)
        if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= 0):
            raise ValueError('Caja inválida: se esperan x, y, ancho y alto positivos.')
        b = b * scale
        if b[0] >= width or b[1] >= height or np.any(b[:2] + b[2:] <= 0):
            raise ValueError('Ancla completamente fuera del vídeo.')
        boxes.append(b)
    return Anchors(times, boxes)


def rectangle_mask(shape, box, inset=0.):
    mask = np.zeros(shape, np.uint8)
    if box is None:
        return mask
    x, y, w, h = box
    x0, y0 = np.maximum(0, np.floor([x + inset * w, y + inset * h])).astype(int)
    x1, y1 = np.minimum(shape[::-1], np.ceil([x + (1 - inset) * w,
                                             y + (1 - inset) * h])).astype(int)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def correspondences(prev, cur, mask, max_points=500):
    points = cv2.goodFeaturesToTrack(prev, max_points, .01, 4, mask=mask, blockSize=5)
    empty = np.empty((0, 2), np.float32)
    if points is None:
        return empty, empty, np.empty(0), 0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, points, None,
                                            winSize=(21, 21), maxLevel=3)
    if nxt is None:
        return empty, empty, np.empty(0), len(points)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None,
                                              winSize=(21, 21), maxLevel=3)
    if back is None:
        return empty, empty, np.empty(0), len(points)
    a, b = points.reshape(-1, 2), nxt.reshape(-1, 2)
    error = np.linalg.norm(a - back.reshape(-1, 2), axis=1)
    h, w = prev.shape
    good = ((status.ravel() == 1) & (reverse.ravel() == 1) & (error <= 1.) &
            np.isfinite(b).all(axis=1) & (b[:, 0] >= 0) & (b[:, 0] < w) &
            (b[:, 1] >= 0) & (b[:, 1] < h))
    return a[good], b[good], error[good], len(points)


def transform_points(matrix, points):
    return points @ matrix[:, :2].T + matrix[:, 2]


class GuidedTracker:
    def __init__(self):
        self.box = None
        self.mask = None

    def update(self, image, prev, guide):
        old_box, old_mask = self.box, self.mask
        self.box = None
        self.mask = None
        if guide is None:
            return None, {'reason': 'guide_missing'}
        self.box = guide.copy()
        self.mask = rectangle_mask(image.shape[:2], guide, .2)
        if prev is None or old_box is None or old_mask is None:
            return self.box, {'reason': 'first_frame'}
        cur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.erode(old_mask, np.ones((5, 5), np.uint8))
        a, b, error, total = correspondences(prev, cur, mask, 250)
        if len(a):
            xy = np.rint(b).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, cur.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, cur.shape[0] - 1)
            inside = self.mask[xy[:, 1], xy[:, 0]] > 0
            a, b, error = a[inside], b[inside], error[inside]
        if len(a) < 6:
            return self.box, {'reason': 'rabbit_features_unreliable'}
        return self.box, {'reason': 'visual_flow', 'a': a, 'b': b,
                          'fb_error': float(np.quantile(error, .9)),
                          'quality': float(len(a) / max(total, 1)),
                          'old_box': old_box, 'rabbit_points': len(a)}


def export_guides(anchors, fps, count, path):
    rows = []
    for frame in range(count):
        t = frame / fps
        exact, box = anchors.at_frame(frame, fps)
        if not exact:
            box = anchors.interpolate(t)
        values = [None] * 4 if box is None else list(box)
        rows.append([t, 'unknown' if box is None else 'manual' if exact else 'interpolated', *values])
    pd.DataFrame(rows, columns=['seconds', 'status', 'x', 'y', 'width', 'height']).to_csv(path, index=False)