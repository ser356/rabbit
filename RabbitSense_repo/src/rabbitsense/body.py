import cv2
import numpy as np

from .tracking import rectangle_mask, transform_points


class BodyMotion:
    def reset(self, gray, box):
        self.previous = gray
        self.box = box
        self.invalid = box is None
        self.paths = []
        self.errors = []
        self.points = np.empty((0, 2), np.float32)
        self.foreground = np.empty(0, bool)
        if box is None:
            return
        center = box[:2] + box[2:] / 2
        yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
        background = 255 - rectangle_mask(gray.shape, box, -.2)
        background[(xx - center[0]) ** 2 + (yy - center[1]) ** 2 > (1.5 * max(box[2:])) ** 2] = 0
        for foreground, mask, count in [(True, rectangle_mask(gray.shape, box, .2), 250),
                                         (False, background, 500)]:
            found = cv2.goodFeaturesToTrack(gray, count, .01, 4, mask=mask, blockSize=5)
            if found is not None:
                self.points = np.concatenate([self.points, found.reshape(-1, 2)])
                self.foreground = np.r_[self.foreground, np.full(len(found), foreground)]
        self.paths = [self.points.copy()]

    def update(self, gray, box):
        if self.invalid:
            return
        if box is None:
            self.invalid = True
            return
        ids = np.flatnonzero(np.isfinite(self.points).all(axis=1))
        current = np.full_like(self.points, np.nan)
        errors = np.full(len(self.points), np.nan)
        if len(ids):
            old = self.points[ids].reshape(-1, 1, 2)
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, old, None,
                                                    winSize=(21, 21), maxLevel=3)
            if nxt is not None:
                back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, nxt, None,
                                                          winSize=(21, 21), maxLevel=3)
                if back is not None:
                    b = nxt.reshape(-1, 2)
                    fb = np.linalg.norm(old.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
                    x, y, w, h = box
                    inside = ((b[:, 0] >= x + .1 * w) & (b[:, 0] <= x + .9 * w) &
                              (b[:, 1] >= y + .1 * h) & (b[:, 1] <= y + .9 * h))
                    outside = ((b[:, 0] < x - .2 * w) | (b[:, 0] > x + 1.2 * w) |
                               (b[:, 1] < y - .2 * h) | (b[:, 1] > y + 1.2 * h))
                    good = ((status.ravel() == 1) & (reverse.ravel() == 1) & (fb <= 1.) &
                            np.isfinite(b).all(axis=1) & (b[:, 0] >= 0) & (b[:, 0] < gray.shape[1]) &
                            (b[:, 1] >= 0) & (b[:, 1] < gray.shape[0]) &
                            np.where(self.foreground[ids], inside, outside))
                    current[ids[good]] = b[good]
                    errors[ids[good]] = fb[good]
        self.points = current
        self.paths.append(current.copy())
        self.errors.append(errors)
        self.previous = gray

    def measure(self, duration, threshold):
        row = dict(movement='unknown', reason='persistent_points_unavailable', interval_seconds=duration,
                   evidence_quality=0., valid_fraction=0., rabbit_points=0, background_points=0,
                   speed_bodywidths_s=None, speed_lower=None, speed_upper=None,
                   body_rotation_degrees=None, body_scale_ratio=None, translation_span_bodywidths=None,
                   camera_error_px=None, body_fit_error_px=None)
        if self.invalid or len(self.paths) < 2 or duration <= 0:
            return row
        alive = np.isfinite(self.points).all(axis=1)
        fg, bg = alive & self.foreground, alive & ~self.foreground
        row.update(rabbit_points=int(fg.sum()), background_points=int(bg.sum()))
        if fg.sum() < 12 or bg.sum() < 24:
            return row
        paths = np.stack(self.paths)
        a, background = paths[0, fg], paths[0, bg]
        center = self.box[:2] + self.box[2:] / 2
        sectors = (background[:, 0] > center[0]).astype(int) + 2 * (background[:, 1] > center[1])
        counts = np.bincount(sectors)
        if len(np.unique(sectors)) < 3 or counts[counts > 0].min() < 4 or cv2.pointPolygonTest(cv2.convexHull(background), tuple(center), False) < 0:
            row['reason'] = 'background_not_surrounding_body'
            return row
        if np.any(np.ptp(a, axis=0) < .25 * self.box[2:]):
            row['reason'] = 'body_support_too_local'
            return row
        centers, angles, scales, uncertainty, camera_errors, body_errors, quality = [center], [0.], [1.], [0.], [], [], []
        for index in range(1, len(paths)):
            target = paths[index, bg]
            held_errors = []
            for sector in np.unique(sectors):
                held = sectors == sector
                matrix, _ = cv2.estimateAffine2D(background[~held], target[~held],
                                                 method=cv2.RANSAC, ransacReprojThreshold=2.)
                if matrix is None:
                    row['reason'] = 'camera_spatial_validation_failed'
                    return row
                held_errors.append(np.quantile(np.linalg.norm(target[held] - transform_points(matrix, background[held]), axis=1), .75))
            camera_error = float(max(held_errors))
            camera, inliers = cv2.estimateAffine2D(background, target, method=cv2.RANSAC, ransacReprojThreshold=2.)
            if camera is None or inliers is None or inliers.mean() < .7 or camera_error > 3.:
                row['reason'] = 'camera_spatial_validation_failed'
                return row
            singular = np.linalg.svd(camera[:, :2], compute_uv=False)
            if singular.min() < .5 or singular.max() > 2. or np.linalg.det(camera[:, :2]) <= 0:
                row['reason'] = 'camera_transform_invalid'
                return row
            inverse = cv2.invertAffineTransform(camera)
            stabilized = transform_points(inverse, paths[index, fg]).astype(np.float32)
            body, support = cv2.estimateAffinePartial2D(a, stabilized, method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.)
            if body is None or support is None or support.sum() < 12 or support.mean() < .7:
                row['reason'] = 'body_not_coherent'
                return row
            support = support.ravel().astype(bool)
            if np.any(np.ptp(a[support], axis=0) < .25 * self.box[2:]):
                row['reason'] = 'body_support_too_local'
                return row
            body_error = float(np.quantile(np.linalg.norm(stabilized - transform_points(body, a), axis=1), .75))
            scale = float(np.linalg.norm(body[:, 0]))
            if not .5 <= scale <= 2.:
                row['reason'] = 'body_transform_invalid'
                return row
            fb = float(np.nanmax(self.errors[index - 1][alive]))
            centers.append(transform_points(body, center[None])[0])
            angles.append(float(np.arctan2(body[1, 0], body[0, 0])))
            scales.append(scale)
            uncertainty.append((camera_error + fb) / singular.min() + body_error)
            camera_errors.append(camera_error)
            body_errors.append(body_error)
            quality.append(float(min(inliers.mean(), support.mean())))
        centers = np.array(centers)
        distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        errors = np.array(uncertainty)
        errors = errors[:, None] + errors[None, :]
        span = float(distances.max() / self.box[2])
        lower = float(np.maximum(0., distances - errors).max() / self.box[2] / duration)
        upper = float((distances + errors).max() / self.box[2] / duration)
        row.update(speed_bodywidths_s=span / duration, speed_lower=lower, speed_upper=upper,
                   translation_span_bodywidths=span, body_rotation_degrees=float(np.degrees(np.ptp(np.unwrap(angles)))),
                   body_scale_ratio=float(max(scales) / min(scales)),
                   camera_error_px=float(max(camera_errors)), body_fit_error_px=float(max(body_errors)),
                   valid_fraction=1., evidence_quality=float(alive.sum() / len(alive) * min(quality)))
        if lower >= threshold:
            row.update(movement='locomotion_candidate', reason='persistent_body_translation')
        elif upper < threshold:
            row.update(movement='low_motion_candidate', reason='body_center_stable')
        else:
            row['reason'] = 'body_translation_near_noise'
        return row


def analyze_body(video, anchors, step, threshold, overlay_path=None):
    if not np.isfinite(step) or step <= 0 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('step y threshold deben ser positivos y finitos.')
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    writer = None
    rows, images = [], []
    tracker = BodyMotion()
    last = 0
    try:
        if not cap.isOpened():
            raise ValueError(f'No se puede abrir {video}')
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('Metadatos de vídeo inválidos.')
        stride = max(1, round(fps * step))
        if overlay_path is not None:
            writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (int(cap.get(3)), int(cap.get(4))))
            if not writer.isOpened():
                raise RuntimeError('No se puede crear el vídeo corporal.')
        for frame in range(count):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Error decodificando fotograma {frame}.')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            exact, guide = anchors.at_frame(frame, fps)
            if not exact:
                guide = anchors.interpolate(frame / fps)
            if frame == 0:
                tracker.reset(gray, guide)
            else:
                tracker.update(gray, guide)
            if writer is not None:
                images.append(image)
            if frame % stride == 0 or frame == count - 1:
                row = tracker.measure((frame - last) / fps, threshold)
                row['seconds'] = frame / fps
                rows.append(row)
                for index, image in enumerate(images):
                    if not tracker.invalid and len(tracker.paths) >= len(images):
                        points = tracker.paths[len(tracker.paths) - len(images) + index]
                        alive = np.isfinite(tracker.points).all(axis=1)
                        for point, foreground in zip(points[alive], tracker.foreground[alive]):
                            cv2.circle(image, tuple(np.rint(point).astype(int)), 2,
                                       (40, 230, 60) if foreground else (255, 180, 20), -1)
                    cv2.rectangle(image, (0, 0), (image.shape[1], 83), (15, 15, 15), -1)
                    for line, text in enumerate([row['movement'], row['reason'],
                                                  f'body={row["rabbit_points"]} bg={row["background_points"]} guide-assisted']):
                        cv2.putText(image, text, (8, 22 + line * 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    .43, (240, 240, 240), 1, cv2.LINE_AA)
                    writer.write(image)
                images = []
                tracker.reset(gray, guide)
                last = frame
            if frame % 100 == 0 or frame == count - 1:
                print(f'BODY {frame + 1}/{count}', flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return rows