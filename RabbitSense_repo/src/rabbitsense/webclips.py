"""Clasificador de locomoción entrenado con vídeos de conejos de YouTube y evaluado en Cinna.

1. `sheets`: corta cada vídeo descargado en clips de 4 s (hasta 8 por vídeo, repartidos) y genera hojas de contacto
   de 8 fotogramas por clip para etiquetarlas visualmente.
2. `embed`: características X-CLIP (congelado) de ventanas de 8 fotogramas, imagen completa con bandas para que no
   se recorte nada. YouTube: solo ventanas dentro de clips etiquetados. Cinna: rejilla de 0,5 s completa.
3. `train`: regresión logística L2 sobre las características (numpy), validada agrupando por vídeo de YouTube;
   modelo congelado en benchmarks/webclips_modelo.json antes de tocar a Cinna.
4. `test`: una ejecución sobre los vídeos de Cinna con etiquetas verificadas por una persona; mismos criterios.

Las etiquetas de YouTube salen de inspección visual de fotogramas: ruidosas, solo para entrenar.
"""
import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .motion import sha256

CLIP_SECONDS, CLIPS_PER_VIDEO, SHEET_FRAMES, THUMB = 4., 8, 8, 200
MODEL_ID, FRAMES, SPACING, LEAD, SIZE = 'microsoft/xclip-base-patch32', 8, .25, 1.25, 224
BAND, L2, STEP = (.4, .6), 1., .5  # p >= 0,6 locomoción; p <= 0,4 bajo movimiento; L2 = 1 sobre características estandarizadas
VIDEO_SUFFIXES = ('.mp4', '.mov', '.webm', '.mkv')
MODEL_FILE = Path(__file__).resolve().parents[2] / 'benchmarks' / 'webclips_modelo.json'


def videos_in(folder):
    return sorted(p for p in Path(folder).expanduser().iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def metadata(video):
    capture = cv2.VideoCapture(str(video))
    fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not np.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError(f'Metadatos inválidos: {video}')
    return fps, count


def chosen_clips(duration, limit=CLIPS_PER_VIDEO):
    starts = np.arange(0., duration - CLIP_SECONDS + 1e-6, CLIP_SECONDS)
    if len(starts) > limit:
        starts = starts[np.unique(np.linspace(0, len(starts) - 1, limit).round().astype(int))]
    return [float(s) for s in starts]


def read_at(capture, fps, seconds):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(seconds * fps)))
    ok, image = capture.read()
    return image if ok else None


def sheets(folder, out):
    """Una hoja por vídeo: una fila por clip con 8 fotogramas a 0, 0,5 … 3,5 s del inicio del clip."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for video in videos_in(folder):
        fps, count = metadata(video)
        starts = chosen_clips(count / fps)
        if not starts:
            continue
        capture = cv2.VideoCapture(str(video))
        strips = []
        for start in starts:
            tiles = []
            for k in range(SHEET_FRAMES):
                image = read_at(capture, fps, start + k * CLIP_SECONDS / SHEET_FRAMES)
                if image is None:
                    image = np.zeros((THUMB, THUMB, 3), np.uint8)
                h, w = image.shape[:2]
                tile = cv2.resize(image, (THUMB, max(1, round(h * THUMB / w))) if w >= h else
                                  (max(1, round(w * THUMB / h)), THUMB), interpolation=cv2.INTER_AREA)
                tile = cv2.copyMakeBorder(tile, 0, THUMB - tile.shape[0], 0, THUMB - tile.shape[1],
                                          cv2.BORDER_CONSTANT, value=(30, 30, 30))
                cv2.putText(tile, f'{k * CLIP_SECONDS / SHEET_FRAMES:.1f}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            label = np.full((THUMB, 70, 3), 15, np.uint8)
            cv2.putText(label, f'{start:.0f}s', (6, THUMB // 2), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            strips.append(np.hstack([label] + tiles))
            rows.append(dict(video=video.stem, start_s=start, end_s=start + CLIP_SECONDS))
        capture.release()
        cv2.imwrite(str(out / f'{video.stem}.jpg'), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 80])
    pd.DataFrame(rows).to_csv(out / 'clips.csv', index=False)
    return len(rows)


def letterbox(image, size=SIZE):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    top, left = (size - image.shape[0]) // 2, (size - image.shape[1]) // 2
    image = cv2.copyMakeBorder(image, top, size - image.shape[0] - top, left, size - image.shape[1] - left,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def window_ends(count, fps, step=STEP):
    """Instantes de la rejilla de 0,5 s (mismos que flow/silhouette)."""
    stride = max(1, round(fps * step))
    return [f / fps for f in range(count) if f % stride == 0 or f == count - 1]


def sample_frames(video, needed):
    """Fotogramas con bandas en los instantes k·0,25 s pedidos (`needed`: conjunto de k)."""
    fps, count = metadata(video)
    wanted = {}
    for k in sorted(needed):
        frame = int(round(k * SPACING * fps))
        if frame < count:
            wanted.setdefault(frame, []).append(k)
    frames, capture = {}, cv2.VideoCapture(str(video))
    try:
        for frame in range(max(wanted, default=-1) + 1):
            ok, image = capture.read()
            if not ok:
                break
            for k in wanted.get(frame, []):
                frames[k] = letterbox(image)
    finally:
        capture.release()
    return frames


def window_keys(t):
    first = int(round((t - LEAD) / SPACING))
    return list(range(first, first + FRAMES)) if first >= 0 else None


class XClipEmbedder:
    def __init__(self, device='cpu', model_id=MODEL_ID):
        import torch
        import transformers
        from transformers import XCLIPModel, XCLIPProcessor
        self.torch, self.device = torch, device
        self.processor = XCLIPProcessor.from_pretrained(model_id)
        self.model = XCLIPModel.from_pretrained(model_id).to(device).eval()
        self.info = dict(model_id=model_id, revision=getattr(self.model.config, '_commit_hash', None),
                         torch=torch.__version__, transformers=transformers.__version__, device=device)

    def __call__(self, clips):
        pixels = self.processor(videos=[list(c) for c in clips], return_tensors='pt')['pixel_values'].to(self.device)
        with self.torch.no_grad():
            output = self.model.get_video_features(pixel_values=pixels)
        output = output if isinstance(output, self.torch.Tensor) else output.pooler_output
        return output.float().cpu().numpy()


def embed_video(video, times, embedder, batch=16):
    """Características para las ventanas que terminan en `times`; ventanas fuera del vídeo se omiten."""
    keys = {t: window_keys(t) for t in times}
    frames = sample_frames(video, {k for ks in keys.values() if ks for k in ks})
    usable = [t for t, ks in keys.items() if ks and all(k in frames for k in ks)]
    features = []
    for start in range(0, len(usable), batch):
        chunk = usable[start:start + batch]
        features.append(embedder([np.stack([frames[k] for k in keys[t]]) for t in chunk]))
    return np.array(usable), (np.concatenate(features) if features else np.empty((0, 0)))


def embed(manifest_or_folder, labels, out, embedder):
    """YouTube (`labels` con start_s/end_s): ventanas contenidas en clips etiquetados. Cinna (manifest): rejilla."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if labels is not None:
        table = pd.read_csv(labels)
        table = table[table.label.isin(['locomotion', 'resting'])]
        for video in videos_in(manifest_or_folder):
            clips = table[table.video == video.stem]
            if clips.empty:
                continue
            fps, count = metadata(video)
            times = [t for t in window_ends(count, fps)
                     if any((t - LEAD >= c.start_s - 1e-6) and (t - LEAD + (FRAMES - 1) * SPACING <= c.end_s + 1e-6)
                            for c in clips.itertuples())]
            ends, features = embed_video(video, times, embedder)
            np.savez(out / f'{video.stem}.npz', seconds=ends, features=features)
            print('CARACTERÍSTICAS', video.stem, len(ends), flush=True)
    else:
        manifest = Path(manifest_or_folder).resolve()
        for case in json.loads(manifest.read_text())['cases']:
            video = (manifest.parent / Path(case['video']).expanduser()).resolve()
            fps, count = metadata(video)
            ends, features = embed_video(video, window_ends(count, fps), embedder)
            np.savez(out / f'{video.stem}.npz', seconds=ends, features=features)
            print('CARACTERÍSTICAS', case['name'], len(ends), flush=True)
    (out / 'embedder.json').write_text(json.dumps(getattr(embedder, 'info', {}), indent=2))


def youtube_windows(labels, feature_dir):
    """Ventanas de YouTube con la etiqueta del clip que las contiene y el vídeo como grupo."""
    table = pd.read_csv(labels)
    table = table[table.label.isin(['locomotion', 'resting'])]
    xs, ys, groups = [], [], []
    for name, clips in table.groupby('video'):
        path = Path(feature_dir) / f'{name}.npz'
        if not path.is_file():
            continue
        data = np.load(path)
        for t, x in zip(data['seconds'], data['features']):
            start = t - LEAD
            for clip in clips.itertuples():
                if start >= clip.start_s - 1e-6 and start + (FRAMES - 1) * SPACING <= clip.end_s + 1e-6:
                    xs.append(x)
                    ys.append(1. if clip.label == 'locomotion' else 0.)
                    groups.append(name)
                    break
    return np.array(xs), np.array(ys), np.array(groups)


def fit(x, y, l2=L2, iterations=50):
    """Regresión logística L2 por Newton, clases equilibradas, sobre características estandarizadas."""
    mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = np.c_[np.ones(len(x)), (x - mean) / scale]
    weights = np.where(y == 1, .5 / max(y.mean(), 1e-9), .5 / max(1 - y.mean(), 1e-9))
    w = np.zeros(z.shape[1])
    penalty = np.r_[0., np.full(z.shape[1] - 1, l2)]
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        gradient = z.T @ (weights * (p - y)) + penalty * w
        hessian = (z * (weights * p * (1 - p))[:, None]).T @ z + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=w.tolist())


def predict(model, x):
    z = (np.asarray(x) - np.array(model['mean'])) / np.array(model['scale'])
    w = np.array(model['weights'])
    return 1 / (1 + np.exp(-np.clip(w[0] + z @ w[1:], -30, 30)))


def auc(p, y):
    positive, negative = p[y == 1], p[y == 0]
    if not len(positive) or not len(negative):
        return None
    return float((positive[:, None] > negative[None]).mean() + .5 * (positive[:, None] == negative[None]).mean())


def train(labels, feature_dir, output=MODEL_FILE, folds=5):
    x, y, groups = youtube_windows(labels, feature_dir)
    if len(set(y)) < 2:
        raise ValueError('Hacen falta ventanas de las dos clases.')
    names = np.array(sorted(set(groups)))
    order = np.random.default_rng(0).permutation(len(names))
    fold_of = {names[i]: j % folds for j, i in enumerate(order)}
    held = np.zeros(len(y))
    for fold in range(folds):
        test = np.array([fold_of[g] == fold for g in groups])
        if test.any() and len(set(y[~test])) == 2:
            held[test] = predict(fit(x[~test], y[~test]), x[test])
    model = fit(x, y)
    decided = (held >= BAND[1]) | (held <= BAND[0])
    report = dict(model=model, band=BAND, l2=L2, windows=int(len(y)), locomotion_windows=int(y.sum()),
                  videos=int(len(names)),
                  grouped_cross_validation=dict(auc=auc(held, y), folds=folds,
                                                decided_fraction=float(decided.mean()),
                                                accuracy_on_decided=float(((held[decided] >= .5) == y[decided]).mean())
                                                if decided.any() else None),
                  labels_sha256=sha256(labels), protocol='Pesos congelados antes de evaluar a Cinna; CV agrupada '
                  'por vídeo de YouTube; etiquetas de YouTube por inspección visual de fotogramas (ruidosas).')
    Path(output).write_text(json.dumps(report) + '\n')
    return report


def classify(seconds_all, ends, features, model):
    p = dict(zip(np.round(ends, 3), predict(model, features))) if len(ends) else {}
    rows = []
    for index, t in enumerate(seconds_all):
        row = dict(seconds=t, interval_seconds=t - seconds_all[index - 1] if index else 0., movement='unknown',
                   p_locomotion=None, evidence_quality=0.)
        value = p.get(round(t, 3))
        if index and value is not None:
            row.update(p_locomotion=float(value), evidence_quality=abs(2 * float(value) - 1),
                       movement='locomotion_candidate' if value >= BAND[1] else
                       'low_motion_candidate' if value <= BAND[0] else 'unknown')
        rows.append(row)
    return rows


def test(manifest, feature_dir, out, model_file=MODEL_FILE):
    from .closure import evaluate_cases
    if not Path(model_file).is_file():
        raise SystemExit('NO SE EJECUTA: el modelo no está congelado.')
    frozen = json.loads(Path(model_file).read_text())

    def analyze(paths):
        fps, count = metadata(paths['video'])
        data = np.load(Path(feature_dir) / f"{paths['video'].stem}.npz")
        return classify(window_ends(count, fps), data['seconds'], data['features'], frozen['model'])

    info = dict(model_file_sha256=sha256(model_file), band=BAND, frames=FRAMES, spacing=SPACING, lead=LEAD,
                model_id=MODEL_ID, python=platform.python_version())
    result = evaluate_cases(manifest, out, 'webclips', analyze, info)
    r = result['methods']['webclips']
    result.update(original_closure_status=result['status'], test_set='cinna_all_verified_videos_unseen_in_training',
                  status='resolved_independent_test' if r['passed'] else 'not_resolved')
    (Path(out) / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False))
    print(f"\nCinna (test independiente): locomoción {r['locomotion_hits']}/{r['locomotion_clips']}, reposo "
          f"{r['resting_correct']}/{r['resting_clips']}, episodios falsos {r['resting_false_events']} → "
          f"{'PASA' if r['passed'] else 'NO PASA'}\nEstado: {result['status']}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Locomoción aprendida con vídeos de YouTube; test en Cinna')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('sheets')
    command.add_argument('--videos', required=True)
    command.add_argument('--out', required=True)
    command = sub.add_parser('embed')
    command.add_argument('--videos', help='Carpeta de YouTube (con --labels)')
    command.add_argument('--manifest', help='Manifest de Cinna')
    command.add_argument('--labels')
    command.add_argument('--out', required=True)
    command.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    command = sub.add_parser('train')
    for name in ('labels', 'features'):
        command.add_argument(f'--{name}', required=True)
    command.add_argument('--output', default=str(MODEL_FILE))
    command = sub.add_parser('test')
    for name in ('manifest', 'features', 'out'):
        command.add_argument(f'--{name}', required=True)
    args = parser.parse_args()
    if args.command == 'sheets':
        print('Clips:', sheets(args.videos, args.out))
    elif args.command == 'embed':
        if bool(args.videos) == bool(args.manifest) or (args.videos and not args.labels):
            parser.error('Use --videos con --labels (YouTube) o --manifest (Cinna).')
        embed(args.videos or args.manifest, args.labels if args.videos else None, args.out, XClipEmbedder(args.device))
    elif args.command == 'train':
        report = train(args.labels, args.features, args.output)
        print(json.dumps({k: report[k] for k in ('windows', 'locomotion_windows', 'videos', 'grouped_cross_validation')},
                         indent=2))
    else:
        test(args.manifest, args.features, args.out)


if __name__ == '__main__':
    main()
