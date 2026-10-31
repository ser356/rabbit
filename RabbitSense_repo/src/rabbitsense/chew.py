"""Detector de masticación por ritmo, sin entrenar (fijado el 2026-10-31, antes de calcular nada).

Comer falló con el modelo de actos (3/31 y 1/34): el recorte del cuerpo entero diluye la boca. Un conejo mastica a un
ritmo regular, de unas 2–4 veces por segundo, y ese ritmo se puede medir sin aprender nada:

1. Recorte cuadrado fijo de la ventana de 2 s, con la caja del detector (el mismo de `auto`), a 256 px, en gris.
2. Media por bloques de 8×8 px; a cada bloque se le quita la tendencia lineal y se le aplica una ventana de Hann.
3. Espectro de cada bloque. Proporción = potencia en 1,5–4,5 Hz / potencia en 0,5–12 Hz (o hasta la mitad de la
   frecuencia de fotogramas).
4. Puntuación de la ventana = esa proporción sumando la potencia del 5 % de bloques que más se mueven: la boca es una
   zona pequeña y, con todos los bloques, manda el ruido (comprobado en sintético antes de tocar datos reales).
5. Si el conejo se desplaza (desplazamiento neto ≥ 0,5 tamaños en la ventana), la puntuación es 0: los saltos también
   tienen ritmo. Sin medida de desplazamiento, la puntuación se usa tal cual.

Evaluación (`evaluate`):
- Desarrollo: los 17 vídeos anteriores con las etiquetas de actos de Claude. El umbral es el percentil 90 de las
  ventanas que no son comer (10 % de falsas alarmas en desarrollo). Se informa la AUC comer frente al resto.
- Prueba: los dos vídeos nuevos de Cinna comiendo (34 ventanas). Se adopta si la AUC de desarrollo es ≥ 0,7 y
  reconoce al menos la mitad de las ventanas de comer de la prueba. Una sola evaluación; no se reajusta.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .motion import sha256

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'runs' / 'masticacion'
CHEW = dict(band=(1.5, 4.5), total=(.5, 12.), block=8, size=256, min_frames=16, top=.05, still=.5,
            false_alarms=.1, min_auc=.7, min_recall=.5)


def chew_score(crops, times):
    """crops: (T, S, S, 3) BGR o (T, S, S) gris; times: segundos reales de cada fotograma → (puntuación, fps)."""
    import cv2
    crops = np.asarray(crops)
    if len(crops) < CHEW['min_frames']:
        return np.nan, np.nan
    gray = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c for c in crops]).astype(np.float32)
    t = np.asarray(times, float)
    fps = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
    b = CHEW['block']
    s = gray.shape[1] // b * b
    blocks = gray[:, :s, :s].reshape(len(gray), s // b, b, s // b, b).mean(axis=(2, 4)).reshape(len(gray), -1)
    x = np.arange(len(blocks), dtype=float)
    fit = np.polyfit(x, blocks, 1)
    signal = (blocks - (np.outer(x, fit[0]) + fit[1])) * np.hanning(len(blocks))[:, None]
    power = np.abs(np.fft.rfft(signal, axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(blocks), 1 / fps)
    top = min(CHEW['total'][1], fps / 2)
    total = (freqs >= CHEW['total'][0]) & (freqs <= top)
    band = (freqs >= CHEW['band'][0]) & (freqs <= min(CHEW['band'][1], top))
    energy = power[total].sum(axis=0)
    if not band.any() or not (energy > 0).any():
        return np.nan, float(fps)
    moving = np.argsort(energy)[-max(1, int(round(CHEW['top'] * len(energy)))):]
    return float(power[band][:, moving].sum() / max(energy[moving].sum(), 1e-12)), float(fps)


def video_scores(video, detector, starts=None):
    """Puntuación de masticación de cada ventana de 2 s (todas, o las que empiezan en `starts`)."""
    from .acts import square
    from .auto import AUTO, square_around, stream_windows
    from .pointtrack import frame_times
    from .ytacts import probe_frames, window_detection
    times = frame_times(video)
    wanted = None if starts is None else {round(float(s), 3) for s in starts}
    rows = []
    for k, frames, images in stream_windows(video, times):
        start = round(k * AUTO['window'], 3)
        if (wanted is not None and start not in wanted) or len(frames) < 2:
            continue
        summary = window_detection(detector([images[f] for f in probe_frames(frames)]))
        if not summary['union']:
            rows.append(dict(start=start, found=0, score=np.nan, fps=np.nan))
            continue
        box = square_around(*summary['union'], images[frames[0]].shape)
        score, fps = chew_score([square(images[f], box, CHEW['size']) for f in frames], times[frames])
        rows.append(dict(start=start, found=summary['found'], score=score, fps=fps))
    return pd.DataFrame(rows)


def run(manifest, windows, videos, out, detector):
    """Puntuaciones de desarrollo (ventanas etiquetadas de los 17 vídeos) y de prueba (vídeos nuevos, todas)."""
    from .actmodel import cases_with_windows
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(windows, dtype={'case': str})
    parts = []
    for name, video, _, mine in cases_with_windows(manifest, windows):
        parts.append(video_scores(video, detector, mine.start).assign(case=name))
        print(f'{name}: {len(parts[-1])} ventanas', flush=True)
    pd.concat(parts, ignore_index=True).to_csv(out / 'desarrollo.csv', index=False)
    parts = []
    for video in [Path(v).resolve() for v in videos]:
        parts.append(video_scores(video, detector).assign(video=video.stem))
        print(f'{video.name}: {len(parts[-1])} ventanas', flush=True)
    pd.concat(parts, ignore_index=True).to_csv(out / 'prueba.csv', index=False)


def effective(scores, displacement):
    """Sin masticación si el conejo se desplaza; sin medida de desplazamiento, la puntuación tal cual."""
    moving = np.asarray(displacement, float) >= CHEW['still']
    return np.where(moving, 0., np.asarray(scores, float))


def evaluate(scores_dir, dev_labels, dev_features, test_labels, analysis, out):
    from .actmodel import GROUPS
    from .auto import wilson
    from .pointtrack import auc
    scores_dir, out = Path(scores_dir), Path(out)
    dev = pd.read_csv(scores_dir / 'desarrollo.csv', dtype={'case': str})
    labels = pd.read_csv(dev_labels, dtype={'case': str})
    features = pd.read_csv(dev_features, dtype={'case': str})[['case', 'start', 'net_displacement']]
    for frame in (dev, labels, features):
        frame['start'] = frame.start.round(3)
    dev = dev.merge(labels[['case', 'start', 'act']], on=['case', 'start']).merge(features, on=['case', 'start'], how='left')
    dev = dev[dev.act.isin(GROUPS) & dev.score.notna()].assign(eating=lambda d: d.act == 'eating')
    dev['effective'] = effective(dev.score, dev.net_displacement)
    threshold = float(np.quantile(dev.effective[~dev.eating], 1 - CHEW['false_alarms']))
    dev_auc = auc(dev.effective, dev.eating)
    test = pd.read_csv(scores_dir / 'prueba.csv', dtype={'video': str})
    truth = pd.read_csv(test_labels, dtype={'video': str})
    parts = []
    for video, group in test.groupby('video'):
        blocks = pd.read_csv(Path(analysis) / video / 'ventanas.csv')[['start', 'displacement']]
        parts.append(group.merge(blocks, on='start', how='left'))
    test = pd.concat(parts, ignore_index=True)
    for frame in (test, truth):
        frame['start'] = frame.start.round(3)
    test = test.merge(truth[['video', 'start', 'act']], on=['video', 'start'])
    test['effective'] = effective(test.score, test.displacement)
    eating = test[test.act == 'eating']
    hits = int((eating.effective >= threshold).sum())
    recall = hits / len(eating) if len(eating) else None
    result = dict(
        criterion=CHEW, threshold=threshold, development=dict(
            windows=int(len(dev)), eating=int(dev.eating.sum()), auc=dev_auc,
            recall_at_threshold=float((dev.effective[dev.eating] >= threshold).mean()),
            false_alarms_by_act={a: float((g.effective >= threshold).mean()) for a, g in dev[~dev.eating].groupby('act')},
            median_score={a: float(g.effective.median()) for a, g in dev.groupby('act')}),
        test=dict(windows=int(len(eating)), hits=hits, recall=recall, interval=wilson(hits, len(eating)),
                  median_score=float(eating.effective.median()) if len(eating) else None),
        adopted=bool(dev_auc is not None and dev_auc >= CHEW['min_auc'] and recall is not None and
                     recall >= CHEW['min_recall']),
        source_sha256=sha256(Path(__file__)),
        scope='Etiquetas de Claude; prueba con dos vídeos de una sola sesión, todos de comer (solo mide sensibilidad).')
    out.mkdir(parents=True, exist_ok=True)
    dev.to_csv(out / 'desarrollo.csv', index=False)
    test.to_csv(out / 'prueba.csv', index=False)
    (out / 'evaluacion.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    print(f"Desarrollo: AUC {dev_auc} | umbral {threshold:.3f} | prueba: {hits}/{len(eating)} ventanas de comer → "
          f"{'se adopta' if result['adopted'] else 'no se adopta'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Detector de masticación por ritmo (sin entrenar)')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('run', help='Puntuaciones de desarrollo y de prueba (Mac)')
    command.add_argument('--manifest', required=True)
    command.add_argument('--windows', default=str(REPO / 'cierre' / 'actos' / 'ventanas_indice.csv'))
    command.add_argument('--videos', nargs='+', required=True)
    command.add_argument('--out', default=str(OUT))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command = sub.add_parser('evaluate')
    command.add_argument('--scores', default=str(OUT))
    command.add_argument('--dev-labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_ventanas_claude_v2.csv'))
    command.add_argument('--dev-features', default=str(REPO / 'runs' / 'actos_modelo' / 'caracteristicas.csv'))
    command.add_argument('--test-labels', default=str(REPO.parent / 'RabbitSense_ARCHIVO' / 'actos_comiendo_claude.csv'))
    command.add_argument('--analysis', default=str(REPO / 'runs' / 'analisis'))
    command.add_argument('--out', default=str(REPO / 'cierre' / 'masticacion'))
    args = parser.parse_args()
    if args.command == 'run':
        from .ytacts import Detector
        run(args.manifest, args.windows, args.videos, args.out, Detector(args.device))
    else:
        evaluate(args.scores, args.dev_labels, args.dev_features, args.test_labels, args.analysis, args.out)


if __name__ == '__main__':
    main()
