"""Actividad continua de Cinna: desplazamiento respecto al suelo medido con CoTracker3 (2026-10-28).

En lugar de clasificar clips en locomoción/reposo (no validado: `not_resolved`), se informa de una medida continua:
- por intervalo de 0,5 s: desplazamiento del conejo respecto al suelo, en tamaños corporales (√área de la máscara SAM2),
  con límites inferior y superior según el error de reproyección del fondo;
- acumulado: desplazamiento total estimado y su límite inferior conservador, y desplazamiento por minuto medido;
- tiempo sin medida (sin máscara, puntos insuficientes o cámara no compensable), que nunca cuenta como reposo;
- como referencia, el estado de la regla física congelada de `pointtrack` (diámetro en ±1 s ≥ 0,5 / ≤ 0,25).

Validación disponible: frente a etiquetas ciegas con definición física, el diámetro máximo por clip ordena locomoción y
reposo con AUC 0,91 (tiempo real por fotograma). Las unidades no son métricas ni están calibradas; la suma de pasos
incluye ruido de seguimiento, por eso se da también el límite inferior.
"""
import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import pointtrack
from .motion import sha256

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'runs' / 'actividad'
PLOT_CAP = 2.  # tamaños por intervalo representados como barra completa
NET_STEP = 2.  # desplazamiento neto por bloques de 2 s: el temblor de una Cinna quieto no se acumula paso a paso
# Criterio fijado antes de ver resultados (2026-10-28): los bloques de 2 s pasan a ser la medida principal si la
# mediana de la tasa en los clips de reposo grabados en mano (IMG_519x, WhatsApp) baja al menos un 30 % y la AUC por
# clip de la tasa frente a las etiquetas ciegas no baja más de 0,02 respecto a los pasos de 0,5 s.
ADOPT = dict(handheld_rest_median_drop=.3, auc_tolerance=.02)


def load_chunks(folder, times):
    """Tramos guardados con su tiempo real y su trayectoria estabilizada (None si no hay medida)."""
    chunks = {}
    for path in sorted(Path(folder).glob('chunk_*.npz')):
        chunk = dict(np.load(path))
        frames = np.asarray(chunk['frames'])
        if not len(frames) or frames.max() >= len(times):
            continue
        chunk['times'] = times[frames]
        k = int(path.stem.split('_')[1])
        chunks[k] = (chunk, pointtrack.stabilised(chunk)) if int(chunk['query']) >= 0 else (chunk, None)
    return chunks


def position(chunk, stable, instant):
    """Desplazamiento estabilizado en `instant` y su error, o None.

    Interpola entre los fotogramas válidos que lo rodean (los fotogramas analizados no caen en los límites de 0,5 s);
    si solo hay uno a menos de `tolerance`, usa ese."""
    offsets, errors = stable[0], stable[1]
    times = np.asarray(chunk['times'])
    spacing = float(np.median(np.diff(times))) if len(times) > 1 else 0.
    tolerance = max(.1, .6 * spacing)
    valid = np.flatnonzero(np.isfinite(offsets[:, 0]))
    before, after = valid[times[valid] <= instant + 1e-9], valid[times[valid] >= instant - 1e-9]
    if len(before) and len(after) and times[after[0]] - times[before[-1]] <= 2 * tolerance:
        i, j = before[-1], after[0]
        w = 0. if j == i else (instant - times[i]) / (times[j] - times[i])
        return (1 - w) * offsets[i] + w * offsets[j], float(max(errors[i], errors[j]))
    near = valid[np.abs(times[valid] - instant) <= tolerance]
    if not len(near):
        return None
    j = near[np.argmin(np.abs(times[near] - instant))]
    return offsets[j], float(errors[j])


def block_grid(times, step):
    """Límites de bloques de `step` s en tiempo real hasta el último fotograma (0,5 s: la rejilla de pointtrack)."""
    if step == .5:
        return pointtrack.time_grid(times)
    end = float(times[-1])
    grid = [float(g) for g in np.arange(0., end + 1e-9, step)]
    if end - grid[-1] > 1e-6:
        grid.append(end)
    return grid


def step_rows(times, folder, step=.5):
    """Desplazamiento neto de cada bloque (a, b] de `step` s dentro del tramo cuyo centro está más cerca.

    Con 0,5 s se suman pasos cortos (más resolución, más ruido acumulado); con 2 s, el desplazamiento neto del bloque."""
    starts, grid = pointtrack.chunk_starts(pointtrack.duration(times)), block_grid(times, step)
    chunks = load_chunks(folder, times)
    rows = []
    for i, b in enumerate(grid):
        row = dict(seconds=b, interval_seconds=b - grid[i - 1] if i else 0., displacement=np.nan,
                   displacement_lower=np.nan, displacement_upper=np.nan, measured=False)
        if i:
            a = grid[i - 1]
            chunk, stable = chunks.get(pointtrack.nearest_chunk(starts, (a + b) / 2), (None, None))
            if stable is not None:
                start, end = position(chunk, stable, a), position(chunk, stable, b)
                if start is not None and end is not None:
                    unit = float(chunk['unit'])
                    d = float(np.linalg.norm(end[0] - start[0]) / unit)
                    margin = (start[1] + end[1]) / unit
                    row.update(displacement=d, displacement_lower=max(0., d - margin), displacement_upper=d + margin,
                               measured=True)
        rows.append(row)
    return rows


def totals(rows):
    frame = pd.DataFrame(rows)
    measured = frame[(frame.interval_seconds > 0) & frame.measured]
    minutes = measured.interval_seconds.sum() / 60
    return dict(measured_seconds=float(measured.interval_seconds.sum()),
                displacement_body_units=dict(estimate=float(measured.displacement.sum()),
                                             lower=float(measured.displacement_lower.sum()),
                                             upper=float(measured.displacement_upper.sum())),
                displacement_per_measured_minute=dict(
                    estimate=float(measured.displacement.sum() / minutes) if minutes else None,
                    lower=float(measured.displacement_lower.sum() / minutes) if minutes else None))


def summarize(rows, video, times):
    frame = pd.DataFrame(rows)
    intervals = frame[frame.interval_seconds > 0]
    measured = intervals[intervals.measured]
    minutes = measured.interval_seconds.sum() / 60
    seconds_in = {state: float(intervals.loc[intervals.movement == state, 'interval_seconds'].sum())
                  for state in ('locomotion_candidate', 'low_motion_candidate', 'unknown')}
    return dict(
        video=str(video), video_sha256=sha256(video), duration_seconds=pointtrack.duration(times),
        measured_seconds=float(measured.interval_seconds.sum()),
        unmeasured_seconds=float(intervals.interval_seconds.sum() - measured.interval_seconds.sum()),
        measured_fraction=float(measured.interval_seconds.sum() / max(intervals.interval_seconds.sum(), 1e-9)),
        displacement_body_units=dict(estimate=float(measured.displacement.sum()),
                                     lower=float(measured.displacement_lower.sum()),
                                     upper=float(measured.displacement_upper.sum())),
        displacement_per_measured_minute=dict(estimate=float(measured.displacement.sum() / minutes) if minutes else None,
                                              lower=float(measured.displacement_lower.sum() / minutes) if minutes else None),
        rule_seconds=dict(clear_displacement=seconds_in['locomotion_candidate'], low_motion=seconds_in['low_motion_candidate'],
                          undetermined=seconds_in['unknown']),
        units=('Tamaños corporales = √(área de la máscara SAM2) en el fotograma de consulta de cada tramo; no son '
               'centímetros ni longitudes corporales calibradas.'),
        caveats=['La clasificación locomoción/reposo por clips no está validada (not_resolved); se da como referencia.',
                 'La suma de pasos incluye ruido de seguimiento: el límite inferior descuenta el error del fondo.',
                 'El tiempo sin medida no es reposo.',
                 'Validación: AUC 0,91 por clip frente a etiquetas ciegas con definición física (tiempo real).',
                 'No es un indicador de bienestar.'])


def plot(rows, summary, path, width=1200, height=320):
    frame = pd.DataFrame(rows)
    frame = frame[frame.interval_seconds > 0]
    image = np.full((height, width, 3), 255, np.uint8)
    left, right, top, bottom = 60, width - 20, 40, height - 70
    end = max(float(summary['duration_seconds']), 1e-6)
    x = lambda t: int(left + (right - left) * t / end)
    y = lambda v: int(bottom - (bottom - top) * min(v, PLOT_CAP) / PLOT_CAP)
    cv2.line(image, (left, bottom), (right, bottom), (0, 0, 0), 1)
    cv2.line(image, (left, top), (left, bottom), (0, 0, 0), 1)
    for value in (0, 1, 2):
        cv2.putText(image, str(value), (30, y(value) + 4), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    cv2.putText(image, 'tamanos por 0,5 s', (left + 6, top + 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (90, 90, 90), 1)
    colours = dict(locomotion_candidate=(60, 140, 255), low_motion_candidate=(200, 160, 60), unknown=(190, 190, 190))
    for r in frame.itertuples():
        a, b = x(r.seconds - r.interval_seconds), max(x(r.seconds) - 1, x(r.seconds - r.interval_seconds) + 1)
        if r.measured:
            cv2.rectangle(image, (a, y(r.displacement_upper)), (b, bottom), (225, 225, 225), -1)
            cv2.rectangle(image, (a, y(r.displacement)), (b, bottom), (120, 120, 120), -1)
            cv2.rectangle(image, (a, y(r.displacement_lower)), (b, bottom), (40, 40, 40), -1)
        else:
            cv2.rectangle(image, (a, bottom - 6), (b, bottom), (0, 0, 220), -1)
        cv2.rectangle(image, (a, bottom + 10), (b, bottom + 24), colours.get(r.movement, (190, 190, 190)), -1)
    for t in range(0, int(end) + 1, 5 if end <= 60 else 10):
        cv2.putText(image, f'{t}s', (x(t) - 8, bottom + 42), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    d, per = summary['displacement_body_units'], summary['displacement_per_measured_minute']
    text = (f"{Path(summary['video']).name}: desplazamiento {d['estimate']:.1f} (min. {d['lower']:.1f}) tamanos; "
            f"{(per['estimate'] or 0):.1f}/min medido; sin medida {summary['unmeasured_seconds']:.0f} s")
    cv2.putText(image, text, (left, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.putText(image, 'gris: estimado / oscuro: minimo / rojo: sin medida; franja: regla (naranja desplazamiento claro, '
                'azul bajo movimiento)', (left, height - 8), cv2.FONT_HERSHEY_SIMPLEX, .38, (60, 60, 60), 1)
    cv2.imwrite(str(path), image)


def measure(video, masks, out, tracks=None, tracker=None):
    """Actividad de un vídeo completo; si se da `tracker`, sigue antes los tramos que falten."""
    video, out = Path(video).resolve(), Path(out)
    tracks = Path(tracks) if tracks else pointtrack.TRACKS / video.stem
    times = pointtrack.frame_times(video, tracks.parent / 'tiempos')
    if tracker is not None:
        pointtrack.track_case(video, masks, tracks, tracker, None, times)
    steps = step_rows(times, tracks)
    blocks = step_rows(times, tracks, NET_STEP)
    rule = pointtrack.analyze(video, tracks, times)
    rows = [dict(s, movement=r['movement'], span=r['span'], span_lower=r['span_lower'], span_upper=r['span_upper'],
                 valid_fraction=r['valid_fraction']) for s, r in zip(steps, rule)]
    summary = summarize(rows, video, times)
    summary.update(source_sha256={name: sha256(Path(__file__).parent / f'{name}.py') for name in ('activity', 'pointtrack')},
                   python=platform.python_version(), tracks=str(tracks),
                   net_2s=dict(totals(blocks), note='Desplazamiento neto por bloques de 2 s (actividad_2s.csv).'))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'actividad.csv', index=False)
    pd.DataFrame(blocks).to_csv(out / 'actividad_2s.csv', index=False)
    (out / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    plot(rows, summary, out / 'actividad.png')
    d, n = summary['displacement_body_units'], summary['net_2s']['displacement_body_units']
    print(f"{video.name}: {d['estimate']:.1f} tamaños en pasos de 0,5 s, {n['estimate']:.1f} netos en 2 s, "
          f"{summary['measured_seconds']:.0f} s medidos de {summary['duration_seconds']:.0f} s → {out}", flush=True)
    return summary


def clip_rates(table, clips):
    """Tasa (tamaños/min medido) de cada clip con los bloques cuyo centro cae dentro."""
    centre = table.seconds - table.interval_seconds / 2
    rates = []
    for c in clips.itertuples():
        part = table[(table.interval_seconds > 0) & table.measured.astype(bool) &
                     (centre >= c.start_seconds) & (centre <= c.start_seconds + c.duration_seconds)]
        rates.append(part.displacement.sum() / (part.interval_seconds.sum() / 60) if len(part) else np.nan)
    return np.array(rates)


def evaluate(out, manifest):
    """Compara pasos de 0,5 s y bloques de 2 s frente a etiquetas verificadas con el criterio fijado (ADOPT)."""
    manifest, out = Path(manifest).resolve(), Path(out)
    rows = []
    for case in json.loads(manifest.read_text())['cases']:
        video = (manifest.parent / Path(case['video']).expanduser()).resolve()
        labels = pd.read_csv((manifest.parent / Path(case['labels']).expanduser()).resolve())
        clips = labels[(labels.video_id == video.name) & labels.label.isin(['locomotion', 'resting']) &
                       labels.reviewed.astype(str).str.lower().isin(['si', 'sí', 'yes', 'true'])]
        if clips.empty:
            continue
        fine = clip_rates(pd.read_csv(out / case['name'] / 'actividad.csv'), clips)
        net = clip_rates(pd.read_csv(out / case['name'] / 'actividad_2s.csv'), clips)
        for c, a, b in zip(clips.itertuples(), fine, net):
            rows.append(dict(case=case['name'], clip_id=c.clip_id, label=c.label, rate_05=a, rate_2s=b,
                             handheld=case['name'].startswith(('IMG_519', 'WhatsApp'))))
    table = pd.DataFrame(rows)
    table.to_csv(out / 'evaluacion_clips.csv', index=False)
    result = {}
    for column in ('rate_05', 'rate_2s'):
        scored = table[table[column].notna()]
        rest = scored[(scored.label == 'resting') & scored.handheld][column]
        result[column] = dict(auc=pointtrack.auc(scored[column], scored.label == 'locomotion'),
                              handheld_rest_median=float(rest.median()) if len(rest) else None,
                              fixed_rest_median=float(scored[(scored.label == 'resting') & ~scored.handheld][column].median()),
                              locomotion_median=float(scored[scored.label == 'locomotion'][column].median()),
                              clips=int(len(scored)))
    a, b = result['rate_05'], result['rate_2s']
    drop = 1 - b['handheld_rest_median'] / a['handheld_rest_median'] if a['handheld_rest_median'] else None
    adopted = bool(drop is not None and drop >= ADOPT['handheld_rest_median_drop'] and
                   b['auc'] is not None and a['auc'] is not None and b['auc'] >= a['auc'] - ADOPT['auc_tolerance'])
    result.update(criterion=ADOPT, handheld_rest_median_drop=drop, adopt_2s=adopted, manifest_sha256=sha256(manifest),
                  source_sha256=sha256(Path(__file__)))
    (out / 'evaluacion.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    print(f"0,5 s: AUC {a['auc']:.3f}, mediana reposo en mano {a['handheld_rest_median']:.1f}/min | "
          f"2 s: AUC {b['auc']:.3f}, mediana reposo en mano {b['handheld_rest_median']:.1f}/min | "
          f"bajada {drop:.0%} → {'se adopta 2 s' if adopted else 'se mantiene 0,5 s'}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Actividad continua: desplazamiento respecto al suelo con CoTracker3')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', help='Vídeo a analizar (con --masks)')
    source.add_argument('--manifest', help='Todos los casos de un manifest, una carpeta por caso')
    parser.add_argument('--masks', help='Carpeta de máscaras SAM2 del vídeo (masks/000000.png…)')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--tracks', help='Carpeta de tramos (por defecto runs/pointtrack/<vídeo>)')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--no-track', action='store_true', help='Usar solo los tramos ya guardados')
    parser.add_argument('--evaluate', help='Manifest con etiquetas para comparar 0,5 s y 2 s (criterio ADOPT)')
    args = parser.parse_args()
    if args.video and not args.masks:
        parser.error('--video requiere --masks')
    holder = {}

    def tracker():
        if 'model' not in holder:
            holder['model'] = pointtrack.CoTracker(args.device)
            pointtrack.smoke(holder['model'])
        return holder['model']

    class Lazy:
        def __call__(self, frames, queries):
            return tracker()(frames, queries)

        def memory(self):
            return tracker().memory()
    model = None if args.no_track else Lazy()
    if args.video:
        measure(args.video, args.masks, args.out, args.tracks, model)
        return
    manifest = Path(args.manifest).resolve()
    for case in json.loads(manifest.read_text())['cases']:
        video = (manifest.parent / Path(case['video']).expanduser()).resolve()
        masks = (manifest.parent / Path(case['masks']).expanduser()).resolve()
        measure(video, masks, Path(args.out) / case['name'], None, model)
    if args.evaluate:
        evaluate(args.out, args.evaluate)


if __name__ == '__main__':
    main()
