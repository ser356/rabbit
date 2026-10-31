"""Desplazamiento con cámara fija a partir del centro de la máscara (2026-10-31; sin validar).

Con la cámara quieta no hay movimiento de cámara que descontar: la posición de Cinna es el centro de su máscara SAM2
y su tamaño, √(área). Es la referencia de `FixedCamera.swift`; se validará en la prueba con cámara fija.

- Posición en un instante: interpolación lineal entre las dos muestras válidas que lo rodean, si distan ≤ 2 ×
  tolerancia; si no, la muestra válida más cercana a ≤ tolerancia (como `activity.position`).
- Desplazamiento neto de [a, b]: distancia entre las posiciones en a y en b dividida por la mediana de √(área) de las
  muestras válidas del intervalo (tamaños corporales, como en `activity`).
"""
import numpy as np

TOLERANCE = .25


def position(times, xs, ys, instant, tolerance=TOLERANCE):
    times, xs, ys = (np.asarray(v, float) for v in (times, xs, ys))
    valid = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
    before, after = valid[times[valid] <= instant + 1e-9], valid[times[valid] >= instant - 1e-9]
    if len(before) and len(after) and times[after[0]] - times[before[-1]] <= 2 * tolerance:
        i, j = before[-1], after[0]
        w = 0. if j == i else (instant - times[i]) / (times[j] - times[i])
        return (1 - w) * xs[i] + w * xs[j], (1 - w) * ys[i] + w * ys[j]
    near = valid[np.abs(times[valid] - instant) <= tolerance]
    if not len(near):
        return None
    j = near[np.argmin(np.abs(times[near] - instant))]
    return xs[j], ys[j]


def net_displacement(times, xs, ys, areas, start, end, tolerance=TOLERANCE):
    """Desplazamiento neto en tamaños corporales entre `start` y `end`, o None si falta posición o tamaño."""
    times, areas = np.asarray(times, float), np.asarray(areas, float)
    a, b = position(times, xs, ys, start, tolerance), position(times, xs, ys, end, tolerance)
    inside = (times >= start - 1e-9) & (times <= end + 1e-9) & np.isfinite(areas) & (areas > 0)
    if a is None or b is None or not inside.any():
        return None
    unit = float(np.median(np.sqrt(areas[inside])))
    return float(np.hypot(b[0] - a[0], b[1] - a[1]) / unit)
