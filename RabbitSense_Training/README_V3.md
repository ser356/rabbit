# RabbitSense V3 — movimiento relativo a la cámara (experimental)

Esta versión NO es un detector automático de conejos ni un clasificador validado. Reutiliza las cajas de seguimiento aproximadas que ya produjo V2.2 y estima movimiento relativo al fondo mediante puntos de Lucas–Kanade y transformación afín RANSAC. No entrena con los 19 clips ni necesita vídeos nuevos.

## Instalación

Desde `~/Downloads/RabbitSense_Training`:

```bash
unzip -o ~/Downloads/RabbitSense_V3_movimiento.zip
```

## Prueba sobre cinna.mov

```bash
.venv/bin/python scripts/rabbitsense_v3.py \
  --video "$HOME/Downloads/cinna.mov" \
  --tracks runs/predictions_v23/cinna/cinna_behaviours_v22.csv \
  --out runs/predictions_v3/cinna \
  --video-out
```

## Prueba sobre IMG_5177.mp4

```bash
.venv/bin/python scripts/rabbitsense_v3.py \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --tracks runs/predictions_v24/5177/IMG_5177_behaviours_v22.csv \
  --out runs/predictions_v3/5177 \
  --video-out
```

Salidas: `motion_v3.csv`, `summary_v3.json` y opcionalmente `motion_v3.mp4`.

## Lectura de etiquetas

- `locomotion_candidate`: desplazamiento de la caja respecto al fondo y flujo visual interno elevados, **solo candidato**.
- `low_motion_candidate`: poca evidencia de movimiento, **no implica descanso**.
- `unknown`: evidencia insuficiente o seguimiento dudoso.

El algoritmo se abstiene en cajas interpoladas y cuando no puede compensar el movimiento de cámara. No estima exploración, dolor, estrés ni bienestar. Los umbrales son heurísticos y el análisis requiere revisar el vídeo anotado; la perspectiva, los objetos móviles, los recortes grandes y las oclusiones pueden inducir errores. Los resultados no deben presentarse como una mejora demostrada frente a V2.3.

## Requisitos

Python 3.12, `opencv-python`, `numpy` (ya usados en V2.2), y FFmpeg no es necesario para este script. El vídeo y CSV deben corresponder al mismo archivo y misma orientación/resolución original.
