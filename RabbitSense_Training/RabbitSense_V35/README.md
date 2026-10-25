# RabbitSense V3.5

Separa la disponibilidad de caja (`TRACKING: BOX_OK / BOX_MISSING`) de la inferencia de movimiento (`MOVEMENT: locomotion_candidate / low_motion_candidate / unknown`). Usa las cajas guiadas de V3.2 y seguimiento de puntos del fondo con LK + RANSAC para compensar la cámara.

## Instalar

Descomprime en `~/Downloads/RabbitSense_Training`.

## Ejecutar

```bash
.venv/bin/python RabbitSense_V35/rabbitsense_v35.py \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --tracks runs/v32/cinna_trayectoria_v32.csv \
  --out runs/predictions_v35/5177 \
  --video-out
```

Requiere `opencv-python`, `numpy` y `pandas`. Produce `motion_v35.csv`, `summary_v35.json`, `motion_v35.mp4`.

**Limitaciones:** las cajas son manuales/interpoladas, no detecciones independientes; la compensación de cámara puede fallar por perspectiva, falta de textura o movimientos de cámara; las etiquetas son candidatos no validados. No mide estrés, dolor ni bienestar.
