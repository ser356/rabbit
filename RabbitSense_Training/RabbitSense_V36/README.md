# RabbitSense V3.6

Mejora incremental de V3.5: mantiene las cajas manuales/interpoladas de V3.2, calcula compensación de cámara con puntos del fondo **cercanos al conejo**, usa el fondo global como alternativa y registra una medida de flujo óptico dentro de la caja (`body_flow_bodywidths_s`). Esta última es una señal auxiliar, no un detector de patas ni una etiqueta de locomoción verificada.

## Uso

```bash
cd ~/Downloads/RabbitSense_Training
unzip -o ~/Downloads/RabbitSense_V36.zip
.venv/bin/python RabbitSense_V36/rabbitsense_v36.py \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --tracks runs/v32/cinna_trayectoria_v32.csv \
  --out runs/predictions_v36/5177 \
  --video-out
```

Requiere `opencv-python`, `numpy`, `pandas` en el entorno virtual. Produce `motion_v36.csv`, `summary_v36.json` y, con `--video-out`, `motion_v36.mp4`.

## Validación ejecutada

Con el vídeo original IMG_5177.mp4 y `cinna_trayectoria_v32.csv`: 84 muestras; 81 BOX_OK; 20 locomotion_candidate; 43 low_motion_candidate; 21 unknown. 63 transformaciones locales válidas; 17 casos de referencia de cámara no fiable. **Estos números no son precisión ni sensibilidad**: faltan etiquetas verificadas. Las cajas interpoladas no son detecciones independientes. No permite inferir estrés, dolor ni bienestar.
