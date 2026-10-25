# RabbitSense V3.3 — Movimiento con compensación de cámara

Analiza cajas **revisadas manualmente** de V3.2. Estima el movimiento del fondo mediante ORB y RANSAC (transformación afín) excluyendo la región de Cinna. Compara la posición observada con la posición que cabría esperar si el conejo permaneciera quieto respecto al entorno.

## Instalación

Desde `~/Downloads/RabbitSense_Training`:

```bash
unzip -o ~/Downloads/RabbitSense_V33.zip
.venv/bin/python -m pip install opencv-python numpy pandas
```

## Ejecución

```bash
.venv/bin/python RabbitSense_V33/rabbitsense_v33.py \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --tracks runs/v32/cinna_trayectoria_v32.csv \
  --out runs/predictions_v33/5177 \
  --video-out
open runs/predictions_v33/5177/motion_v33.mp4
```

Si el CSV de V3.2 está en otra ruta, ajusta `--tracks`. El MP4 de salida es un **resumen a 2 fps**, no el vídeo completo a 30 fps.

`locomotion_candidate` indica movimiento relativo del recuadro respecto al fondo; `low_motion_candidate` indica movimiento bajo; `unknown` indica estimación no fiable. No son etiquetas validadas de comportamiento. La perspectiva y las cajas interpoladas pueden inducir errores. Umbral ajustable mediante `--threshold` (anchos corporales por segundo).
