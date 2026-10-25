# RabbitSense V3.1 — detección visual de Cinna

Esta versión sustituye las cajas aproximadas V2.2 por un **detector visual de vocabulario abierto** YOLO-World, consultado con la clase `rabbit`. Cada fotograma se detecta de nuevo; si no aparece un conejo con confianza suficiente, muestra `unknown` y **no reutiliza cajas antiguas**. Es una prueba de detección espacial, todavía **no** un clasificador de locomoción. El vídeo de salida sirve para comprobar si realmente localiza a Cinna.

## Instalar en macOS

```bash
cd ~/Downloads/RabbitSense_Training
unzip -o ~/Downloads/RabbitSense_V31_detector.zip
.venv/bin/python -m pip install 'ultralytics>=8.3,<9'
```

## Ejecutar con el vídeo de la persecución

```bash
.venv/bin/python scripts/rabbitsense_v31.py \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --out runs/predictions_v31/5177 \
  --device mps
open runs/predictions_v31/5177/rabbit_detected_v31.mp4
```

La primera ejecución descargará los pesos del modelo (`yolov8s-worldv2.pt`) de Ultralytics; requiere conexión a Internet. Si falla MPS, repite con `--device cpu`. Para vídeo de alta resolución y lentitud excesiva, usa `--imgsz 512`.

## Entregables

- `rabbit_detected_v31.mp4`: vídeo con cajas verdes únicamente cuando se detecta un conejo.
- `rabbit_tracks_v31.csv`: detecciones con confianza y coordenadas por fotograma.
- `summary_v31.json`: proporción de detecciones.

**Limitaciones:** YOLO-World no está especializado en Cinna, puede fallar con oclusiones o planos difíciles. No presupone seguimiento correcto ni interpreta conductas. No se entrena ni modifica V2.3–V2.5. La siguiente etapa es verificar visualmente las detecciones y entonces calcular desplazamiento relativo a la cámara.
