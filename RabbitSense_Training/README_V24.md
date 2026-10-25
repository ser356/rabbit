# RabbitSense V2.4 — tres acciones, revisión manual y evaluación independiente

Este parche **no trae un modelo V2.4 entrenado**. No hay suficientes ejemplos revisados de `locomotion` en dos grabaciones para entrenarlo de forma defendible. Tampoco fabrica etiquetas a partir del movimiento o de las predicciones de V2.3.

## Instalar

Descomprime este ZIP **sobre** `~/Downloads/RabbitSense_Training`. Conserva los modelos anteriores. Necesita V2.2 y V2.3 instalados, su `.venv`, ffmpeg y los vídeos reducidos de V2.3.

## 1. Crear clips para revisar

```bash
cd ~/Downloads/RabbitSense_Training
.venv/bin/python scripts/prepare_v24.py --make-clips
```

Abre `runs/review_v24/clips/` para ver cada clip. En `runs/review_v24/etiquetas_para_revisar.csv` rellena **manualmente** `label` y `reviewed` (`si` o `no`). Las sugerencias de V2.3 figuran en `notes`, pero **no se aceptan automáticamente**.

Etiquetas:

- `locomotion`: desplazamiento del cuerpo observable, sin una exploración/olfateo claramente dominante.
- `exploring_sniffing`: investigación evidente del entorno con el hocico (puede haber desplazamiento).
- `resting`: quietud sostenida, sin desplazamiento observable. No basta una postura agachada.
- `unknown`: dudoso, recorte erróneo, varias acciones simultáneas o categoría no cubierta.

Se recomienda tener locomoción, exploración y reposo en **varios vídeos**, y al menos un vídeo con las tres clases para validación. No marques `reviewed=si` si el clip no es inequívoco. Si el conjunto no tiene datos suficientes, el entrenamiento se detendrá explicando qué falta.

## 2. Entrenar cuando hayas revisado las etiquetas

```bash
.venv/bin/python scripts/train_v24.py
```

Guarda un checkpoint nuevo en `runs/temporal_v24/best.pt` sin tocar V2.3. Valida con una grabación completa distinta de las usadas en entrenamiento. Si necesitas nuevos ejemplos de locomoción, **graba más vídeos**, prepara sus características con `predict_v22.py` y añade sus intervalos al CSV revisado; no incluyas `cinna.mov` en el entrenamiento.

## 3. Probar `cinna.mov` solo después de entrenar

```bash
.venv/bin/python scripts/predict_v22.py \
  --input "$HOME/Downloads/cinna.mov" \
  --checkpoint runs/behaviour/best.pt \
  --temporal runs/temporal_v24/best.pt \
  --anchors runs/anchors/cinna.json \
  --out runs/predictions_v24/cinna \
  --interval 0.5 --action-conf 0.75 --action-margin 0.25
```

Se utiliza la abstención `unknown` cuando la confianza o la separación entre clases son bajas. **El umbral no está calibrado**; más `unknown` no significa automáticamente un mejor modelo.

## 4. Comparar V2.3 y V2.4

```bash
.venv/bin/python scripts/compare_cinna_v24.py
```

Compara distribuciones y cambios de predicción. Para obtener aciertos, prepara **manualmente**, antes de consultar V2.4, `runs/review_v24/cinna_test_labels.csv` con columnas `start_seconds,duration_seconds,label,reviewed` y `reviewed=si`. No uses las predicciones del propio modelo como verdad. Esta evaluación no mide bienestar, estrés o dolor.

## Limitaciones

- Un único conejo, pocos vídeos, anclas aproximadas y vídeos reducidos.
- El seguimiento sigue siendo LK con anclas: no se ha implementado un detector robusto de conejos.
- La clasificación de acciones no equivale a medir emociones, dolor ni bienestar.
- El nuevo entrenamiento no se ejecuta hasta que existan suficientes etiquetas humanas verificadas.
