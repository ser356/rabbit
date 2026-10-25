# RabbitSense V2.3 — Entrenamiento con siete grabaciones diferentes de Cinna

## Qué hace

1. Utiliza los **MP4 reducidos** que ya habías enviado (720 px, 12 FPS). Van incluidos en el parche: no necesita los originales de cientos de MB.
2. Usa **anclas distintas para cada vídeo**, anotadas aproximadamente sobre sus muestras, y seguimiento LK entre ellas. **No** reutiliza `runs/anchors/cinna.json` de tu vídeo anterior.
3. Extrae embeddings del **ViT entrenado** y señales de movimiento a 2 FPS. La caché se guarda en `runs/features_v22/` con una clave única por ruta.
4. Entrena una GRU temporal nueva en `runs/temporal_v23/best.pt`, sin sobrescribir ni el ViT ni `runs/temporal_v22/best.pt`.
5. Conserva completamente separados los vídeos usados en entrenamiento y validación.

El material etiquetado original consta de 11 vídeos/40 tramos, pero 4 vídeos se etiquetaron exclusivamente `unknown`. El entrenador solo usa los 7 vídeos con 19 tramos etiquetados, de los cuales 18 están aprobados para entrenar: 7 de exploración probable y 11 de reposo. La validación se realiza sobre 2 grabaciones completas (6 tramos), dejando 5 grabaciones (12 tramos) para entrenamiento. Las clases son únicamente `exploring_sniffing` y `resting`.

**Limitaciones serias**: el recorte se basa en anclas aproximadas, el material está comprimido y las cámaras se mueven y se acercan. No hay un test independiente sin utilizar para seleccionar el modelo. Todo procede de un mismo conejo y hay muy pocos clips. Las métricas serán demostrativas y no permiten inferir estrés ni dolor. Revisa siempre los recortes/CSV, porque los errores de tracking contaminan las etiquetas.

## Instalación

Descomprime el ZIP **dentro** de `~/Downloads/RabbitSense_Training`, sin crear otra carpeta nueva (en Terminal):

```bash
cd ~/Downloads/RabbitSense_Training
unzip -o ~/Downloads/RabbitSense_V23_multivideo.zip
bash train_multivideo_v23.command
```

El procesamiento usa los pesos `runs/behaviour/best.pt` ya presentes y la configuración V2.2. No vuelve a bajar modelos ni datasets de Internet.

El primer proceso extrae las características; posteriores ejecuciones las reutilizan. Si quieres repetir **solo entrenamiento**, ejecuta `.venv/bin/python scripts/train_multivideo_v23.py --overwrite`. Si necesitas regenerar características y conservar el modelo, usa `.venv/bin/python scripts/process_multivideo_v23.py --force`.

### Evaluar con una grabación nueva

Esta versión aún no trae un detector automático genérico de conejos: `run_v22.command` necesita anclas propias para cualquier nuevo vídeo. **No uses la anotación `cinna.json` para otros vídeos**. Los resultados sobre los vídeos que se usaron para entrenar no son una prueba independiente. Hay que reservar otro vídeo que nunca se haya utilizado ni para crear etiquetas ni para elegir modelo.

### Archivos relevantes

- `data/cinna_review_v23/`: vídeos revisados y etiquetas, respaldos reproducibles.
- `runs/anchors_v23/*.json`: cajas de posición aproximadas, una por vídeo (revisables).
- `runs/review/cinna_v23_labels.csv`: etiquetas listas para el entrenamiento.
- `runs/features_v22/`: embeddings y movimiento, con ruta/hashes por vídeo.
- `runs/temporal_v23/best.pt`: GRU nueva de dos clases.
- `runs/temporal_v23/results.json`: origen, separación por vídeo y limitaciones.

No elimina nada de la V1, V2.1 ni V2.2.
