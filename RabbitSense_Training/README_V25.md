# RabbitSense V2.5 — baseline para pocos datos

Instalar desde la carpeta del proyecto:

```bash
cd ~/Downloads/RabbitSense_Training
unzip -o ~/Downloads/RabbitSense_V25_sin_videos_nuevos.zip
.venv/bin/python scripts/rabbitsense_v25.py train
.venv/bin/python scripts/rabbitsense_v25.py cinna
```

Reutiliza `runs/review_v24/etiquetas_para_revisar.csv` y las características `runs/features_v22/*.npz`.

- Validación **leave-one-recording-out**: no se mezclan clips de una misma grabación entre entrenamiento y validación.
- Si falta alguna clase en un fold de entrenamiento, ese fold se marca como no evaluable.
- Comparación con baseline de clase mayoritaria.
- Después de evaluar, entrena modelo final con todos los clips revisados, pero nunca con `cinna.mov`.
- Guarda `runs/temporal_v25/baseline.joblib` y `runs/temporal_v25/evaluation.json`.
- Predice `cinna.mov` en ventanas de 4 segundos: `runs/predictions_v25/cinna/cinna_v25_windows.csv`.
- La clase `unknown` depende de umbrales heurísticos de scores **no calibrados**. No es diagnóstico de estrés, dolor o bienestar.

No sustituye el predictor V2.3/V2.4, no crea vídeo anotado ni modifica pesos antiguos. Los resultados pueden ser malos; el objetivo es comprobar si las características existentes aportan señal generalizable antes de invertir en otro modelo profundo.
