# RabbitSense V2.4 — incorporación de IMG_5177.mp4

Este parche **no crea un modelo entrenado**. Añade tres fragmentos candidatos a locomoción de la grabación IMG_5177.mp4, sin incluir `cinna.mov` en entrenamiento. Conserva copia de las etiquetas existentes.

1. Descomprime en `~/Downloads/RabbitSense_Training`.
2. Ejecuta `.venv/bin/python scripts/add_5177_v24.py --input "$HOME/Downloads/IMG_5177.mp4"`.
3. Revisa los tres clips `runs/review_v24/clips/v24_5177_*.mp4`. En `runs/review_v24/etiquetas_para_revisar.csv`, marca `reviewed=si` únicamente en los fragmentos donde el conejo camina de forma clara. Si ya has comprobado los tres intervalos visualmente, puedes usar `--confirm` al crear las filas.
4. Ejecuta el comando de extracción de características que imprime el script. Para ello el predictor necesita los modelos de V2.2/V2.3 ya instalados.
5. Ejecuta `.venv/bin/python scripts/train_v24.py`. Este entrenador exige **al menos dos grabaciones por clase, un vídeo de validación con las tres clases y al menos tres clips de entrenamiento por clase**. Si no se cumplen esas condiciones, detendrá el entrenamiento; no significa que haya fallado la instalación.

**Limitación**: la inspección de fotogramas indica locomoción en torno a 30–41,7 s, pero puede haber pausas y oclusiones; los tres intervalos son candidatos y deben verificarse antes de autorizar su uso. No usar el vídeo completo como locomoción.
