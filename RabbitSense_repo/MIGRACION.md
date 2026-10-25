# Migración sin perder datos

**No borres `~/Downloads/RabbitSense_Training`**. La estructura nueva se crea aparte.

1. Descomprime `RabbitSense_repo.zip` en Descargas.
2. Desde `~/Downloads`, ejecuta:

```bash
mkdir -p RabbitSense_repo/data/cinna_review_v24 RabbitSense_repo/runs/v32
cp RabbitSense_Training/data/cinna_review_v24/IMG_5177.mp4 RabbitSense_repo/data/cinna_review_v24/
cp RabbitSense_Training/runs/v32/cinna_trayectoria_v32.csv RabbitSense_repo/runs/v32/
cd RabbitSense_repo
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m rabbitsense --version v36 --video data/cinna_review_v24/IMG_5177.mp4 --tracks runs/v32/cinna_trayectoria_v32.csv --out runs/predictions_v36/5177 --video-out
```

3. Comprueba el MP4/CSV generados antes de decidir qué carpetas viejas archivar. El repositorio no elimina ningún archivo.
