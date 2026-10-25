# RabbitSense: exportador de vídeos para revisión

Este complemento funciona sobre el proyecto `RabbitSense_Training` ya existente. No entrena modelos, no sube archivos y no modifica tus grabaciones originales.

1. Descomprime el ZIP en `~/Downloads/RabbitSense_Training/`.
2. Pon los vídeos en `~/Downloads/Cinna_Videos/`.
3. Desde `~/Downloads/RabbitSense_Training` ejecuta:

```bash
.venv/bin/python scripts/export_review_pack.py "$HOME/Downloads/Cinna_Videos"
```

La carpeta `~/Downloads/Cinna_Para_ChatGPT/` contendrá un ZIP por vídeo (o por fragmento de hasta 120 segundos). Adjunta esos ZIP aquí. Cada ZIP incluye el MP4 comprimido sin audio, fotogramas JPG de muestreo, mosaicos y una plantilla de intervalos de 4 segundos con tiempos referidos al original.

Puedes ajustar calidad/tamaño con `--size 960 --fps 15 --crf 28` (más detalle, más peso) o `--size 480 --fps 8 --crf 34` (más pequeño). Para vídeos muy largos puedes cambiar `--chunk-seconds 60`. Repetir el comando no reexportará ZIPs existentes si no pasas `--overwrite`.

## Sobre el entrenamiento actual

El modelo V2.2 necesita una caché `.npz` de características por cada vídeo, creada actualmente con un seguidor de recuadros que requiere anclas de ese vídeo. No debes usar `runs/anchors/cinna.json` en grabaciones nuevas distintas. Los ZIP de revisión sirven para que podamos etiquetar los vídeos y preparar el siguiente entrenamiento, pero no sustituyen esa fase de extracción de características.
