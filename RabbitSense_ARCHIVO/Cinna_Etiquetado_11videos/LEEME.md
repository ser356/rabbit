# RabbitSense: revisión de los 11 vídeos adicionales de Cinna

He revisado los vídeos MP4 reducidos contenidos dentro de los 11 ZIP individuales del exportador. Duración combinada: **140,884 segundos**. Cada acción se ha evaluado por intervalos de hasta cuatro segundos, siguiendo las divisiones que generó el exportador.

Se entregan **40 intervalos**. Las etiquetas observadas son:

- `resting`: 12 tramos. Reposo visual suficientemente reconocible; algunos con certeza media.
- `exploring_sniffing`: 7 tramos. Inspección de mano, suelo o un envase, con cabeza/hocico orientados hacia el objeto. No se afirma olfato fisiológicamente medido.
- `unknown`: 21 tramos. Actividad ambigua, caricias humanas fuera del vocabulario actual, hocico ocluido, barrotes, o intervalos finales demasiado cortos.

**No aparecen ejemplos inequívocos para entrenar `eating`, `grooming`, `jumping_binkying` o `locomotion` por separado.** Tampoco se observan comportamientos de los que deba inferirse una medida cuantitativa de estrés.

## Contenido

- `etiquetas_40_tramos.csv`: las 40 anotaciones con nombre del vídeo original, segundos inicial y final, etiqueta, comentario y confianza cualitativa.
- `por_video/`: un CSV por vídeo con esas mismas columnas.
- `importar_etiquetas.py`: localiza los vídeos originales dentro de `~/Downloads/Cinna_Videos` y produce el CSV que espera `train_temporal_v22.py`, usando las rutas exactas de tu Mac.
- `resumen.json`: recuentos y límites de lo observado.

## Importar en tu Mac

Descomprime este ZIP en `~/Downloads` y ejecuta:

```bash
cd ~/Downloads/RabbitSense_Training
.venv/bin/python ~/Downloads/Cinna_Etiquetado_11videos/importar_etiquetas.py \
  --videos "$HOME/Downloads/Cinna_Videos" \
  --out runs/review/cinna_multivideo_labels.csv
```

Si has guardado los vídeos en otra carpeta, cambia `--videos`. El script no duplica archivos, no toca tus pesos y falla de forma visible si falta alguno.

## Lo que falta ANTES de entrenar con V2.2

Las etiquetas **no sustituyen las características del vídeo**. El entrenador V2.2 lee cachés `runs/features_v22/<nombre>_<hash>.npz` extraídas mediante `run_v22.command`. Esas cachés aún no existen para estos 11 vídeos. Cada vídeo requiere anclas propias de posición (o mejorar el sistema de detección), y **no** deben reutilizarse las coordenadas de `runs/anchors/cinna.json` de la primera grabación. Si las usases, entrenarías con mobiliario, paredes o imágenes mal recortadas.

El video `IMG_4944.MOV` tiene los ejemplos más nítidos de reposo tumbado. `AE22D736-2B56-419D-886E-7926A182891E.mp4` y `IMG_5006.MOV` contienen ejemplos más útiles de exploración. Los vídeos de caricias se quedan `unknown` porque todavía no tenemos la clase `human_interaction`.

**Límites de calidad:** se revisaron los MP4 comprimidos a 12 FPS sin audio, no los originales. Algunas etiquetas son de certeza media; son anotaciones visuales provisionales, no verdad de terreno clínica. Clips cercanos dentro del mismo vídeo están correlacionados; nunca mezclarlos entre entrenamiento y prueba.
