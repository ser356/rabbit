# RabbitSense V2.2 — acciones temporales de Cinna

**Parche para la carpeta existente `RabbitSense_Training`**. No modifica `runs/behaviour/best.pt`, los datasets ni los scripts V1/V2/V2.1.

## Instalar en macOS M4

```bash
cd ~/Downloads
unzip -o RabbitSense_V22_patch.zip -d RabbitSense_Training
cd RabbitSense_Training
bash run_v22.command "$HOME/Downloads/cinna.mov"
```

Debe existir el antiguo `runs/behaviour/best.pt`, `.venv/` y el archivo `runs/anchors/cinna.json` de V2.1. El parche no sobrescribe este JSON por si has retocado las cajas. El MP4 de 800 MB no sale del ordenador. El comando procesa fotogramas a 2 FPS y genera:

- `runs/predictions_v22/cinna_behaviours_v22.csv`: postura fotográfica, acción temporal (si está entrenada), movimiento visual y procedencia del recorte.
- `runs/predictions_v22/cinna_summary_v22.json`: incidencias, procedencias, advertencias.
- `runs/predictions_v22/cinna_annotated_v22.mp4`: vídeo ligero con resultados y recuadros (H.264 funciona con resoluciones impares).
- `runs/features_v22/cinna_<hash>.npz`: embeddings ViT + características de movimiento, necesarios para aprender acciones; no son etiquetas.

### Diferencias reales frente a V2.1

- **Seguimiento**: puntos LK con verificación hacia atrás y control de coherencia, reanclados en instantes clave. Si falla, recurre a interpolación *explícitamente identificada*. No es una red detectora YOLO ni segmentación validada. El fondo y la falta de textura aún pueden causar errores.
- **Postura**: mantiene el ViT de siete clases y un `unknown` cuando su confianza es baja. La confianza no está calibrada.
- **Movimiento**: flujo óptico en el cuerpo respecto al fondo, con indicador de posible movimiento de cámara. No es una etiqueta `exploring_sniffing`.
- **Acción temporal**: si **no** existe el modelo de secuencias entrenado, la salida será `untrained`; nunca infiere una acción mediante reglas ocultas. Si lo entrenas, utiliza una GRU bidireccional que combina características ViT y movimiento en ventanas de ocho muestras.

## Enseñar acciones REALES al modelo

1. Genera clips etiquetables con `bash label_v22.command "$HOME/Downloads/cinna.mov"`.
2. El navegador abrirá **`runs/review/cinna/review.html`** con clips de cuatro segundos. Míralos completos y selecciona una sola acción dominante por clip (o `unknown` si no está clara). No etiquetes un cuerpo encogido como descansando si el conejo está caminando; `exploring_sniffing` solo cuando sea apreciable la exploración/olfateo, `locomotion` si únicamente se desplaza, `resting` cuando realmente permanece en reposo.
3. Pulsa «Descargar rabbitsense_labels.csv» y colócalo en:

   ```bash
   cp "$HOME/Downloads/rabbitsense_labels.csv" runs/review/cinna/rabbitsense_labels.csv
   ```
4. Entrena un **prototipo** con:

   ```bash
   bash train_temporal_v22.command
   ```

   Esto reutiliza los embeddings extraídos, **sin** volver a procesar los 800 MB ni tocar el ViT. Necesitas al menos ocho clips claros, al menos dos clases distintas y dos clips de entrenamiento por clase. Si hay un solo vídeo, el launcher permite un **prototipo no validado** y lo advertirá expresamente. No se calcula precisión sobre un test independiente que no existe. El nuevo checkpoint queda en `runs/temporal_v22/best.pt`.

5. Vuelve a analizar el original con `bash run_v22.command "$HOME/Downloads/cinna.mov"`. Ahora aparecerán predicciones temporales y `unknown` cuando no se supere el umbral.

Si tienes más vídeos, ejecuta `run_v22.command` con su archivo de anclas correspondiente (`--anchors ruta.json`), etiqueta cada vídeo, coloca sus CSV de etiquetas en sus carpetas y entrena pasando todos:

```bash
.venv/bin/python scripts/train_temporal_v22.py --labels \
  runs/review/cinna/rabbitsense_labels.csv \
  runs/review/otrovideo/rabbitsense_labels.csv
```

Con dos o más vídeos, retiene uno entero para validación (no mezcla clips del mismo vídeo en train y val). Para evaluar generalización real, aún necesita vídeos adicionales no usados para ajustar modelo ni umbrales, idealmente de otros conejos y casas.

## Problemas frecuentes

- **Solo aparece `untrained`**: normal si no has etiquetado y entrenado clips. Ejecuta las instrucciones anteriores.
- **`low_classifier_confidence` en V2.1 / muchos `unknown` en V2.2**: `unknown` puede ser una respuesta adecuada cuando no hay evidencia. Comprueba los recuadros antes de bajar umbrales.
- **LK recupera pocas cajas**: inspecciona el MP4 y corrige / añade anclas en `runs/anchors/cinna.json` (coordenadas relativas a `reference_width`, `reference_height`). El seguimiento usa las anclas como referencias, nunca etiqueta acciones automáticamente.
- **Tu otro vídeo tiene fondo y escala distintos**: genera anclas propias. No reutilices las de Cinna.
- **La grabación dura un minuto**: no permite validar todas las acciones, ni demuestra reconocimiento de bienestar o estrés.

## Límite científico

El dataset IEEE original proviene de fotos clasificadas y no contiene etiquetas veterinarias de estrés. Esta V2.2 es **investigación/prototipo** de comportamientos observables, no un diagnóstico de dolor, miedo, ansiedad, bienestar ni «porcentaje de felicidad». Las clases solo están entrenadas donde existen ejemplos humanos etiquetados. Revisa licencias de las fotografías de origen antes de distribuir modelos y datasets.
