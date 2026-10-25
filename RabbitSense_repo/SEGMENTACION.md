# Segmentación temporal: mejora medida, alcance limitado

## Resultado útil

SAM 2.1 Small sigue a Cinna en IMG_5177 utilizando **solo la caja original del primer fotograma**. Se procesaron los 1251 fotogramas, incluyendo cambios de escala, ocultación parcial detrás de la silla y desplazamientos finales. Las máscaras estuvieron presentes en todos; eso no prueba por sí solo identidad correcta. Se revisaron visualmente muestras distribuidas y los desplazamientos finales, no se anotaron máscaras de referencia.

- [Vídeo con máscara y candidatos de movimiento](runs/segmentation/5177/motion_overlay.mp4).
- [Cronología final](runs/segmentation/5177/flow_timeline.json).
- [Comparación verificable](runs/segmentation/comparison.json).

En las **mismas diez anclas reservadas** del benchmark anterior, IoU medio de caja: interpolación **0,441**, puente persistente **0,462**, SAM2 **0,674**. Error mediano del centro: **0,071 anchos de caja** con SAM2 frente a **0,190** del puente. SAM2 recibió menos anclas: solo la inicial, no los extremos ni las correcciones intermedias. IoU mide cajas, no precisión de píxeles; evaluación de desarrollo en el mismo vídeo.

Los tres clips revisados del vídeo principal pasan de **0/3 a 2/3** con la misma regla de mayoría temporal estricta. Los episodios candidatos principales son **31–34,5 s** y **39–41 s**. También aparecen candidatos 0–0,5 s y 41,5–41,667 s. Son límites predichos, no eventos anotados ni precisión temporal demostrada. El clip 34–37,5 s sigue sin concordar por mayoría: contiene una breve salida y después una pausa.

## Qué cambió

1. SAM2 conserva memoria entre fotogramas; no depende de reconocer la especie con YOLO.
2. Los puntos de flujo deben permanecer dentro del interior erosionado de dos máscaras consecutivas. El fondo ya no entra por una caja holgada.
3. Se reutilizan compensación local de cámara, umbral 0,2 anchos de caja/s y controles de ruido/cobertura del flujo existente.
4. Un positivo requiere también desplazamiento del centro de la máscara tras compensar cámara. Movimiento del pelaje con silueta estable queda **desconocido**, no reposo.
5. La medida rígida anterior sigue separada como diagnóstico. Su muestreo de fondo con máscaras reparte puntos por cuadrantes: evita que la alfombra monopolice los puntos, sin rebajar validación espacial.

La concordancia del centro rechazó dos falsos candidatos del control de reposo, en 6–6,5 s y 12,5–13 s, y uno del vídeo principal en 41–41,5 s. Es una corrección desarrollada sobre estos vídeos, **no una prueba independiente**. La corrección por ruido del centro solo resta residuo de cámara: no acota errores de segmentación, paralaje ni cambios de silueta. La velocidad sigue normalizada por ancho de caja, no por longitud anatómica ni metros.

## Controles: no resueltos

| Caso | Fotogramas con máscara | IoU en anclas reservadas anteriores | Candidatos / bajo movimiento / desconocido | Clips correctos |
|---|---:|---:|---:|---:|
| IMG_5177 | 1251/1251 | 0,674, n=10 | 13 / 27 / 45 | 2/3 |
| Locomoción V2.3 | 166/166 | 0,572, n=3 | 1 / 10 / 18 | 0/2 |
| Reposo V2.3, corregido | 181/211 | 0,468, n=4 | 0 / 0 / 36 | 0/4 |

Conteos sobre filas completas: 85/29/36, incluyendo instante inicial sin duración y último intervalo parcial. No confundir con rejilla histórica 84/28/36. Los controles tenían IoU del puente 0,714 y 0,748 respectivamente; SAM2 no mejora esas métricas de caja. Sus cajas originales son holgadas, por lo que una máscara ajustada no equivale a mayor IoU.

El control de reposo **se abstiene todo el tiempo**: cero candidatos no significa especificidad validada ni reposo reconocido. El control de locomoción tampoco supera la evaluación por clips. No se cambia el algoritmo predeterminado ni se declara un clasificador general fiable.

## Correcciones con datos existentes

La primera caja del control de reposo inicializó una máscara incorrecta; la propagación sin correcciones solo produjo 18 máscaras no vacías. Se ensayaron correcciones con anclas pares, reservando impares. Una caja de ancho completo a 12 s hizo seleccionar suelo.

La versión final excluye correcciones que abarcan todo el ancho o alto de la imagen. En ese control solo se usaron fotogramas **0, 48 y 96** (0, 4 y 8 s), nunca las cuatro anclas de evaluación. Recupera el animal desde 4 s, con limitaciones en encuadres muy cercanos. El resto del vídeo no recibe nuevas cajas. Los fallos anteriores permanecen en `runs/segmentation/resting` y `runs/segmentation_guided/resting`; el resultado corregido está en `runs/segmentation_localized/resting`.

## Reproducción local

- Pesos: `weights/sam2.1_s.pt`, descargados de https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt. SHA-256 registrado por ejecución. No se añadieron dependencias al entorno CPU.
- Inferencia: entorno existente `RabbitSense_Training/.venv`, Python 3.12.15, torch 2.14.1, ultralytics 8.4.188, MPS, tamaño 1024. No se entrenaron pesos ni se enviaron vídeos a servicios externos. No se acredita velocidad de tiempo real.
- Tarea **RabbitSense: SAM2 single seed**: vídeo principal, caja inicial, máscaras y movimiento.
- Tarea **RabbitSense: SAM2 controls**: controles sin correcciones; conserva el fallo inicial de reposo.
- Tarea **RabbitSense: SAM2 localized resting**: control de reposo con correcciones localizadas.
- Tarea **RabbitSense: SAM2 motion review**: reutiliza máscaras finales; regenera CSV, cronologías, vídeos y comparación. Utiliza Python 3.15/OpenCV, sin cargar PyTorch.

CLI opcional `python -m rabbitsense.segment`: requiere `--video`, `--anchors`, `--model`, `--out`; `--device mps` selecciona MPS y `--guided` activa correcciones retenidas. Por defecto solo caja inicial. Las anclas impares se excluyen antes de cualquier corrección; el evaluador rechaza coincidencias entre prompts y referencias reservadas.

Cada ejecución guarda máscaras PNG por fotograma, cajas, resumen de seguimiento, medida rígida, flujo con concordancia, cronologías y dos vídeos a FPS/resolución originales, sin audio. Los resultados previos a la concordancia se conservan en `runs/segmentation_before_centroid`. Los hashes de código de las primeras ejecuciones de una semilla se capturaron al finalizar; no prueban por sí solos qué módulos estaban cargados durante la inferencia. Las ejecuciones posteriores capturan código antes de inferencia. La comparación final vincula hashes de los artefactos utilizados.

## Verificaciones

40 pruebas automatizadas: anclas reservadas, correcciones en fotograma correcto, cajas de imagen completa, máscaras ausentes, puntos fuera del animal, rechazo de articulación con centro estable, intervalos parciales y decodificación de vídeos. **Estas pruebas no sustituyen validación conductual.**

Los seis vídeos finales se decodifican completos: dos por caso, 1251/166/211 fotogramas. Los doce CSV anteriores (v36/v37/flow/body × tres vídeos) son idénticos byte a byte al repetirlos en `runs/segmentation_regression`; datos originales y resultados históricos se conservan.

No se infieren estrés, dolor, enfermedad, intención ni bienestar. No se solicitan grabaciones ni anotaciones nuevas.