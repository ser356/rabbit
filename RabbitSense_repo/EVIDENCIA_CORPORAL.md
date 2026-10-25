# Evidencia corporal persistente: implementación y evaluación

## Decisión

**Experimento ejecutado, no solución de locomoción.** Se implementó seguimiento persistente, separación geométrica de traslación/giro/escala y validación espacial del fondo. Pasa las pruebas sintéticas, pero su cobertura real es insuficiente. Se conserva como diagnóstico reproducible, exclusivamente en el evaluador; no añade coste ni modifica resultados del análisis habitual `flow` o la referencia predeterminada V3.6.

También se probaron los modelos locales YOLO-World y CLIP con MPS. No proporcionan identidad fiable de Cinna. No se integran sus detecciones como guías ni pseudoetiquetas. No se pidieron grabaciones ni anotaciones, no se cambiaron etiquetas existentes y no se entrenaron modelos.

## Implementación corporal

[body.py](src/rabbitsense/body.py) sigue los mismos identificadores de puntos durante cada intervalo de 0.5 s. Los puntos perdidos no reaparecen hasta el siguiente intervalo. Se procesan todos los fotogramas, incluido el intervalo parcial final.

- Puntos de primer plano: interior de la caja guía. Fondo: exterior ampliado y entorno local. **Son máscaras geométricas, no segmentación de conejo/suelo.**
- Flujo Lucas–Kanade con comprobación ida/vuelta. Se exigen al menos 12 puntos interiores y 24 exteriores persistentes.
- Fondo distribuido en al menos tres cuadrantes; centro de guía dentro de su envolvente convexa. Cada ajuste afín se valida dejando fuera un cuadrante.
- Tras compensar la cámara, una transformación de similitud separa traslación del centro, giro y escala. Se rechazan soporte demasiado local, deformación incompatible y cámara inconsistente.
- `translation_span_bodywidths`: diámetro de la trayectoria del centro estabilizado, normalizado por anchura inicial de caja. `speed_bodywidths_s`: ese diámetro dividido por duración, **no distancia recorrida ni velocidad anatómica**.
- `body_rotation_degrees` y `body_scale_ratio`: excursiones geométricas relativas al inicio. El centro de caja no es un punto anatómico.
- `speed_lower`, `speed_upper` y `evidence_quality`: diagnósticos **no calibrados**, no intervalos estadísticos ni probabilidades.
- Centro estable no implica reposo: puede coexistir con giro, postura o movimiento local. Ausencia de evidencia produce `unknown`, nunca un diagnóstico de comportamiento o bienestar.

## Tres vídeos reales

Resultados finales: [suite.json](runs/body_verified/suite.json). Mismos vídeos, etiquetas, anclas y umbral 0.2 que las referencias. Se conserva la comparación anterior y el primer intento en directorios separados.

Rejilla común; locomoción candidata / bajo movimiento / desconocido. Incluye primera muestra de duración cero por compatibilidad.

| Vídeo | Flow existente | Body nuevo | Cobertura temporal Body |
|---|---|---|---:|
| IMG_5177, 84 muestras | 7 / 40 / 37 | 0 / 4 / 80 | 4.82% |
| Reposo, 36 muestras | 0 / 3 / 33 | 0 / 1 / 35 | 2.86% |
| Locomoción, 28 muestras | 2 / 10 / 16 | 0 / 0 / 28 | 0% |

Mayoría estricta por clip, contando abstenciones: **0/3, 0/4 y 0/2** respectivamente. No se convierte ausencia de positivos en ausencia de locomoción real. Las etiquetas mixtas por clip siguen sin aportar verdad temporal por evento.

Motivos sobre salidas completas, incluyendo el intervalo parcial final: 85/36/29 filas, distintas de la rejilla común anterior.

| Motivo | IMG_5177 | Reposo | Locomoción |
|---|---:|---:|---:|
| Fondo no rodea el cuerpo | 37 | 15 | 23 |
| Puntos persistentes insuficientes/intervalo inicial inválido | 23 | 19 | 2 |
| Cuerpo incompatible con modelo coherente | 13 | 0 | 2 |
| Validación espacial de cámara fallida | 6 | 1 | 2 |
| Soporte corporal demasiado local | 2 | 0 | 0 |
| Centro estable | 4 | 1 | 0 |

### Inspección visual

Se revisó [el mosaico corporal](runs/body_verified/review.jpg) y los mosaicos neuronales indicados abajo. No equivalen a nuevas etiquetas.

- IMG_5177, 31.5 s: ningún punto interior sobrevive al intervalo. En 33 s quedan nueve, concentrados cerca de la cabeza; no sustentan traslación de todo el cuerpo.
- En 37.5 s los puntos interiores cubren pelaje y el centro se mantiene estable en el modelo. No demuestra reposo durante todo el clip.
- En los controles, algunas guías incluyen superficie ajena a Cinna y los puntos verdes también la siguen. El fondo disponible queda a un lado o debajo del animal; una extrapolación afín no queda validada alrededor de su centro.
- En IMG_5177 a 5 s se observaron 4/3/212/281 puntos exteriores por cuadrante: concentración fuerte en la alfombra. En el control locomoción a 5 s: 0/0/45/70, sin referencia superior. No se rebajaron requisitos para convertir estos intervalos en positivos.

Vídeos corporales completos, decodificados para comprobar salida: [IMG_5177](runs/body_verified/5177/body/overlay.mp4), 1251 fotogramas a 30 FPS; [reposo](runs/body_verified/resting/body/overlay.mp4), 211 a 12 FPS; [locomoción](runs/body_verified/locomotion/body/overlay.mp4), 166 a 12 FPS. Sin audio. Verde/azul = puntos candidatos interiores/exteriores supervivientes, no segmentación validada.

## Auditoría neuronal real

[detector_audit.py](src/rabbitsense/detector_audit.py) carga exclusivamente pesos locales YOLO-World y CLIP. Inferencia ejecutada en MPS desde el entorno de entrenamiento: Python 3.12.15, torch 2.14.1, ultralytics 8.4.188, clip 1.0, OpenCV 4.14.0.94. No se instalaron paquetes. El entorno normal del paquete sigue en Python 3.15, sin requerir torch.

Protocolo guardado con hashes de pesos, vídeos, anclas y código en [audit.json](runs/detector_audit/audit.json): imagen completa, tamaño 640, confianza mínima 0.25, NMS independiente de clase; selección por puntuación máxima, **sin consultar la caja anotada**. Comparación contra todas las anclas visibles originales: 21/10/8. No es el holdout de interpolación, por lo que sus IoU no se comparan directamente con aquel protocolo.

| Prompts | Vídeo | Predicción disponible / anclas | IoU media, ausencias = 0 |
|---|---|---:|---:|
| rabbit | IMG_5177 | 0 / 21 | 0.0000 |
| rabbit | Reposo | 0 / 10 | 0.0000 |
| rabbit | Locomoción | 0 / 8 | 0.0000 |
| rabbit, bunny, animal, dog, cat | IMG_5177 | 14 / 21 | 0.4530 |
| rabbit, bunny, animal, dog, cat | Reposo | 7 / 10 | 0.3442 |
| rabbit, bunny, animal, dog, cat | Locomoción | 8 / 8 | 0.5769 |

La variante multiclase surgió al diagnosticar IMG_5177; **no es una validación independiente ni prueba de generalización**. Clasifica visualmente Cinna como gato/perro. En 40 s detecta también un objeto oscuro y le da mayor puntuación que al conejo: selección máxima con IoU cero. Hay pérdidas con contraluz y movimiento.

Mosaicos: [IMG_5177](runs/detector_audit/proposals_5177.jpg), [reposo](runs/detector_audit/proposals_resting.jpg), [locomoción](runs/detector_audit/proposals_locomotion.jpg). Conservar la clase original evita fingir que el modelo reconoce conejos. No se reutiliza ninguna predicción como etiqueta de entrenamiento.

## Reproducción y verificación

Desde VS Code en la raíz del workspace:

1. **RabbitSense: persistent body benchmark**: ejecuta V3.6/V3.7/flow/body sobre el manifest existente; escribe resultados corporales separados y sus motivos en la comparación.
2. **RabbitSense: audit local detector**: usa el intérprete de entrenamiento y pesos locales explícitos; escribe la auditoría y mosaicos. No modifica ese entorno ni conecta descargas como alternativa.

34 pruebas pasan: cinco combinaciones de geometría corporal/cámara, seguimiento con imágenes, persistencia de bajas, paralaje discrepante, soporte local, entradas inválidas, duración/vídeo corporal y regresiones existentes. El análisis habitual sigue sin ejecutar Body. Las pruebas sintéticas demuestran propiedades geométricas, **no precisión conductual**.

La conclusión técnica queda acotada: persistir puntos dentro de cajas y exigir un modelo rígido no basta con este encuadre y estas guías. La identidad de torso y la referencia de suelo siguen sin establecerse. Estos experimentos se conservan para impedir que una mejora sintética o una detección multiclase se presente como solución validada.