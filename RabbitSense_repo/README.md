# RabbitSense

Seguimiento asistido de Cinna y análisis experimental de movimiento. **No es detección autónoma ni diagnóstico de estrés, dolor o bienestar.**

## Estado verificado

**SAM2 ejecutado sobre vídeos reales:** seguimiento de IMG_5177 desde una caja inicial; IoU **0,674** en las mismas diez anclas reservadas, frente a **0,462** del puente. Flujo sobre máscaras con concordancia del centro: **2/3 clips de locomoción** del vídeo principal, frente a 0/3. **No generaliza aún a los dos controles.** Vídeo, límites y reproducción en [SEGMENTACION.md](SEGMENTACION.md).

Tres vídeos ejecutados; resultados y regresiones en [AUDITORIA.md](AUDITORIA.md).

La corrección temporal posterior, episodios candidatos y regresión de cobertura están en [LOCOMOCION.md](LOCOMOCION.md). No convierten las etiquetas mixtas por clip en verdad por fotograma.

**Locomoción: cerrada, no resuelta (2026-10-26).** En el test independiente con cámara en mano (IMG_5190–5193) fallan dos enfoques distintos: flujo óptico (flow 7/12 locomoción, 3/12 reposo) y X-CLIP zero-shot preentrenado (9/12, 1/12, 11 episodios falsos). No se graban más vídeos. Detalle en [LOCOMOCION.md](LOCOMOCION.md).

El seguimiento corporal persistente y YOLO-World local ya se ejecutaron: **ninguno resuelve la locomoción en estos vídeos**. Métodos, resultados negativos y reproducción en [EVIDENCIA_CORPORAL.md](EVIDENCIA_CORPORAL.md). Permanecen aislados del análisis habitual.

- **v36** permanece como referencia predeterminada, no como verdad-terreno.
- **flow** analiza fotogramas consecutivos y propone correcciones de centro con puntos persistentes entre anclas. Mejora geométrica limitada en IMG_5177, pero **regresa en otros vídeos**. Selección explícita.
- **v37** se conserva para comparación; sus predicciones nunca se utilizan como etiquetas.
- Detección sin anclas, interfaz macOS y bienestar no se declaran resueltos. La locomoción fiable está cerrada como no resoluble con estos datos.

## Instalación

Desde este directorio, con Python >=3.11:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Probado en Python 3.15.0, macOS arm64. [requirements-tested.txt](requirements-tested.txt) fija las versiones observadas; esos pins no implican compatibilidad con todos los Python admitidos por el paquete.

```bash
.venv/bin/python -m pip install -r requirements-tested.txt
.venv/bin/python -m pip install -e .
```

Un entorno virtual trasladado no es portable: reinstalar el paquete editable después de mover el proyecto. Este análisis usa CPU/OpenCV, sin necesitar PyTorch/MPS ni cargar los modelos originales.

## Analizar IMG_5177

```bash
.venv/bin/python -m rabbitsense --version flow \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --anchors ../RabbitSense_ARCHIVO/cinna_marcas_v32.json \
  --out runs/analysis/5177 --video-out
```

Requiere el JSON original, no los registros `manual` del CSV derivado. Admite marcas V3.2 (`box` xywh) y anclas V2.3 (`bbox` xyxy), escaladas desde su resolución de referencia.

Salidas:

- `tracks.csv`: guía original en `x,y,width,height`; propuesta en `refined_x,refined_y`; `bridge_points` y procedencia explícita. Las correcciones **no sobrescriben** las guías.
- `motion.csv`: intervalo terminado en `seconds`, candidatos, abstenciones, velocidad en anchos de caja/s, banda de incertidumbre y calidad de evidencia.
- En `flow`, velocidad = diámetro de la trayectoria residual acumulada / duración completa; errores por fotograma en cuadratura y huecos sin puentear. Con evidencia ausente, `speed_upper` queda vacío y no se declara bajo movimiento. La banda no está calibrada.
- `timeline.json`: intervalos contiguos; calidad ponderada por duración, **no probabilidad calibrada**.
- `summary.json`: versiones, parámetros, hashes de entradas/código y advertencias.
- `overlay.mp4`: fotogramas y FPS originales; gris = guía, verde = propuesta. Sin audio.

Sin guía, se mantiene desconocido; no se extrapola más allá de la última ancla. La cronología termina en el tiempo del último fotograma decodificado, incluyendo el intervalo parcial final.

## Referencias

```bash
.venv/bin/python -m rabbitsense --version v36 \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --tracks runs/v32/cinna_trayectoria_v32.csv \
  --out runs/reference/v36
```

Cambiar `v36` por `v37` para reproducir su lógica. En `flow`, `--step` controla la rejilla de resultados, **no reduce la frecuencia del análisis visual**. Se conserva el umbral `.2` de las referencias; no se ajustó para aumentar candidatos.

## Comparación reproducible

```bash
.venv/bin/python -m rabbitsense.evaluate \
  --suite benchmarks/local.json --out runs/suite --video-out
```

También disponible como tarea VS Code **RabbitSense: suite existente** desde la raíz del workspace. El manifest referencia datos originales sin copiarlos.

Para un caso concreto:

```bash
.venv/bin/python -m rabbitsense.evaluate \
  --video data/cinna_review_v24/IMG_5177.mp4 \
  --tracks runs/v32/cinna_trayectoria_v32.csv \
  --anchors ../RabbitSense_ARCHIVO/cinna_marcas_v32.json \
  --labels ../RabbitSense_ARCHIVO/etiquetas_v24_revision_final.csv \
  --out runs/comparison
```

Recalcula V3.6/V3.7/flow/body con semilla 0 y un hilo OpenCV; compara rejilla común y reserva anclas originales antes de interpolar. Evalúa etiquetas revisadas **por clip**, sin convertirlas en verdad por fotograma. Excluye `exploring_sniffing` y etiquetas no revisadas de la evaluación binaria.

`body` solo se ejecuta en la comparación: escribe movimiento, cronología y vídeo en su propio subdirectorio. No altera la CLI habitual `--version flow` ni incorpora PyTorch. La tarea **RabbitSense: persistent body benchmark** conserva esta comparación en [runs/body_verified/suite.json](runs/body_verified/suite.json).

## Arquitectura

| Módulo | Responsabilidad |
|---|---|
| [tracking.py](src/rabbitsense/tracking.py) | Anclas, correspondencias ida/vuelta y guías |
| [bridge.py](src/rabbitsense/bridge.py) | Puntos persistentes y corrección temporal candidata |
| [motion.py](src/rabbitsense/motion.py) | Cámara, movimiento relativo, agregación, cronología y vídeo |
| [body.py](src/rabbitsense/body.py) | Experimento de traslación, giro y escala con puntos persistentes |
| [evaluate.py](src/rabbitsense/evaluate.py) | Holdout geométrico, comparación y clips |
| [detector_audit.py](src/rabbitsense/detector_audit.py) | Auditoría opcional de pesos locales; no genera guías ni etiquetas |
| [segment.py](src/rabbitsense/segment.py) | SAM2 opcional, máscaras, concordancia silueta/flujo, vídeo y evaluación reservada |
| [closure.py](src/rabbitsense/closure.py) | Veredicto de locomoción con criterios fijados de antemano |
| [handheld.py](src/rabbitsense/handheld.py) | Cámara en mano: semillas, anclas SAM2, verificación de etiquetas y cierre con puerta de suficiencia |
| [zeroshot.py](src/rabbitsense/zeroshot.py) | Locomoción zero-shot con X-CLIP (preentrenado, sin ajuste); mismo veredicto |
| [silhouette.py](src/rabbitsense/silhouette.py) | Locomoción respecto al suelo (homografía del anillo de suelo) y forma de la máscara SAM2 |
| [__main__.py](src/rabbitsense/__main__.py) | CLI único |

V3.6/V3.7 son referencias dentro del paquete, sin nuevas carpetas de desarrollo por versión; comparten funciones idénticas. Los scripts legacy y el directorio original de entrenamiento se conservan, pero no son la API pública.

## Límites

Las cajas pueden contener fondo; los puntos no tienen identidad semántica garantizada. Oclusión, paralaje, desenfoque y guías holgadas causan errores. El puente necesita anclas inicial y final: análisis offline, no seguimiento causal. Las anclas reservadas del mismo vídeo son diagnóstico de desarrollo, no test independiente. No se infieren intenciones, estrés, dolor, enfermedad ni bienestar.
