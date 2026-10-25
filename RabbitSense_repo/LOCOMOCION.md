# Locomoción: diagnóstico y corrección temporal

## Resultado y decisión

**Locomoción fiable no resuelta.** Se corrigieron dos defectos matemáticos y se separaron episodios candidatos de la clasificación mayoritaria de un clip. No se cambiaron umbrales, etiquetas ni anclas. V3.6 sigue como referencia predeterminada; `flow` continúa experimental.

La corrección no aumenta los candidatos positivos del vídeo principal. Reduce la cobertura de bajo movimiento en los controles. No se presenta como mejora global ni se selecciona por sus recuentos.

## Qué mostraron los vídeos

Se inspeccionaron las secuencias existentes, no se crearon etiquetas de entrenamiento. Los mosaicos están en [main.jpg](runs/locomotion_diagnosis/main.jpg), [control.jpg](runs/locomotion_diagnosis/control.jpg) y [points.jpg](runs/locomotion_diagnosis/points.jpg).

- IMG_5177: desplazamiento visible alrededor de 33 s y 39–40 s; las imágenes de 35–38 s mantienen aproximadamente la posición de Cinna respecto al suelo. Las capturas no establecen una etiqueta para cada fotograma intermedio.
- Los puntos examinados en 35 y 37 s están mayoritariamente sobre el pelaje, no sobre el suelo. Por tanto, atribuir todo bajo movimiento de ese tramo a contaminación del fondo era una hipótesis incorrecta.
- En 31 s sí se observa mezcla de puntos sobre Cinna y la alfombra. No existe segmentación semántica garantizada.
- El control denominado `locomotion` contiene imágenes donde Cinna mantiene posición mientras se aproxima la cámara. El nombre del vídeo y su etiqueta de clip no prueban desplazamiento continuo.

Diagnóstico por pares consecutivos concretos, no media del intervalo ni métricas de precisión:

| Tiempo IMG_5177 | Puntos | Mediana residual x,y (px/fotograma) | Error cámara (px) |
|---|---:|---|---:|
| 31 s | 83 | 0.191, 0.003 | 0.586 |
| 33 s | 165 | 3.795, 20.410 | 0.474 |
| 35 s | 249 | 0.075, -0.412 | 0.549 |
| 37 s | 250 | -0.085, 0.012 | 0.170 |
| 39 s | 166 | 11.112, -3.216 | 0.261 |
| 40 s | 210 | 2.418, 0.854 | 0.796 |

Esos valores respaldan una diferencia local de movimiento, no prueban una compensación válida en todo el vídeo. El paralaje y el movimiento articulado siguen sin quedar separados de la traslación corporal.

## Defectos corregidos

1. Antes se medía la longitud de la suma de vectores. Ida y vuelta se cancelaban: la prueba reproducida devolvía bajo movimiento pese a un recorrido sintético de 0.8 anchos/s. Ahora se suman longitudes residuales.
2. Antes se dividía por duración × cobertura, extrapolando a los fotogramas ausentes. Ahora se divide por la duración completa: velocidad observada, sin imputar desplazamiento desconocido.
3. Sin cobertura completa no existe cota superior del recorrido del intervalo. `speed_upper` queda vacío y no se declara bajo movimiento; `incomplete_interval` explica la abstención cuando la cobertura supera el mínimo.

La banda inferior suma `max(0, distancia - error)`; la superior suma `distancia + error` solo con evidencia completa. Ambas dependen de una estimación de error **no calibrada**. No son intervalos estadísticos de confianza. La suma de longitudes también incluye ruido y movimiento de cabeza/cuerpo: no equivale automáticamente a locomoción.

## Comparación ejecutada

Tarea VS Code **RabbitSense: locomotion regression**. Resultados nuevos en [suite.json](runs/locomotion_suite/suite.json); los resultados anteriores permanecen en [suite.json](runs/suite/suite.json). Misma rejilla común, mismos vídeos y umbral 0.2. Recuentos incluyen la primera muestra de duración cero por compatibilidad.

| Vídeo | Flow anterior: locomoción / bajo / desconocido | Corregido: locomoción / bajo / desconocido |
|---|---|---|
| IMG_5177, 84 muestras | 7 / 42 / 35 | 7 / 40 / 37 |
| Reposo, 36 muestras | 0 / 12 / 24 | 0 / 3 / 33 |
| Locomoción, 28 muestras | 2 / 17 / 9 | 2 / 10 / 16 |

La mayoría estricta por clip se conserva: IMG_5177 sigue en **0/3**, control locomoción **0/2**; reposo pasa de **1/4 a 0/4** por más abstenciones. No se oculta esta regresión. La corrección geométrica y su holdout no cambian.

`candidate_events` añade episodios por clip sin alterar `prediction`, `correct` ni las métricas anteriores. En los tres clips finales de IMG_5177 aparecen:

- Clip 30–33.5: 31.5–32 y 32.5–33.5 s.
- Clip 34–37.5: 34–34.5 s.
- Clip 38–41.3: 39–40 s.

Son **predicciones**, no tres aciertos demostrados. Los límites están cuantizados a la rejilla; las anclas terminan en 40 s y el algoritmo se abstiene después. No se rellenan huecos desconocidos. Las etiquetas de clip no permiten medir precisión temporal por evento.

## Alternativa descartada: separar más los fotogramas

Se probaron 1, 2, 3 y 5 pares por intervalo de 0.5 s, contra el análisis de todos los fotogramas. Los CSV de diagnóstico contienen el vídeo completo y su intervalo final parcial; por ello tienen 85/36/29 filas, no la rejilla común de la tabla anterior.

| Pares por intervalo | IMG_5177: positivos | Reposo: positivos / bajo | Control locomoción: positivos |
|---|---:|---|---:|
| 1 | 0 | 0 / 14 | 1 |
| 2 | 0 | 0 / 12 | 3 |
| 3 | 1 | 0 / 9 | 2 |
| 5 | 4 | 0 / 5 | 2 |

Separar observaciones recupera bajo movimiento en reposo, pero pierde episodios positivos del principal. No se incorpora ese cambio ni se elige una separación distinta para cada vídeo. Diagnóstico exploratorio, sin afirmación de generalización.

## Requisito identificado y experimento ejecutado

El objetivo correcto es detectar **traslación corporal respecto al suelo**, localizar sus episodios y abstenerse durante oclusión, cámara no compensable o identidad visual perdida. No asignar locomoción a cada fotograma de un clip mixto ni inferir bienestar.

Este diagnóstico llevó a implementar correspondencias persistentes y validación espacial del fondo, separando traslación, giro y escala. También se ejecutaron los pesos locales YOLO-World/CLIP. La evaluación posterior y sus límites están en [EVIDENCIA_CORPORAL.md](EVIDENCIA_CORPORAL.md): cobertura corporal insuficiente, identidad semántica no resuelta y ninguna mejora demostrada de locomoción. No se instalaron ni entrenaron modelos; el detector usó el entorno de entrenamiento existente.

Se puede continuar con vídeos y anotaciones existentes, sin pedir nuevas grabaciones ni etiquetado manual. Con las etiquetas actuales solo se verifican geometría, coherencia, invariancia a cámara y observaciones por clip; no una precisión temporal completa del comportamiento.

## Verificación

20 pruebas automáticas pasan: cámara/animal sintéticos, ida/vuelta, ruido, ausencia de evidencia, episodios frente a mayoría, anclas, holdout, duración y exportación. Sin errores del editor y `git diff --check` limpio. Los hashes del código e inputs están en los resultados de la suite.