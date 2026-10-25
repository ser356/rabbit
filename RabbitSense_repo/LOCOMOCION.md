# Locomoción: diagnóstico, corrección temporal y cierre

## Veredicto: cerrada, no resoluble con los datos existentes

Ejecutado el 2026-10-25 en el Mac (M4, MPS para SAM2) con el código de v0.5.0. Los hashes de `motion.py`, `closure.py`, `evaluate.py`, `segment.py`, `body.py` y del manifest en los veredictos coinciden con el commit. Evidencia: [cierre/verdict_cpu.json](cierre/verdict_cpu.json) y [cierre/verdict_sam2.json](cierre/verdict_sam2.json).

| Método | Locomoción (≥4/5) | Reposo (≥8/10) | Episodios falsos en reposo | Resultado |
|---|---:|---:|---:|---|
| V3.6 | 4/5 | 3/10 | 3 | No pasa |
| V3.7 | 1/5 | 4/10 | 0 | No pasa |
| **flow** | **4/5** | 5/10 | **0** | No pasa |
| body | 0/5 | 0/10 | 0 | No pasa |
| SAM2 + flow | 3/5 | 5/10 | 0 | No pasa |

Estado: `closed_not_resolvable_with_existing_data`. **Ningún método cumple los criterios fijados antes de ejecutar.**

### Lectura

- **flow detecta locomoción** en 4/5 clips, con al menos un acierto en cada vídeo de locomoción, y **ningún episodio falso** en los 10 clips de reposo.
- **Falla por abstención, no por error.** Con cero episodios falsos, sus cinco fallos de reposo son clips sin mayoría de bajo movimiento medido. La causa principal registrada es `camera_reference_unreliable`: 16/36 muestras en el control de reposo y 14/21 en IMG_4944 (motivos de V3.6, que comparte la referencia de cámara). En esos vídeos Cinna ocupa casi todo el encuadre de una cámara en mano; no queda fondo suficiente para separar el movimiento de la cámara del del animal.
- La corrección de la medida funcionó donde podía. Flow en el control de reposo pasó de 0/3/33 a 0/14/22 (locomoción/bajo/desconocido) y en IMG_5177 de 7/40/37 a 8/58/18, sin aumentar los candidatos en reposo. Es la mejora que se esperaba de quitar el ruido acumulado, pero no basta para el criterio.
- SAM2 no resuelve el problema: pierde un acierto en el control de locomoción (0/2) y no mejora la especificidad.

### Qué no se hace

- **No se cambian los criterios.** Contar la abstención como acierto daría a flow 10/10 en reposo; está prohibido de antemano porque un sistema que nunca decide pasaría ese criterio.
- No se ajustan umbrales, ventanas ni anclas para pasar el veredicto.
- No se presenta flow como detector validado. Se conserva como análisis experimental, igual que antes.

### Condición para reabrir

Solo con **datos nuevos** que eliminen la causa de la abstención, no con otra iteración sobre estos vídeos:

1. Grabaciones con cámara fija (trípode) o encuadre amplio con suelo visible alrededor del animal; o
2. anotación por evento (inicio/fin de cada desplazamiento) sobre vídeos donde la cámara sea compensable.

La reapertura vuelve a ejecutar `rabbitsense.closure` con los mismos criterios sobre un manifest ampliado.

---

## Cierre (v0.5.0): método

### Qué faltaba

1. **Un defecto de medida que impedía clasificar reposo.** `aggregate` sumaba longitudes y bandas de error fotograma a fotograma. El ruido y la banda crecían linealmente con los FPS: con un conejo quieto, la velocidad medida era ≈0,10 / 0,25 / 0,50 anchos/s a 12 / 30 / 60 FPS (simulación con ruido de 1 px y error estimado de 2 px, ancho 150 px). A 30 FPS el ruido ya superaba el umbral 0,2 y `low_motion` era prácticamente inalcanzable. Explica la abstención casi total de los controles de reposo, también con SAM2 (0/0/36).
2. **Una métrica que no correspondía a las etiquetas.** Los clips de locomoción contienen pausas; la mayoría estricta penaliza un clip con desplazamiento real seguido de pausa. Faltaba una métrica de detección por clip.
3. **Especificidad sustentada en un único vídeo** (4 clips de reposo).
4. **Un criterio de cierre fijado de antemano.** Sin él, cada iteración producía otra comparación sin decisión.

### Qué se cambió

- **Medida.** Velocidad = diámetro de la trayectoria acumulada del residuo compensado por cámara, dividido por la duración del intervalo; misma definición que `body.py`. Errores por fotograma en cuadratura: el ruido crece con √n, no con n. Los huecos cortan la trayectoria; nunca se puentean ni extrapolan. La ida y vuelta no se cancela: cuenta la excursión máxima. Se aplica igual a la concordancia del centro de máscara SAM2. No se cambiaron umbral (0,2), rejilla (0,5 s), etiquetas ni anclas.
- **Métrica de detección por clip** (`detection` en `clip_metrics`), además de la mayoría estricta, que se conserva:
  - clip de locomoción acertado = al menos un episodio candidato dentro del clip;
  - clip de reposo acertado = mayoría estricta de bajo movimiento medido **y** ningún episodio. La abstención nunca cuenta como acierto.
- **Manifest de cierre** [benchmarks/closure.json](benchmarks/closure.json): los tres casos anteriores más dos controles de reposo con anclas V2.3 existentes (972F531F, 3 clips; IMG_4944, 3 clips). Total: **5 clips de locomoción en 2 vídeos y 10 de reposo en 3 vídeos.** `local.json` no cambia.
- **Comando de cierre** `python -m rabbitsense.closure`: ejecuta la suite, opcionalmente mide máscaras SAM2 sobre **copias** (no sobrescribe resultados previos) y escribe `verdict.json` con hashes de código y manifest.

### Criterios fijados antes de ejecutar (`closure.CRITERIA`)

Un método **cierra la locomoción como resuelta en el benchmark de desarrollo** si cumple todo:

| Criterio | Valor |
|---|---|
| Clips de locomoción evaluados | ≥ 5 |
| Clips de reposo evaluados | ≥ 10 |
| Sensibilidad por clip (agregada) | ≥ 0,8 (≥ 4/5) |
| Especificidad por clip (agregada) | ≥ 0,8 (≥ 8/10) |
| Cada vídeo con clips de locomoción | ≥ 1 acierto |

Resultado posible, ambos cierran el tema:

- `closed_resolved_on_development_benchmark`: se adopta el mejor método que pase como análisis de locomoción candidata. Alcance: vídeos de desarrollo y etiquetas por clip; no validación independiente, precisión temporal por evento ni indicador de bienestar.
- `closed_not_resolvable_with_existing_data`: se archiva la locomoción con los datos actuales. Reabrirla exige datos nuevos (anotación por evento o vídeos con cámara fija), no otra iteración de umbrales.

**No se modifican los criterios después de ver resultados.** Si se cambian, el veredicto anterior queda invalidado y se documenta.

### Ejecutar (Mac, desde la raíz del workspace)

1. Tarea VS Code **RabbitSense: cierre locomoción** → `runs/closure/verdict.json` (V3.6, V3.7, flow, body; CPU).
2. Opcional, **RabbitSense: cierre locomoción con SAM2** → segmenta antes los dos controles nuevos con el protocolo guiado (anclas impares reservadas) y añade `sam2_masked_flow` en `runs/closure_sam2/verdict.json`. Requiere el entorno de entrenamiento con MPS y `weights/sam2.1_s.pt`.

Equivalente por terminal:

```bash
RabbitSense_repo/.venv/bin/python -m rabbitsense.closure \
  --suite RabbitSense_repo/benchmarks/closure.json --out RabbitSense_repo/runs/closure
```

### Estado de verificación

- 53 pruebas pasan, incluida una ejecución extremo a extremo del cierre con vídeo y máscaras sintéticas.
- Pruebas nuevas que **fallan con la medida anterior**: reposo a 30 FPS, traslación a 30/60 FPS, hueco no puenteado, ida/vuelta como diámetro.
- Los recuentos de las secciones siguientes corresponden a la medida anterior (v0.4.0) y no deben citarse como resultado de v0.5.0.

---

## Diagnóstico previo (v0.4.0)

### Resultado y decisión (v0.4.0)

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