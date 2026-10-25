# Auditoría y primera mejora verificable — 2026-10-25

## Dictamen

La instalación, CLI, pruebas y análisis de vídeos funcionan. Se incorpora una mejora **limitada al diagnóstico geométrico de IMG_5177**, no un sistema de locomoción fiable. El método nuevo no supera globalmente las referencias; V3.6 permanece predeterminada. No se han creado etiquetas, grabaciones ni inferencias de bienestar.

## Estado real encontrado

- Código activo: paquete src-layout con V3.6/V3.7, tres scripts legacy y una prueba de ayuda. El runner correcto era pytest; unittest descubría **cero pruebas**.
- El entorno editable importaba desde `/Users/sergiogarciaseco/Downloads/RabbitSense_repo/src`, inexistente. Reparado con reinstalación editable en el workspace actual.
- Git no tenía commits. Se conservó el código recibido en `2d0e692` antes de cambios funcionales.
- Vídeo principal: 512 × 910, 30 FPS, **1.251 fotogramas**, 41,7 s. CSV V3.2: 1.251 filas, 1.169 interpoladas, 32 con estado manual y 50 desconocidas.
- El JSON original contiene **21 anclas visibles distintas**, no 32 cajas humanas independientes. El CSV marca también fotogramas adyacentes como manuales. No se usa ese estado para dividir entrenamiento/evaluación.
- V3.6/V3.7 tenían copias idénticas de lectura de cajas, máscaras, estabilización y flujo denso. Se comparten las implementaciones; se mantiene la lógica de referencia.
- Las referencias comparaban imágenes separadas 0,5 s, almacenaban todos los fotogramas muestreados y V3.7 vetaba resultados cuando discrepaban con centros interpolados.
- Sus MP4 muestreados se escribían a 10 FPS, acelerando la reproducción. Corregidos a FPS de muestreo; el exportador nuevo conserva todos los fotogramas a FPS original.
- Scripts `.command`, CSV y JSON históricos incluyen rutas a Descargas. No son una API portable. El flujo nuevo usa argumentos explícitos o manifest relativo, sin mover datos.

### Recursos existentes

No falta ningún archivo necesario para reproducir esta entrega:

- Principal y trayectoria: [vídeo](data/cinna_review_v24/IMG_5177.mp4), [CSV](runs/v32/cinna_trayectoria_v32.csv).
- [Marcas originales](../RabbitSense_ARCHIVO/cinna_marcas_v32.json).
- [Etiquetas revisadas](../RabbitSense_ARCHIVO/etiquetas_v24_revision_final.csv): incluye tres clips finales de IMG_5177 etiquetados como locomoción, con aviso de posibles pausas.
- [Anclas V2.3](../RabbitSense_Training/runs/anchors_v23) y vídeos existentes permiten dos controles adicionales; rutas exactas en [manifest](benchmarks/local.json).
- [Modelo de postura](../RabbitSense_Training/runs/behaviour/best.pt) y [métricas guardadas](../RabbitSense_Training/runs/behaviour/results.json): accuracy 0,80, macro-F1 0,80243475, balanced accuracy 0,80696537. **No se volvió a ejecutar ese entrenamiento ni a validar el checkpoint.**
- Existen checkpoints temporales V2.2/V2.3/V2.4 y baseline V2.5. No se cargaron, reentrenaron ni usaron sus predicciones como verdad.
- PyTorch no está instalado en el entorno activo. No se afirma haber probado MPS; el trabajo ejecutado usa OpenCV/CPU.

Faltan anclas V2.3 y features V2.2 para IMG_4946.MOV, IMG_5071.MOV, IMG_5104.MOV y e9ba3304-dcd0-43d0-84bf-52a746df196b.mp4. Los vídeos sí existen. Esos recursos serían necesarios para reutilizar directamente el antiguo pipeline en esas secuencias; **no se solicita grabar ni anotar más**. Tampoco existe validación independiente densa de locomoción ni ground truth de bienestar.

## Implementación retenida

1. **Seguimiento guiado y corrección offline.** Se conservan cajas originales. Se inicializan puntos en el interior de cada ancla; se siguen las mismas identidades entre fotogramas consecutivos con comprobación ida/vuelta. Solo puntos que sobreviven hasta el ancla final y terminan dentro de ella permiten proponer un centro corregido. Se preservan ambos extremos. Sin seis correspondencias persistentes, no se aplica corrección.
2. **Movimiento visual independiente del centro de caja.** Se siguen puntos dentro de guías consecutivas, sin usar la velocidad interpolada como etiqueta. Aún pueden incluir fondo.
3. **Cámara local.** Ajuste afín RANSAC sobre entorno exterior al conejo. Una de cada cuatro correspondencias se reserva para evaluar el error del modelo, sin participar en el ajuste. Se rechazan modelos insuficientes o geométricamente inverosímiles.
4. **Agregación temporal.** Vectores compensados en intervalos de 0,5 s; banda de error basada en cámara, flujo ida/vuelta y dispersión corporal. Cobertura insuficiente o banda que cruza el umbral produce desconocido. La banda y calidad son diagnósticos, no intervalos/probabilidades calibrados.
5. **Reproducibilidad.** Semilla 0, un hilo OpenCV, versiones y SHA-256 de entradas/código. CSV, cronología JSON, vídeo y comparación se generan mediante CLI o tareas VS Code.

El umbral de locomoción sigue en 0,2 anchos de caja/s. Las constantes geométricas/de calidad siguen siendo heurísticas experimentales; no se presentaron como parámetros científicamente calibrados.

### Iteraciones rechazadas

- Propagación de máscara rígida: pérdida de puntos por articulación y deriva. Descartada.
- GrabCut con guías: colapsaba a fondo o confundía pelaje y suelo. Con semilla central, IoU reservado principal cayó a **0,249** frente a 0,441. Descartado.
- Puente con puntos redetectados en cada fotograma: IoU principal llegó a **0,587**, pero la inspección visual descubrió una corrección hacia la silla a los 31 s y regresiones importantes en controles. **Ese 0,587 no corresponde al método entregado.** Sustituido por identidad persistente hasta el ancla final; a los 31 s se conserva la guía sin esa corrección falsa.

## Evaluación geométrica final

Se retienen extremos y anclas alternas; las anclas interiores impares se reservan **antes** de interpolar. No se usa el CSV derivado para reconstruir guías de evaluación. Todas las propuestas se comparan con las mismas anclas reservadas; ausencia de caja puntúa IoU 0. El error de centro se normaliza por ancho de la caja de referencia.

| Vídeo | Anclas reservadas | IoU interpolación | IoU puente persistente | Error centro interpolación | Error centro puente |
|---|---:|---:|---:|---:|---:|
| IMG_5177 | 10 | 0,440947 | **0,461935** | 0,359667 | **0,190276** |
| 6BE6… / reposo | 4 | 0,753477 | 0,747592 | 0,020331 | 0,028946 |
| AE22… / desplazamiento | 3 | 0,753616 | 0,714487 | 0,032919 | 0,108955 |

Todas las cajas comparadas están disponibles. En IMG_5177 mejora el centro mediano y ligeramente la superposición; los controles **regresan**. Se trata de diagnóstico sobre vídeos de desarrollo, con anclas aproximadas y sin intervalos de confianza estadísticos; no prueba generalización. Por ello la corrección queda como propuesta separada, no reemplazo automático de las guías.

## Movimiento: comparación final

Rejilla común: 84 muestras en IMG_5177, 36 en reposo y 28 en desplazamiento. Los controles reales están a **12 FPS**, no 30. El nuevo exportador incluye además el intervalo final parcial; este no se cuenta en la comparación común. Se incluye la muestra inicial sin duración para reproducir contadores históricos; la cobertura temporal ponderada la excluye. Las anclas se evalúan en el fotograma decodificado más cercano y el informe registra ambos tiempos.

| Vídeo | Método | Locomoción candidata | Poco movimiento | Desconocido |
|---|---|---:|---:|---:|
| IMG_5177 | V3.6 | 20 | 43 | 21 |
| IMG_5177 | V3.7 | 2 | 43 | 39 |
| IMG_5177 | flow | 7 | 42 | 35 |
| Reposo | V3.6 | 1 | 17 | 18 |
| Reposo | V3.7 | 0 | 17 | 19 |
| Reposo | flow | 0 | 12 | 24 |
| Desplazamiento | V3.6 | 6 | 20 | 2 |
| Desplazamiento | V3.7 | 0 | 17 | 11 |
| Desplazamiento | flow | 2 | 17 | 9 |

En el tramo 30–41 s de IMG_5177: V3.6 = **6/6/11**, V3.7 = **1/6/16**, flow = **7/8/8** (locomoción/poco/desconocido). Menos abstención no equivale a mayor exactitud.

La cifra histórica V3.7 1/43/40 no se reproduce en este entorno: se obtiene 2/43/39. Se informa lo ejecutado, no se fuerza el resultado antiguo cambiando umbrales.

### Etiquetas de comportamiento existentes

Evaluación binaria únicamente de clips `reviewed=si` con `locomotion` o `resting`. No se considera exploración como clase negativa: puede coexistir con desplazamiento. Solo se votan intervalos totalmente contenidos en el clip, ponderados por duración. Se exige mayoría estricta, incluyendo abstenciones; empates producen desconocido. No se asigna la etiqueta del clip a todos sus fotogramas como verdad.

| Subconjunto | Clips | Aciertos V3.6 | Aciertos V3.7 | Aciertos flow |
|---|---:|---:|---:|---:|
| IMG_5177, locomoción | 3 | 0 | 0 | **0** |
| Control reposo | 4 | 2 | 2 | 1 |
| Control desplazamiento | 2 | 0 | 0 | 0 |

Las abstenciones cuentan como no acierto. Cada subconjunto contiene una sola clase: no se calcula balanced accuracy por vídeo. Estas etiquetas son revisiones previas, con posibles pausas internas, no una validación independiente exhaustiva. **La locomoción fiable sigue sin demostrarse.**

## Ejecuciones y artefactos

- [Suite completa](runs/suite/suite.json).
- [Comparación principal con cada ancla reservada](runs/suite/5177/comparison.json).
- [Trayectoria principal](runs/suite/5177/flow/tracks.csv).
- [Cronología principal](runs/suite/5177/flow/timeline.json).
- [Vídeo principal](runs/suite/5177/flow/overlay.mp4): 512 × 910, 30 FPS, 1.251 fotogramas; sin audio.
- Entorno: Python 3.15.0, OpenCV 4.14.0.94, NumPy 2.5.4, pandas 2.3.3, pytest 9.1.1. Instalación editable y comprobación de dependencias ejecutadas.
- **14 pruebas automatizadas pasan**: separación cámara/animal sintética, transformación afín, imagen sin textura, incertidumbre, validación/unidades de anclas, exclusión de pseudoetiquetas, puente persistente no lineal, extremos, intervalos finales y duración de vídeos de referencia.

Los artefactos voluminosos están ignorados por Git y se regeneran con el manifest. Este documento conserva los resultados observados. No se ejecutó entrenamiento de postura/temporal ni se construyó interfaz macOS; se priorizó seguimiento y medición reproducible.

### Huellas principales

- Vídeo: `3ab323f706e18eb8073088c7d0570ce21d1e740da5e12685a37497eaf673d0d7`.
- Anclas: `026886cff0a06d34c21fbe62cd0b7bfb19fdb78fb1dd7e4227dad20fb097ef58`.
- Etiquetas: `9fc173ae74a2fd4d8d3f16c5542950758565681755052609887e4031704b47ff`.