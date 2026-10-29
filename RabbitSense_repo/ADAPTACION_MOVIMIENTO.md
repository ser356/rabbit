# Adaptación sin nuevos vídeos

## Protocolo fijado antes de ejecutar esta línea

Los 17 vídeos de Cinna son desarrollo. Se reutilizan exclusivamente los vídeos de YouTube ya descargados y las etiquetas existentes, sin cambiarlas. Ningún aprobado de esta línea será un test independiente ni una garantía de generalización.

Hipótesis: las características de X-CLIP contienen apariencia y escena, además de movimiento. Se compara cada ventana real con ocho repeticiones de su fotograma central. La diferencia absoluta de embeddings normalizados elimina la respuesta idéntica a una escena estática; no garantiza invariancia al movimiento de cámara.

Experimentos cerrados:

1. Control de apariencia: X-CLIP congelado, normalizado, sin sustracción.
2. Contraste temporal: valor absoluto de la diferencia entre ventana real y fotograma central repetido.
3. Cada representación se entrena con YouTube solo o YouTube + Cinna. No se ajusta el codificador.
4. Regresión logística con C en {0,01; 0,1; 1}. Pesos equilibran clases, vídeos dentro de clase y dominios; las ventanas solapadas no multiplican el peso de un vídeo.
5. Media temporal de probabilidades: ventana de 1 o 3 predicciones consecutivas. No se usan límites de clips ni etiquetas al inferir; no se atraviesan huecos. Inferencia offline, con contexto futuro.
6. Validación externa dejando una sesión completa fuera. Selección de representación, origen, C y suavizado mediante otra validación por sesión, exclusivamente dentro del entrenamiento externo.
7. Se conservan banda 0,4/0,6 y todos los criterios de `closure`. No se cambian etiquetas, requisitos, métricas ni informes históricos.
8. Se publican selección anidada, ablaciones de todos los candidatos y predicciones fuera de pliegue. Las ablaciones no sustituyen el resultado de la selección anidada.

Antes de evaluar, probar controles sintéticos: un vídeo constante produce contraste cero; exclusión de la sesión externa e interna; invariancia de predicciones externas a sus etiquetas; suavizado sin cruzar huecos; salida compatible con las métricas históricas.

Límites: YouTube tiene etiquetas ruidosas; únicamente tres vídeos de Cinna contienen ambas clases; siete sesiones agrupadas conservadoramente. Las decisiones metodológicas ya están influidas por resultados anteriores de estos vídeos. La validación anidada evita fuga directa durante selección, no convierte datos reutilizados en un test nuevo.

## Segunda hipótesis, fijada antes de evaluar el contraste

Las etiquetas son por clip: locomoción significa que existe desplazamiento, no que todos sus fotogramas se desplacen. El entrenamiento anterior asignaba locomoción a todas las ventanas contenidas en un clip positivo, aunque incluyera pausas. Se añade aprendizaje de instancias múltiples (MIL): cada clip es una bolsa; su logit es el máximo de los logits de sus ventanas. Una bolsa positiva exige al menos una ventana positiva; una negativa penaliza cualquier ventana positiva. No se inventan etiquetas por fotograma.

MIL reutiliza las dos representaciones, los dos orígenes, los tres valores de regularización y los dos suavizados anteriores. Pesos por vídeo/clase/dominio a nivel de bolsa; escalado calculado solo en entrenamiento, Adam determinista, 300 pasos, learning rate 0,03. Selección MIL separada y comparación supervisado/MIL también dentro de los pliegues internos. Se conservan todos los fallos y no se cambia ningún criterio después de ver resultados.

## Resultado de contraste y MIL

48 configuraciones, ninguna cumple los criterios. Selección anidada: 11/25 desplazamientos, 21/32 reposos, 11 reposos con falsas alarmas, AUC por clip 0,64. Los resultados completos y predicciones están en `cierre/adapt_motion`.

## Tercera hipótesis, fijada tras ese resultado y antes de evaluar fusión

La respuesta temporal no separa movimiento de cámara y desplazamiento. Se reutilizan las medidas físicas existentes, sin sus decisiones o reglas seleccionadas con etiquetas: velocidad corporal relativa al suelo, límite inferior/superior, fracción válida, calidad de evidencia, error de cámara, fracción de máscara y cambio de elongación/área. Magnitudes no negativas se transforman con log1p; valores ausentes se codifican con cero más un indicador de ausencia, no como prueba de reposo.

Se comparan medidas físicas solas y su concatenación con cuatro probabilidades YouTube (apariencia/contraste × supervisado/MIL, C=0,1 fijo). Esos cuatro modelos nunca entrenan con Cinna. La fusión entrena exclusivamente con las sesiones de Cinna permitidas por cada pliegue. C en {0,01; 0,1; 1}, pérdidas supervisada/MIL, suavizado 1/3; 24 candidatos. Selección anidada por sesión y criterios idénticos. Esta ampliación es exploratoria, posterior al resultado anterior, y no reemplaza ni borra ese fallo.

## Resultado de fusión lineal y cuarta hipótesis

Las 24 configuraciones fallan. Selección anidada: 13/25 desplazamientos, 22/32 reposos, 9 reposos con falsas alarmas. Se reprodujeron ambos veredictos desde las predicciones guardadas y se verificaron hashes y exclusión de sesiones. Una advertencia numérica de sklearn al escalar columnas constantes quedó corregida eliminándolas solo a partir del entrenamiento; la repetición conserva los resultados.

Último contraste de esta batería: las medidas físicas y su fiabilidad requieren interacciones, no una suma lineal. Se añaden 128 características aleatorias de Fourier para aproximar un kernel RBF (`RBFSampler`, semilla 0, gamma = 1/número de características variables). Solo se ajusta la cabeza, con las mismas C, pérdidas, pesos y suavizados. Se evalúan 24 candidatos adicionales; la selección externa compara lineal/no lineal exclusivamente mediante validación interna. Esta fase posterior no se presenta como confirmación independiente.

La primera ejecución RBF no selecciona ningún candidato no lineal. Con C≤0,1 varios quedan enteramente en abstención: AUC hasta 0,74 no se convierte en decisiones. Las características RBF tienen norma aproximadamente 1, frente a la escala de múltiples variables estandarizadas del modelo lineal. Se amplía únicamente C de RBF a {10; 100}, 16 candidatos adicionales, seleccionados otra vez solo dentro de cada entrenamiento externo. Se conserva la ejecución previa en `cierre/adapt_nonlinear` y la ampliación se escribe aparte, sin sustituir el resultado.

## Resultado final: no resuelto

**112 configuraciones distintas ejecutadas: 48 temporales y 64 físicas/fusión, contando ampliaciones sin duplicar repeticiones. Ninguna cumple los criterios.** No se modificaron etiquetas, banda, criterios, modelos históricos ni el comportamiento predeterminado.

| Selección exclusivamente dentro del entrenamiento | Locomoción | Reposo | Reposos con alarma | AUC clip |
|---|---:|---:|---:|---:|
| Apariencia/contraste, supervisado/MIL, YouTube/mixto | 11/25 | 21/32 | 11 | 0,64 |
| Física/fusión lineal | 13/25 | 22/32 | 9 | 0,52 |
| Física/fusión incluyendo RBF, C≤1 | 13/25 | 22/32 | 9 | 0,52 |
| Física/fusión incluyendo RBF, C≤100 | 14/25 | 19/32 | 13 | 0,57 |

Estos son resultados agregados fuera de pliegue por sesión, no los mejores candidatos elegidos tras ver el resultado externo. Los informes conservan también todas las ablaciones. `resting_false_events` cuenta **clips de reposo con alguna alarma**, no el número total de episodios ni una tasa por hora.

El candidato exploratorio más equilibrado de la primera batería alcanza 15/25 desplazamientos y 23/32 reposos, pero elegirlo mirando esta tabla sería selección sobre la evaluación externa: no sustituye al 11/25 y 21/32 de la selección anidada. Las variantes con 24/25 desplazamientos aciertan solo 4/32 reposos. Las que aciertan 30/32 reposos detectan apenas 6/25 desplazamientos. No hay solución escondida en una sola cifra alta.

El desajuste entre etiquetas por clip y entrenamiento por ventana existe, y MIL lo aborda, pero corregirlo no basta. La sustracción estática tampoco separa por sí sola el movimiento de cámara. Las interacciones no lineales no rescatan las medidas disponibles. **No se demuestra imposibilidad matemática; se descartan estas recetas con estos datos.** No se despliega ningún candidato como detector fiable, ni se promete que iterar indefinidamente sobre los mismos 57 clips produzca generalización.

### Artefactos y controles

- [Contraste y MIL](cierre/adapt_motion/report.json), [fusión lineal](cierre/adapt_fusion/report.json), [RBF inicial](cierre/adapt_nonlinear/report.json), [RBF ampliado](cierre/adapt_nonlinear_extended/report.json).
- Cada informe contiene criterios, resultados por vídeo, predicciones en su subcarpeta, configuración seleccionada por sesión, métricas internas, pesos, hashes de datos/código y copia del módulo utilizado.
- Los hashes de fuentes anteriores corresponden a sus copias archivadas dentro del informe, no a versiones posteriores del módulo. Los archivos de datos y etiquetas permanecen idénticos.
- 101 pruebas pasan, 1 omitida; seis avisos preexistentes de pandas en pruebas de `handheld`. Ningún aviso de escalado en las ejecuciones finales.
- Pruebas nuevas: contraste cero en vídeo estático, separación de clases no lineal, entrenamiento MIL con pausas, pesos por vídeo/clase, límites temporales, constantes numéricas, serialización y predicciones invariantes al cambiar las etiquetas de la sesión excluida.
- Los veredictos se recalculan a partir de las predicciones serializadas, no solo de la salida de consola.

### Reproducción

Desde la raíz del workspace, usar el entorno existente `RabbitSense_Training/.venv` con torch, transformers y scikit-learn. Se añadió scikit-learn como extra opcional `adapt`; pytest se instaló en ese entorno para ejecutar también las pruebas con torch.

Tareas VS Code:

- **RabbitSense: pruebas adaptación temporal**: suite completa.
- **RabbitSense: completar evaluación temporal**: 48 candidatos sobre características ya extraídas.
- **RabbitSense: fusión física y YouTube**: fusión lineal, 24 candidatos.
- **RabbitSense: fusión no lineal validada por sesión**: fusión con RBF ampliado, 64 candidatos.

Para regenerar características, sin descargar vídeos:

```bash
PYTHONPATH=RabbitSense_repo/src PYTORCH_ENABLE_MPS_FALLBACK=1 \
RabbitSense_Training/.venv/bin/python -u -m rabbitsense.adapt_motion embed \
	--manifest RabbitSense_repo/benchmarks/pose_desarrollo.json \
	--labels RabbitSense_ARCHIVO/etiquetas_youtube.csv \
	--features RabbitSense_repo/runs/adapt_motion/features \
	--youtube-features RabbitSense_repo/runs/webclips/youtube \
	--youtube-dir RabbitSense_Training/data/youtube/youtube_conejos \
	--youtube-dir RabbitSense_Training/data/youtube/youtube_conejos_2 --device mps
```

La extracción utiliza vídeos completos de Cinna y exactamente los instantes existentes de YouTube. Inferencia offline, no causal. No usa nombres de vídeo, etiquetas ni límites de clip para construir características visuales.