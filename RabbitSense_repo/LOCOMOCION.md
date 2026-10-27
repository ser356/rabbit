# Locomoción: diagnóstico, corrección temporal y cierre

## Línea experimental nueva: adaptación del modelo temporal a Cinna (pre-registrada el 2026-10-27)

Esta línea no modifica los veredictos `not_resolved` anteriores ni sus modelos congelados. Es una evaluación exploratoria con validación cruzada agrupada, no un test independiente.

### Auditoría de datos

Los datos son los 17 vídeos de [pose_desarrollo.json](benchmarks/pose_desarrollo.json) con sus etiquetas verificadas ([adapt.py](src/rabbitsense/adapt.py) `audit`). La auditoría con vídeo y máscaras se genera en el Mac, en `cierre/adapt/auditoria.json`.

| | Locomoción | Reposo |
|---|---:|---:|
| Clips verificados de 4 s | 25 (97 s) | 32 (126 s) |
| Vídeos con la clase | 9 | 10 |
| Sesiones con la clase (agrupación conservadora) | **4** | **5** |
| Ventanas cada 0,5 s (solapadas, abarcan 1,75 s) | 140 | 137 |
| Ventanas sin fotogramas compartidos | 56 | 53 |

- **Unidad independiente.** Las 277 ventanas solapadas no son 277 ejemplos. El máximo sin solapamiento es 109, y tampoco son independientes, porque las de un mismo vídeo comparten escena, luz y cámara.
- **Sesiones.** Se agrupan los 8 vídeos de WhatsApp en una sesión y IMG_5190–5193 en otra, porque pueden compartir sala y momento. Con esa agrupación, la locomoción procede de solo 4 sesiones y el reposo de 5.
- **Vídeos con etiquetas.** Hay 16; IMG_5193 no tiene clips verificados. IMG_5190 aporta el 28 % de los clips.
- **Confusión escena–etiqueta.** 13 de los 16 vídeos, y 5 de las 7 sesiones, tienen una sola clase. Solo IMG_5190, IMG_5191 y WhatsApp 18.44.34 tienen las dos. Un modelo entrenado con Cinna puede acertar reconociendo la escena en lugar del desplazamiento. Por eso se informa también de la AUC intra-vídeo en esos tres vídeos.
- **Precisión alcanzable.** Con 25 clips, una sensibilidad observada de 0,8 tiene un intervalo de Wilson del 95 % de 0,61 a 0,91. Para que el límite inferior supere 0,8 hacen falta 54 clips si se acierta el 90 %, o 25 clips si se acierta el 95 %.

**Conclusión de la auditoría.** Hay información suficiente para un experimento exploratorio con pocos parámetros por pliegue: sondas lineales sobre X-CLIP congelado y un ajuste parcial con hiperparámetros fijados de antemano. No la hay para declarar resuelta la locomoción, salvo con resultados casi perfectos: unos 24/25 desplazamientos y 31/32 reposos. Ajustar el modelo completo no es viable: son millones de parámetros frente a unas 109 ventanas disjuntas de 4–5 sesiones.

### Protocolo (fijado antes de ejecutar)

1. **Caché.** Fotogramas cada 0,25 s, con el lado largo a 448 px, y la caja de la máscara SAM2 de cada fotograma.
2. **Dos vistas por ventana de 8 fotogramas.** La imagen completa con bandas y un recorte del conejo con entorno: la unión de las cajas de la ventana, ampliada medio tamaño por lado y cuadrada. Así se conservan el suelo y el fondo cercanos, que delatan el desplazamiento aunque la cámara siga al conejo.
3. **Sondas sobre X-CLIP congelado.**
   - Métodos: `lr_full` (la regresión logística actual, reentrenada con Cinna), `lr_crop`, `lr_full_crop` y, opcionalmente, `lr_full_youtube` (YouTube más Cinna con igual peso total).
   - La L2 se elige entre {1, 10, 100} con validación interna agrupada, dentro de cada pliegue.
   - Los modelos YouTube v1 y v2 congelados se incluyen solo como referencia.
4. **Ajuste parcial** (`finetune_two_view`).
   - Se entrenan la última capa del codificador visual de X-CLIP, su normalización final, la proyección y el MIT, más una capa lineal sobre las dos vistas.
   - Hiperparámetros: 4 épocas, lote 8, AdamW con lr 2e-5 para el codificador y 1e-3 para la cabeza, weight decay 0,05 y semilla 0.
   - Muestreo equilibrado por clase. Aumentos: volteo, brillo y contraste, y desplazamiento del recorte.
   - Un modelo nuevo por pliegue; no se guardan pesos.
5. **Validación cruzada.**
   - Por vídeo (deja uno fuera) y, más estricta, por sesión.
   - Estandarización, L2 y pesos salen solo del entrenamiento del pliegue.
   - Solo se entrena con ventanas contenidas enteras en clips verificados.
   - El vídeo evaluado se clasifica completo con la misma rejilla y la misma banda 0,4/0,6 que el cierre.
6. **Métricas.**
   - Detección por clip con `CRITERIA`: sensibilidad, especificidad y episodios falsos, con intervalos de Wilson.
   - AUC por ventana, por clip e intra-vídeo.
   - Cobertura: fracción de clips y de segundos con decisión.
7. **Diagnóstico.**
   - Vídeos con la caja del recorte, p(locomoción) y franjas de verdad y predicción, para el mejor método por vídeo.
   - Cada error se clasifica por la primera causa que se cumple:
     1. pérdida de seguimiento (menos del 50 % de fotogramas con máscara);
     2. abstención;
     3. transición entre clases;
     4. cámara en movimiento (más de 0,02 del ancho por cada 0,25 s);
     5. conejo pequeño (menos del 1 % del área);
     6. desplazamiento corto (menos de un cuerpo dentro del clip);
     7. otra.

### Resultado (ejecutado en el Mac M4, 2026-10-27)

El código y el manifest de [finetune.json](cierre/adapt/finetune.json) coinciden por hash con los subidos (`bdc61090…` y `63aa160e…`). Las predicciones están en [cierre/adapt](cierre/adapt/). En la tabla, «intra-vídeo» es la AUC por clip dentro de los tres vídeos con ambas clases: IMG_5190, IMG_5191 y WhatsApp 18.44.34.

| Método (validación cruzada) | Locomoción | Reposo | Reposos con alarma | AUC clip | Intra-vídeo |
|---|---:|---:|---:|---:|---|
| `lr_full`, por vídeo | 20/25 | 12/32 | 18 | 0,67 | 0,56 / 0,11 / 0,00 |
| `lr_full`, por sesión | 22/25 | 13/32 | 17 | 0,72 | 0,76 / 0,67 / 0,00 |
| `lr_full_crop`, por vídeo | 20/25 | 11/32 | 21 | 0,64 | 0,63 / 0,67 / 0,00 |
| `lr_full_youtube`, por vídeo | 19/25 | 13/32 | 18 | 0,73 | 0,48 / 0,44 / 1,00 |
| `finetune_two_view`, por vídeo | 21/25 | 13/32 | 19 | 0,61 | 0,56 / 1,00 / 0,00 |
| Referencia: YouTube v1 congelado | 15/25 | 23/32 | 9 | 0,64 | 0,51 / 0,33 / 1,00 |
| Referencia: YouTube v2 congelado | 11/25 | 25/32 | 7 | 0,72 | 0,49 / 0,78 / 1,00 |

- **Ningún método pasa.** Los entrenados con Cinna suben la sensibilidad a costa de la especificidad. La AUC por clip (0,61–0,73) no mejora la de los modelos YouTube congelados (0,64–0,72): es el mismo poder de separación con otro umbral.
- **Recorte del conejo.** Solo, empeora (AUC 0,52–0,56). Combinado con la imagen completa, no aporta.
- **Ajuste parcial de X-CLIP.** No supera a la sonda lineal.
- **Confusión escena–etiqueta confirmada.** Dentro de los vídeos con ambas clases, el orden es casi aleatorio o invertido: el modelo distingue escenas, no desplazamiento.
- **Errores.** Con `lr_full_youtube`, 18 reposos tienen alarmas sin causa geométrica clara (10 «otra», 6 en transiciones). De las locomociones perdidas, 4 son desplazamientos cortos, de menos de un cuerpo.
- **Cobertura de máscara.** IMG_5191 tiene máscara solo en el 46 % de los fotogramas etiquetados; el resto de vídeos, el 82–100 %.

La batería posterior de contraste temporal, MIL y fusión física, con 112 configuraciones, tampoco pasa; está en [ADAPTACION_MOVIMIENTO.md](ADAPTACION_MOVIMIENTO.md). **Esta línea queda como no resuelta.**

## Estado final (2026-10-27): cerrada, no resuelta

| Conjunto | Enfoque | Naturaleza | Resultado |
|---|---|---|---|
| 5 vídeos de desarrollo | Flujo óptico (V3.6/V3.7/flow/body/SAM2) | Veredicto | Ningún método pasa |
| **IMG_5190–5193, cámara en mano** | **Flujo óptico** | **Veredicto, test independiente** | **Ningún método pasa** |
| **IMG_5190–5193, cámara en mano** | **X-CLIP zero-shot (preentrenado)** | **Veredicto, test independiente** | **No pasa** |
| 9 vídeos (validación cruzada) | Suelo + silueta SAM2 | Desarrollo | Sensibilidad validada 0,59 |
| **Vídeos nuevos de WhatsApp** | **Suelo + silueta SAM2** | **Veredicto, test independiente** | **No pasa: 4/8 locomoción, 6/10 reposo, 1 episodio falso** |
| 17 vídeos (validación cruzada) | Marcha, DeepLabCut SuperAnimal | Desarrollo | Señal ≈ azar (AUC 0,60); reposo 3/32 |
| **Los 17 vídeos de Cinna** | **Clasificador entrenado con YouTube** | **Veredicto, test independiente** | **No pasa: 14/25 locomoción, 22/32 reposo, 10 episodios falsos** |
| **Los 17 vídeos de Cinna** | **Clasificador YouTube v2 (2,6× más datos)** | **Veredicto, segunda mirada al test** | **No pasa: 11/25 locomoción, 24/32 reposo, 8 episodios falsos** |

Cinco enfoques distintos fallan:

- **Flujo óptico:** falla en la compensación de cámara y pierde el seguimiento.
- **X-CLIP zero-shot:** no discrimina.
- **Marcha con DeepLabCut:** los puntos de las patas son demasiado inestables en estos planos.
- **Suelo + silueta:** SAM2 pierde a Cinna, los desplazamientos lentos quedan bajo cualquier umbral y la máscara salta en reposo. En los vídeos nuevos de WhatsApp es el más equilibrado (1 episodio falso), pero detecta solo la mitad de los desplazamientos.
- **Clasificador entrenado con YouTube:** es el que mejor generaliza a los 17 vídeos (sensibilidad 0,56, especificidad 0,69), pero queda lejos del 0,8 y da 10 episodios falsos. Con 2,6 veces más clips de locomoción (v2), mejora dentro de YouTube (AUC 0,70 → 0,78), pero no en Cinna: el reposo sube a 0,75 y la sensibilidad baja a 0,44. **La locomoción queda cerrada como no resuelta.** Reabrirla exige vídeos con locomoción etiquetada en cantidad suficiente, al menos decenas de clips, para desarrollar y validar con datos que no se hayan usado.

## Reapertura: suelo + silueta con vídeos nuevos (pre-registrado el 2026-10-26)

**Por qué otro intento.** SAM2 sigue a Cinna, pero su posición en la imagen mezcla su movimiento y el de la cámara: si la cámara la sigue, parece quieta; si la cámara gira, parece moverse. Los métodos anteriores estimaban la cámara con todo lo que rodea la caja, muebles incluidos, y con la caja, que contiene suelo. El método nuevo ([silhouette.py](src/rabbitsense/silhouette.py)):

1. **Suelo.** Homografía RANSAC solo con puntos del anillo que rodea la máscara, entre 0,15 y 1,0 × √área desde su borde. Exige 20 puntos en al menos 3 cuadrantes, validación con puntos reservados (≤ 3 px) y una deformación plausible.
2. **Conejo.** Centroide de la **máscara** frente al centroide anterior proyectado por esa homografía, normalizado por √área. Se agrega con el diámetro de trayectoria y los errores en cuadratura de `flow`; umbral 0,2/s.
3. **Silueta.** Rango de elongación y de log-área de la máscara por intervalo: cambian al saltar aunque la cámara se mueva. Se registran como características para el desarrollo.

Con vídeo sintético, si la cámara se desplaza y el conejo está quieto, la velocidad en imagen es 0,6/s y la medida respecto al suelo 0,0/s. Si el conejo avanza y la cámara lo sigue, en imagen es 0,0/s y respecto al suelo 0,6/s.

**Protocolo: test mixto (opción 1, decidida el 2026-10-26 antes de ver resultados).** Los vídeos nuevos son casi todos de reposo, así que:

- **Desarrollo:** los 9 vídeos ya usados ([silueta_desarrollo.json](benchmarks/silueta_desarrollo.json)), incluidos IMG_5190–5193, que dejan de ser test.
- **Reglas candidatas, lista cerrada** (`CANDIDATES`, 28):
  - `floor`: solo suelo, umbral en 0,15 / 0,2 / 0,3 / 0,4;
  - `floor_shape`: además exige cambio de elongación de 0,05 / 0,1 / 0,2 para declarar locomoción;
  - `floor_or_shape`: si el suelo no decide, un cambio de elongación de 0,1 / 0,2 / 0,3 con máscara completa declara locomoción.
- **Selección:** maximizar min(sensibilidad, especificidad) de detección por clip; después, menos episodios falsos; después, la primera de la lista (la más simple).
- **Validación cruzada por vídeo:** para cada vídeo de desarrollo se elige la regla con los otros 8 y se evalúa en él. La sensibilidad agregada de esos vídeos excluidos es la **sensibilidad validada**. La regla final se elige con los 9 y se congela en `benchmarks/silueta_regla.json` (`silhouette develop`), commiteada antes de procesar los vídeos nuevos con cualquier método de locomoción. Sin ese archivo, `--decisive` no se ejecuta.
- **Test decisivo:** vídeos nuevos, etiquetados desde cero (`handheld grid` + `revision.html`), verificados y commiteados antes de ejecutar.
  - **Especificidad** siempre en los vídeos nuevos: al menos 10 clips de reposo y ≥ 0,8.
  - **Sensibilidad** en los vídeos nuevos si tienen al menos 5 clips de locomoción; si no, la validada. En ambos casos ≥ 0,8, al menos 5 clips y un acierto por vídeo con locomoción.
- **Estados:**
  - `resolved_independent_test`;
  - `resolved_specificity_independent_sensitivity_cross_validated`: conclusión más débil, porque la sensibilidad viene de vídeos de desarrollo aunque cada uno se evalúe con una regla elegida sin él;
  - `not_resolved`;
  - `descriptive_only_insufficient_sample`: menos de 10 clips de reposo nuevos.
- **Una ejecución:** si el test falla, no se ajusta la regla sobre esos vídeos.

## Resultado del segundo intento con YouTube (2026-10-27)

Modelo congelado en [webclips_modelo_v2.json](benchmarks/webclips_modelo_v2.json) (commit `6560485`) antes del test; su `labels_sha256` (`f3193194…`) coincide con [etiquetas_youtube.csv](../RabbitSense_ARCHIVO/etiquetas_youtube.csv). Test ejecutado una vez sobre las mismas características de Cinna.

| | v1 (536 clips, 74 vídeos) | v2 (1.702 clips, 228 vídeos) |
|---|---:|---:|
| Ventanas de entrenamiento | 965 | 2.490 |
| AUC con validación cruzada en YouTube | 0,70 | **0,78** |
| Acierto en YouTube (sobre lo decidido) | 0,64 | 0,71 |
| **Cinna: locomoción** | 14/25 (0,56) | **11/25 (0,44)** |
| **Cinna: reposo** | 22/32 (0,69) | **24/32 (0,75)** |
| **Cinna: episodios falsos** | 10 | 8 |

- **Estado `not_resolved`.** Por protocolo, no hay tercer reentrenamiento.
- **Más datos mejoran el modelo en YouTube, pero no se transfieren a Cinna.** El AUC dentro de YouTube sube 0,08, y en Cinna solo se desplaza el compromiso: algo más de reposo y menos locomoción. Con 25 y 32 clips, ninguna de las dos diferencias frente a v1 es significativa.
- **Fallo sistemático en el vídeo de locomoción.** En `locomotion` (IMG_5177) v2 declara locomoción en solo 2 de 29 intervalos, frente a 4 con v1. Los desplazamientos lentos y cortos de una belier en interior se parecen poco a las carreras y saltos que dominan los clips de YouTube.
- **Conclusión.** El cuello de botella ya no es la cantidad de datos de YouTube, sino la distancia entre ese dominio y los vídeos de Cinna. Lo que separa clases en YouTube no basta para Cinna. Mejorar haría falta ejemplos etiquetados de la propia Cinna para entrenar, y entonces habría que reservar otros para el test.

## Segundo intento con YouTube: mismo método, más datos (pre-registrado el 2026-10-27)

**Por qué.** En YouTube la validación cruzada solo llega a AUC 0,70 con 82 clips de locomoción. La hipótesis es que el límite son los datos, no el método, así que solo cambian los datos.

**Qué cambia:**

- Se descargan más vídeos con [youtube_busquedas_2.txt](benchmarks/youtube_busquedas_2.txt): 20 búsquedas orientadas a grabaciones caseras continuas con desplazamiento. Incluye búsquedas en japonés (部屋んぽ, «paseo por la habitación»), francés, alemán, coreano y español.
- Se reutiliza el archivo de descargas de yt-dlp para no repetir vídeos.
- Las etiquetas nuevas se añaden a [etiquetas_youtube.csv](../RabbitSense_ARCHIVO/etiquetas_youtube.csv) con los mismos criterios, mirando solo las hojas de contacto.

**Qué no cambia:** características X-CLIP, regresión logística (L2 = 1, clases equilibradas), banda 0,4/0,6, `CRITERIA` y los 17 vídeos de Cinna con sus etiquetas.

**Protocolo:**

1. El modelo se entrena con todas las etiquetas de YouTube (las de antes y las nuevas). Se congela en `benchmarks/webclips_modelo_v2.json` y se commitea antes del test. El modelo anterior queda intacto.
2. Hay una sola ejecución del test sobre las mismas características de Cinna (`runs/webclips/cinna`), con `--model` apuntando al v2.
3. **Límite de la conclusión.** Es la segunda mirada al mismo conjunto de test tras un fallo. Solo cambian los datos de YouTube, que no se eligen mirando a Cinna, pero un aprobado sería más débil que uno a la primera y se documentaría así.
4. Si falla, no hay tercer reentrenamiento con YouTube.

**Etiquetas del segundo lote (2026-10-27).** Se descargaron 154 vídeos nuevos, sin repetir ninguno, y se etiquetaron sus 1.166 clips con los mismos criterios mirando solo las hojas de contacto. Se añaden a [etiquetas_youtube.csv](../RabbitSense_ARCHIVO/etiquetas_youtube.csv).

| | Locomoción | Reposo | Desconocido | Vídeos útiles |
|---|---:|---:|---:|---:|
| Primer lote (74 vídeos) | 82 | 127 | 327 | 58 |
| Segundo lote (154 vídeos) | 136 | 192 | 838 | 94 |
| **Total** | **218** | **319** | **1.165** | **152** |

En el segundo lote, el 72 % de los clips es `unknown`. Hay muchas animaciones, imágenes generadas por IA, videoclips sin conejo, varios conejos a la vez y planos demasiado cercanos. Las búsquedas en japonés (部屋んぽ) aportaron sobre todo vídeos con texto superpuesto.

## Resultado del clasificador entrenado con YouTube (2026-10-27)

Ejecutado en el Mac (MPS). Modelo congelado en [webclips_modelo.json](benchmarks/webclips_modelo.json) (commit `dc09ec9`) antes de procesar a Cinna; su `labels_sha256` coincide con [etiquetas_youtube.csv](../RabbitSense_ARCHIVO/etiquetas_youtube.csv).

| | Locomoción | Reposo | Episodios falsos |
|---|---:|---:|---:|
| Validación cruzada en YouTube (58 vídeos, 965 ventanas) | AUC 0,70 | acierto 0,64 en el 96 % decidido | — |
| **Test independiente: los 17 vídeos de Cinna** | **14/25 (0,56)** | **22/32 (0,69)** | **10** |

- **Estado `not_resolved`.** Ni la sensibilidad ni la especificidad llegan a 0,8. Por protocolo, no se reentrena ni se ajusta la banda mirando a Cinna.
- **Es el mejor resultado en un test amplio.** La especificidad de 0,69 sobre 32 reposos de 17 vídeos supera la de cualquier método anterior en un conjunto comparable: 3/32 con la marcha y 5/10 con flujo en desarrollo. Aun así, deja 10 episodios falsos.
- **Errores por vídeo.** En `locomotion` (IMG_5177) declara bajo movimiento en la mayoría de los intervalos (19 frente a 4), y en `resting` declara locomoción en 18 de 36. El modelo responde sobre todo a la escena y al encuadre, no al desplazamiento.
- **Límite de los datos.** Con 82 clips de locomoción ruidosos, el modelo solo separa las clases en YouTube con AUC 0,70. Esto concuerda con un rendimiento medio en Cinna: no faltan reglas, faltan ejemplos limpios de desplazamiento.

## Quinto enfoque: clasificador entrenado con vídeos de YouTube (pre-registrado el 2026-10-27)

**Idea.** Los vídeos caseros de conejos en YouTube se parecen a los de Cinna: móvil en mano, interiores y razas variadas. Un clasificador aprendido con ellos puede separar desplazamiento de movimiento de cámara sin reglas hechas a mano. **Ninguno de los 17 vídeos de Cinna se usa para entrenar ni ajustar nada**, así que todos son test independiente: 25 clips de locomoción y 32 de reposo, verificados por una persona.

**Método** ([webclips.py](src/rabbitsense/webclips.py)):

1. **Datos:** descarga en el Mac con `yt-dlp` usando las búsquedas fijas de [youtube_busquedas.txt](benchmarks/youtube_busquedas.txt): 6 vídeos por búsqueda, de 15 a 180 s, hasta 480p.
2. **Etiquetas de entrenamiento:** hasta 8 clips de 4 s repartidos por vídeo. Las etiqueta Claude mirando hojas de contacto de 8 fotogramas (cada 0,5 s): `locomotion` si cambia de sitio sobre el suelo, `resting` si no se desplaza, `unknown` si hay dudas u oclusión. Son **etiquetas ruidosas solo para entrenar**; las de test son las verificadas de Cinna.
3. **Características:** X-CLIP `microsoft/xclip-base-patch32` congelado (`get_video_features`), ventanas de 8 fotogramas cada 0,25 s (de t−1,25 a t+0,5 s), imagen completa con bandas para no recortar.
4. **Clasificador:** regresión logística L2 (λ = 1) sobre características estandarizadas, con clases equilibradas. Validación cruzada de 5 bloques agrupando por vídeo de YouTube, solo informativa. Pesos congelados en `benchmarks/webclips_modelo.json` antes de procesar a Cinna.
5. **Decisión:** p ≥ 0,6 locomoción; p ≤ 0,4 bajo movimiento; entre ambas, desconocido.
6. **Test:** una ejecución sobre los 17 vídeos de Cinna ([pose_desarrollo.json](benchmarks/pose_desarrollo.json)) con `CRITERIA`: al menos 5 clips de locomoción y 10 de reposo, sensibilidad y especificidad ≥ 0,8, y al menos un acierto por vídeo con locomoción. Estado `resolved_independent_test` o `not_resolved`. Si falla, no se reentrena mirando a Cinna.

**Etiquetas de YouTube (2026-10-27):** [etiquetas_youtube.csv](../RabbitSense_ARCHIVO/etiquetas_youtube.csv), 536 clips de 74 vídeos, etiquetados por Claude mirando las hojas de contacto antes de calcular ninguna característica:

- 82 clips de `locomotion` (de 32 vídeos), 127 de `resting` (de 45 vídeos) y 327 `unknown`.
- 58 vídeos aportan al menos un clip útil.
- Van a `unknown` los cortes de plano, títulos y cartelas, el conejo sujetado o acariciado, varios conejos, el conejo diminuto o en visión nocturna, la animación, la cámara caótica y los desplazamientos dudosos.
- Comer, acicalarse, girar la cabeza o tumbarse sin desplazarse cuentan como `resting`, aunque se mueva la cámara.
- La columna `notes` explica cada decisión.

## Resultado del desarrollo de la marcha (DeepLabCut, 2026-10-27)

Ejecutado en el Mac (MPS) con el código commiteado (hashes verificados) y el modelo `superanimal_quadruped_hrnet_w32` (SHA-256 `a42e58d5…`). Salidas y mosaicos en [cierre/pose_dev](cierre/pose_dev/) y [cierre/pose_keypoints](cierre/pose_keypoints/). Regla congelada en [benchmarks/pose_regla.json](benchmarks/pose_regla.json).

| | Locomoción | Reposo | Episodios falsos |
|---|---:|---:|---:|
| **Validación cruzada por vídeo (17 vídeos)** | 13/25 (0,52) | **3/32 (0,09)** | 7 |

- **Regla elegida:** `pose_or_floor`, extremidades traseras, umbrales 0,5 / 0,05. La mejor de las 40 candidatas solo alcanza 0,09 en min(sensibilidad, especificidad), incluso sobre sus propios datos.
- **La señal de marcha casi no separa las clases:** AUC por intervalo de 0,59 (todas las extremidades) y 0,60 (traseras); 0,5 es azar. En reposo, la excursión mediana de las patas es 0,18 longitudes corporales (percentil 90: 0,43–0,50), similar a la de locomoción (mediana 0,25). El ruido de los puntos domina.
- **Causa visible en los mosaicos:** el modelo encuentra patas en vistas laterales nítidas (AE22), pero en planos cenitales, lejanos o con las patas ocultas bajo el pelo del belier las coloca de forma inestable. En IMG_5191, 972F y dos vídeos de WhatsApp apenas detecta cuerpo (cobertura 0–0,17).
- **Consecuencia:** la regla congelada casi nunca puede declarar reposo, así que la especificidad en vídeos nuevos no alcanzaría 0,8. **No se piden vídeos de test para este método:** el resultado sería `not_resolved`.

## Cuarto enfoque: marcha con DeepLabCut SuperAnimal-Quadruped (pre-registrado el 2026-10-27)

**Idea.** Al desplazarse, las patas se mueven respecto al cuerpo; en reposo no, aunque la cámara se mueva. Esa señal no depende de compensar la cámara.

**Método** ([pose.py](src/rabbitsense/pose.py)):

- **Modelo:** SuperAnimal-Quadruped `hrnet_w32` de DeepLabCut 3.0.2, preentrenado en cuadrúpedos, sin ajuste fino ni adaptación a nuestros vídeos.
- **Entrada:** modo *top-down* con la caja de la máscara SAM2 ampliada un 10 %, en lugar del detector de DeepLabCut.
- **Marcha:** posiciones de patas y rodillas con puntuación ≥ 0,4, en el eje del cuerpo (cuello → cola) y en longitudes corporales. Así no influyen la traslación, el giro ni el zoom de la cámara; comprobado en pruebas sintéticas: excursión 0 con cámara en movimiento. Por intervalo de 0,5 s se toma la mediana, entre extremidades, del diámetro de su trayectoria.
- **Candidatas, lista cerrada (40):** `pose` (solo marcha) y `pose_or_floor` (marcha combinada con la regla de suelo ya congelada); extremidades `all` o `hind` (traseras, para no confundir el acicalado con locomoción); umbral alto 0,15 / 0,2 / 0,3 / 0,4 / 0,5; umbral bajo 0,05 / 0,1.

**Protocolo.**

- **Desarrollo:** los 17 vídeos ya usados ([pose_desarrollo.json](benchmarks/pose_desarrollo.json)), incluidos los de WhatsApp, que dejan de ser test. Validación cruzada por vídeo con el mismo objetivo que en suelo + silueta; regla congelada en `benchmarks/pose_regla.json` antes de procesar vídeos nuevos.
- **Test:** vídeos nuevos aún no grabados o no usados, etiquetados desde cero y commiteados antes de ejecutar. Protocolo mixto y criterios sin cambios. Una sola ejecución.
- **Comprobación visual previa:** por cada vídeo se guarda un mosaico con los puntos detectados. Si el modelo no encuentra patas en Cinna, se verá antes de cualquier cifra.

## Test con vídeos nuevos de WhatsApp: suelo + silueta (2026-10-27)

Vídeos nuevos no usados en ningún desarrollo. Etiquetas desde cero, verificadas y commiteadas antes de ejecutar (`56feb7b`). Regla congelada antes (`5658aa1`): solo suelo, umbral 0,3. Hubo 8 clips de locomoción (≥ 5), así que la sensibilidad se midió en los vídeos nuevos: **test independiente completo**.

| Método (test independiente) | Locomoción | Reposo | Episodios falsos en reposo |
|---|---:|---:|---:|
| **Suelo + silueta, WhatsApp** | **4/8 (0,50; IC95 % 0,22–0,78)** | **6/10 (0,60; 0,31–0,83)** | **1** |
| Flujo óptico, IMG_5190–5193 | 7/12 (0,58; 0,32–0,81) | 3/12 (0,25; 0,09–0,53) | 6 |
| X-CLIP, IMG_5190–5193 | 9/12 (0,75; 0,47–0,91) | 1/12 (0,08; 0,01–0,35) | 11 |

Estado: `not_resolved`. No alcanza 0,8 en ninguna de las dos clases. Intervalos de Wilson al 95 %; los conjuntos de test difieren entre filas.

**Lectura.** Es el método más equilibrado y el único que casi no inventa desplazamientos: 1 episodio falso en 10 clips de reposo, frente a 6 y 11. Pierde la mitad de los desplazamientos y no decide en 3 de 10 reposos. Con 8 y 10 clips la incertidumbre es grande. No se ajusta nada sobre estos vídeos.

## Resultado del desarrollo de suelo + silueta (2026-10-26)

Ejecución de desarrollo en el Mac con el código commiteado (hashes verificados). Características y veredicto en [cierre/silueta_dev](cierre/silueta_dev/). La selección y la validación cruzada (`silhouette develop`) se ejecutaron sin cambios en la lista de candidatas. La regla congelada está en [benchmarks/silueta_regla.json](benchmarks/silueta_regla.json).

| | Locomoción | Reposo | Episodios falsos |
|---|---:|---:|---:|
| Regla por defecto (suelo, 0,2), 9 vídeos | 12/17 | 7/22 | 11 |
| **Validación cruzada por vídeo** | **10/17 (0,59)** | 10/22 (0,45) | 7 |

**Regla congelada:** solo suelo, umbral 0,3. Incluso sobre todos los datos de desarrollo, la mejor de las 28 candidatas solo alcanza 0,45 en min(sensibilidad, especificidad): las características no separan las clases en estos vídeos, y no es cuestión de ajustar umbrales. La sensibilidad validada no llega a 0,8 y hay vídeos de locomoción sin ningún acierto (AE22 0/2, IMG_5191 0/3).

**Consecuencia pre-registrada:** los vídeos nuevos son casi todos de reposo. Con menos de 5 clips de locomoción en ellos, la sensibilidad sale de la validación cruzada (0,59 < 0,8) y el estado es `not_resolved`, sea cual sea la especificidad. Ejecutar el test solo mediría la especificidad, como dato informativo.

**Causas observadas, clip a clip:**

1. **SAM2 pierde a Cinna:** IMG_5191 no tiene máscara entre 12 y 24 s, justo los 3 clips de locomoción.
2. **Desplazamiento lento:** AE22 avanza despacio olfateando a 0,04–0,08 tamaños/s, indistinguible del reposo con cualquier umbral razonable.
3. **Saltos de la máscara en reposo:** intervalos aislados con 0,6–3 tamaños/s y cambio grande de elongación. Compatible con la máscara ganando o perdiendo orejas, mano o sombra de golpe.

Donde la máscara es estable y hay suelo visible, sí separa reposo y desplazamiento; por ejemplo, IMG_5190 44–48 s frente a 20–36 s.

No se añaden candidatas a la vista de estos resultados, por ejemplo exigir dos intervalos consecutivos para filtrar los saltos aislados: la validación cruzada dejaría de ser honesta. Serían hipótesis para datos nuevos con locomoción etiquetada.

## Resultado X-CLIP zero-shot (2026-10-26)

Ejecutado en el Mac (MPS) según el protocolo pre-registrado (`c0aecc4`), sin cambios de frases, recorte ni umbrales.

| Conjunto | Locomoción | Reposo | Episodios falsos en reposo | Resultado |
|---|---:|---:|---:|---|
| **IMG_5190–5193 (decisivo)** | 9/12 | **1/12** | **11** | **No pasa** |

Muestras (locomoción / bajo movimiento / desconocido): IMG_5190 102/13/20, IMG_5191 18/0/31, IMG_5192 40/0/5, IMG_5193 4/8/8.

Informativo, vídeos de desarrollo: también predice locomoción en los tres controles de reposo (8, 6 y 9 muestras) y casi nunca bajo movimiento (1, 8 y 7).

**Lectura.** X-CLIP asigna locomoción por defecto. Su sensibilidad de 9/12 no tiene valor sin especificidad: marca episodios en 11 de 12 clips de reposo. Las frases sobre conejos no separan desplazamiento de reposo en estos recortes; el modelo no vio conejos domésticos con cámara en mano durante su entrenamiento. Como estaba pre-registrado, **no se prueban otras frases ni umbrales**: elegirlos viendo estos resultados convertiría el test en ajuste.

## Protocolo X-CLIP zero-shot (pre-registrado el 2026-10-26, antes de ejecutar)

El flujo óptico falla en la compensación de cámara. Se prueba un enfoque distinto: un modelo vídeo-texto ya entrenado clasifica cada ventana sin entrenamiento ni ajuste con nuestros vídeos.

- **Modelo:** `microsoft/xclip-base-patch32` (X-CLIP, Transformers). Sin ajuste fino. Se registra la revisión descargada.
- **Entrada:** 8 fotogramas cada 0,25 s, desde t−1,25 s hasta t+0,5 s, para cada intervalo de 0,5 s que termina en t. Recorte cuadrado de 2 × max(ancho, alto) de la guía, centrado en ella y escalado a 224 px. Sin guía o fuera del vídeo: desconocido.
- **Frases** (fijadas en [zeroshot.py](src/rabbitsense/zeroshot.py)):
  - locomoción: *hopping across the floor*, *walking across the room*, *running*;
  - quieto: *sitting still*, *lying down resting*, *sitting and looking around*, *grooming itself*.

  p(locomoción) = suma del softmax de las frases de locomoción.
- **Decisión:** p ≥ 0,6 locomoción; p ≤ 0,4 bajo movimiento; entre ambas, desconocido. Sin ajustar.
- **Veredicto:** IMG_5190–5193, etiquetas verificadas `2ee024c`, mismos `CRITERIA`. Los 5 vídeos de desarrollo solo son informativos. Un único método nuevo: no se prueban variantes de frases, recortes ni umbrales tras ver resultados.
- **Limitación conocida:** X-CLIP no se entrenó con conejos domésticos y clasifica apariencia y movimiento corporal; un salto en el sitio podría verse como locomoción. Quien diseña esto ha visto los resultados agregados de flow y las hojas de contacto de estos vídeos, pero ninguna salida de X-CLIP.

## Veredicto con cámara en mano sobre vídeos nuevos (2026-10-26)

Primer test independiente: ningún método ni constante se ajustó con IMG_5190–5193. Etiquetas verificadas en vídeo continuo y commiteadas antes de ejecutar (`2ee024c`, 2026-10-26 10:52). Anclas automáticas SAM2 desde una caja inicial por vídeo. Mismos criterios.

Suficiencia: 12 clips de locomoción y 12 de reposo verificados (mínimos 5 y 10). La ejecución es un **veredicto decisivo**, no descriptivo.

| Método | Locomoción (≥ 0,8) | Reposo (≥ 0,8) | Episodios falsos en reposo | Resultado |
|---|---:|---:|---:|---|
| V3.6 | 7/12 | 3/12 | 6 | No pasa |
| V3.7 | 1/12 | 5/12 | 0 | No pasa |
| **flow** | 7/12 | 3/12 | **6** | No pasa |
| body | 0/12 | 0/12 | 0 | No pasa |

Estado: `closed_not_resolvable_with_existing_data`.

Informativo (5 vídeos de desarrollo + 4 nuevos; no decide): flow 11/17 en locomoción, 8/22 en reposo, 6 episodios falsos. V3.6 11/17, 6/22, 9.

### Lectura

- **flow falla en los dos criterios.** La sensibilidad (7/12 = 0,58) no llega a 0,8 con independencia de cómo se etiquetara el reposo.
- **No generaliza la ausencia de falsos positivos.** En los vídeos de desarrollo flow no producía episodios en reposo (0/10); en los nuevos produce 6. La conclusión anterior de que flow "falla por abstención, no por error" no se sostiene fuera de los vídeos de desarrollo.
- **Abstención y pérdida de seguimiento.** `camera_reference_unreliable` sigue dominando en IMG_5192 (30/44 muestras, motivos V3.6). En IMG_5191, SAM2 pierde a Cinna en 26 de 48 muestras (`rabbit_not_marked`).
- **Origen de las etiquetas de reposo.** De los 12 clips verificados, 9 eran `unknown` en la revisión por fotogramas de ChatGPT y se marcaron como reposo al verlos en continuo. Tres reposos propuestos quedaron sin verificar y no se evaluaron. Se registra como hecho; no altera el veredicto, porque la sensibilidad falla por sí sola.
- El IoU de seguimiento ≈ 0,98 de estas ejecuciones compara contra guías de SAM2 y **no valida el seguimiento**.

---

## Veredicto sobre vídeos de desarrollo (2026-10-25)

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

### Protocolo de reapertura con cámara fija (fijado el 2026-10-25, antes de grabar)

1. **Conjunto de decisión:** `benchmarks/closure_fija.json`, solo con vídeos nuevos grabados con cámara fija. Mismos `CRITERIA` sin cambios: ≥ 5 clips de locomoción, ≥ 10 de reposo, sensibilidad y especificidad ≥ 0,8, al menos un acierto por vídeo con locomoción. Método principal: `flow`.
2. **Informativo, no decisorio:** `closure.json` + vídeos nuevos. Se publica igualmente.
3. **Etiquetas a ciegas:** clips de 4 s en la rejilla 0, 4, 8… s, revisados viendo el vídeo, **antes de ejecutar ningún método**, y commiteados en Git antes de la ejecución. `locomotion` = cambia de sitio sobre el suelo dentro del clip; `resting` = no se desplaza en todo el clip (movimientos de cabeza permitidos). Lo dudoso va como `unknown` o `exploring_sniffing` y no se evalúa.
4. **Anclas:** `RabbitSense_V32/rabbitsense_v32.py prepare --every 2`, cajas cada 2 s, primera en 0 s.
5. **Alcance si pasa:** locomoción candidata en grabaciones con cámara fija y encuadre amplio. La cámara en mano sigue cerrada como no resoluble.

### Protocolo de reapertura con cámara en mano (2026-10-25; sustituye al de cámara fija)

Decisión del responsable técnico: evaluar locomoción **con movimiento de cámara**, que es el uso real. Sustituye al protocolo de cámara fija. Se registra **antes de ejecutar ningún método** sobre los vídeos nuevos.

1. **Conjunto de decisión:** solo los vídeos nuevos IMG_5190–IMG_5193, más los que se añadan antes de ejecutar. Es el primer **test independiente**: ningún método ni constante se ajustó con ellos. Manifest `benchmarks/closure_mano.json`. Mismos `CRITERIA`, sin cambios. Método principal: `flow`; el resto se publica.
2. **Informativo, no decisorio:** `closure.json` + vídeos nuevos.
3. **Etiquetas provisionales archivadas sin cambios** en [etiquetas_mano_provisionales](../RabbitSense_ARCHIVO/etiquetas_mano_provisionales/). Las preparó ChatGPT a partir de fotogramas muestreados cada 2 s, sin consultar ningún modelo. **No son ground truth.** Los tiempos se reconstruyeron desde clips recodificados.
4. **Puerta de ejecución:** las etiquetas entran al CSV de evaluación con `reviewed=no`. Solo pasan a `reviewed=si` tras revisión humana a velocidad normal sobre el **archivo original**, con tiempos medidos desde su primer fotograma. Se commitean antes de ejecutar. Cualquier cambio de etiquetas después de ejecutar invalida el veredicto.
5. **Clips de menos de 3 s no se evalúan** (rejilla de 0,5 s: demasiado cortos para mayoría o episodio).
6. **Suficiencia antes de ejecutar:** si tras la revisión hay menos de 5 clips de locomoción o menos de 10 de reposo, **no se ejecuta**: se graba más, porque el criterio fallaría por construcción. Con las provisionales hay 13 / 6 (≥ 3 s): falta reposo.
7. **Anclas:** cajas cada 2 s sobre los originales con `RabbitSense_V32/rabbitsense_v32.py prepare --every 2`.
8. **Alcance:** si pasa, locomoción candidata con cámara en mano en vídeos no vistos durante el desarrollo. Si no pasa, la locomoción con cámara en mano queda cerrada definitivamente con este enfoque.

#### Enmienda (2026-10-25, antes de ejecutar): procedimiento automático

Sustituye el punto 7. Sigue sin haberse ejecutado ningún método sobre IMG_5190–5193.

- **Anclas automáticas.** Una caja manual en el primer fotograma de cada vídeo; SAM2 sigue a Cinna y cada máscara produce una guía por fotograma. Las máscaras vacías o por debajo del 0,1 % de la imagen dejan la guía ausente y el método se abstiene. Son **guías, no verdad-terreno**. En este conjunto, el IoU de seguimiento de `comparison.json` compara contra guías de SAM2 y **no valida el seguimiento**. Se revisa `runs/handheld/segmentation/<vídeo>/overlay.mp4` antes de verificar las etiquetas.
- **Método decisorio:** `flow` sobre esas guías. No se añade `sam2_masked_flow` al veredicto: usaría las mismas máscaras dos veces.
- **Verificación de etiquetas** en `revision.html`: cada clip se repite en bucle sobre el **original**; una tecla por clip. Solo los clips confirmados ahí pasan a `revisado_video_continuo`; `prepare` los convierte en `reviewed=si` si duran al menos 3 s.
- **Puerta mecánica:** `handheld run` comprueba la suficiencia y **no ejecuta** el cierre si faltan clips verificados.

#### Estado de suficiencia (2026-10-25)

Revisión visual de ChatGPT archivada sin cambios en [etiquetas_mano_revision_visual](../RabbitSense_ARCHIVO/etiquetas_mano_revision_visual/): fotogramas cada 0,5 s, sin revisión de vídeo continuo. Respecto a las provisionales, pasa IMG_5191 12–16 s e IMG_5193 4–8 s de locomoción a `unknown`. Convertida a [etiquetas_mano.csv](../RabbitSense_ARCHIVO/etiquetas_mano.csv) con `reviewed=no` en todas las filas. Informe: [cierre/suficiencia_mano.json](cierre/suficiencia_mano.json).

| Clase (≥ 3 s) | Verificados | Propuestos pendientes | Mínimo | ¿Alcanzable con las propuestas actuales? |
|---|---:|---:|---:|---|
| Locomoción | 0 | 11 | 5 | Sí |
| Reposo | 0 | 6 | 10 | **No** |

#### Decisión: no se graba más (2026-10-25)

El responsable técnico decide no grabar más. Consecuencia, registrada antes de ejecutar:

- *Superado el 2026-10-26: la verificación en vídeo continuo alcanzó 12 clips de reposo y hubo veredicto decisivo (ver arriba).* **No habrá veredicto decisivo con cámara en mano.** El criterio de 10 clips de reposo no se rebaja; hacerlo ahora sería ajustarlo a los datos disponibles. Estado oficial: **locomoción con cámara en mano no resuelta y sin veredicto**.
- **Evaluación descriptiva** (`handheld run --descriptive`): mismas métricas sobre IMG_5190–5193, con estado `descriptive_only_insufficient_sample`. No puede producir `closed_resolved…`. El campo `passed` se renombra a `meets_thresholds_not_a_verdict`.
- **Evaluación informativa** (`--informative benchmarks/closure_mano_informativo.json`): los cinco vídeos de desarrollo más los cuatro nuevos, con estado `informative_not_decisive`. Combina material ya usado para desarrollar; no es un test independiente.
- **Etiquetas sin verificar** (`--include-unverified`, tarea 5b): evalúa también las propuestas de ChatGPT de 3 s o más. El resultado registra `label_source=includes_unverified_proposals`. `etiquetas_mano.csv` no se modifica: la copia usada se escribe en la carpeta de salida.

**Reposo insuficiente.** Aunque se confirmaran los seis clips propuestos, faltarían al menos cuatro. No se reetiquetan `unknown` como reposo para alcanzar el mínimo. Hace falta grabar al menos 16 s más de reposo con cámara en mano, en tramos de 4 s. En la hoja de contacto de IMG_5192, Cinna parece desplazarse entre 20 y 22 s, un tramo propuesto como reposo. Al durar menos de 3 s no se evalúa, pero confirma que el muestreo no sustituye la verificación.

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