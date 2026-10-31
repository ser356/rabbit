# RabbitSenseKit

Lógica de RabbitSense en Swift para la app de macOS. Usa la cámara fija y los modelos en Core ML. Replica el código Python validado, y sus tests comprueban que da lo mismo.

| Archivo | Qué hace | Equivale a |
|---|---|---|
| `Image.swift` | Recortes, reescalado por áreas, caja cuadrada fija, fotogramas muestreados | `acts.square`, `acts.square_around`, `ytacts.probe_frames` |
| `Detection.swift` | Detecciones de OWLv2 y resumen por ventana | `ytacts.detections`, `ytacts.window_detection` |
| `OwlPreprocess.swift` | Preprocesado de OWLv2 | `Owlv2ImageProcessor` |
| `Classifier.swift` | Clasificador de actos congelado | `ytacts.probabilities` |
| `FixedCamera.swift` | Desplazamiento con cámara fija desde el centro de la máscara (sin validar) | `fixedcam.py` |
| `Models.swift` | OWLv2, V-JEPA 2 y SAM2 en Core ML | `coreml.py`, `queryseg.CoreMLSAM` |
| `WindowAnalyzer.swift` | Acto de una ventana de 2 s (con OWLv2 o con las cajas del seguidor) | `auto.act_windows` |
| `Journal.swift` | Diario por día (JSONL), resumen de actividad y avisos | — (nuevo, sin validar) |

La app (`Sources/RabbitSenseApp`) mira la cámara en directo, sin guardar vídeo: el seguidor de Vision sigue a Cinna en cada fotograma, OWLv2 lo recoloca al empezar, al perderlo y cada 15 s, y cada 2 s V-JEPA 2 y el clasificador dan el acto en segundo plano. Si el análisis va con retraso se salta ventanas («sin analizar»), nunca se acumula. El diario queda en `~/Library/Application Support/RabbitSense/diario/AAAA-MM-DD.jsonl`.

Avisos en el Mac (como mucho uno por hora de cada tipo): sin cámara 5 min, Cinna sin verse 60 min, y actividad menor que la mitad de la mediana de las mismas horas de días anteriores (a partir de 7 días de diario). Son reglas provisionales, sin validar; comer no es fiable.

## Probarlo en el Mac

Requisito: las Command Line Tools de Xcode con Swift 6 (o Xcode 16 o posterior). Los tests usan Swift Testing, que no necesita Xcode completo.

1. **Partes sin modelos.** Los datos de referencia ya están en el repositorio:
   ```zsh
   cd RabbitSense_repo/swift/RabbitSenseKit && swift test
   ```
2. **Modelos** (Swift + Core ML frente a PyTorch). Necesitan los modelos convertidos (`rabbitsense.coreml convert`) y unas ventanas de referencia. Desde la raíz del espacio de trabajo:
   ```zsh
   export PYTHONPATH=RabbitSense_repo/src PYTORCH_ENABLE_MPS_FALLBACK=1
   RabbitSense_Training/.venv/bin/python -m rabbitsense.swiftgolden models RabbitSense_Training/data/cinna_comiendo/* "RabbitSense_Training/data/cinna_mano/IMG_5190.MOV"
   RabbitSense_Training/.venv/bin/python -m rabbitsense.queryseg download-coreml   # opcional: SAM2 de Core ML de Apple
   cd RabbitSense_repo/swift/RabbitSenseKit && swift test
   RABBITSENSE_COMPUTE=ne swift test   # forzar el Neural Engine y comparar tiempos
   ```
   Las ventanas de referencia (`Tests/RabbitSenseKitTests/Fixtures/models`) no se suben a GitHub.

Si falta algo, los tests de modelos se saltan y lo dicen.

## La app

Necesita los modelos convertidos en `RabbitSense_repo/weights/coreml` (o la carpeta de `RABBITSENSE_WEIGHTS` al construir).

```zsh
cd RabbitSense_repo/swift/RabbitSenseKit
zsh scripts/build_app.sh
open build/RabbitSense.app
```

La primera vez compila los modelos (un par de minutos) y pide permiso de cámara y de notificaciones. Con el iPhone cerca, bloqueado, apoyado y en horizontal, macOS lo ofrece como cámara (Continuity Camera); la app lo prefiere a la integrada. Mientras observa, el Mac no entra en reposo solo.
