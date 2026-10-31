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
| `WindowAnalyzer.swift` | Acto de una ventana de 2 s | `auto.act_windows` |

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
