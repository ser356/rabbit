#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] || { echo 'Falta .venv: instala este parche en la carpeta RabbitSense_Training ORIGINAL' >&2; exit 1; }
[[ -f runs/behaviour/best.pt ]] || { echo 'No existe runs/behaviour/best.pt. No lo borres: es el ViT entrenado.' >&2; exit 1; }
[[ -f scripts/predict_v22.py && -f scripts/train_temporal_v22.py && -f scripts/anchor_boxes.py ]] || { echo 'Falta V2.2: este parche se instala encima de RabbitSense_Training V2.2.' >&2; exit 1; }
if ! .venv/bin/python -c 'import cv2' >/dev/null 2>&1; then .venv/bin/python -m pip install 'opencv-python>=4.8,<5'; fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/process_multivideo_v23.py "$@"
.venv/bin/python scripts/train_multivideo_v23.py
