#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-$HOME/Downloads/cinna.mov}"
if [[ $# -gt 0 ]]; then shift; fi
[[ -x .venv/bin/python ]] || { echo 'Falta el entorno original .venv' >&2; exit 1; }
[[ -f runs/behaviour/best.pt ]] || { echo 'Falta runs/behaviour/best.pt' >&2; exit 1; }
[[ -f runs/anchors/cinna.json ]] || { echo 'Falta runs/anchors/cinna.json de la V2.1' >&2; exit 1; }
if ! .venv/bin/python -c 'import cv2' >/dev/null 2>&1; then
  .venv/bin/python -m pip install 'opencv-python>=4.8,<5'
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/predict_v22.py --input "$VIDEO" "$@"
