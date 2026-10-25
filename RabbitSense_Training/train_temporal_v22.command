#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] || { echo 'Falta .venv' >&2; exit 1; }
LABELS="${1:-runs/review/cinna/rabbitsense_labels.csv}"
if [[ $# -gt 0 ]];then shift;fi
[[ -f "$LABELS" ]] || { echo "Falta etiquetas $LABELS" >&2; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python scripts/train_temporal_v22.py --labels "$LABELS" --allow-single-video-prototype "$@"
