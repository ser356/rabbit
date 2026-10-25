#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-$HOME/Downloads/cinna.mov}"
if [[ $# -gt 0 ]];then shift;fi
[[ -x .venv/bin/python ]] || { echo 'Falta .venv' >&2; exit 1; }
.venv/bin/python scripts/make_review_v22.py --input "$VIDEO" --clip-seconds 4 "$@"
open "runs/review/$(basename "${VIDEO%.*}")/review.html"
echo 'Al terminar: exporta el CSV y colócalo en runs/review/<video>/rabbitsense_labels.csv'
