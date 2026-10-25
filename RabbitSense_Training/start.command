#!/bin/bash
# RabbitSense: instalación, datos y entrenamiento de comportamientos con ViT.
set -Eeuo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

note(){ printf '\n\033[1;36m%s\033[0m\n' "$*"; }
fail(){ printf '\nERROR: %s\n' "$*" >&2; exit 1; }

choose_python(){
  if [[ -n "${PYTHON:-}" ]]; then command -v "$PYTHON" >/dev/null || fail "PYTHON=$PYTHON no existe"; printf '%s' "$PYTHON"; return; fi
  for bin in python3.12 python3.11 python3.13 python3; do
    if command -v "$bin" >/dev/null 2>&1 && "$bin" -c 'import sys; assert (3,11)<=sys.version_info[:2]<=(3,13)' >/dev/null 2>&1; then
      printf '%s' "$bin"; return
    fi
  done
  if command -v brew >/dev/null 2>&1; then
    note 'Instalando Python 3.12 mediante Homebrew...' >&2
    brew install python@3.12 >&2
    local prefix; prefix="$(brew --prefix python@3.12)"
    printf '%s' "$prefix/bin/python3.12"; return
  fi
  fail 'Se necesita Python 3.11, 3.12 o 3.13. Instala Python 3.12 o Homebrew y vuelve a ejecutarlo.'
}

note '1/4 · Preparando entorno Python'
PYBIN="$(choose_python)"
if [[ ! -f .venv/bin/python ]]; then "$PYBIN" -m venv .venv; fi
PY="$PWD/.venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

note '2/4 · Comprobando GPU'
"$PY" - <<'PY'
import torch, platform
print('SO:', platform.platform())
print('PyTorch:',torch.__version__)
print('MPS (GPU Apple):',torch.backends.mps.is_available())
print('CUDA:',torch.cuda.is_available())
PY

note '3/4 · Descargando y organizando Dataset C (7 comportamientos)'
"$PY" scripts/prepare.py "$@"

note '4/4 · Entrenando Vision Transformer'
"$PY" scripts/train.py \
  --data data/processed \
  --out runs/behaviour \
  --model "${RABBITSENSE_MODEL:-vit_small_patch16_224.augreg_in21k_ft_in1k}" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --accumulate "${ACCUMULATE:-2}" \
  --require-mps-if-apple-silicon

if [[ "${TRAIN_DETECTOR:-0}" == "1" ]]; then
  note 'Extra · Descargando Roboflow B y entrenando detector YOLOv8'
  "$PY" -m pip install 'roboflow>=1.1,<2' 'ultralytics>=8.3,<9'
  "$PY" scripts/train_detector.py
fi

note 'ENTRENAMIENTO COMPLETADO'
echo 'Mejor modelo: runs/behaviour/best.pt'
echo 'Métricas:     runs/behaviour/results.json'
echo 'Para analizar una imagen o vídeo: .venv/bin/python scripts/predict.py --input "/ruta/al/video.mov"'
