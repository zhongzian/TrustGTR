#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN=/mnt/data/zza/anaconda3/bin/conda
ENV_PREFIX="$PROJECT_DIR/.conda"
mkdir -p "$PROJECT_DIR/logs" /tmp/trustgtr_install /tmp/trustgtr_pip_cache
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  "$CONDA_BIN" create -p "$ENV_PREFIX" python=3.10.13 pip --offline -y
fi
TMPDIR=/tmp/trustgtr_install PIP_CACHE_DIR=/tmp/trustgtr_pip_cache \
  "$ENV_PREFIX/bin/python" -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r "$PROJECT_DIR/configs/environment_pip.txt"
"$ENV_PREFIX/bin/python" -m pip freeze > "$PROJECT_DIR/configs/environment_lock.txt"
"$CONDA_BIN" list -p "$ENV_PREFIX" --explicit > "$PROJECT_DIR/configs/conda_explicit.txt"
"$ENV_PREFIX/bin/python" -c 'import sys, torch; print(sys.version); print("torch", torch.__version__, "runtime", torch.version.cuda, "cuda_available", torch.cuda.is_available())'
