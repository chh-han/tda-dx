#!/usr/bin/env bash
set -euo pipefail
# Stage 1: download MAESTRO v2.0.0 + dattri's generated_music test set.
# Requires TDADX_VENV, TDADX_DATA_ROOT.

: "${TDADX_VENV:?set TDADX_VENV}"
: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"

DATA_PATH="${TDADX_DATA_ROOT}/maestro"
mkdir -p "${DATA_PATH}"

# shellcheck disable=SC1091
source "${TDADX_VENV}/bin/activate"

export PYTHONUNBUFFERED=1
echo "DATA_PATH=${DATA_PATH}"

python - <<PY
from dattri.benchmark.datasets.maestro import create_maestro_datasets
train_ds, test_ds = create_maestro_datasets("${DATA_PATH}", generated_music=True)
print(f"train size: {len(train_ds)}")
print(f"test  size: {len(test_ds)}")
PY

echo "DONE: maestro v2.0.0 ready at ${DATA_PATH}"
