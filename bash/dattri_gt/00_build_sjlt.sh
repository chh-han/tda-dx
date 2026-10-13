#!/usr/bin/env bash
set -euo pipefail
# (Optional setup) Rebuild the sjlt CUDA extension against the active
# torch ABI. Run once per environment, on a node with nvcc + GPU
# available — the extension is compiled in source mode so the .so
# matches the torch wheel currently installed.
#
# Requires:
#   TDADX_VENV       path to the activated venv (with torch installed)
# Optional:
#   UV_BIN          path to the `uv` binary (default: `uv` on PATH)
#   CUDA_MODULE     module name to load via `module load` (default CUDA/13.0.0)
#   CUDA_ARCH_LIST  TORCH_CUDA_ARCH_LIST to compile for (default 8.0;8.6;9.0)

: "${TDADX_VENV:?set TDADX_VENV (path to venv with torch installed)}"
: "${UV_BIN:=uv}"
: "${CUDA_MODULE:=CUDA/13.0.0}"
: "${CUDA_ARCH_LIST:=8.0;8.6;9.0}"

PY="${TDADX_VENV}/bin/python"

if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE}"
fi

export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
export FORCE_CUDA_BUILD=1

echo "nvcc: $(which nvcc 2>/dev/null || echo NOT_FOUND)"
echo "torch: $("${PY}" -c 'import torch; print(torch.__version__, torch.version.cuda)')"

"${UV_BIN}" pip install --python "${PY}" --reinstall --no-cache \
    --no-binary sjlt --no-build-isolation -v sjlt 2>&1 | tail -80

"${PY}" -c "
import sjlt, sjlt._C
print('sjlt   :', sjlt.__file__)
print('sjlt._C:', sjlt._C.__file__)
print('SJLTProjection import OK')
"

echo "DONE: sjlt rebuilt for ${TDADX_VENV}"
