#!/usr/bin/env bash
set -euo pipefail
# Stage 2: run TRAK-10 / TracIn / Grad-Dot / Grad-Cos sequentially in
# one GPU job. Requires TDADX_VENV, TDADX_DATA_ROOT, TDADX_RESULTS_DIR.
# Optional: PROJ_DIM (2048), PROJ_MAX_BATCH (1024).

: "${TDADX_VENV:?set TDADX_VENV}"
: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

PROJ_DIM="${PROJ_DIM:-2048}"
PROJ_MAX_BATCH="${PROJ_MAX_BATCH:-1024}"

BENCH_CACHE="${TDADX_DATA_ROOT}/maestro/dattri_benchmark"
OUT_DIR="${TDADX_RESULTS_DIR}/dattri_gt/scores"
mkdir -p "${BENCH_CACHE}" "${OUT_DIR}"

# shellcheck disable=SC1091
source "${TDADX_VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_method() {
    local METHOD="$1"
    echo "##### ${METHOD} #####"
    python scripts/dattri_gt/run_attribution.py \
        --method "${METHOD}" \
        --device cuda \
        --output_dir "${OUT_DIR}" \
        --cache_path "${BENCH_CACHE}" \
        --proj_dim "${PROJ_DIM}" \
        --proj_max_batch_size "${PROJ_MAX_BATCH}"
}

run_method "TRAK-10"
run_method "TracIn"
run_method "Grad-Dot"
run_method "Grad-Cos"

echo "DONE: all 4 methods -> ${OUT_DIR}"
