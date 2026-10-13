#!/usr/bin/env bash
set -euo pipefail
# Stage 7: re-measure dattri's LDS on the original and rank-1 residual scores.
# Requires TDADX_VENV, TDADX_DATA_ROOT, TDADX_RESULTS_DIR.

: "${TDADX_VENV:?set TDADX_VENV}"
: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

BENCH_CACHE="${TDADX_DATA_ROOT}/maestro/dattri_benchmark"
OUT_DIR="${TDADX_RESULTS_DIR}/dattri_gt/diagnostics"
mkdir -p "${OUT_DIR}"

# shellcheck disable=SC1091
source "${TDADX_VENV}/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/dattri_gt/recompute_lds.py \
    --score_dir "${TDADX_RESULTS_DIR}/dattri_gt/scores" \
    --cache_path "${BENCH_CACHE}" \
    --output "${OUT_DIR}/lds_original.json"

python scripts/dattri_gt/recompute_lds.py \
    --score_dir "${TDADX_RESULTS_DIR}/dattri_gt/scores_r1" \
    --cache_path "${BENCH_CACHE}" \
    --output "${OUT_DIR}/lds_r1_residual.json"
