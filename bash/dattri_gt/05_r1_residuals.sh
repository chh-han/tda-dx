#!/usr/bin/env bash
set -euo pipefail
# Stage 5: rank-1 SVD residual for each score matrix.
# Requires TDADX_VENV, TDADX_RESULTS_DIR.

: "${TDADX_VENV:?set TDADX_VENV}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

SCORE_DIR="${TDADX_RESULTS_DIR}/dattri_gt/scores"
R1_DIR="${TDADX_RESULTS_DIR}/dattri_gt/scores_r1"
mkdir -p "${R1_DIR}"

# shellcheck disable=SC1091
source "${TDADX_VENV}/bin/activate"
export PYTHONUNBUFFERED=1

for METHOD in TRAK10 TracIn GradDot GradCos; do
    echo ""
    echo "##### ${METHOD} #####"
    python scripts/dattri_gt/compute_r1_residual.py \
        --score     "${SCORE_DIR}/dattri_mt_${METHOD}_scores.npy" \
        --out_score "${R1_DIR}/dattri_mt_${METHOD}_scores.npy" \
        --out_info  "${R1_DIR}/dattri_mt_${METHOD}_r1info.json"
done

echo "DONE: r1 residuals -> ${R1_DIR}"
