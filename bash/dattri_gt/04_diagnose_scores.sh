#!/usr/bin/env bash
set -euo pipefail
# Stage 4: score-matrix reliability diagnostics (r_i, p, |rho|_bar).
# Re-run on rank-1 residuals by setting SCORE_DIR + OUT_JSON.
# Requires TDADX_VENV, TDADX_RESULTS_DIR.

: "${TDADX_VENV:?set TDADX_VENV}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

SCORE_DIR="${SCORE_DIR:-${TDADX_RESULTS_DIR}/dattri_gt/scores}"
OUT_JSON="${OUT_JSON:-${TDADX_RESULTS_DIR}/dattri_gt/diagnostics/score_matrix_diagnostics.json}"
mkdir -p "$(dirname "${OUT_JSON}")"

# shellcheck disable=SC1091
source "${TDADX_VENV}/bin/activate"

export PYTHONUNBUFFERED=1

python scripts/07_diagnose_score_matrix.py \
    --targets \
        "${SCORE_DIR}/dattri_mt_TRAK10_scores.npy" \
        "${SCORE_DIR}/dattri_mt_TracIn_scores.npy" \
        "${SCORE_DIR}/dattri_mt_GradDot_scores.npy" \
        "${SCORE_DIR}/dattri_mt_GradCos_scores.npy" \
    --out "${OUT_JSON}"

echo "DONE: diagnostics -> ${OUT_JSON}"
