#!/usr/bin/env bash
set -euo pipefail
# Stage 6: within-group homogeneity z-scores, K-sweep.
# Requires TDADX_VENV, TDADX_RESULTS_DIR, METHOD (TRAK10/TracIn/GradDot/GradCos).
# Optional: SCORE_DIR, TAG, K_LIST, NULL_B.

: "${TDADX_VENV:?set TDADX_VENV}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"
: "${METHOD:?set METHOD (TRAK10 / TracIn / GradDot / GradCos)}"

SCORE_DIR="${SCORE_DIR:-${TDADX_RESULTS_DIR}/dattri_gt/scores}"
FEAT_DIR="${TDADX_RESULTS_DIR}/dattri_gt/midi_features"
TAG="${TAG:-}"
K_LIST="${K_LIST:-10 20 50 100 200 300 400 500}"
NULL_B="${NULL_B:-200}"

OUT_DIR="${TDADX_RESULTS_DIR}/dattri_gt/homogeneity"
mkdir -p "${OUT_DIR}"

# shellcheck disable=SC1091
source "${TDADX_VENV}/bin/activate"
export PYTHONUNBUFFERED=1

echo "METHOD=${METHOD}  TAG=${TAG}  SCORE_DIR=${SCORE_DIR}  K_LIST=${K_LIST}"

# shellcheck disable=SC2086
python scripts/dattri_gt/compute_homogeneity.py \
    --score "${SCORE_DIR}/dattri_mt_${METHOD}_scores.npy" \
    --train_features "${FEAT_DIR}/maestro_train_features.pkl" \
    --method_name "${METHOD}${TAG}" \
    --K ${K_LIST} \
    --null_B "${NULL_B}" \
    --output "${OUT_DIR}/${METHOD}${TAG}.json"

echo "DONE: ${METHOD}${TAG} homogeneity -> ${OUT_DIR}/${METHOD}${TAG}.json"
