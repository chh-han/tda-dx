#!/usr/bin/env bash
set -euo pipefail
# Stage 3: extract jSymbolic MIDI features per segment.
# Requires TDADX_VENV, TDADX_DATA_ROOT, TDADX_RESULTS_DIR.

: "${TDADX_VENV:?set TDADX_VENV}"
: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

DATA_PATH="${TDADX_DATA_ROOT}/maestro"
OUT_DIR="${TDADX_RESULTS_DIR}/dattri_gt/midi_features"
mkdir -p "${OUT_DIR}"

# shellcheck disable=SC1091
source "${TDADX_VENV}/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/dattri_gt/extract_midi_features.py \
    --data_path "${DATA_PATH}" \
    --split train \
    --output "${OUT_DIR}/maestro_train_features.pkl"

python scripts/dattri_gt/extract_midi_features.py \
    --data_path "${DATA_PATH}" \
    --split test \
    --output "${OUT_DIR}/maestro_test_features.pkl"

echo "DONE: features -> ${OUT_DIR}"
