#!/usr/bin/env bash
set -euo pipefail
# Stage 4: Extract per-track musicological features (rhythmic, harmonic,
# timbral) used by the analysis pipeline (Stage 7) for feature-alignment.
#
# Requires:
#   TDADX_DATA_ROOT, TDADX_RESULTS_DIR
#
# Inputs:
#   $TDADX_RESULTS_DIR/fma-48k/                (or raw mp3s under $TDADX_DATA_ROOT/fma_audio/)
#   $TDADX_DATA_ROOT/fma_metadata/             (tracks.csv etc.)
# Output:
#   $TDADX_RESULTS_DIR/fma_features/features.tsv

: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

python scripts/04_extract_features.py \
    --audio-root    "${TDADX_RESULTS_DIR}/fma-48k" \
    --metadata-root "${TDADX_DATA_ROOT}/fma_metadata" \
    --output-dir    "${TDADX_RESULTS_DIR}/fma_features" \
    --audio-format  npy \
    --sample-rate   48000 \
    --subset        large \
    --split         all
