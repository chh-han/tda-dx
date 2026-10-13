#!/usr/bin/env bash
set -euo pipefail
# Stage 7: Run the full TDA-Dx analysis pipeline on attribution outputs.
# Produces within-group homogeneity tables, per-channel feature alignment,
# and reliability diagnostics.
#
# Works equally on:
#   - attribution outputs (e.g. $TDADX_RESULTS_DIR/attribution/trak/)
#   - retrieval outputs   (e.g. $TDADX_RESULTS_DIR/clap_retrieval/)
#
# Requires:
#   TDADX_RESULTS_DIR

: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

INPUT="${INPUT:-${TDADX_RESULTS_DIR}/attribution/trak}"
TOPK="${TOPK:-50}"
FEATURES_TSV="${FEATURES_TSV:-${TDADX_RESULTS_DIR}/fma_features/features.tsv}"
OUTPUT="${OUTPUT:-${INPUT}/analysis-top${TOPK}}"

mkdir -p "${OUTPUT}"

python scripts/07_analyze.py \
    --input-results-dir   "${INPUT}" \
    --output-dir          "${OUTPUT}" \
    --features-tsv        "${FEATURES_TSV}" \
    --feature-set         v1 \
    --top-k               "${TOPK}" \
    --num-workers         8 \
    --rank-selection      top \
    --track-aggregation   mean \
    --score-normalization zscore_per_test \
    --skip-loo \
    --skip-appendix
