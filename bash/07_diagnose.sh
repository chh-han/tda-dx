#!/usr/bin/env bash
set -euo pipefail
# Stage 7 (diagnostics): canonical reliability checks on attribution score
# matrices. Three independent diagnostics:
#   (1) score_matrix       — global popularity / column collinearity stats
#   (2) u1_channel_corr    — alignment of u_1 with rhythmic/harmonic/timbral
#                            features (explains channel z-scores)
#   (3) rank1_removed      — recompute scores with the dominant rank-1 axis
#                            removed, then re-run analyze on the residual
#
# Requires:
#   TDADX_RESULTS_DIR

: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

ATTRIB_ROOT="${TDADX_RESULTS_DIR}/attribution"
TRAK_ROOT="${ATTRIB_ROOT}/trak"
FEATURES_TSV="${TDADX_RESULTS_DIR}/fma_features/features.tsv"
DIAG_OUT="${ATTRIB_ROOT}/score_matrix_diagnostics"
mkdir -p "${DIAG_OUT}"

# (1) score-matrix diagnostics → JSON summary
python scripts/07_diagnose_score_matrix.py \
    --targets "${TRAK_ROOT}:trak" \
    --out     "${DIAG_OUT}/trak_score_matrix.json"

# (2) u_1 ↔ channel correlations
python scripts/07_compute_u1_channel_corr.py \
    --features_tsv "${FEATURES_TSV}" \
    --target  "trak:${TRAK_ROOT}" \
    --out_root "${DIAG_OUT}"

# (3) rank-1-removed scores → re-run analyze on residual
R1_OUT="${ATTRIB_ROOT}/rank1_removed"
python scripts/07_compute_rank1_removed_scores.py \
    --target  "trak:${TRAK_ROOT}" \
    --out_root "${R1_OUT}" \
    --stages semantic coarse fine

python scripts/07_analyze.py \
    --input-results-dir   "${R1_OUT}/trak-r1" \
    --output-dir          "${R1_OUT}/trak-r1/analysis-top50" \
    --features-tsv        "${FEATURES_TSV}" \
    --feature-set v1 --top-k 50 --num-workers 8 \
    --rank-selection top --track-aggregation mean \
    --score-normalization zscore_per_test \
    --skip-loo --skip-appendix
