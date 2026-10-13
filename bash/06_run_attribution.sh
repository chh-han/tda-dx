#!/usr/bin/env bash
set -euo pipefail
# Stage 6a: Run TDA attribution on the trained hierarchical model.
# This example uses TRAK with proj_dim=4096, regularization=10. To switch
# methods, change --method to one of: trak | grad_cos | logra | factgrass.
#
# For large training pools, shard with --num_shards N --shard_id i, run
# `--mode cache_shard` per shard, then run once with `--mode merge_and_score`
# to combine. This script shows the single-shot `--mode full` path.
#
# Requires:
#   TDADX_RESULTS_DIR
#
# Inputs:
#   $TDADX_RESULTS_DIR/train/{semantic,coarse,fine}/*.pt   (Stage 5 checkpoints)
#   $TDADX_RESULTS_DIR/fma_tokens_mmap/                    (Stage 3 token pool)
#   tracks.csv with the FMA train_subset of interest
# Output:
#   $TDADX_RESULTS_DIR/attribution/<method>/<stage>/

: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

METHOD=trak
CKPT_DIR="${TDADX_RESULTS_DIR}/train"
OUT_ROOT="${TDADX_RESULTS_DIR}/attribution/${METHOD}"

for stage in semantic coarse fine; do
  python scripts/06_run_attribution.py \
      --stage "${stage}" \
      --mode  full \
      --method "${METHOD}" \
      --results_folder "${OUT_ROOT}" \
      --model_config ./configs/model/musiclm_large.json \
      --preprocessed_data_folder "${TDADX_RESULTS_DIR}/fma_tokens_mmap" \
      --tracks_csv "${TDADX_DATA_ROOT}/fma_metadata/tracks.csv" \
      --fma_train_subset large \
      --semantic_checkpoint "${CKPT_DIR}/semantic/semantic.best.pt" \
      --coarse_checkpoint   "${CKPT_DIR}/coarse/coarse.best.pt" \
      --fine_checkpoint     "${CKPT_DIR}/fine/fine.best.pt" \
      --samples_per_track 7 \
      --batch_size 1 \
      --proj_dim 4096 \
      --regularization 10 \
      --seed 42
done
