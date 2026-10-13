#!/usr/bin/env bash
set -euo pipefail
# Stage 5: Train the hierarchical transformer in three stages.
# Each stage saves checkpoints to its own results subdirectory.
#
# Requires:
#   TDADX_RESULTS_DIR
#
# To run distributed: launch with `torchrun --standalone --nproc_per_node=N`
# instead of `python` and pass `--expect_distributed`.

: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

DATA_FOLDER="${TDADX_RESULTS_DIR}/fma_tokens_mmap"
RVQ_PATH="${TDADX_RESULTS_DIR}/clap_rvq/clap.rvq.350.pt"
KMEANS_PATH="${TDADX_RESULTS_DIR}/hubert_kmeans/kmeans.joblib"
MODEL_CONFIG=./configs/model/musiclm_large.json
TRAINING_CONFIG=./configs/training/train_musiclm_fma.json

run_stage () {
    local stage="$1"
    python scripts/05_train.py \
        --stage           "${stage}" \
        --results_folder  "${TDADX_RESULTS_DIR}/train/${stage}" \
        --logging_dir     "${TDADX_RESULTS_DIR}/train/logs/${stage}" \
        --model_config    "${MODEL_CONFIG}" \
        --training_config "${TRAINING_CONFIG}" \
        --rvq_path        "${RVQ_PATH}" \
        --kmeans_path     "${KMEANS_PATH}" \
        --data_folder     "${DATA_FOLDER}" \
        --use_preprocessed_data true \
        --batch_size 4 \
        --grad_accum_every 8 \
        --log_with tensorboard
}

run_stage semantic
run_stage coarse
run_stage fine
