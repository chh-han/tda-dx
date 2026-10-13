#!/usr/bin/env bash
set -euo pipefail
# Stage 2: Train the two tokenizers used by TDA-Dx's hierarchical model.
#   (a) CLAP residual vector quantizer  → semantic conditioning tokens
#   (b) HuBERT k-means                  → semantic target tokens
#
# Requires:
#   TDADX_RESULTS_DIR — where to write tokenizer checkpoints
#
# Configs:
#   configs/model/musiclm_large.json
#   configs/training/train_musiclm_fma.json
#
# Output checkpoints:
#   $TDADX_RESULTS_DIR/clap_rvq/clap.rvq.<step>.pt
#   $TDADX_RESULTS_DIR/hubert_kmeans/kmeans.joblib

: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

MODEL_CONFIG=./configs/model/musiclm_large.json
TRAINING_CONFIG=./configs/training/train_musiclm_fma.json

# (a) CLAP RVQ
python scripts/02_train_clap_rvq.py \
    --results_folder  "${TDADX_RESULTS_DIR}/clap_rvq" \
    --model_config    "${MODEL_CONFIG}" \
    --training_config "${TRAINING_CONFIG}"

# (b) HuBERT k-means
python scripts/02_train_hubert_kmeans.py \
    --results_folder  "${TDADX_RESULTS_DIR}/hubert_kmeans" \
    --model_config    "${MODEL_CONFIG}" \
    --training_config "${TRAINING_CONFIG}"
