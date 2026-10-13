#!/usr/bin/env bash
set -euo pipefail
# Stage 3: Tokenize the audio dataset using the trained CLAP RVQ + HuBERT
# k-means tokenizers + EnCodec acoustic tokens. Output is a memmap pool
# consumed by the trainer.
#
# Requires:
#   TDADX_DATA_ROOT, TDADX_RESULTS_DIR
#
# Inputs:
#   $TDADX_RESULTS_DIR/fma-48k/                  (from 01_preprocess_audio.sh)
#   $TDADX_RESULTS_DIR/clap_rvq/clap.rvq.350.pt  (from 02_train_tokenizers.sh)
#   $TDADX_RESULTS_DIR/hubert_kmeans/kmeans.joblib
# Output:
#   $TDADX_RESULTS_DIR/fma_tokens_mmap/

: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

python scripts/03_preprocess_tokens.py \
    --model_config    ./configs/model/musiclm_large.json \
    --training_config ./configs/training/train_fma_preprocess.json \
    --rvq_path        "${TDADX_RESULTS_DIR}/clap_rvq/clap.rvq.350.pt" \
    --kmeans_path     "${TDADX_RESULTS_DIR}/hubert_kmeans/kmeans.joblib" \
    --result_folder   "${TDADX_RESULTS_DIR}/fma_tokens_mmap" \
    --tracks_csv      "${TDADX_DATA_ROOT}/fma_metadata/tracks.csv" \
    --input_audio_format npy \
    --npy_sample_hz 48000 \
    --output_format memmap
