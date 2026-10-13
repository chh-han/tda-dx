#!/usr/bin/env bash
set -euo pipefail
# Stage 1: Convert raw audio to normalized .npy (48kHz mono, LUFS-normalized).
#
# Requires:
#   TDADX_DATA_ROOT  — root containing tracks metadata + raw audio
#   TDADX_RESULTS_DIR — where to write outputs
#
# Inputs:
#   $TDADX_DATA_ROOT/fma_metadata/tracks.csv
#   $TDADX_DATA_ROOT/fma_audio/<track>.mp3
# Outputs:
#   $TDADX_RESULTS_DIR/fma-48k/<track>.npy
#
# This script processes the full dataset in one shot. To shard across many
# CPU workers, set --shard_count N --shard_index i (0..N-1) and launch in
# parallel.

: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

TRACKS_CSV="${TDADX_DATA_ROOT}/fma_metadata/tracks.csv"
AUDIO_ROOT="${TDADX_DATA_ROOT}/fma_audio"
OUTPUT_DIR="${TDADX_RESULTS_DIR}/fma-48k"

mkdir -p "${OUTPUT_DIR}"

python scripts/01_preprocess_audio.py \
    --tracks_csv   "${TRACKS_CSV}" \
    --audio_root   "${AUDIO_ROOT}" \
    --output_dir   "${OUTPUT_DIR}" \
    --target_sample_hz 48000 \
    --target_channels 1 \
    --target_lufs -14.0 \
    --subset large \
    --num_workers 8 \
    --log_every 100
