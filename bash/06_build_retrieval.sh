#!/usr/bin/env bash
set -euo pipefail
# Stage 6b: Build a CLAP-cosine retrieval baseline. Two steps:
#   (1) Extract CLAP audio embeddings for the train pool and for generated
#       audio (one .npz per set; track_id -> 512-dim vector).
#   (2) Build per-stage cosine score matrices in the same layout as
#       attribution outputs (so analyze.py can be run on them directly).
#
# To switch encoder, replace `clap` with `clews` or `mert` in the extract
# step. Only CLAP feeds into the retrieval matrix builder.
#
# Requires:
#   TDADX_DATA_ROOT, TDADX_RESULTS_DIR

: "${TDADX_DATA_ROOT:?set TDADX_DATA_ROOT}"
: "${TDADX_RESULTS_DIR:?set TDADX_RESULTS_DIR}"

EMB_DIR="${TDADX_RESULTS_DIR}/clap_embeddings"
mkdir -p "${EMB_DIR}"

# (1) Train-pool embeddings
python scripts/06_extract_embeddings.py --encoder clap \
    --audio-dir "${TDADX_RESULTS_DIR}/fma-48k" \
    --tracks-csv "${TDADX_DATA_ROOT}/fma_metadata/tracks.csv" \
    --fma-split  large \
    --checkpoint "${TDADX_RESULTS_DIR}/checkpoints/clap_music_audioset_epoch_15_esc_90.14.pt" \
    --output     "${EMB_DIR}/fma_train_clap_512.npz" \
    --audio-ext  npy --audio-sr 48000

# (1) Generated-audio embeddings
python scripts/06_extract_embeddings.py --encoder clap \
    --audio-dir "${TDADX_RESULTS_DIR}/generated_audio" \
    --checkpoint "${TDADX_RESULTS_DIR}/checkpoints/clap_music_audioset_epoch_15_esc_90.14.pt" \
    --output     "${EMB_DIR}/generated_clap_512.npz" \
    --audio-ext  wav

# (2) Build retrieval score matrices (mirrors attribution layout)
python scripts/06_build_retrieval.py \
    --train_clap     "${EMB_DIR}/fma_train_clap_512.npz" \
    --generated_clap "${EMB_DIR}/generated_clap_512.npz" \
    --trak_metadata  "${TDADX_RESULTS_DIR}/attribution/trak/fine/fine_trak_metadata.json" \
    --output_dir     "${TDADX_RESULTS_DIR}/clap_retrieval"
