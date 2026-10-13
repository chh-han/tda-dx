# TDA-Dx

*(formerly ARIA)*

TDA-Dx is a diagnostic framework for training data attribution (TDA) in music
generation. It characterises attribution outputs along musical dimensions
(rhythmic, harmonic, timbral) by measuring within-group homogeneity of
attributed groups standardised against random reference groups, and pairs that
with SVD-based score-matrix reliability diagnostics. The framework, the four
attribution methods compared, the three embedding-retrieval baselines, and the
canonical results are described in the accompanying paper.

## Installation

```bash
pip install -e . --no-build-isolation
# Optional extras
pip install -e ".[attribution]"   # adds trak
pip install -e ".[features]"      # adds BeatNet, omnizart
pip install -e ".[dev]"           # adds pytest, ruff
```

`--no-build-isolation` is needed for `sjlt` to find the active `torch`
when compiling its CUDA extension; install `torch` first, and run on a
node with `nvcc` and a visible GPU.

A few feature/eval dependencies are not installable from PyPI in the standard
way and must be set up separately:

- `imagebind` (audio evaluation) — install from the upstream Meta repository.
- `BeatNet` — beat tracking for the rhythmic channel.
- `omnizart` — chord estimation for the harmonic channel.

CUDA-capable GPUs are required for stages 5 and 6.

## Repository layout

```
tda_dx/        Python package (models, data, preprocessing, tokenizers,
               training, inference, features, attribution, analysis, utils)
configs/       JSON configs for model/, training/, attribution/
scripts/       Numbered CLI entry points (01_..07_*) and diagnostics
bash/          Reference shell wrappers around the numbered scripts
tests/         Smoke tests
third_party/   Vendored dependencies made importable through tda_dx/__init__.py
```

## The 7-stage pipeline

| Stage | Script | Bash example | Description |
|---|---|---|---|
| 1 | `scripts/01_preprocess_audio.py` | `bash/01_preprocess_audio.sh` | Decode raw audio to 48 kHz mono `.npy`. |
| 2 | `scripts/02_train_clap_rvq.py`, `scripts/02_train_hubert_kmeans.py` | `bash/02_train_tokenizers.sh` | Optional: fit the CLAP RVQ codebook and the MERT k-means tokenizer. The paper run skips this stage and uses the pretrained open-musiclm checkpoints (see [Pre-trained models](#pre-trained-models)). |
| 3 | `scripts/03_preprocess_tokens.py` | `bash/03_preprocess_tokens.sh` | Tokenise the corpus into MERT semantic, CLAP RVQ, and EnCodec tokens. |
| 4 | `scripts/04_extract_features.py` | `bash/04_extract_features.sh` | Compute rhythmic/harmonic/timbral features per training segment. |
| 5 | `scripts/05_train.py` | `bash/05_train.sh` | Train the three-stage hierarchical transformer (semantic, coarse, fine). |
| 5b | `scripts/05b_generate_testset.py` | — | Run the trained model on FMA test conditioning to produce per-query generated audio and the (semantic, coarse, fine) token sequences that stage 6 attribution scores against. |
| 6 | `scripts/06_run_attribution.py`, `scripts/06_build_retrieval.py`, `scripts/06_extract_embeddings.py`, `scripts/06_find_topk_similar_train.py` | `bash/06_run_attribution.sh`, `bash/06_build_retrieval.sh` | Run a TDA method (TRAK, FactGraSS, GradCos, LoGra) or build an embedding retrieval baseline (CLAP, CLEWS, MERT). |
| 7 | `scripts/07_analyze.py` and diagnostics (`07_diagnose_score_matrix.py`, `07_diagnose_top_svd.py`, `07_compute_u1_channel_corr.py`, `07_compute_rank1_removed_scores.py`, `07_prepare_listening.py`) | `bash/07_analyze.sh`, `bash/07_diagnose.sh` | Compute within-group homogeneity, reliability diagnostics, and rank-1 residual analyses; emit summary tables. |

## Quick start

```bash
export TDADX_DATA_ROOT=/path/to/datasets       # contains fma_metadata, fma_audio
export TDADX_RESULTS_DIR=/path/to/results      # outputs go here

bash bash/01_preprocess_audio.sh
bash bash/02_train_tokenizers.sh              # optional; or use open-musiclm's pretrained tokenizers
bash bash/03_preprocess_tokens.sh
bash bash/04_extract_features.sh
bash bash/05_train.sh                         # GPU; multi-day per stage
bash bash/06_run_attribution.sh
bash bash/07_analyze.sh
bash bash/07_diagnose.sh
```

Stage 5 trains three transformer stages sequentially and is the dominant
compute cost (multiple GPUs, several days end-to-end). Stage 6 attribution
runs are also GPU-bound but shorter; analysis (stage 7) is CPU-only.

## Symbolic-music ground-truth evaluation (`dattri_gt/`)

Audio extraction is unavoidably noisy (Omnizart chord estimates, librosa
beat tracking, etc.). To verify TDA-Dx's reliability + homogeneity
diagnostics independently, `scripts/dattri_gt/` and `bash/dattri_gt/`
run the same diagnostics on the
[dattri](https://github.com/TRAIS-Lab/dattri) MAESTRO + MusicTransformer
benchmark, where every input is exact MIDI and the four attribution
methods come with a published LDS ground truth. See
`scripts/dattri_gt/README.md` for the full pipeline (download MAESTRO →
attribute → jSymbolic V2 features → reliability diagnostics → rank-1
residual ablation → within-group homogeneity → LDS recomputation).

```bash
export TDADX_VENV=/path/to/venv                # the venv `pip install -e .` ran in
export TDADX_DATA_ROOT=/path/to/datasets
export TDADX_RESULTS_DIR=/path/to/results

bash bash/dattri_gt/01_download_maestro.sh
bash bash/dattri_gt/02_attribution.sh
bash bash/dattri_gt/03_extract_midi_features.sh
bash bash/dattri_gt/04_diagnose_scores.sh
bash bash/dattri_gt/05_r1_residuals.sh
METHOD=TRAK10 TAG=_K500  bash bash/dattri_gt/06_homogeneity.sh
bash bash/dattri_gt/07_recompute_lds.sh
```

## Configuration

Configs are JSON files under `configs/`. The canonical configurations used to
produce the paper results are shipped in this repository:

- `configs/model/musiclm_large.json` — canonical transformer architecture
- `configs/training/train_musiclm_fma.json` — canonical training schedule on FMA
- `configs/attribution/default.json` — canonical TDA hyperparameters
  (projection dimension, regularisation, damping, blowup)
Variant configs are not included; the three files above are sufficient to
reproduce the canonical run.

The train / validation / test split is FMA's official `set.split` column in
`fma_metadata/tracks.csv`. Stage 3 records it per segment in
`idx_to_split.json` next to the preprocessed tokens.

## Pre-trained models

The CLAP RVQ and MERT k-means tokenizers are not trained here: the paper run
uses the pretrained checkpoints released by
[open-musiclm](https://github.com/zhvng/open-musiclm). Point `--rvq_path` and
`--kmeans_path` (stages 3 and 5) at those files.

The three transformer checkpoints (semantic, coarse, fine), the tokenized FMA
Large corpus with its split, the generated test queries and the embedding-baseline
score matrices are available at
[huggingface.co/buckets/chh-han/tda-dx](https://huggingface.co/buckets/chh-han/tda-dx).

## Reproducing the paper

The canonical attribution run uses TRAK with projection dimension 4096 and
ridge regularisation `lambda = 0.01`, evaluated at `K = 300` attributed
neighbours per query. The canonical analysis path is

```bash
export TDADX_DATA_ROOT=/path/to/datasets
export TDADX_RESULTS_DIR=/path/to/results
bash bash/07_analyze.sh
bash bash/07_diagnose.sh
```

These emit per-method summaries, including the `paper_table*.tsv` family
(stage-channel z-scores, consistency, enrichment) and aggregate
`summary_by_method.csv`-style files used to populate the paper tables.

### Canonical results (K = 300)

Stage-channel z-scores produced by the analysis pipeline above (see paper
for the full description of metrics and protocol).

| Method | Stage | Rhythmic | Harmonic | Timbral |
|---|---|---:|---:|---:|
| TRAK | semantic | +1.017 | -3.185 | -2.325 |
| TRAK | coarse | -2.120 | -4.959 | -7.788 |
| TRAK | fine | +0.281 | +0.931 | +0.416 |
| CLAP cosine | - | +0.962 | -2.324 | +1.178 |
| CLEWS cosine | - | +0.317 | -2.565 | +0.394 |
| MERT cosine | - | +0.760 | +5.200 | +4.500 |

The complete reliability diagnostics ($r_1$, $r_2$, popularity fraction $p$)
per (method, stage) cell are written to
`<TDADX_RESULTS_DIR>/score_matrix_diagnostics/summary_by_method.csv` by
`bash/07_diagnose.sh`.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

Smoke tests cover module imports, CLI entry-point help, and reliability /
residual numerical behaviour.

## Citation

```bibtex
@article{han2026aria,
  title         = {ARIA: A Diagnostic Framework for Music Training Data Attribution},
  author        = {Han, Changheon and Panahi, Ashkan and Tatar, K{\i}van{\c{c}}},
  journal       = {arXiv preprint arXiv:2605.16181},
  year          = {2026},
  eprint        = {2605.16181},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2605.16181},
}
```

## Acknowledgements

This codebase builds on the following open-source projects. Citations
appear in the paper; the links here are for the implementations on which
this repository directly depends.

**Upstream codebase**

- [open-musiclm](https://github.com/zhvng/open-musiclm) — the MusicLM
  reimplementation used as the base for the model, training, and tokenization
  pipelines. The hierarchical transformer, CLAP RVQ tokenizer, HuBERT k-means
  tokenizer, and EnCodec wrapper are derived from this project.

**Attribution methods**

- [dattri](https://github.com/TRAIS-Lab/dattri) — we vendor the
  `block_projected_if` subpackage under `tda_dx/third_party/dattri/`, which
  provides the FactGraSS implementation. TRAK and GradCos in `tda_dx/attribution/methods/`
  are reimplemented for this repository's pipeline shape but follow the
  formulations in the corresponding papers.

**Audio encoders and features**

- [laion-clap](https://github.com/LAION-AI/CLAP) — vendored under
  `tda_dx/third_party/laion_clap/` (Python sources only; checkpoints are
  fetched separately by the user).
- [encodec](https://github.com/facebookresearch/encodec) — fine acoustic
  tokenizer.
- MERT — semantic tokenizer (HuggingFace).
- [librosa](https://librosa.org) — chroma, MFCC, CQT, onset, Tonnetz features.
- [BeatNet](https://github.com/mjhydri/BeatNet) — beat tracking for the
  rhythmic channel.
- [omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) —
  chord estimation for the harmonic channel.

**Retrieval baselines**

- CLAP, CLEWS, MERT cosine baselines use the encoders above without
  modification.

## License and asset use

This repository is released under the MIT License (see `LICENSE`). Vendored
third-party code under `tda_dx/third_party/` retains its original license.
The table below lists every external asset used in this work along with its
license and the basis on which it is used here. All non-commercial-licensed
assets are used strictly for non-commercial academic research.

| Asset                    | License                                                              | Use here |
|--------------------------|----------------------------------------------------------------------|----------|
| FMA Large (audio)        | Per-track Creative Commons (varies by track)                         | Training corpus and test queries. Track-level CC terms allow research redistribution of derived features and embeddings; raw audio is not redistributed in this repository. |
| FMA dataset code         | MIT                                                                  | Metadata-loading utilities. |
| FMA metadata             | CC-BY-4.0                                                            | Track / artist / genre tables used for split construction. |
| open-musiclm             | MIT                                                                  | Base model, training, and tokenization pipeline (refactored). |
| dattri (block_projected_if subset) | MIT                                                        | FactGraSS implementation, vendored under `tda_dx/third_party/dattri/`. |
| laion-clap               | CC0-1.0 (code), CC-BY-4.0 (model weights)                            | Conditioning encoder, vendored under `tda_dx/third_party/laion_clap/`. |
| EnCodec                  | CC-BY-NC-4.0                                                         | Fine acoustic tokenizer (non-commercial research use). |
| MERT                     | CC-BY-NC-4.0                                                         | Semantic tokenizer and retrieval baseline encoder (non-commercial research use). |
| BeatNet                  | AGPL-3.0                                                             | Beat tracking for the rhythmic channel; invoked at inference only, not redistributed. |
| Omnizart                 | MIT                                                                  | Chord estimation for the harmonic channel. |
| librosa, scikit-learn, scipy, numpy, pandas, torch, transformers, einops | BSD / MIT / Apache-2.0 / PSF | Standard scientific-Python dependencies. |
