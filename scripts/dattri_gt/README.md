# `dattri_gt` — symbolic-music ground-truth evaluation for TDA-Dx

Runs TDA-Dx's reliability + homogeneity diagnostics on the dattri
MAESTRO + MusicTransformer benchmark, where attribution methods have a
published LDS ground truth and the training data is exact MIDI. The
agreement between LDS and TDA-Dx's diagnostics is the central sanity
check this sub-pipeline produces.

## Reliability metrics

| Symbol | Field name in JSON | Description |
|---|---|---|
| `r_i`            | `raw_score_spectrum.r_per_rank_S[i-1]` | Singular energy ratio = σ_i² / ‖S‖_F² |
| `p`              | `mean_concentration_ratio.mean`        | Mean concentration ratio (paper Eq. p) |
| `\bar{|\rho|}` | `test_test_correlation.off_diag_abs_mean` | Mean absolute inter-query correlation (newly proposed in this experiment) |

`popularity_fraction.mean` is kept as a legacy alias of `p`.

## jSymbolic feature schema

Strict jSymbolic 2.2 subset (McKay 2010 App. B). Distinct from the
audio analyzer's feature set in `scripts/07_analyze.py`; same
channel-z framework, different descriptors. 12 features, 3 per channel:

```
harmonic   P-2  pitch_class_hist        12-d
           C-2  vert_int_hist           12-d   (interval class on 50 Hz grid)
           M-1  mel_int_hist            25-d   (±12 semitones)
rhythmic   R-3  note_density            scalar
           R-15 mean_rhythmic_value     scalar (avg duration in beats)
           R-23 rhythmic_value_hist    11-d   (32nd .. double whole)
dynamic    D-1  avg_dvelocity           scalar
           D-2  var_dynamics            scalar
           D-3  dynamic_range           scalar
texture    T-2  mean_polyphony          scalar (50 Hz polyphony grid)
           T-7  var_polyphony           scalar
           P-6  pitch_range             scalar (semitones)
```

## Pipeline order

| Stage | Script (`scripts/dattri_gt/`) | Bash (`bash/dattri_gt/`) | Output (`${TDADX_RESULTS_DIR}/dattri_gt/`) |
|-------|------------------------------|--------------------------|--------|
| 0 setup    | —                                          | `00_build_sjlt.sh`             | rebuilds the `sjlt` CUDA extension in the venv (one-off) |
| 1 download | (uses `dattri.benchmark.datasets.maestro`) | `01_download_maestro.sh`       | maestro-v2.0.0 MIDI + dattri's generated_music test set |
| 2 attribute| `run_attribution.py`                       | `02_attribution.sh`            | `scores/dattri_mt_<METHOD>_scores.npy` for TRAK-10 / TracIn / Grad-Dot / Grad-Cos |
| 3 features | `extract_midi_features.py`                 | `03_extract_midi_features.sh`  | `midi_features/maestro_{train,test}_features.pkl` |
| 4 diagnose | (`scripts/07_diagnose_score_matrix.py`)    | `04_diagnose_scores.sh`        | `diagnostics/score_matrix_diagnostics{,_r1}.json` (`r_i`, `p`, $\bar{\|\rho\|}$, ...) |
| 5 r1 residual | `compute_r1_residual.py`                | `05_r1_residuals.sh`           | `scores_r1/dattri_mt_<METHOD>_scores.npy` (S − σ₁u₁v₁ᵀ) |
| 6 homogeneity | `compute_homogeneity.py`                | `06_homogeneity.sh`            | `homogeneity/<METHOD><TAG>.json` over K = 10..500 |
| 7 LDS      | `recompute_lds.py`                         | `07_recompute_lds.sh`          | `diagnostics/lds_{original,r1_residual}.json` |

Bash scripts are env-driven (`TDADX_VENV`, `TDADX_DATA_ROOT`,
`TDADX_RESULTS_DIR`); per-stage overrides such as `METHOD`, `SCORE_DIR`,
`TAG`, `K_LIST`, `NULL_B`, `PROJ_DIM`, `PROJ_MAX_BATCH` cover the
original-vs-r1 variants and the K-sweep.

## Reading the output

1. **Reliability** — `score_matrix_diagnostics.json`: $\bar{|\rho|}$ is
   usually the most discriminative single number; $p$ catches a global
   offset; cumulative $\Sigma r_{1:5}$ catches multi-axis collapse.
2. **Homogeneity** — `homogeneity/<METHOD><TAG>.json`: healthy
   attribution shows moderate, balanced positive top-`z̄` across
   channels; failing attribution shows abnormally large or below-null
   `z̄` on at least one channel.
3. **r1 ablation** — compare a JSON to its `_r1` sibling. A small delta
   means the metric does not live on the rank-1 axis.
4. **LDS** — `lds_original.json` vs `lds_r1_residual.json` for the
   ablation on the attribution-quality side.
