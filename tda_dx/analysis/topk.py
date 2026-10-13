"""Top-k training-track similarity and enrichment computation."""

import json, os
import multiprocessing as mp
from itertools import combinations
from typing import Any
import numpy as np
from tqdm import tqdm
from tda_dx.analysis.io_utils import _mean, _std
from tda_dx.analysis._common import FEATURE_DIMS, EVIDENCE_CHANNELS, N_BG_SAMPLES, _select_rank_indices, _cosine_similarity_np
from tda_dx.analysis.homogeneity import (compute_group_profile_strength, compute_track_similarity,
    _pairwise_similarity_from_prepared, _compute_contrast_stats,
    _sample_random_profile_values, build_feature_context)
from tda_dx.analysis.summary import build_enrichment_summary


_TOPK_WORKER_CONTEXT: dict[str, Any] | None = None


def _compute_topk_similarity_row(
    test_idx: int,
    stage_data: dict[str, Any],
    top_k: int,
    rank_selection: str,
    feature_table: dict[str, dict[str, Any]],
    feature_context: dict[str, Any],
    compute_loo: bool,
    shared_bg_k: dict[str, list[float]] | None = None,
    shared_bg_kminus1: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    scores = stage_data["scores"]
    train_track_ids = stage_data["track_ids"]
    test_track_ids = stage_data["test_track_ids"]
    test_genres = stage_data["test_genres"]
    stage = stage_data["stage"]
    condition = stage_data["condition"]
    method = stage_data["method"]

    prepared_feature_table = feature_context["prepared_tracks"]
    stft_zstats = feature_context["stft_zstats"]
    mfcc_zstats = feature_context["mfcc_zstats"]
    cqt_zstats = feature_context["cqt_zstats"]
    vqt_zstats = feature_context["vqt_zstats"]
    chroma_zstats = feature_context["chroma_zstats"]
    tonnetz_zstats = feature_context["tonnetz_zstats"]
    beat_hist_zstats = feature_context["beat_hist_zstats"]
    onset_hist_zstats = feature_context["onset_hist_zstats"]
    beat_interval_hist_zstats = feature_context.get("beat_interval_hist_zstats")

    test_track_id = test_track_ids[test_idx]
    col_scores = np.asarray(scores[:, test_idx], dtype=np.float64)
    effective_k = min(max(1, int(top_k)), len(train_track_ids))
    top_indices = _select_rank_indices(col_scores, effective_k, rank_selection)
    top_track_ids = [train_track_ids[i] for i in top_indices.tolist()]
    n_top_with_features = sum(1 for track_id in top_track_ids if track_id in feature_table)

    row: dict[str, Any] = {
        "condition": condition, "stage": stage, "method": method,
        "test_index": test_idx, "test_track_id": test_track_id,
        "test_genre": test_genres[test_idx] if test_idx < len(test_genres) else "",
        "rank_selection": rank_selection,
        "top_k": effective_k,
        "topk_train_track_ids": json.dumps(top_track_ids),
        "feature_table_hit": n_top_with_features > 0,
    }

    top_profile = compute_group_profile_strength(
        top_track_ids,
        feature_table=feature_table,
        stft_zstats=stft_zstats,
        mfcc_zstats=mfcc_zstats,
        cqt_zstats=cqt_zstats,
        vqt_zstats=vqt_zstats,
        chroma_zstats=chroma_zstats,
        tonnetz_zstats=tonnetz_zstats,
        beat_hist_zstats=beat_hist_zstats,
        onset_hist_zstats=onset_hist_zstats,
        beat_interval_hist_zstats=beat_interval_hist_zstats,
        prepared_feature_table=prepared_feature_table,
    )
    if shared_bg_k is not None:
        random_profiles_k = shared_bg_k
    else:
        bg_rng = np.random.default_rng(seed=(10007 * (test_idx + 1)))
        random_profiles_k = _sample_random_profile_values(
            train_track_ids=train_track_ids,
            group_size=effective_k,
            num_samples=N_BG_SAMPLES,
            rng=bg_rng,
            feature_table=feature_table,
            stft_zstats=stft_zstats,
            mfcc_zstats=mfcc_zstats,
            cqt_zstats=cqt_zstats,
            vqt_zstats=vqt_zstats,
            chroma_zstats=chroma_zstats,
            tonnetz_zstats=tonnetz_zstats,
            beat_hist_zstats=beat_hist_zstats,
            onset_hist_zstats=onset_hist_zstats,
            beat_interval_hist_zstats=beat_interval_hist_zstats,
            prepared_feature_table=prepared_feature_table,
        )

    for dim in FEATURE_DIMS:
        top_value = top_profile.get(dim)
        row[f"profile_top_{dim}"] = top_value
        row[f"topk_mean_{dim}"] = top_value

        z, bg_mean, bg_std, pct = _compute_contrast_stats(top_value, random_profiles_k[dim])
        row[f"profile_contrast_z_{dim}"] = z
        row[f"profile_contrast_bg_mean_{dim}"] = bg_mean
        row[f"profile_contrast_bg_std_{dim}"] = bg_std
        row[f"profile_contrast_pct_{dim}"] = pct
        row[f"enrichment_z_{dim}"] = z
        row[f"enrichment_attr_{dim}"] = top_value
        row[f"enrichment_bg_mean_{dim}"] = bg_mean
        row[f"enrichment_bg_std_{dim}"] = bg_std

    for channel, dims in EVIDENCE_CHANNELS.items():
        standardized_bg: list[np.ndarray] = []
        standardized_query: list[float] = []
        for d in dims:
            query_z = row.get(f"profile_contrast_z_{d}")
            bg_mean_d = row.get(f"profile_contrast_bg_mean_{d}")
            bg_std_d = row.get(f"profile_contrast_bg_std_{d}")
            bg_vals = random_profiles_k.get(d) or []
            if query_z is None or bg_mean_d is None or bg_std_d is None or not bg_vals:
                continue
            if bg_std_d <= 1e-12:
                continue
            standardized_bg.append((np.asarray(bg_vals, dtype=np.float64) - bg_mean_d) / bg_std_d)
            standardized_query.append(float(query_z))
        if not standardized_bg:
            ch_z = None
        else:
            bg_chan = np.stack(standardized_bg, axis=0).mean(axis=0)
            mu_c = float(bg_chan.mean())
            sigma_c = float(bg_chan.std(ddof=1)) if len(bg_chan) > 1 else 0.0
            raw_chan = sum(standardized_query) / len(standardized_query)
            ch_z = (raw_chan - mu_c) / sigma_c if sigma_c > 1e-12 else None
        row[f"channel_profile_z_{channel}"] = ch_z
        row[f"channel_z_{channel}"] = ch_z

    topk_feats = [(track_id, prepared_feature_table.get(track_id), feature_table.get(track_id)) for track_id in top_track_ids]
    pw_sims_per_dim: dict[str, list[float]] = {dim: [] for dim in FEATURE_DIMS}
    pw_counts_per_dim: dict[str, int] = {dim: 0 for dim in FEATURE_DIMS}
    for (_, fa_prepared, fa_raw), (_, fb_prepared, fb_raw) in combinations(topk_feats, 2):
        if fa_prepared is None or fb_prepared is None:
            if fa_raw is None or fb_raw is None:
                continue
            pair_sim = compute_track_similarity(
                fa_raw,
                fb_raw,
                stft_zstats=stft_zstats,
                mfcc_zstats=mfcc_zstats,
                cqt_zstats=cqt_zstats,
                vqt_zstats=vqt_zstats,
                chroma_zstats=chroma_zstats,
                tonnetz_zstats=tonnetz_zstats,
                beat_hist_zstats=beat_hist_zstats,
                onset_hist_zstats=onset_hist_zstats,
            )
        else:
            pair_sim = _pairwise_similarity_from_prepared(fa_prepared, fb_prepared)
            if fa_prepared.get("onset_interval_hist") is not None and fb_prepared.get("onset_interval_hist") is not None:
                pair_sim["onset_hist_sim"] = _cosine_similarity_np(
                    fa_prepared.get("onset_interval_hist"),
                    fb_prepared.get("onset_interval_hist"),
                )
        for dim in FEATURE_DIMS:
            value = pair_sim.get(dim)
            if value is None:
                continue
            pw_sims_per_dim[dim].append(float(value))
            pw_counts_per_dim[dim] += 1
    for dim in FEATURE_DIMS:
        row[f"topk_pairwise_{dim}"] = _mean(pw_sims_per_dim[dim]) if pw_sims_per_dim[dim] else None
        row[f"topk_pairwise_count_{dim}"] = pw_counts_per_dim[dim]

    if compute_loo and len(top_track_ids) >= 2:
        if shared_bg_kminus1 is not None:
            random_profiles_kminus1 = shared_bg_kminus1
        else:
            loo_rng = np.random.default_rng(seed=(20011 * (test_idx + 1)))
            random_profiles_kminus1 = _sample_random_profile_values(
                train_track_ids=train_track_ids,
                group_size=len(top_track_ids) - 1,
                num_samples=N_BG_SAMPLES,
                rng=loo_rng,
                feature_table=feature_table,
                stft_zstats=stft_zstats,
                mfcc_zstats=mfcc_zstats,
                cqt_zstats=cqt_zstats,
                vqt_zstats=vqt_zstats,
                chroma_zstats=chroma_zstats,
                tonnetz_zstats=tonnetz_zstats,
                beat_hist_zstats=beat_hist_zstats,
                onset_hist_zstats=onset_hist_zstats,
                beat_interval_hist_zstats=beat_interval_hist_zstats,
                prepared_feature_table=prepared_feature_table,
            )
        for dim in FEATURE_DIMS:
            full_z = row.get(f"profile_contrast_z_{dim}")
            impacts: list[float] = []
            if full_z is None:
                row[f"loo_max_contribution_{dim}"] = None
                row[f"loo_contribution_std_{dim}"] = None
                continue
            for i in range(len(top_track_ids)):
                loo_group = top_track_ids[:i] + top_track_ids[i + 1:]
                loo_profile = compute_group_profile_strength(
                    loo_group,
                    feature_table=feature_table,
                    stft_zstats=stft_zstats,
                    mfcc_zstats=mfcc_zstats,
                    cqt_zstats=cqt_zstats,
                    vqt_zstats=vqt_zstats,
                    chroma_zstats=chroma_zstats,
                    tonnetz_zstats=tonnetz_zstats,
                    beat_hist_zstats=beat_hist_zstats,
                    onset_hist_zstats=onset_hist_zstats,
                    beat_interval_hist_zstats=beat_interval_hist_zstats,
                    prepared_feature_table=prepared_feature_table,
                )
                loo_z, _, _, _ = _compute_contrast_stats(loo_profile.get(dim), random_profiles_kminus1[dim])
                if loo_z is not None:
                    impacts.append(float(full_z) - float(loo_z))
            if impacts:
                row[f"loo_max_contribution_{dim}"] = max(impacts)
                row[f"loo_contribution_std_{dim}"] = _std(impacts)
            else:
                row[f"loo_max_contribution_{dim}"] = None
                row[f"loo_contribution_std_{dim}"] = None
    else:
        for dim in FEATURE_DIMS:
            row[f"loo_max_contribution_{dim}"] = None
            row[f"loo_contribution_std_{dim}"] = None

    return row


def _compute_topk_similarity_row_from_global(test_idx: int) -> dict[str, Any]:
    if _TOPK_WORKER_CONTEXT is None:
        raise RuntimeError("Top-k worker context was not initialized.")
    return _compute_topk_similarity_row(
        test_idx=test_idx,
        stage_data=_TOPK_WORKER_CONTEXT["stage_data"],
        top_k=_TOPK_WORKER_CONTEXT["top_k"],
        rank_selection=_TOPK_WORKER_CONTEXT["rank_selection"],
        feature_table=_TOPK_WORKER_CONTEXT["feature_table"],
        feature_context=_TOPK_WORKER_CONTEXT["feature_context"],
        compute_loo=_TOPK_WORKER_CONTEXT["compute_loo"],
        shared_bg_k=_TOPK_WORKER_CONTEXT.get("shared_bg_k"),
        shared_bg_kminus1=_TOPK_WORKER_CONTEXT.get("shared_bg_kminus1"),
    )


def build_topk_train_similarities(
    stage_data: dict[str, Any],
    top_k: int,
    rank_selection: str,
    feature_table: dict[str, dict[str, Any]],
    feature_context: dict[str, Any] | None = None,
    num_workers: int = 1,
    compute_loo: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_track_ids = stage_data["test_track_ids"]
    stage = stage_data["stage"]
    condition = stage_data["condition"]
    method = stage_data["method"]

    feature_context = feature_context or build_feature_context(feature_table)

    train_track_ids_for_bg = stage_data["track_ids"]
    effective_k_shared = min(max(1, int(top_k)), len(train_track_ids_for_bg))
    print(f"[analyze]       precomputing {N_BG_SAMPLES} shared bg samples (k={effective_k_shared}) ...", flush=True)
    shared_bg_k = _sample_random_profile_values(
        train_track_ids=train_track_ids_for_bg,
        group_size=effective_k_shared,
        num_samples=N_BG_SAMPLES,
        rng=np.random.default_rng(seed=42),
        feature_table=feature_table,
        stft_zstats=feature_context["stft_zstats"],
        mfcc_zstats=feature_context["mfcc_zstats"],
        cqt_zstats=feature_context["cqt_zstats"],
        vqt_zstats=feature_context["vqt_zstats"],
        chroma_zstats=feature_context["chroma_zstats"],
        tonnetz_zstats=feature_context["tonnetz_zstats"],
        beat_hist_zstats=feature_context["beat_hist_zstats"],
        onset_hist_zstats=feature_context["onset_hist_zstats"],
        prepared_feature_table=feature_context["prepared_tracks"],
    )
    shared_bg_kminus1: dict[str, list[float]] | None = None
    if compute_loo and effective_k_shared >= 2:
        shared_bg_kminus1 = _sample_random_profile_values(
            train_track_ids=train_track_ids_for_bg,
            group_size=effective_k_shared - 1,
            num_samples=N_BG_SAMPLES,
            rng=np.random.default_rng(seed=43),
            feature_table=feature_table,
            stft_zstats=feature_context["stft_zstats"],
            mfcc_zstats=feature_context["mfcc_zstats"],
            cqt_zstats=feature_context["cqt_zstats"],
            vqt_zstats=feature_context["vqt_zstats"],
            chroma_zstats=feature_context["chroma_zstats"],
            tonnetz_zstats=feature_context["tonnetz_zstats"],
            beat_hist_zstats=feature_context["beat_hist_zstats"],
            onset_hist_zstats=feature_context["onset_hist_zstats"],
            prepared_feature_table=feature_context["prepared_tracks"],
        )

    effective_workers = max(1, int(num_workers))
    if effective_workers == 1 or len(test_track_ids) <= 1:
        per_test_rows = [
            _compute_topk_similarity_row(
                test_idx=test_idx,
                stage_data=stage_data,
                top_k=top_k,
                rank_selection=rank_selection,
                feature_table=feature_table,
                feature_context=feature_context,
                compute_loo=compute_loo,
                shared_bg_k=shared_bg_k,
                shared_bg_kminus1=shared_bg_kminus1,
            )
            for test_idx in tqdm(range(len(test_track_ids)), desc="queries", unit="q")
        ]
    else:
        worker_count = min(effective_workers, len(test_track_ids), os.cpu_count() or effective_workers)
        global _TOPK_WORKER_CONTEXT
        _TOPK_WORKER_CONTEXT = {
            "stage_data": stage_data,
            "top_k": top_k,
            "rank_selection": rank_selection,
            "feature_table": feature_table,
            "feature_context": feature_context,
            "compute_loo": compute_loo,
            "shared_bg_k": shared_bg_k,
            "shared_bg_kminus1": shared_bg_kminus1,
        }
        try:
            with mp.get_context("fork").Pool(processes=worker_count) as pool:
                per_test_rows = list(tqdm(
                    pool.imap(_compute_topk_similarity_row_from_global, range(len(test_track_ids))),
                    total=len(test_track_ids),
                    desc="queries",
                    unit="q",
                ))
        finally:
            _TOPK_WORKER_CONTEXT = None

    per_test_rows.sort(key=lambda row: int(row["test_index"]))

    summary = build_enrichment_summary(per_test_rows, condition=condition, stage=stage, method=method)
    summary["rank_selection"] = rank_selection
    return per_test_rows, summary
