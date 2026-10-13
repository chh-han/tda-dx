#!/usr/bin/env python3
"""
Build CLAP-cosine-similarity "attribution" results in the format expected by
analyzer/analyze.py, so the full feature alignment pipeline can run on them as
a comparison baseline against TRAK.

Top-K selection: generated audio CLAP × train CLAP cosine similarity.

This retrieval is stage-agnostic (no gradient computation, no model stages) —
it only uses the final generated audio embedding. Written as a single pseudo-stage
called "fine" so analyze.py can pick it up without modification.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

STAGE = "fine"   # single pseudo-stage; CLAP retrieval has no stage concept


def _load_clap_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    return {k: data[k].astype(np.float32) for k in data.files}


def _build_score_matrix(
    train_track_ids: list[str],
    test_track_ids: list[str],
    train_clap: dict[str, np.ndarray],
    gen_clap: dict[str, np.ndarray],
) -> np.ndarray:
    """L2-normalized cosine similarity: shape (N_train, N_test), float32."""
    n_train = len(train_track_ids)
    n_test = len(test_track_ids)

    # Infer embedding dim from data (supports any encoder, not just 512-dim CLAP)
    _all_embs = list(train_clap.values()) + list(gen_clap.values())
    if not _all_embs:
        raise ValueError("Both train and generated embedding dicts are empty.")
    emb_dim = _all_embs[0].shape[0]

    train_mat = np.zeros((n_train, emb_dim), dtype=np.float32)
    for i, tid in enumerate(train_track_ids):
        if tid in train_clap:
            train_mat[i] = train_clap[tid]

    test_mat = np.zeros((n_test, emb_dim), dtype=np.float32)
    missing_test = 0
    for j, tid in enumerate(test_track_ids):
        if tid in gen_clap:
            test_mat[j] = gen_clap[tid]
        else:
            missing_test += 1
    if missing_test:
        print(f"  WARNING: {missing_test}/{n_test} test tracks not found in generated embedding cache")

    # L2 normalize rows
    train_norms = np.linalg.norm(train_mat, axis=1, keepdims=True)
    test_norms = np.linalg.norm(test_mat, axis=1, keepdims=True)
    train_normed = train_mat / np.where(train_norms > 0, train_norms, 1.0)
    test_normed = test_mat / np.where(test_norms > 0, test_norms, 1.0)

    print(f"  Computing ({n_train} × {n_test}) cosine similarity (dim={emb_dim}) ...", flush=True)
    scores = train_normed @ test_normed.T  # (N_train, N_test)
    return scores.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CLAP cosine retrieval results in attribution format for analyze.py."
    )
    parser.add_argument(
        "--train_clap", required=True,
        help="Path to fma_train_clap_512.npz (track_id → 512-dim embedding).",
    )
    parser.add_argument(
        "--generated_clap", required=True,
        help="Path to generated_clap_512.npz (test track_id → 512-dim embedding of generated audio).",
    )
    parser.add_argument(
        "--trak_metadata", required=True,
        help="Path to any TRAK stage metadata JSON — used to get the canonical test_track_ids ordering.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root output directory. Will create semantic/, coarse/, fine/ subdirs inside.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Idempotency check ────────────────────────────────────────────────────────
    _stage_dir = output_dir / STAGE
    _meta = _stage_dir / f"{STAGE}_cosine_metadata.json"
    _scores = _stage_dir / f"{STAGE}_cosine_scores.npy"
    if _meta.exists() and _scores.exists() and _meta.stat().st_size > 0:
        print("[clap-retrieval] Output already exists, skipping.", flush=True)
        print(f"  Output root: {output_dir}")
        print(f"  Next: python analyzer/analyze.py --input-results-dir {output_dir} --feature-set v1 --skip-baseline-generation ...")
        return

    # ── Load embeddings ──────────────────────────────────────────────────────────
    print("[clap-retrieval] Loading CLAP embeddings ...", flush=True)
    train_clap = _load_clap_npz(Path(args.train_clap))
    gen_clap = _load_clap_npz(Path(args.generated_clap))
    print(f"  Train CLAP cache: {len(train_clap)} tracks", flush=True)
    print(f"  Generated CLAP cache: {len(gen_clap)} tracks", flush=True)

    # ── Test track IDs from TRAK metadata (canonical ordering) ──────────────────
    with open(args.trak_metadata) as f:
        trak_meta = json.load(f)
    test_track_ids: list[str] = [str(v) for v in trak_meta["test_track_ids"]]
    print(f"  Test queries (from TRAK metadata): {len(test_track_ids)}", flush=True)

    # ── Train track IDs from CLAP npz (unique, sorted) ──────────────────────────
    # Sort numerically where possible for reproducibility.
    def _sort_key(tid: str) -> tuple:
        try:
            return (0, int(tid))
        except ValueError:
            return (1, tid)

    train_track_ids: list[str] = sorted(train_clap.keys(), key=_sort_key)
    print(f"  Train tracks (CLAP cache): {len(train_track_ids)}", flush=True)

    # ── Compute score matrix ─────────────────────────────────────────────────────
    scores = _build_score_matrix(train_track_ids, test_track_ids, train_clap, gen_clap)
    print(f"  Scores: shape={scores.shape}, dtype={scores.dtype}, "
          f"min={scores.min():.3f}, max={scores.max():.3f}", flush=True)

    # ── Write attribution-format files (single pseudo-stage) ────────────────────
    stage_dir = output_dir / STAGE
    stage_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(stage_dir / f"{STAGE}_cosine_scores.npy"), scores)

    metadata = {
        "run_id": f"clap_retrieval__cosine",
        "condition": "main",
        "stage": STAGE,
        "method": "cosine",
        "train_track_ids": train_track_ids,
        "test_track_ids": test_track_ids,
        "scores_shape": list(scores.shape),
        "source": "clap_cosine_retrieval",
        "description": (
            "CLAP cosine similarity between generated audio and training tracks. "
            "Stage-agnostic — no attribution method, no model stages."
        ),
    }
    with open(stage_dir / f"{STAGE}_cosine_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Written: {stage_dir}", flush=True)
    print("[clap-retrieval] Done.", flush=True)
    print(f"  Output root: {output_dir}")
    print(f"  Next: python analyzer/analyze.py --input-results-dir {output_dir} --feature-set v1 --skip-baseline-generation ...")


if __name__ == "__main__":
    main()
