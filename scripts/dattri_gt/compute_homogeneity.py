"""Within-group homogeneity on jSymbolic MIDI features. See CHANNELS below."""
from __future__ import annotations

import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from tda_dx.analysis.io_utils import _cosine_similarity


CHANNELS: Dict[str, List[Tuple[str, str]]] = {
    "harmonic": [
        ("pitch_class_hist", "cosine"),
        ("vert_int_hist",    "cosine"),
        ("mel_int_hist",     "cosine"),
    ],
    "rhythmic": [
        ("note_density",        "scalar"),
        ("mean_rhythmic_value", "scalar"),
        ("rhythmic_value_hist", "cosine"),
    ],
    "dynamic": [
        ("avg_dvelocity", "scalar"),
        ("var_dynamics",  "scalar"),
        ("dynamic_range", "scalar"),
    ],
    "texture": [
        ("mean_polyphony", "scalar"),
        ("var_polyphony",  "scalar"),
        ("pitch_range",    "scalar"),
    ],
}


def _scalar_inv_diff(a: float, b: float) -> float:
    return float(1.0 - abs(a - b) / (abs(a) + abs(b) + 1e-12))


def _pair_sim(fa: Dict[str, Any], fb: Dict[str, Any], dim: str, kind: str):
    if kind == "cosine":
        return _cosine_similarity(fa.get(dim, []), fb.get(dim, []))
    if kind == "scalar":
        return _scalar_inv_diff(float(fa.get(dim, 0.0)), float(fb.get(dim, 0.0)))
    raise ValueError(f"unknown kind: {kind}")


def _within_group_mean(feats, dim, kind):
    if len(feats) < 2:
        return None
    vals: List[float] = []
    for a, b in combinations(feats, 2):
        s = _pair_sim(a, b, dim, kind)
        if s is not None:
            vals.append(float(s))
    if not vals:
        return None
    return float(np.mean(vals))


def _select_groups(score_col, K, n_random, rng, valid_idx):
    order = np.argsort(score_col)
    out = {"top": order[-K:][::-1], "bottom": order[:K]}
    for i in range(n_random):
        out[f"random_{i}"] = rng.choice(valid_idx, size=K, replace=False)
    return out


def _build_feature_null(records, valid_idx, K, B, rng, dims):
    raw = {dim: [] for (dim, _) in dims}
    for _ in range(B):
        sel = rng.choice(valid_idx, size=K, replace=False)
        feats = [records[int(i)] for i in sel]
        for (dim, kind) in dims:
            g = _within_group_mean(feats, dim, kind)
            if g is not None:
                raw[dim].append(g)
    out = {}
    for dim, vals in raw.items():
        if len(vals) >= 2:
            v = np.asarray(vals, dtype=np.float64)
            out[dim] = (float(v.mean()), float(v.std(ddof=1)))
        else:
            out[dim] = (0.0, 1.0)
    return out


def _build_channel_null(records, valid_idx, K, B, rng, feat_null):
    out = {}
    for ch_name, ch_dims in CHANNELS.items():
        per_chan = []
        for _ in range(B):
            sel = rng.choice(valid_idx, size=K, replace=False)
            feats = [records[int(i)] for i in sel]
            zs = []
            for (dim, kind) in ch_dims:
                g = _within_group_mean(feats, dim, kind)
                if g is None:
                    continue
                mu, sd = feat_null[dim]
                zs.append((g - mu) / max(sd, 1e-12))
            if zs:
                per_chan.append(float(np.mean(zs)))
        if len(per_chan) >= 2:
            v = np.asarray(per_chan, dtype=np.float64)
            out[ch_name] = (float(v.mean()), float(v.std(ddof=1)))
        else:
            out[ch_name] = (0.0, 1.0)
    return out


def _zc_for_group(records, idx, feat_null, ch_null):
    feats = [records[int(i)] for i in idx]
    out = {}
    for ch_name, ch_dims in CHANNELS.items():
        zs = []
        for (dim, kind) in ch_dims:
            g = _within_group_mean(feats, dim, kind)
            if g is None:
                continue
            mu, sd = feat_null[dim]
            zs.append((g - mu) / max(sd, 1e-12))
        if not zs:
            out[ch_name] = float("nan")
            continue
        gtilde = float(np.mean(zs))
        mu_c, sd_c = ch_null[ch_name]
        out[ch_name] = (gtilde - mu_c) / max(sd_c, 1e-12)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--train_features", required=True)
    ap.add_argument("--method_name", required=True)
    ap.add_argument("--K", type=int, nargs="+",
                    default=[10, 20, 50, 100, 200, 300, 400, 500])
    ap.add_argument("--n_random_per_query", type=int, default=1)
    ap.add_argument("--null_B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    score = np.load(args.score)
    with open(args.train_features, "rb") as f:
        records = pickle.load(f)

    n_train, n_test = score.shape
    if len(records) > n_train:
        print(f"slicing records {len(records)} -> {n_train}")
        records = records[:n_train]
    assert len(records) == n_train

    valid_idx = np.array([i for i, r in enumerate(records) if r.get("valid", False)])
    print(f"score: {score.shape}  valid_train={len(valid_idx) / n_train:.3f}")

    all_dims = sorted(
        {(dim, kind) for ch in CHANNELS.values() for (dim, kind) in ch},
        key=lambda x: x[0],
    )

    out_doc = {
        "method": args.method_name,
        "version": "jsymbolic",
        "n_train": int(n_train), "n_test": int(n_test),
        "valid_train": float(len(valid_idx) / n_train),
        "channels": {ch: [d for (d, _) in dims] for ch, dims in CHANNELS.items()},
        "by_K": {},
    }

    for K in args.K:
        print(f"\n=== K={K} ===", flush=True)
        feat_null = _build_feature_null(records, valid_idx, K, args.null_B, rng, all_dims)
        ch_null   = _build_channel_null(records, valid_idx, K, args.null_B, rng, feat_null)

        for d, (mu, sd) in feat_null.items():
            print(f"  null  {d:22s} mu={mu:+.4f}  sd={sd:.4f}")
        for ch, (mu, sd) in ch_null.items():
            print(f"  Cnull {ch:10s}            mu={mu:+.4f}  sd={sd:.4f}")

        z_per = {"top":    {c: [] for c in CHANNELS},
                 "bottom": {c: [] for c in CHANNELS},
                 "random": {c: [] for c in CHANNELS}}

        for q in range(n_test):
            col = score[:, q]
            grp = _select_groups(col, K=K, n_random=args.n_random_per_query, rng=rng,
                                 valid_idx=valid_idx)
            zt = _zc_for_group(records, grp["top"],    feat_null, ch_null)
            zb = _zc_for_group(records, grp["bottom"], feat_null, ch_null)
            zrs = [_zc_for_group(records, grp[f"random_{i}"], feat_null, ch_null)
                   for i in range(args.n_random_per_query)]
            for ch in CHANNELS:
                z_per["top"][ch].append(zt[ch])
                z_per["bottom"][ch].append(zb[ch])
                z_per["random"][ch].append(float(np.nanmean([z[ch] for z in zrs])))

        agg = {g: {} for g in z_per}
        for g, by_ch in z_per.items():
            for ch, vals in by_ch.items():
                z = np.asarray(vals, dtype=np.float64)
                z = z[~np.isnan(z)]
                agg[g][ch] = {
                    "z_bar":    float(z.mean())   if len(z) else float("nan"),
                    "z_median": float(np.median(z)) if len(z) else float("nan"),
                    "Pos":      float((z > 0).mean())    if len(z) else float("nan"),
                    "Sig":      float((z > 1.96).mean()) if len(z) else float("nan"),
                    "n":        int(len(z)),
                }

        print(f"\n  channel    | top  z̄ / Pos / Sig         | random z̄ / Pos / Sig         | bottom z̄ / Pos / Sig")
        for ch in CHANNELS:
            t, r, b = agg["top"][ch], agg["random"][ch], agg["bottom"][ch]
            print(f"  {ch:10s} | {t['z_bar']:+.3f} / {t['Pos']:.3f} / {t['Sig']:.3f}  |  "
                  f"{r['z_bar']:+.3f} / {r['Pos']:.3f} / {r['Sig']:.3f}  |  "
                  f"{b['z_bar']:+.3f} / {b['Pos']:.3f} / {b['Sig']:.3f}")

        out_doc["by_K"][str(K)] = {
            "feat_null":    {d: {"mu": mu, "sd": sd} for d, (mu, sd) in feat_null.items()},
            "channel_null": {c: {"mu": mu, "sd": sd} for c, (mu, sd) in ch_null.items()},
            "agg": agg,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_doc, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
