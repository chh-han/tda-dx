"""α_c = |corr(u₁, channel-c per-track feature)|."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


CHANNEL_FEATURES: dict[str, dict[str, list[str]]] = {
    "rhythmic": {
        "beat_interval_hist":  ["beat_interval_hist"],
        "onset_interval_hist": ["onset_interval_hist"],
    },
    "harmonic": {
        "chroma":  ["chroma_mean"],
        "tonnetz": ["tonnetz_mean"],
    },
    "timbral": {
        "mfcc": ["mfcc_mean", "mfcc_var"],
        "cqt":  ["cqt_mean",  "cqt_var"],
    },
}

RAW_COLUMNS: list[str] = sorted({
    col
    for ch in CHANNEL_FEATURES.values()
    for cols in ch.values()
    for col in cols
})


def parse_features_tsv(features_tsv: Path):
    print(f"loading features.tsv: {features_tsv}", flush=True)
    t0 = time.time()
    with open(features_tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        col_idx = {name: i for i, name in enumerate(header)}
        if "track_id" not in col_idx:
            raise ValueError("features.tsv missing track_id column")
        wanted = [c for c in RAW_COLUMNS if c in col_idx]
        missing = [c for c in RAW_COLUMNS if c not in col_idx]
        if missing:
            print(f"  WARN: raw columns missing in tsv: {missing}", flush=True)

        out: dict[str, dict[str, np.ndarray]] = {}
        n = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            tid = parts[col_idx["track_id"]].strip()
            row: dict[str, np.ndarray] = {}
            for col in wanted:
                raw = parts[col_idx[col]].strip()
                if raw == "" or raw.lower() == "nan":
                    row[col] = None
                    continue
                try:
                    v = json.loads(raw)
                    row[col] = np.asarray(v, dtype=np.float64).flatten()
                except (json.JSONDecodeError, ValueError):
                    row[col] = None
            out[tid] = row
            n += 1
    print(f"  loaded {n} tracks in {time.time()-t0:.1f}s", flush=True)
    return out


def streaming_StS(score_path: Path, chunk_rows: int):
    S = np.load(score_path, mmap_mode="r")
    N, n_test = S.shape
    Gtt = np.zeros((n_test, n_test), dtype=np.float64)
    t0 = time.time()
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        block = np.asarray(S[s:e], dtype=np.float64)
        Gtt += block.T @ block
        if (s // chunk_rows) % 16 == 0:
            print(f"    pass1 rows {e}/{N}  elapsed {time.time()-t0:.1f}s",
                  flush=True)
    return N, n_test, Gtt


def stream_compute_u1(score_path: Path, v1: np.ndarray, sigma1: float,
                      chunk_rows: int) -> np.ndarray:
    S = np.load(score_path, mmap_mode="r")
    N, _ = S.shape
    u1 = np.empty(N, dtype=np.float64)
    v1_f64 = v1.astype(np.float64)
    t0 = time.time()
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        block = np.asarray(S[s:e], dtype=np.float64)
        u1[s:e] = block @ v1_f64 / max(sigma1, 1e-30)
        if (s // chunk_rows) % 16 == 0:
            print(f"    pass2 rows {e}/{N}  elapsed {time.time()-t0:.1f}s",
                  flush=True)
    return u1


def build_feature_arrays(features_db: dict, train_track_ids: list[str]):
    logical_to_cols: dict[str, list[str]] = {
        fn: cols for ch in CHANNEL_FEATURES.values() for fn, cols in ch.items()
    }
    N = len(train_track_ids)

    raw_dim: dict[str, int] = {}
    for tid in train_track_ids[:200]:
        rec = features_db.get(str(tid))
        if not rec:
            continue
        for col, val in rec.items():
            if val is None or col in raw_dim:
                continue
            raw_dim[col] = int(val.shape[0])

    dim_of: dict[str, int] = {}
    for fn, cols in logical_to_cols.items():
        if not all(c in raw_dim for c in cols):
            print(f"  WARN: cannot resolve dim for logical feature {fn} "
                  f"(raw cols {cols}, known dims {list(raw_dim)}); using 1",
                  flush=True)
            dim_of[fn] = 1
        else:
            dim_of[fn] = sum(raw_dim[c] for c in cols)
    print(f"  logical feature dims: {dim_of}", flush=True)

    feat_arr = {fn: np.full((N, dim_of[fn]), np.nan, dtype=np.float64)
                for fn in logical_to_cols}
    n_missing = 0
    for i, tid in enumerate(train_track_ids):
        rec = features_db.get(str(tid))
        if not rec:
            n_missing += 1
            continue
        for fn, cols in logical_to_cols.items():
            parts = []
            ok = True
            for c in cols:
                v = rec.get(c)
                if v is None or v.shape[0] != raw_dim.get(c, -1):
                    ok = False
                    break
                parts.append(v)
            if not ok:
                continue
            feat_arr[fn][i] = np.concatenate(parts)
    if n_missing:
        print(f"    missing features for {n_missing}/{N} train tracks", flush=True)
    valid_mask = {fn: ~np.isnan(feat_arr[fn]).any(axis=1)
                  for fn in logical_to_cols}
    return feat_arr, valid_mask


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    num = float((xm * ym).sum())
    den = float(np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))
    if den == 0.0:
        return 0.0
    return num / den


def regression_R(u1: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0 or X.shape[0] < 3:
        return 0.0
    u_c = u1 - u1.mean()
    Xc = X - X.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(Xc, u_c, rcond=None)
    pred = Xc @ coef
    ss_res = float(((u_c - pred) ** 2).sum())
    ss_tot = float((u_c ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    R2 = 1.0 - ss_res / ss_tot
    R2 = max(0.0, min(1.0, R2))
    return float(np.sqrt(R2))


def per_channel_correlations(u1: np.ndarray, feat_arr: dict, valid_mask: dict):
    out = {}
    for ch, ch_feats in CHANNEL_FEATURES.items():
        ch_record = {"per_feature": {}, "alpha_max_per_dim": 0.0,
                     "alpha_max_regression": 0.0}
        for fn in ch_feats:
            if fn not in feat_arr:
                continue
            X = feat_arr[fn]
            m = valid_mask[fn]
            if m.sum() < 100:
                continue
            X_v = X[m]
            u_v = u1[m]
            per_dim = []
            for d in range(X_v.shape[1]):
                col = X_v[:, d]
                if col.std() == 0 or u_v.std() == 0:
                    per_dim.append(0.0)
                    continue
                per_dim.append(abs(_pearson(u_v, col)))
            per_dim_arr = np.asarray(per_dim, dtype=np.float64)
            R = regression_R(u_v, X_v)
            ch_record["per_feature"][fn] = {
                "n_dims": int(X_v.shape[1]),
                "n_valid": int(m.sum()),
                "per_dim_abs_corr": per_dim_arr.tolist(),
                "max_dim_abs_corr": float(per_dim_arr.max() if per_dim_arr.size else 0.0),
                "regression_R": float(R),
            }
            ch_record["alpha_max_per_dim"] = max(
                ch_record["alpha_max_per_dim"], float(per_dim_arr.max() if per_dim_arr.size else 0.0)
            )
            ch_record["alpha_max_regression"] = max(
                ch_record["alpha_max_regression"], float(R)
            )
        out[ch] = ch_record
    return out


def process_cell(method: str, stage: str, score_path: Path, meta_path: Path,
                 features_db: dict, chunk_rows: int, label: str | None = None):
    display = label if label else method
    print(f"\n=== {display} / {stage} ===", flush=True)
    if not score_path.exists():
        print(f"  MISSING {score_path} — skip", flush=True)
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    train_track_ids = [str(t) for t in meta.get("train_track_ids", [])]
    if not train_track_ids:
        print(f"  no train_track_ids in {meta_path}", flush=True)
        return None
    N_meta = len(train_track_ids)

    t0 = time.time()
    N, n_test, Gtt = streaming_StS(score_path, chunk_rows)
    if N != N_meta:
        print(f"  WARN: N mismatch (scores={N} vs meta={N_meta})", flush=True)

    G_sym = 0.5 * (Gtt + Gtt.T)
    eigvals, eigvecs = np.linalg.eigh(G_sym)
    eigvals = eigvals[::-1]; eigvecs = eigvecs[:, ::-1]
    sigma1 = float(np.sqrt(max(eigvals[0], 0.0)))
    v1 = eigvecs[:, 0].astype(np.float64)
    fro2 = float(np.clip(eigvals, 0.0, None).sum())
    top1_ratio = float(eigvals[0] / max(fro2, 1e-30))
    print(f"  σ_1²/Frob² = {top1_ratio:.4f}  σ_1 = {sigma1:.3e}", flush=True)

    u1 = stream_compute_u1(score_path, v1, sigma1, chunk_rows)
    print(f"  u_1 shape: {u1.shape}  mean={u1.mean():+.4f}  std={u1.std():.4f}",
          flush=True)

    feat_arr, valid_mask = build_feature_arrays(features_db, train_track_ids)
    cors = per_channel_correlations(u1, feat_arr, valid_mask)
    elapsed = time.time() - t0
    summary = {
        "method": label if label else method,
        "stage": stage,
        "N": int(N),
        "n_test": int(n_test),
        "sigma1_sq_over_frob_sq": top1_ratio,
        "sigma1": sigma1,
        "u1_mean": float(u1.mean()),
        "u1_std": float(u1.std()),
        "channels": cors,
        "elapsed_seconds": elapsed,
    }
    for ch, rec in cors.items():
        print(f"    α_{ch:9s}  max|dim corr|={rec['alpha_max_per_dim']:.4f}  "
              f"regression R={rec['alpha_max_regression']:.4f}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_tsv", type=Path, required=True)
    p.add_argument("--target", action="append", required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--stages", nargs="+", default=["semantic", "coarse", "fine"])
    p.add_argument("--chunk_rows", type=int, default=8192)
    args = p.parse_args()

    features_db = parse_features_tsv(args.features_tsv)
    args.out_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for tgt in args.target:
        parts = tgt.split(":", 2)
        if len(parts) == 3:
            label, file_method, orig = parts[0], parts[1], Path(parts[2])
        elif len(parts) == 2:
            label, file_method, orig = parts[0], parts[0], Path(parts[1])
        else:
            raise ValueError(f"--target must be 'label:path' or 'label:file_method:path', got: {tgt!r}")
        for stage in args.stages:
            score_path = orig / stage / f"{stage}_{file_method}_scores.npy"
            meta_path = orig / stage / f"{stage}_{file_method}_metadata.json"
            summary = process_cell(file_method, stage, score_path, meta_path,
                                   features_db, args.chunk_rows, label=label)
            if summary:
                runs.append(summary)

    json_out = args.out_root / "u1_channel_correlations.json"
    with open(json_out, "w") as f:
        json.dump({"runs": runs}, f, indent=2)
    print(f"\nwrote {json_out}", flush=True)

    csv_out = args.out_root / "u1_channel_correlations.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "stage", "N", "n_test", "sigma1_sq_over_frob_sq",
            "alpha_rhythmic_max_dim", "alpha_rhythmic_regression",
            "alpha_harmonic_max_dim", "alpha_harmonic_regression",
            "alpha_timbral_max_dim",  "alpha_timbral_regression",
            "beat_interval_hist_max_dim_corr", "onset_interval_hist_max_dim_corr",
            "chroma_max_dim_corr", "tonnetz_max_dim_corr",
            "mfcc_max_dim_corr", "cqt_max_dim_corr",
        ])
        for r in runs:
            ch = r["channels"]
            def get_max(ch_name, fname):
                if ch_name not in ch:
                    return ""
                feat = ch[ch_name]["per_feature"].get(fname)
                if not feat:
                    return ""
                return f"{feat['max_dim_abs_corr']:.4f}"
            row = [
                r["method"], r["stage"], r["N"], r["n_test"],
                f"{r['sigma1_sq_over_frob_sq']:.4f}",
                f"{ch.get('rhythmic',{}).get('alpha_max_per_dim', 0):.4f}",
                f"{ch.get('rhythmic',{}).get('alpha_max_regression', 0):.4f}",
                f"{ch.get('harmonic',{}).get('alpha_max_per_dim', 0):.4f}",
                f"{ch.get('harmonic',{}).get('alpha_max_regression', 0):.4f}",
                f"{ch.get('timbral',{}).get('alpha_max_per_dim', 0):.4f}",
                f"{ch.get('timbral',{}).get('alpha_max_regression', 0):.4f}",
                get_max("rhythmic", "beat_interval_hist"),
                get_max("rhythmic", "onset_interval_hist"),
                get_max("harmonic", "chroma"),
                get_max("harmonic", "tonnetz"),
                get_max("timbral",  "mfcc"),
                get_max("timbral",  "cqt"),
            ]
            w.writerow(row)
    print(f"wrote {csv_out}", flush=True)


if __name__ == "__main__":
    main()
