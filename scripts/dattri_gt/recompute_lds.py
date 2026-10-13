"""Re-measure dattri's LDS on a directory of score matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dattri.benchmark.load import load_benchmark
from dattri.metric import lds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_dir", required=True)
    ap.add_argument("--cache_path", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["TRAK10", "TracIn", "GradDot", "GradCos"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    _, groundtruth = load_benchmark(
        model="musictransformer", dataset="maestro", metric="lds",
        download_path=args.cache_path,
    )

    score_dir = Path(args.score_dir)
    out = {"score_dir": str(score_dir), "by_method": {}}

    print(f"\n{'method':<10} | LDS    | n_valid_test")
    print("-" * 40)
    for m in args.methods:
        path = score_dir / f"dattri_mt_{m}_scores.npy"
        if not path.exists():
            print(f"{m:<10} | MISSING")
            continue
        s = np.load(path)
        s_t = torch.as_tensor(s.astype(np.float32))
        per_test = lds(s_t, groundtruth)[0]
        mask = ~torch.isnan(per_test)
        if mask.any():
            mean = float(torch.mean(per_test[mask]))
            n_valid = int(mask.sum())
        else:
            mean, n_valid = float("nan"), 0
        print(f"{m:<10} | {mean:+.4f} | {n_valid}/{per_test.numel()}")
        out["by_method"][m] = {"lds": mean, "n_valid": n_valid,
                               "n_total": int(per_test.numel())}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
