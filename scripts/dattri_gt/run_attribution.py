"""Run dattri's MAESTRO+MusicTransformer attribution and save the score matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dattri.algorithm.tracin import TracInAttributor
from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.load import load_benchmark
from dattri.benchmark.models.MusicTransformer.utilities.constants import TOKEN_PAD
from dattri.metric import lds
from dattri.task import AttributionTask


METHOD_CHOICES = ["TRAK-1", "TRAK-10", "TRAK-50", "TracIn", "Grad-Dot", "Grad-Cos"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="TRAK-10", choices=METHOD_CHOICES)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--cache_path", type=str, default=None,
                    help="dattri benchmark download path; defaults to dattri's own cache")
    ap.add_argument("--proj_dim", type=int, default=2048,
                    help="TRAK random projection dim. dattri benchmark uses 2048.")
    ap.add_argument("--proj_max_batch_size", type=int, default=32,
                    help="Forces param chunking when raised. 1024 splits MT params "
                         "into chunks small enough to fit dense projector on 80GB GPU.")
    ap.add_argument("--proj_type", type=str, default="sjlt",
                    choices=["sjlt", "normal", "rademacher"],
                    help="Random projection type. 'sjlt' is sparse and fits 80GB.")
    args = ap.parse_args()
    print(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_kwargs = {"model": "musictransformer", "dataset": "maestro", "metric": "lds"}
    if args.cache_path is not None:
        bench_kwargs["download_path"] = args.cache_path
    model_details, groundtruth = load_benchmark(**bench_kwargs)

    train_loader = DataLoader(
        model_details["train_dataset"],
        batch_size=64,
        shuffle=False,
        sampler=model_details["train_sampler"],
    )
    test_loader = DataLoader(
        model_details["test_dataset"],
        batch_size=64,
        shuffle=False,
    )

    if args.method.startswith("TRAK"):
        ind = int(args.method.split("-")[1])

        def loss_trak(params, data_target_pair):
            x, y = data_target_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction="none")
            output = torch.func.functional_call(model_details["model"], params, x_t)
            output_last = output[:, -1, :]
            y_last = y_t[:, -1]
            logp = -loss_fn(output_last, y_last)
            logit_func = logp - torch.log(1 - torch.exp(logp))
            return logit_func.squeeze(0)

        def correctness_p(params, data_target_pair):
            x, y = data_target_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction="none")
            output = torch.func.functional_call(model_details["model"], params, x_t)
            output_last = output[:, -1, :]
            y_last = y_t[:, -1]
            logp = -loss_fn(output_last, y_last)
            return torch.exp(logp)

        task = AttributionTask(
            loss_func=loss_trak,
            model=model_details["model"].to(args.device),
            checkpoints=model_details["models_half"][0:ind],
        )
        projector_kwargs = {
            "proj_dim": args.proj_dim,
            "proj_max_batch_size": args.proj_max_batch_size,
            "proj_type": args.proj_type,
            "device": args.device,
        }
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=correctness_p,
            device=args.device,
            projector_kwargs=projector_kwargs,
        )
        with torch.no_grad():
            attributor.cache(train_loader)
            score = attributor.attribute(test_loader)

    else:  # TracIn / Grad-Dot / Grad-Cos
        normalized_grad = args.method == "Grad-Cos"
        ensemble = 10 if args.method == "TracIn" else 1

        def loss_tracin(params, data_target_pair):
            x, y = data_target_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction="none")
            output = torch.func.functional_call(model_details["model"], params, x_t)
            output_last = output[:, -1, :]
            y_last = y_t[:, -1]
            return loss_fn(output_last, y_last).squeeze(0)

        ckpts = (
            model_details["models_full"][0:10]
            if args.method == "TracIn"
            else model_details["models_full"][0]
        )
        task = AttributionTask(
            model=model_details["model"].to(args.device),
            loss_func=loss_tracin,
            checkpoints=ckpts,
        )
        attributor = TracInAttributor(
            task=task,
            weight_list=torch.ones(ensemble) * 1e-3,
            normalized_grad=normalized_grad,
            device=args.device,
        )
        with torch.no_grad():
            score = attributor.attribute(train_loader, test_loader)

    score_t = score.detach().cpu()
    score_np = score_t.numpy().astype(np.float32)
    method_tag = args.method.replace("-", "")
    score_path = out_dir / f"dattri_mt_{method_tag}_scores.npy"
    np.save(score_path, score_np)
    print(f"saved score: {score_path}  shape={score_np.shape}")

    lds_per_test = lds(score_t, groundtruth)[0]
    lds_mean = float(torch.mean(lds_per_test[~torch.isnan(lds_per_test)]))
    print(f"LDS: {lds_mean:.4f}")
    (out_dir / f"dattri_mt_{method_tag}_lds.txt").write_text(f"{lds_mean}\n")


if __name__ == "__main__":
    main()
