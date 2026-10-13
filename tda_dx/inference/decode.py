#!/usr/bin/env python3
"""Decode a random subset of generated token npz files to WAV audio."""

import argparse
import os
import random

import numpy as np
import soundfile as sf
import torch


def load_encodec(device: torch.device):
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    return model


def decode(npz_path: str, model, device: torch.device) -> np.ndarray:
    d = np.load(npz_path)
    coarse = torch.from_numpy(d["coarse"].astype(np.int64))  # (T, 3)
    fine   = torch.from_numpy(d["fine"].astype(np.int64))    # (T, 5)
    codes  = torch.cat([coarse, fine], dim=-1).unsqueeze(0).permute(0, 2, 1).to(device)  # (1, 8, T)
    with torch.no_grad():
        wave = model.decode([(codes, None)])
    return wave.squeeze().cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--token-dir", required=True, help="Directory containing .npz token files")
    p.add_argument("--audio-dir", required=True, help="Output directory for decoded .wav files")
    p.add_argument("--n",    type=int, default=20, help="Number of files to decode (default: all when >= total)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    all_files = sorted(f for f in os.listdir(args.token_dir) if f.endswith(".npz"))
    if not all_files:
        raise RuntimeError(f"No .npz files found in {args.token_dir}")

    selected = random.Random(args.seed).sample(all_files, k=min(args.n, len(all_files)))
    print(f"Decoding {len(selected)} / {len(all_files)} files  (seed={args.seed}, device={device})")

    os.makedirs(args.audio_dir, exist_ok=True)
    model = load_encodec(device)

    ok = err = skipped = 0
    for fname in selected:
        out_path = os.path.join(args.audio_dir, fname.replace(".npz", ".wav"))
        if os.path.exists(out_path):
            skipped += 1
            continue
        try:
            wave = decode(os.path.join(args.token_dir, fname), model, device)
            sf.write(out_path, wave, 24000, subtype="FLOAT")
            print(f"  OK  {fname}  ({len(wave)/24000:.1f}s)")
            ok += 1
        except Exception as e:
            print(f"  ERR {fname}: {e}")
            err += 1

    print(f"Done.  ok={ok}  skipped={skipped}  err={err}")


if __name__ == "__main__":
    main()
