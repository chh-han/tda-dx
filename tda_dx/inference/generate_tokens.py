"""
Generate and save intermediate tokens for MusicLM attribution.

Instead of generating full audio, saves (clap_tokens, generated_semantic_tokens)
pairs as a .pt file that attribution.py can load directly.

For the semantic stage:
  - Input: conditioning audio (wav files)
  - Output: (clap_token_ids, generated_semantic_token_ids) per track

Usage:
    python scripts/infer_tokens.py \
        --stage semantic \
        --checkpoint_path ./results/semantic/semantic.transformer.200000.pt \
        --rvq_path ./checkpoints/clap.rvq.350.pt \
        --kmeans_path ./results/hubert_kmeans/kmeans.joblib \
        --selected_json ./data/selected_test_topk_train_sim.json \
        --conditioning_audio_folder /path/to/conditioning_wavs \
        --output_path ./results/generated_tokens.pt
"""

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import torch
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tda_dx.config import (
    load_model_config,
    create_semantic_transformer_from_config,
    create_coarse_transformer_from_config,
    create_fine_transformer_from_config,
    create_clap_quantized_from_config,
    create_hubert_kmeans_from_config,
)
from tda_dx.models.musiclm import (
    SemanticStage,
    CoarseStage,
    FineStage,
    get_or_compute_clap_token_ids,
)
from tda_dx.utils import prepare_audio


def resolve_conditioning_audio(track_id, conditioning_path, audio_folder, exts=('wav', 'mp3', 'flac', 'npy')):
    """Find conditioning audio file, trying JSON path first, then audio_folder by track_id."""
    if conditioning_path:
        p = Path(conditioning_path)
        if p.exists():
            return p

    if audio_folder:
        folder = Path(audio_folder)
        # try basename from JSON path
        if conditioning_path:
            by_name = folder / Path(conditioning_path).name
            if by_name.exists():
                return by_name

        # try track_id with various extensions
        for ext in exts:
            for pattern in [f'{track_id}.{ext}', f'*{track_id}*.{ext}']:
                matches = list(folder.glob(pattern))
                if matches:
                    return matches[0]

    return None


def load_audio(path, target_sample_rate, target_length_seconds=None):
    """Load audio file and resample to target rate."""
    path = str(path)
    if path.endswith('.npy'):
        import numpy as np
        waveform = torch.from_numpy(np.load(path)).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        # assume 48kHz for npy files unless specified
        source_sr = 48000
    else:
        waveform, source_sr = torchaudio.load(path)

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if source_sr != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sample_rate)

    # truncate
    if target_length_seconds is not None:
        max_samples = int(target_length_seconds * target_sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

    return waveform


def generate_semantic_tokens(args, model_config, device):
    """Generate semantic tokens for each conditioning track."""
    # load CLAP
    print(f'Loading CLAP from {args.rvq_path}...')
    clap = create_clap_quantized_from_config(model_config, args.rvq_path, device)

    # load semantic transformer
    print(f'Loading semantic transformer from {args.checkpoint_path}...')
    transformer = create_semantic_transformer_from_config(
        model_config, args.checkpoint_path, device,
    )

    semantic_stage = SemanticStage(
        semantic_transformer=transformer,
        clap=clap,
    )
    semantic_stage.eval()

    # read conditioning tracks from JSON
    with open(args.selected_json, 'r') as f:
        selected = json.load(f)

    semantic_window_seconds = model_config.global_cfg.semantic_audio_length_seconds
    semantic_steps_per_second = model_config.hubert_kmeans_cfg.output_hz
    max_time_steps = int(semantic_window_seconds * semantic_steps_per_second)
    clap_sample_rate = clap.sample_rate

    print(f'Semantic window: {semantic_window_seconds}s, steps/sec: {semantic_steps_per_second}, '
          f'max_time_steps: {max_time_steps}')
    print(f'CLAP sample rate: {clap_sample_rate}')

    tokens_list = []
    track_ids = []
    skipped = []

    for i, (query_key, payload) in enumerate(selected.items()):
        cond_id = payload['conditioning_track_id']
        cond_path = payload.get('conditioning_path')

        audio_path = resolve_conditioning_audio(
            cond_id, cond_path, args.conditioning_audio_folder,
        )
        if audio_path is None:
            print(f'  [{i+1}/{len(selected)}] SKIP {cond_id} — audio not found')
            skipped.append(cond_id)
            continue

        print(f'  [{i+1}/{len(selected)}] {cond_id} — {audio_path}')

        # load and prepare audio for CLAP
        waveform = load_audio(
            audio_path,
            target_sample_rate=clap_sample_rate,
            target_length_seconds=semantic_window_seconds,
        )
        waveform = waveform.to(device)

        # compute CLAP token IDs
        clap_token_ids = get_or_compute_clap_token_ids(
            None, clap, conditioning_audio=waveform, conditioning_text=None,
        )  # shape: (1, 12)

        # generate semantic tokens
        with torch.no_grad():
            generated_semantic = semantic_stage.generate(
                clap_token_ids=clap_token_ids,
                max_time_steps=max_time_steps,
                temperature=args.temperature,
                filter_thres=args.filter_thres,
                include_eos_in_output=False,
                append_eos_to_conditioning_tokens=True,
            )  # shape: (1, T)

        # squeeze batch dim and store
        clap_ids = clap_token_ids.squeeze(0).cpu()  # (12,)
        semantic_ids = generated_semantic.squeeze(0).cpu()  # (T,)

        tokens_list.append((clap_ids, semantic_ids))
        track_ids.append(cond_id)

        print(f'    clap: {clap_ids.shape}, semantic: {semantic_ids.shape}')

    if skipped:
        print(f'WARNING: skipped {len(skipped)} tracks: {skipped}')

    return tokens_list, track_ids


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 60)
    print(f'Token Generation — {args.stage} stage')
    print('=' * 60)
    print(f'  Device:            {device}')
    print(f'  Model config:      {args.model_config}')
    print(f'  Checkpoint:        {args.checkpoint_path}')
    print(f'  RVQ path:          {args.rvq_path}')
    print(f'  KMeans path:       {args.kmeans_path}')
    print(f'  Selected JSON:     {args.selected_json}')
    print(f'  Cond audio folder: {args.conditioning_audio_folder or "(not set)"}')
    print(f'  Output path:       {args.output_path}')
    print(f'  Temperature:       {args.temperature}')
    print(f'  Filter thres:      {args.filter_thres}')
    print(f'  Seed:              {args.seed}')
    print('=' * 60)

    model_config = load_model_config(args.model_config)

    torch.manual_seed(args.seed)

    start = time.time()

    if args.stage == 'semantic':
        tokens_list, track_ids = generate_semantic_tokens(args, model_config, device)
    elif args.stage == 'coarse':
        raise NotImplementedError('Coarse token generation not yet implemented')
    elif args.stage == 'fine':
        raise NotImplementedError('Fine token generation not yet implemented')
    else:
        raise ValueError(f'Unknown stage: {args.stage}')

    elapsed = time.time() - start

    if len(tokens_list) == 0:
        print('ERROR: no tokens generated (all conditioning tracks skipped)')
        return

    # save
    output_path = Path(args.output_path)
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / 'generated_tokens.pt'
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'stage': args.stage,
        'tokens': tokens_list,
        'track_ids': track_ids,
        'config': {
            'model_config': args.model_config,
            'checkpoint_path': args.checkpoint_path,
            'rvq_path': args.rvq_path,
            'kmeans_path': args.kmeans_path,
            'temperature': args.temperature,
            'filter_thres': args.filter_thres,
            'seed': args.seed,
            'selected_json': args.selected_json,
        },
    }
    torch.save(output, str(output_path))

    print(f'\nSaved {len(tokens_list)} token pairs to {output_path} ({elapsed:.0f}s)')
    for i, (clap_ids, sem_ids) in enumerate(tokens_list):
        print(f'  [{i}] track={track_ids[i]} clap={clap_ids.shape} semantic={sem_ids.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and save tokens for MusicLM attribution')

    parser.add_argument('--stage', type=str, required=True, choices=['semantic', 'coarse', 'fine'])
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained transformer checkpoint')
    parser.add_argument('--model_config', type=str, default='./configs/model/musiclm_large.json')
    parser.add_argument('--rvq_path', type=str, required=True,
                        help='Path to CLAP RVQ checkpoint')
    parser.add_argument('--kmeans_path', type=str, default=None,
                        help='Path to HuBERT kmeans checkpoint (needed for coarse/fine stages)')

    # conditioning input
    parser.add_argument('--selected_json', type=str, required=True,
                        help='Path to selected_test_topk_train_sim.json')
    parser.add_argument('--conditioning_audio_folder', type=str, default=None,
                        help='Folder with conditioning audio files (resolved by basename or track_id)')

    # generation params
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--filter_thres', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)

    # output
    parser.add_argument('--output_path', type=str, default='./results/generated_tokens.pt',
                        help='Output .pt file path')

    args = parser.parse_args()
    run(args)
