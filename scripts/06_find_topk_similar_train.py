"""
Compute CLAP audio embeddings for training set and find top-k similar training samples
for each conditioning audio in a selected folder.
"""

import os
import sys
import csv
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from threading import Thread
from queue import Queue

import torch
import torchaudio
import torchaudio.functional as AF

# Optional workspace-scoped cache redirection. Set TDADX_WORKSPACE to redirect torch / huggingface caches.
_workspace = os.environ.get("TDADX_WORKSPACE")
if _workspace:
    os.environ.setdefault('TORCH_HOME', f"{_workspace}/.cache/torch")
    os.environ.setdefault('HF_HOME', f"{_workspace}/.cache/huggingface")
    os.environ.setdefault('TRANSFORMERS_CACHE', f"{_workspace}/.cache/huggingface/transformers")
    os.environ.setdefault('HF_HUB_CACHE', f"{_workspace}/.cache/huggingface/hub")
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', f"{_workspace}/.cache/huggingface/hub")
    os.environ.setdefault('XDG_CACHE_HOME', f"{_workspace}/.cache")
    os.environ.setdefault('TORCH_HUB_DIR', f"{_workspace}/.cache/torch/hub")
    os.environ.setdefault('TORCHAUDIO_HOME', f"{_workspace}/.cache/torchaudio")
    os.environ.setdefault('MPLCONFIGDIR', f"{_workspace}/.cache/matplotlib")


# ---------------------------------------------------------------------------
# Reproduce SoundDataset.process_audio logic without needing SoundDataset
# (normalize=False, single target_sample_hz, no seq_len_multiple_of)
# ---------------------------------------------------------------------------

def float32_to_int16(x):
    return (x * 32767).to(torch.int16)

def int16_to_float32(x):
    return x.to(torch.float32) / 32767.0

def process_audio_standalone(data, sample_hz, target_sample_hz, pad_to_seconds, pad_to_target_length=False):
    """Replicates SoundDataset.process_audio for normalize=False, single output."""
    # 1. mono
    if data.shape[0] > 1:
        data = torch.mean(data, dim=0).unsqueeze(0)

    # 2. crop/pad to max_length_seconds at original sample rate
    if pad_to_seconds is not None:
        target_length = int(pad_to_seconds * sample_hz)
        audio_length = data.size(1)
        if audio_length > target_length:
            # no random crop for inference
            data = data[:, :target_length]
        elif pad_to_target_length:
            data = torch.nn.functional.pad(data, (0, target_length - audio_length), 'constant')

    # 3. resample
    if target_sample_hz is not None and sample_hz != target_sample_hz:
        data = AF.resample(data, sample_hz, target_sample_hz)

    # 4. int16 quantization (normalize=False path)
    data = int16_to_float32(float32_to_int16(data))

    # 5. remove channel dim: (1, T) -> (T,)
    data = data.squeeze(0).float()

    return data


# ---------------------------------------------------------------------------
# Worker functions for multiprocessing
# ---------------------------------------------------------------------------

_WORKER_TARGET_HZ = None

def _init_worker(target_sample_hz):
    global _WORKER_TARGET_HZ
    _WORKER_TARGET_HZ = target_sample_hz
    torch.set_num_threads(1)


def _load_and_process(args):
    path_str, pad_to_seconds = args
    global _WORKER_TARGET_HZ
    import os
    pid = os.getpid()
    data, sample_hz = torchaudio.load(path_str)
    wave = process_audio_standalone(
        data, sample_hz,
        target_sample_hz=_WORKER_TARGET_HZ,
        pad_to_seconds=pad_to_seconds,
        pad_to_target_length=False,
    )
    return path_str, wave


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_split_track_ids(tracks_csv_path, split_name):
    ids = []
    with open(tracks_csv_path, newline='') as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)

        split_idx = None
        for i, (a, b) in enumerate(zip(header1, header2)):
            if a.strip().lower() == 'set' and b.strip().lower() == 'split':
                split_idx = i
                break
        if split_idx is None:
            raise ValueError('set.split column not found in tracks.csv')

        for row in reader:
            if len(row) <= split_idx:
                continue
            if row[split_idx].strip().lower() == split_name:
                track_id = int(row[0])
                ids.append(track_id)
    return ids


def track_id_to_path(track_id, audio_root):
    filename = f'{track_id:06d}.mp3'
    p1 = audio_root / filename
    if p1.exists():
        return p1
    subdir = f'{track_id:06d}'[:3]
    p2 = audio_root / subdir / filename
    return p2


def load_audio_for_clap(path, pad_to_seconds, target_sample_hz):
    """Single-process audio loading (used for query embeddings and fallback)."""
    data, sample_hz = torchaudio.load(path)
    return process_audio_standalone(
        data, sample_hz,
        target_sample_hz=target_sample_hz,
        pad_to_seconds=pad_to_seconds,
        pad_to_target_length=False,
    )


def compute_clap_audio_embedding(clap_wave_batch, musiclm):
    with torch.no_grad():
        emb = musiclm.clap.clap.get_audio_embedding_from_data(clap_wave_batch)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb


def parse_conditioning_files(selected_dir):
    selected_dir = Path(selected_dir)
    files = sorted(selected_dir.glob('*_conditioning.wav'))
    items = []
    for p in files:
        stem = p.stem
        parts = stem.split('_')
        track_id = parts[2] if len(parts) >= 4 else None
        items.append((p, track_id))
    return items


# ---------------------------------------------------------------------------
# Streaming batch iterator with multiprocessing
# ---------------------------------------------------------------------------

def iter_embed_batches(paths, pad_to_seconds, target_sample_hz, musiclm, batch_size=32, num_workers=0, prefetch=2):
    total = len(paths)

    executor = None
    if num_workers and num_workers > 0:
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(target_sample_hz,),
        )

    load_queue = Queue(maxsize=prefetch)

    def loader_thread():
        for i in range(0, total, batch_size):
            batch_paths = paths[i:i + batch_size]
            load_start = time.time()
            batch_waves = []
            batch_kept = []

            if executor is not None:
                futures = {
                    executor.submit(_load_and_process, (str(p), pad_to_seconds)): p
                    for p in batch_paths
                }
                for future in as_completed(futures):
                    try:
                        path_str, wave = future.result()
                        batch_waves.append(wave)
                        batch_kept.append(Path(path_str))
                    except Exception as e:
                        print(f'skip: {e}', flush=True)
            else:
                for p in batch_paths:
                    try:
                        wave = load_audio_for_clap(p, pad_to_seconds=pad_to_seconds, target_sample_hz=target_sample_hz)
                        batch_waves.append(wave)
                        batch_kept.append(p)
                    except Exception as e:
                        print(f'skip {p}: {e}', flush=True)

            load_time = time.time() - load_start
            done = min(i + batch_size, total)
            load_queue.put((batch_kept, batch_waves, done, load_time))
        load_queue.put(None)  # sentinel

    t = Thread(target=loader_thread, daemon=True)
    t.start()

    try:
        while True:
            item = load_queue.get()
            if item is None:
                break
            batch_kept, batch_waves, done, load_time = item

            if not batch_kept:
                yield batch_kept, None, done, load_time, 0.0
                continue

            embed_start = time.time()
            emb = compute_clap_audio_embedding(batch_waves, musiclm).cpu()
            embed_time = time.time() - embed_start
            yield batch_kept, emb, done, load_time, embed_time
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

# def iter_embed_batches(paths, pad_to_seconds, target_sample_hz, musiclm, batch_size=32, num_workers=0):
#     total = len(paths)
#     print(f'[iter] total={total}, batch_size={batch_size}, num_workers={num_workers}', flush=True)

#     executor = None
#     if num_workers and num_workers > 0:
#         print(f'[iter] creating ProcessPoolExecutor with {num_workers} workers...', flush=True)
#         executor = ProcessPoolExecutor(
#             max_workers=num_workers,
#             initializer=_init_worker,
#             initargs=(target_sample_hz,),
#         )
#         print(f'[iter] executor created', flush=True)

#     try:
#         for i in range(0, total, batch_size):
#             batch_paths = paths[i:i + batch_size]
#             load_start = time.time()

#             batch_waves = []
#             batch_kept = []

#             if executor is not None:
#                 if i == 0:
#                     print(f'[iter] submitting first batch ({len(batch_paths)} files)...', flush=True)
#                 futures = {
#                     executor.submit(_load_and_process, (str(p), pad_to_seconds)): p
#                     for p in batch_paths
#                 }
#                 if i == 0:
#                     print(f'[iter] first batch submitted, waiting for results...', flush=True)
#                 for future in as_completed(futures):
#                     p = futures[future]
#                     try:
#                         path_str, wave = future.result()
#                         batch_waves.append(wave)
#                         batch_kept.append(Path(path_str))
#                         if i == 0 and len(batch_kept) == 1:
#                             print(f'[iter] first file done: {path_str} ({time.time()-load_start:.1f}s)', flush=True)
#                     except Exception as e:
#                         print(f'skip {p}: {e}', flush=True)
#                 if i == 0:
#                     print(f'[iter] first batch fully loaded ({time.time()-load_start:.1f}s)', flush=True)
#             else:
#                 for p in batch_paths:
#                     try:
#                         wave = load_audio_for_clap(p, pad_to_seconds=pad_to_seconds, target_sample_hz=target_sample_hz)
#                         batch_waves.append(wave)
#                         batch_kept.append(p)
#                     except Exception as e:
#                         print(f'skip {p}: {e}', flush=True)

#             load_time = time.time() - load_start
#             done = min(i + batch_size, total)

#             if not batch_kept:
#                 yield batch_kept, None, done, load_time, 0.0
#                 continue

#             embed_start = time.time()
#             emb = compute_clap_audio_embedding(batch_waves, musiclm).cpu()
#             embed_time = time.time() - embed_start
#             if i == 0:
#                 print(f'[iter] first batch embedded ({embed_time:.1f}s)', flush=True)
#             yield batch_kept, emb, done, load_time, embed_time
#     finally:
#         if executor is not None:
#             executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    sys.path.append(str(args.project_root))

    from open_musiclm.config import load_model_config, create_musiclm_from_config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}', flush=True)

    model_config = load_model_config(str(args.model_config))
    musiclm = create_musiclm_from_config(
        model_config=model_config,
        semantic_path=str(args.semantic_ckpt),
        coarse_path=str(args.coarse_ckpt),
        fine_path=str(args.fine_ckpt),
        rvq_path=str(args.rvq_ckpt),
        kmeans_path=str(args.kmeans_ckpt),
        device=device,
    )
    musiclm.eval()

    target_sample_hz = musiclm.clap.sample_rate
    semantic_window_seconds = int(model_config.global_cfg.semantic_audio_length_seconds)
    print(f'target_sample_hz: {target_sample_hz}, semantic_window: {semantic_window_seconds}s', flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load training split paths
    train_ids = load_split_track_ids(args.tracks_csv, 'training')
    train_paths = [track_id_to_path(tid, args.audio_root) for tid in train_ids]
    train_paths = [p for p in train_paths if p.exists()]
    print(f'training tracks: {len(train_ids)} ids, {len(train_paths)} files found', flush=True)

    # load conditioning (query) files
    test_items = parse_conditioning_files(args.selected_dir)
    print(f'conditioning files: {len(test_items)}', flush=True)

    # precompute query embeddings
    query_paths = []
    query_track_ids = []
    query_embs = []
    for i, (path, track_id) in enumerate(test_items):
        print(f'[query {i}] {path}', flush=True)
        clap_wave = load_audio_for_clap(path, pad_to_seconds=semantic_window_seconds, target_sample_hz=target_sample_hz)
        q = compute_clap_audio_embedding([clap_wave], musiclm).cpu().squeeze(0)
        query_paths.append(path)
        query_track_ids.append(track_id)
        query_embs.append(q)

    if query_embs:
        query_embs = torch.stack(query_embs, dim=0)  # (Q, D)
    else:
        print('no query embeddings, exiting', flush=True)
        return

    k = args.topk
    topk_vals = [torch.empty(0) for _ in range(len(query_paths))]
    topk_paths = [[] for _ in range(len(query_paths))]
    bottomk_vals = [torch.empty(0) for _ in range(len(query_paths))]
    bottomk_paths = [[] for _ in range(len(query_paths))]

    # stream over training set
    start_time = time.time()
    interval_load = 0.0
    interval_embed = 0.0
    interval_sim = 0.0

    for batch_paths, batch_emb, done, load_time, embed_time in iter_embed_batches(
        train_paths,
        pad_to_seconds=semantic_window_seconds,
        target_sample_hz=target_sample_hz,
        musiclm=musiclm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    ):
        interval_load += load_time
        interval_embed += embed_time

        if batch_emb is None:
            continue

        sim_start = time.time()
        sims = batch_emb @ query_embs.T  # (B, Q)

        for qi in range(len(query_paths)):
            col = sims[:, qi]
            if col.numel() == 0:
                continue

            # top-k
            if topk_vals[qi].numel() == 0:
                tk = min(k, col.numel())
                vals, idx = torch.topk(col, k=tk)
                topk_vals[qi] = vals
                topk_paths[qi] = [batch_paths[j] for j in idx.tolist()]
            else:
                combined_vals = torch.cat([topk_vals[qi], col], dim=0)
                combined_paths = topk_paths[qi] + list(batch_paths)
                tk = min(k, combined_vals.numel())
                vals, idx = torch.topk(combined_vals, k=tk)
                topk_vals[qi] = vals
                topk_paths[qi] = [combined_paths[j] for j in idx.tolist()]

            # bottom-k
            if bottomk_vals[qi].numel() == 0:
                bk = min(k, col.numel())
                vals, idx = torch.topk(-col, k=bk)
                bottomk_vals[qi] = -vals
                bottomk_paths[qi] = [batch_paths[j] for j in idx.tolist()]
            else:
                combined_vals = torch.cat([bottomk_vals[qi], col], dim=0)
                combined_paths = bottomk_paths[qi] + list(batch_paths)
                bk = min(k, combined_vals.numel())
                vals, idx = torch.topk(-combined_vals, k=bk)
                bottomk_vals[qi] = -vals
                bottomk_paths[qi] = [combined_paths[j] for j in idx.tolist()]

        interval_sim += time.time() - sim_start

        if args.log_every > 0 and (done // args.batch_size) % args.log_every == 0:
            pct = done / len(train_paths) * 100.0
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (len(train_paths) - done) / rate if rate > 0 else float('inf')
            print(
                f'progress: {pct:5.1f}%  {done}/{len(train_paths)}  '
                f'elapsed={elapsed:.1f}s  eta={remaining:.1f}s  '
                f'load={interval_load:.1f}s embed={interval_embed:.1f}s sim={interval_sim:.1f}s',
                flush=True,
            )
            interval_load = interval_embed = interval_sim = 0.0

    # save results
    results = {}
    for i, path in enumerate(query_paths):
        topk_list = [
            {'track_id': Path(p).stem, 'path': str(p), 'score': float(s)}
            for s, p in zip(topk_vals[i].tolist(), topk_paths[i])
        ]
        bottomk_list = [
            {'track_id': Path(p).stem, 'path': str(p), 'score': float(s)}
            for s, p in zip(bottomk_vals[i].tolist(), bottomk_paths[i])
        ]
        results[str(path)] = {
            'conditioning_track_id': query_track_ids[i],
            'conditioning_path': str(path),
            'topk': topk_list,
            'bottomk': bottomk_list,
        }

        print(f'\n[{i}] {path}')
        print('  Top-K')
        for rank, item in enumerate(topk_list, 1):
            print(f"    {rank:02d}. {item['track_id']}  sim={item['score']:.4f}  {item['path']}")
        print('  Bottom-K')
        for rank, item in enumerate(bottomk_list, 1):
            print(f"    {rank:02d}. {item['track_id']}  sim={item['score']:.4f}  {item['path']}")

    out_path = output_dir / 'selected_test_topk_train_sim.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved -> {out_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find top-k similar training samples for conditioning audio')

    parser.add_argument('--project_root', type=Path, required=True,
                        help='project root containing checkpoints/ subdirectory')
    parser.add_argument('--tracks_csv', type=Path, required=True,
                        help='path to FMA tracks.csv metadata file')
    parser.add_argument('--audio_root', type=Path, required=True,
                        help='root directory containing source audio files')
    parser.add_argument('--selected_dir', type=Path, required=True,
                        help='directory of conditioning audio for which to find top-k similar training samples')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='output directory for top-k results')

    parser.add_argument('--model_config', type=Path, required=True,
                        help='model config JSON (e.g. configs/model/musiclm_large.json)')
    parser.add_argument('--semantic_ckpt', type=Path, required=True)
    parser.add_argument('--coarse_ckpt', type=Path, required=True)
    parser.add_argument('--fine_ckpt', type=Path, required=True)
    parser.add_argument('--rvq_ckpt', type=Path, required=True)
    parser.add_argument('--kmeans_ckpt', type=Path, required=True)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=5, help='log progress every N batches')
    parser.add_argument('--num_workers', type=int, default=8, help='number of loader worker processes')

    args = parser.parse_args()
    main(args)