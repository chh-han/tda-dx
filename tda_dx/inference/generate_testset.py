"""
Generate audio for test split tracks using CLAP tokens from preprocessed memmap files.

Example:
  python scripts/05b_generate_testset.py \\
    --preprocessed-data-folder /path/to/fma_preprocessed \\
    --split test \\
    --output-dir /path/to/exp_results/test_audio_conditioned

Optional env vars:
  TDADX_WORKSPACE       redirect torch / huggingface caches to the given dir
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torchaudio
from einops import rearrange
from tqdm import tqdm

_workspace = os.environ.get("TDADX_WORKSPACE")
if _workspace:
    os.environ.setdefault("TORCH_HOME", f"{_workspace}/.cache/torch")
    os.environ.setdefault("HF_HOME", f"{_workspace}/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{_workspace}/.cache/huggingface/transformers")
    os.environ.setdefault("HF_HUB_CACHE", f"{_workspace}/.cache/huggingface/hub")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_workspace}/.cache/huggingface/hub")
    os.environ.setdefault("XDG_CACHE_HOME", f"{_workspace}/.cache")
    os.environ.setdefault("TORCH_HUB_DIR", f"{_workspace}/.cache/torch/hub")
    os.environ.setdefault("TORCHAUDIO_HOME", f"{_workspace}/.cache/torchaudio")
    os.environ.setdefault("MPLCONFIGDIR", f"{_workspace}/.cache/matplotlib")

from tda_dx.config import load_model_config, create_musiclm_from_config
import tda_dx.models.musiclm as open_musiclm_module


def log(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)



def _save_generated_output(token_output_dir: str, audio_output_dir: str, file_stem: str, stage: str, token_format: str, token_payload: dict, decode_audio: bool, wave: np.ndarray, sample_rate: int):
    token_output_dir_path = Path(token_output_dir)
    audio_output_dir_path = Path(audio_output_dir)

    token_file_stem = f"{file_stem}_{stage}"
    if token_format == "npz":
        token_out_path = token_output_dir_path / f"{token_file_stem}.npz"
        np.savez_compressed(str(token_out_path), **token_payload)
    else:
        token_out_path = token_output_dir_path / f"{token_file_stem}.pt"
        torch.save({name: torch.from_numpy(arr) for name, arr in token_payload.items()}, str(token_out_path))

    audio_out_path = None
    if decode_audio and wave is not None:
        audio_out_path = audio_output_dir_path / f"{file_stem}.wav"
        torchaudio.save(str(audio_out_path), torch.from_numpy(wave), sample_rate)

    return str(token_out_path), (str(audio_out_path) if audio_out_path is not None else None)


def normalize_track_id(track_id):
    try:
        return str(int(track_id))
    except (TypeError, ValueError):
        return str(track_id)


def extract_track_id_from_path(path_str: str):
    stem = Path(path_str).stem
    try:
        return str(int(stem))
    except ValueError:
        for part in reversed(stem.split("_")):
            try:
                return str(int(part))
            except ValueError:
                continue
    return stem


def _required_memmap_files(preprocessed_folder: Path):
    return [
        preprocessed_folder / "metadata.json",
        preprocessed_folder / "id_mapping.npy",
        preprocessed_folder / "clap_tokens.npy",
    ]


def _validate_memmap_folder(preprocessed_folder: Path):
    missing = [str(p) for p in _required_memmap_files(preprocessed_folder) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required memmap files in {preprocessed_folder}: {missing}. "
            "Run scripts/preprocess_data.py with --output_format memmap."
        )


def _extract_mapping_from_metadata(metadata: dict):
    for key in ("idx_to_track_id", "index_to_track_id"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return {int(k): normalize_track_id(v) for k, v in value.items()}

    for key in ("idx_to_path", "index_to_path"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return {int(k): normalize_track_id(extract_track_id_from_path(str(v))) for k, v in value.items()}

    for key in ("paths", "file_paths"):
        value = metadata.get(key)
        if isinstance(value, list):
            return {
                int(i): normalize_track_id(extract_track_id_from_path(str(v)))
                for i, v in enumerate(value)
            }

    return None


def _sample_clap_token(clap_array, sqlite_idx: int, seed=None):
    clap_token_ids = torch.from_numpy(clap_array.astype(np.int32))
    if clap_token_ids.ndim == 1:
        clap_token_ids = clap_token_ids.unsqueeze(0)
    # preprocessed clap tokens are typically [num_windows, num_quantizers, 1]
    if clap_token_ids.ndim == 3 and clap_token_ids.shape[-1] == 1:
        clap_token_ids = clap_token_ids.squeeze(-1)
    if clap_token_ids.ndim != 2:
        raise RuntimeError(f"Unexpected clap token shape at idx={sqlite_idx}: {tuple(clap_token_ids.shape)}")

    rng = random.Random(seed) if seed is not None else random
    start_idx = rng.randint(0, clap_token_ids.shape[0] - 1)
    return clap_token_ids[start_idx].unsqueeze(0)


class MemmapTokenStore:
    def __init__(self, preprocessed_folder: Path):
        self.preprocessed_folder = preprocessed_folder

        with open(preprocessed_folder / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        raw_mapping = np.load(preprocessed_folder / "id_mapping.npy", allow_pickle=True).item()
        self.id_mapping = {int(k): int(v) for k, v in raw_mapping.items()}
        self.sqlite_indices = sorted(self.id_mapping.keys())

        clap_shape = tuple(self.metadata["clap_shape"])
        total_count = int(self.metadata["total_count"])
        self.clap_mmap = np.memmap(
            preprocessed_folder / "clap_tokens.npy",
            dtype=np.uint16,
            mode="r",
            shape=(total_count, *clap_shape),
        )

        # Prefer explicit map file from preprocessing; fallback to legacy metadata fields.
        self.idx_to_trackid = None
        idx_to_track_id_file = self.metadata.get("idx_to_track_id_file", "idx_to_track_id.json")
        idx_to_track_id_path = preprocessed_folder / idx_to_track_id_file
        if idx_to_track_id_path.exists():
            with open(idx_to_track_id_path, "r") as f:
                raw = json.load(f)
            self.idx_to_trackid = {int(k): normalize_track_id(v) for k, v in raw.items()}
        else:
            self.idx_to_trackid = _extract_mapping_from_metadata(self.metadata)

        self.idx_to_split = None
        idx_to_split_file = self.metadata.get("idx_to_split_file", "idx_to_split.json")
        idx_to_split_path = preprocessed_folder / idx_to_split_file
        if idx_to_split_path.exists():
            with open(idx_to_split_path, "r") as f:
                raw_split = json.load(f)
            self.idx_to_split = {int(k): str(v).strip().lower() for k, v in raw_split.items()}

    def resolve_track_id(self, sqlite_idx: int):
        if self.idx_to_trackid is None:
            return None
        return normalize_track_id(self.idx_to_trackid.get(int(sqlite_idx))) if int(sqlite_idx) in self.idx_to_trackid else None

    def resolve_split(self, sqlite_idx: int):
        if self.idx_to_split is None:
            return None
        return self.idx_to_split.get(int(sqlite_idx))

    def load_clap_tokens_batch(self, sqlite_indices, seeds=None):
        if len(sqlite_indices) == 0:
            return []

        if seeds is None:
            seeds = [None] * len(sqlite_indices)

        result = []
        for idx, seed in zip(sqlite_indices, seeds):
            idx = int(idx)
            if idx not in self.id_mapping:
                raise RuntimeError(f"idx={idx} not found in id_mapping.npy")
            pos = self.id_mapping[idx]
            result.append(_sample_clap_token(self.clap_mmap[pos], sqlite_idx=idx, seed=seed))
        return result


def load_memmap_entries(memmap_store: MemmapTokenStore, split_name: str = "all", cache_path: Path = None):
    if cache_path is not None and cache_path.exists():
        t_cache = time.time()
        entries = []
        with open(cache_path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx_str, track_id = line.split("\t", 1)
                entries.append((int(idx_str), track_id))
        log(f"loaded memmap-entry cache: {len(entries)} rows in {time.time() - t_cache:.2f}s")
        return entries

    split_key = str(split_name).strip().lower()
    if split_key != "all" and memmap_store.idx_to_split is None:
        raise RuntimeError(
            "Split filtering needs idx_to_split metadata, but it was not found in memmap folder. "
            "Re-run preprocessing with current memmap format."
        )

    entries = []
    t_scan = time.time()
    for i, sqlite_idx in enumerate(memmap_store.sqlite_indices, start=1):
        mapped_track_id = memmap_store.resolve_track_id(sqlite_idx)
        track_id = mapped_track_id if mapped_track_id is not None else normalize_track_id(sqlite_idx)

        if split_key != "all":
            row_split = memmap_store.resolve_split(sqlite_idx)
            if row_split is None or row_split != split_key:
                continue
        entries.append((int(sqlite_idx), track_id))

        if i % 500000 == 0:
            log(f"memmap scan progress: scanned={i}, kept={len(entries)}")

    log(f"memmap scan complete: scanned={len(memmap_store.sqlite_indices)}, kept={len(entries)}, elapsed={time.time() - t_scan:.2f}s")

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
            with open(tmp_path, "w") as f:
                for idx, track_id in entries:
                    f.write(f"{idx}\t{track_id}\n")
            os.replace(tmp_path, cache_path)
            log(f"wrote memmap-entry cache -> {cache_path}")
        except Exception as exc:
            log(f"warning: failed to write memmap-entry cache ({cache_path}): {exc}")

    return entries


def _select_header_column(level0, level1, target):
    target = str(target).strip()
    if not target:
        return None

    target_lower = target.lower()
    target_parts = [p.strip().lower() for p in target.replace(".", "|").split("|") if p.strip()]

    for i, (h0, h1) in enumerate(zip(level0, level1)):
        h0_l = h0.strip().lower()
        h1_l = h1.strip().lower()
        if h1_l == target_lower:
            return i
        if f"{h0_l}|{h1_l}" == target_lower:
            return i
        if f"{h0_l}.{h1_l}" == target_lower:
            return i
        if len(target_parts) == 2 and [h0_l, h1_l] == target_parts:
            return i
    return None


def load_track_to_genre_map(tracks_csv_path: Path, genre_column: str = "genre_top"):
    if not tracks_csv_path.exists():
        raise FileNotFoundError(f"tracks csv not found: {tracks_csv_path}")

    with open(tracks_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            level0 = next(reader)
            level1 = next(reader)
            _ = next(reader)  # third header row (track_id, ... )
        except StopIteration as exc:
            raise RuntimeError(f"Invalid tracks csv format: {tracks_csv_path}") from exc

        if len(level0) != len(level1):
            raise RuntimeError(f"Unexpected tracks csv header width mismatch in {tracks_csv_path}")

        track_id_col = _select_header_column(level0, level1, "track_id")
        if track_id_col is None:
            track_id_col = 0

        genre_col = _select_header_column(level0, level1, genre_column)
        if genre_col is None and str(genre_column).strip().lower() == "genre_top":
            genre_col = _select_header_column(level0, level1, "track|genre_top")
        if genre_col is None:
            raise RuntimeError(
                f"Could not find genre column '{genre_column}' in tracks csv header: {tracks_csv_path}"
            )

        track_to_genre = {}
        for row in reader:
            if len(row) <= max(track_id_col, genre_col):
                continue
            raw_track = row[track_id_col].strip()
            if not raw_track:
                continue
            try:
                norm_track = normalize_track_id(raw_track)
            except Exception:
                continue

            genre = row[genre_col].strip()
            if not genre or genre.lower() in {"nan", "none", "null"}:
                continue

            track_to_genre[norm_track] = genre

    return track_to_genre


def select_genre_balanced_samples(entries, track_to_genre: dict, samples_per_genre: int, seed: int):
    if samples_per_genre <= 0:
        raise ValueError("samples_per_genre must be > 0")

    grouped = defaultdict(list)
    missing_genre_count = 0

    for sqlite_idx, track_id in entries:
        norm_track = normalize_track_id(track_id)
        genre = track_to_genre.get(norm_track)
        if genre is None:
            missing_genre_count += 1
            continue
        grouped[genre].append((sqlite_idx, track_id))

    if not grouped:
        raise RuntimeError("No entries matched any genre from tracks metadata")

    rng = random.Random(seed)
    selected = []
    genre_stats = []

    for genre in sorted(grouped.keys()):
        bucket = grouped[genre]
        k = min(samples_per_genre, len(bucket))
        if k == len(bucket):
            picked = list(bucket)
        else:
            picked = rng.sample(bucket, k=k)
        selected.extend(picked)
        genre_stats.append((genre, len(bucket), k))

    rng.shuffle(selected)
    return selected, genre_stats, missing_genre_count


def write_selection_manifest(manifest_path: Path, samples, track_to_genre: dict, base_seed: int):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("order_idx	sqlite_idx	track_id	genre	file_stem	clap_seed\n")
        for order_idx, (sqlite_idx, track_id) in enumerate(samples):
            norm_track_id = normalize_track_id(track_id)
            genre = track_to_genre.get(norm_track_id, "")
            file_stem = f"{track_id}_{sqlite_idx}"
            clap_seed = base_seed + order_idx
            f.write(
                f"{order_idx}\t{sqlite_idx}\t{track_id}\t{genre}\t{file_stem}\t{clap_seed}\n"
            )


def generate_from_audio_conditioning(
    musiclm,
    clap_token_ids,
    output_seconds=4,
    semantic_window_seconds=10,
    coarse_window_seconds=4,
    fine_window_seconds=2,
    semantic_steps_per_second=50,
    acoustic_steps_per_second=75,
    return_coarse_generated_wave=False,
    semantic_sliding_window_step_percent=0.5,
    coarse_sliding_window_step_percent=0.5,
    fine_sliding_window_step_percent=1.0,
    decode_audio=False,
):
    all_audio_condition_coarse_token_ids = None
    all_audio_condition_fine_token_ids = None
    audio_condition_semantic_token_ids = None
    audio_condition_coarse_token_ids = None
    audio_condition_fine_token_ids = None
    semantic_token_adjustment = 0
    coarse_token_adjustment = 0
    fine_token_adjustment = 0

    all_semantic_token_ids = musiclm.semantic.generate(
        clap_token_ids=clap_token_ids,
        semantic_token_ids=audio_condition_semantic_token_ids,
        max_time_steps=int(min(output_seconds, semantic_window_seconds) * semantic_steps_per_second),
        include_eos_in_output=False,
        append_eos_to_conditioning_tokens=True,
    )

    while all_semantic_token_ids.shape[1] < int(output_seconds * semantic_steps_per_second):
        condition_length = int(
            semantic_window_seconds * semantic_steps_per_second * (1 - semantic_sliding_window_step_percent)
        )
        condition_semantic_token_ids = all_semantic_token_ids[:, -condition_length:]
        pred_semantic_token_ids = musiclm.semantic.generate(
            clap_token_ids=clap_token_ids,
            semantic_token_ids=condition_semantic_token_ids,
            max_time_steps=int(semantic_window_seconds * semantic_steps_per_second),
            include_eos_in_output=False,
            append_eos_to_conditioning_tokens=True,
        )
        pred_semantic_token_ids = pred_semantic_token_ids[:, condition_length:]
        all_semantic_token_ids = torch.cat([all_semantic_token_ids, pred_semantic_token_ids], dim=1)

    all_semantic_token_ids = all_semantic_token_ids[:, semantic_token_adjustment:]

    # Keep semantic outputs in [batch, time, quantizer] for return/saving.
    # Use a separate windowed view for coarse-stage generation.
    window_size = int(coarse_window_seconds * semantic_steps_per_second - 1)
    step_size = int(window_size * coarse_sliding_window_step_percent)
    semantic_windows = all_semantic_token_ids.unfold(1, window_size, step_size)
    semantic_windows = rearrange(semantic_windows, "b n q w -> n b w q")

    all_coarse_token_ids = None
    for semantic_token_ids in semantic_windows:
        if all_coarse_token_ids is not None:
            condition_length = int(
                coarse_window_seconds * acoustic_steps_per_second * (1 - coarse_sliding_window_step_percent)
            )
            condition_coarse_token_ids = all_coarse_token_ids[:, -condition_length:]
        else:
            condition_coarse_token_ids = audio_condition_coarse_token_ids

        pred_coarse_token_ids = musiclm.coarse.generate(
            clap_token_ids=clap_token_ids,
            semantic_token_ids=semantic_token_ids,
            coarse_token_ids=condition_coarse_token_ids,
            max_time_steps=int(coarse_window_seconds * acoustic_steps_per_second),
            reconstruct_wave=False,
            include_eos_in_output=False,
            append_eos_to_conditioning_tokens=True,
            temperature=0.95,
        )

        if all_coarse_token_ids is None:
            all_coarse_token_ids = pred_coarse_token_ids
        else:
            pred_coarse_token_ids = pred_coarse_token_ids[:, condition_length:]
            all_coarse_token_ids = torch.cat([all_coarse_token_ids, pred_coarse_token_ids], dim=1)

    if return_coarse_generated_wave:
        wave = musiclm.neural_codec.decode_from_codebook_indices(all_coarse_token_ids)
        return rearrange(wave, "b 1 n -> b n")

    all_coarse_token_ids = all_coarse_token_ids[:, coarse_token_adjustment:]

    fine_window_size = int(fine_window_seconds * acoustic_steps_per_second)
    fine_step_size = int(fine_window_size * fine_sliding_window_step_percent)
    all_coarse_token_ids_unfolded = all_coarse_token_ids.unfold(1, fine_window_size, fine_step_size)
    all_coarse_token_ids_unfolded = rearrange(all_coarse_token_ids_unfolded, "b n q w -> n b w q")

    all_fine_token_ids = None
    for coarse_token_ids in all_coarse_token_ids_unfolded:
        if all_fine_token_ids is not None:
            condition_length = int(fine_window_size * (1 - fine_sliding_window_step_percent))
            condition_fine_token_ids = all_fine_token_ids[:, -condition_length:] if condition_length > 0 else None
        else:
            condition_fine_token_ids = audio_condition_fine_token_ids

        pred_fine_token_ids = musiclm.fine.generate(
            clap_token_ids=clap_token_ids,
            coarse_token_ids=coarse_token_ids,
            fine_token_ids=condition_fine_token_ids,
            max_time_steps=fine_window_size,
            reconstruct_wave=False,
            include_eos_in_output=False,
            append_eos_to_conditioning_tokens=True,
            temperature=0.4,
        )
        if all_fine_token_ids is None:
            all_fine_token_ids = pred_fine_token_ids
        else:
            pred_fine_token_ids = pred_fine_token_ids[:, condition_length:]
            all_fine_token_ids = torch.cat([all_fine_token_ids, pred_fine_token_ids], dim=1)

    all_fine_token_ids = all_fine_token_ids[:, fine_token_adjustment:]

    if all_audio_condition_coarse_token_ids is not None and all_audio_condition_fine_token_ids is not None:
        all_fine_token_ids = torch.cat([all_audio_condition_fine_token_ids, all_fine_token_ids], dim=1)
        all_coarse_token_ids = torch.cat([all_audio_condition_coarse_token_ids, all_coarse_token_ids], dim=1)

    if not decode_audio:
        return all_semantic_token_ids, all_coarse_token_ids, all_fine_token_ids

    all_acoustic_token_ids = torch.cat([all_coarse_token_ids, all_fine_token_ids], dim=-1)
    wave = musiclm.neural_codec.decode_from_codebook_indices(all_acoustic_token_ids)
    return rearrange(wave, "b 1 n -> b n")


def generate_tokens_for_clap_batch(
    musiclm,
    clap_token_ids_batch,
    output_seconds,
    semantic_window_seconds,
    coarse_window_seconds,
    fine_window_seconds,
    semantic_steps_per_second,
    acoustic_steps_per_second,
):
    expected_bs = int(clap_token_ids_batch.shape[0])

    with torch.no_grad():
        semantic_batch, coarse_batch, fine_batch = generate_from_audio_conditioning(
            musiclm,
            clap_token_ids=clap_token_ids_batch,
            output_seconds=output_seconds,
            semantic_window_seconds=semantic_window_seconds,
            coarse_window_seconds=coarse_window_seconds,
            fine_window_seconds=fine_window_seconds,
            semantic_steps_per_second=semantic_steps_per_second,
            acoustic_steps_per_second=acoustic_steps_per_second,
            decode_audio=False,
        )

    got_bs = int(semantic_batch.shape[0])
    if got_bs == expected_bs and int(coarse_batch.shape[0]) == expected_bs and int(fine_batch.shape[0]) == expected_bs:
        return semantic_batch, coarse_batch, fine_batch

    log(
        f"batch generation returned unexpected batch dim (expected={expected_bs}, got={got_bs}); "
        "falling back to per-sample generation"
    )

    semantic_list = []
    coarse_list = []
    fine_list = []

    for i in range(expected_bs):
        clap_single = clap_token_ids_batch[i:i + 1]
        with torch.no_grad():
            sem_i, coarse_i, fine_i = generate_from_audio_conditioning(
                musiclm,
                clap_token_ids=clap_single,
                output_seconds=output_seconds,
                semantic_window_seconds=semantic_window_seconds,
                coarse_window_seconds=coarse_window_seconds,
                fine_window_seconds=fine_window_seconds,
                semantic_steps_per_second=semantic_steps_per_second,
                acoustic_steps_per_second=acoustic_steps_per_second,
                decode_audio=False,
            )
        semantic_list.append(sem_i)
        coarse_list.append(coarse_i)
        fine_list.append(fine_i)

    return torch.cat(semantic_list, dim=0), torch.cat(coarse_list, dim=0), torch.cat(fine_list, dim=0)


def generate_semantic_tokens_for_clap_batch(
    musiclm,
    clap_token_ids_batch,
    output_seconds,
    semantic_window_seconds,
    semantic_steps_per_second,
    semantic_sliding_window_step_percent=0.5,
):
    with torch.no_grad():
        all_semantic_token_ids = musiclm.semantic.generate(
            clap_token_ids=clap_token_ids_batch,
            semantic_token_ids=None,
            max_time_steps=int(min(output_seconds, semantic_window_seconds) * semantic_steps_per_second),
            include_eos_in_output=False,
            append_eos_to_conditioning_tokens=True,
        )

        while all_semantic_token_ids.shape[1] < int(output_seconds * semantic_steps_per_second):
            condition_length = int(
                semantic_window_seconds * semantic_steps_per_second * (1 - semantic_sliding_window_step_percent)
            )
            condition_semantic_token_ids = all_semantic_token_ids[:, -condition_length:]
            pred_semantic_token_ids = musiclm.semantic.generate(
                clap_token_ids=clap_token_ids_batch,
                semantic_token_ids=condition_semantic_token_ids,
                max_time_steps=int(semantic_window_seconds * semantic_steps_per_second),
                include_eos_in_output=False,
                append_eos_to_conditioning_tokens=True,
            )
            pred_semantic_token_ids = pred_semantic_token_ids[:, condition_length:]
            all_semantic_token_ids = torch.cat([all_semantic_token_ids, pred_semantic_token_ids], dim=1)

    return all_semantic_token_ids


def generate_coarse_tokens_for_batch(
    musiclm,
    clap_token_ids_batch,
    semantic_token_ids_batch,
    coarse_window_seconds,
    semantic_steps_per_second,
    acoustic_steps_per_second,
    coarse_sliding_window_step_percent=0.5,
):
    window_size = int(coarse_window_seconds * semantic_steps_per_second - 1)
    step_size = int(window_size * coarse_sliding_window_step_percent)
    semantic_windows = semantic_token_ids_batch.unfold(1, window_size, step_size)
    semantic_windows = rearrange(semantic_windows, "b n q w -> n b w q")

    all_coarse_token_ids = None
    for semantic_token_ids in semantic_windows:
        if all_coarse_token_ids is not None:
            condition_length = int(
                coarse_window_seconds * acoustic_steps_per_second * (1 - coarse_sliding_window_step_percent)
            )
            condition_coarse_token_ids = all_coarse_token_ids[:, -condition_length:]
        else:
            condition_coarse_token_ids = None

        pred_coarse_token_ids = musiclm.coarse.generate(
            clap_token_ids=clap_token_ids_batch,
            semantic_token_ids=semantic_token_ids,
            coarse_token_ids=condition_coarse_token_ids,
            max_time_steps=int(coarse_window_seconds * acoustic_steps_per_second),
            reconstruct_wave=False,
            include_eos_in_output=False,
            append_eos_to_conditioning_tokens=True,
            temperature=0.95,
        )

        if all_coarse_token_ids is None:
            all_coarse_token_ids = pred_coarse_token_ids
        else:
            pred_coarse_token_ids = pred_coarse_token_ids[:, condition_length:]
            all_coarse_token_ids = torch.cat([all_coarse_token_ids, pred_coarse_token_ids], dim=1)

    return all_coarse_token_ids


def generate_fine_tokens_for_batch(
    musiclm,
    clap_token_ids_batch,
    coarse_token_ids_batch,
    fine_window_seconds,
    acoustic_steps_per_second,
    fine_sliding_window_step_percent=1.0,
):
    fine_window_size = int(fine_window_seconds * acoustic_steps_per_second)
    fine_step_size = int(fine_window_size * fine_sliding_window_step_percent)
    all_coarse_token_ids_unfolded = coarse_token_ids_batch.unfold(1, fine_window_size, fine_step_size)
    all_coarse_token_ids_unfolded = rearrange(all_coarse_token_ids_unfolded, "b n q w -> n b w q")

    all_fine_token_ids = None
    for coarse_token_ids in all_coarse_token_ids_unfolded:
        if all_fine_token_ids is not None:
            condition_length = int(fine_window_size * (1 - fine_sliding_window_step_percent))
            condition_fine_token_ids = all_fine_token_ids[:, -condition_length:] if condition_length > 0 else None
        else:
            condition_fine_token_ids = None

        pred_fine_token_ids = musiclm.fine.generate(
            clap_token_ids=clap_token_ids_batch,
            coarse_token_ids=coarse_token_ids,
            fine_token_ids=condition_fine_token_ids,
            max_time_steps=fine_window_size,
            reconstruct_wave=False,
            include_eos_in_output=False,
            append_eos_to_conditioning_tokens=True,
            temperature=0.4,
        )

        if all_fine_token_ids is None:
            all_fine_token_ids = pred_fine_token_ids
        else:
            pred_fine_token_ids = pred_fine_token_ids[:, condition_length:]
            all_fine_token_ids = torch.cat([all_fine_token_ids, pred_fine_token_ids], dim=1)

    return all_fine_token_ids


def _candidate_input_token_stems(stage: str, file_stem: str):
    if stage == "coarse":
        return [f"{file_stem}_semantic", file_stem]
    if stage == "fine":
        return [f"{file_stem}_coarse", file_stem]
    return [file_stem]


def load_saved_token_payload(input_token_dir: Path, file_stem: str, input_token_format: str, stage: str):
    candidates = _candidate_input_token_stems(stage, file_stem)

    chosen = input_token_format
    chosen_stem = None
    if chosen == "auto":
        for cand in candidates:
            if (input_token_dir / f"{cand}.npz").exists():
                chosen = "npz"
                chosen_stem = cand
                break
            if (input_token_dir / f"{cand}.pt").exists():
                chosen = "pt"
                chosen_stem = cand
                break
        if chosen_stem is None:
            raise FileNotFoundError(
                f"No token file found for {file_stem} in {input_token_dir}. Tried stems: {candidates}"
            )
    else:
        for cand in candidates:
            if (input_token_dir / f"{cand}.{chosen}").exists():
                chosen_stem = cand
                break
        if chosen_stem is None:
            raise FileNotFoundError(
                f"Missing token file for {file_stem} in {input_token_dir} with format={chosen}. Tried stems: {candidates}"
            )

    if chosen == "npz":
        npz_path = input_token_dir / f"{chosen_stem}.npz"
        with np.load(str(npz_path), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    pt_path = input_token_dir / f"{chosen_stem}.pt"
    data = torch.load(str(pt_path), map_location="cpu")
    payload = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            payload[k] = v.detach().cpu().numpy()
        else:
            payload[k] = np.asarray(v)
    return payload


def _token_array_to_batch_tensor(name: str, arr: np.ndarray):
    t = torch.from_numpy(arr.astype(np.int64, copy=False))

    if name == "clap":
        if t.ndim == 1:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.ndim != 2:
            raise RuntimeError(f"Unexpected clap token shape from saved file: {tuple(t.shape)}")
        return t

    if name in {"semantic", "coarse", "fine"}:
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected {name} token shape from saved file: {tuple(t.shape)}")
        return t

    raise ValueError(f"Unsupported token key: {name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate test-set outputs using CLAP tokens from preprocessed memmap data"
    )

    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional path to project root (legacy compatibility; not required for normal use).",
    )
    parser.add_argument(
        "--preprocessed-data-folder",
        required=True,
        help="Folder containing memmap outputs (metadata.json/id_mapping.npy/clap_tokens.npy).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help='Split name from memmap metadata (e.g., "training", "validation", "test"). Use "all" to disable split filtering.',
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save generated tokens/audio files",
    )

    parser.add_argument("--model-config", required=True,
                        help="Model config JSON (e.g. configs/model/musiclm_large.json)")
    parser.add_argument("--semantic-path", required=True,
                        help="Semantic transformer checkpoint")
    parser.add_argument("--coarse-path", required=True,
                        help="Coarse transformer checkpoint")
    parser.add_argument("--fine-path", required=True,
                        help="Fine transformer checkpoint")
    parser.add_argument("--rvq-path", required=True,
                        help="CLAP RVQ checkpoint")
    parser.add_argument("--kmeans-path", required=True,
                        help="HuBERT k-means checkpoint")

    parser.add_argument("--num-samples", type=int, default=15, help="Number of test tracks to sample. <=0 means use all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracks-csv", default=None, help="Path to tracks metadata CSV (required with --genre-selective)")
    parser.add_argument("--genre-selective", action="store_true", help="Enable per-genre balanced sampling using tracks metadata")
    parser.add_argument("--genre-samples-per-genre", type=int, default=0, help="When --genre-selective is set, sample this many per genre")
    parser.add_argument("--genre-column", default="genre_top", help="Genre column in tracks CSV. Supports values like genre_top or track|genre_top")
    parser.add_argument("--output-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of conditioning tracks to generate per forward pass")
    parser.add_argument(
        "--stage",
        choices=["full", "semantic", "coarse", "fine"],
        default="full",
        help="Generation stage. Use semantic->coarse->fine for stage-wise pipeline.",
    )
    parser.add_argument(
        "--input-token-dir",
        default=None,
        help="Directory containing prior-stage token files for coarse/fine stage runs.",
    )
    parser.add_argument(
        "--input-token-format",
        choices=["auto", "npz", "pt"],
        default="auto",
        help="Format for input token files when --stage is coarse/fine.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of dataset shards (for multi-process / multi-GPU runs).",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard index for this process, in [0, num-shards).",
    )
    parser.add_argument(
        "--decode-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decode and save waveform (.wav). Only valid for stage=fine/full.",
    )
    parser.add_argument(
        "--token-format",
        choices=["npz", "pt"],
        default="npz",
        help="Format for saved token files when generating test outputs.",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=2,
        help="Number of worker processes for token/audio file saving. 1 disables multiprocessing.",
    )
    parser.add_argument(
        "--max-pending-saves",
        type=int,
        default=128,
        help="Upper bound of queued asynchronous save tasks before backpressure.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_output_dir = output_dir / "tokens"
    audio_output_dir = output_dir / "audio"
    token_output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.save_workers < 1:
        raise ValueError("--save-workers must be >= 1")
    if args.max_pending_saves < 1:
        raise ValueError("--max-pending-saves must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")
    if args.decode_audio and args.stage in {"semantic", "coarse"}:
        raise ValueError("--decode-audio is only supported for --stage fine/full")
    if args.genre_selective:
        if args.genre_samples_per_genre <= 0:
            raise ValueError("--genre-samples-per-genre must be > 0 when --genre-selective is enabled")
        if not args.tracks_csv:
            raise ValueError("--tracks-csv is required when --genre-selective is enabled")

    input_token_dir = Path(args.input_token_dir) if args.input_token_dir else None
    if args.stage in {"coarse", "fine"}:
        if input_token_dir is None:
            raise ValueError("--input-token-dir is required for --stage coarse/fine")
        if not input_token_dir.exists():
            raise FileNotFoundError(f"input token dir not found: {input_token_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    log("loading model config...")
    model_config = load_model_config(str(args.model_config))
    log("loading checkpoints and building MusicLM (this may take a while)...")
    t_model_start = time.time()
    musiclm = create_musiclm_from_config(
        model_config=model_config,
        semantic_path=str(args.semantic_path),
        coarse_path=str(args.coarse_path),
        fine_path=str(args.fine_path),
        rvq_path=str(args.rvq_path),
        kmeans_path=str(args.kmeans_path),
        device=device,
    )
    musiclm.eval()
    log(f"model ready in {time.time() - t_model_start:.1f}s")
    open_musiclm_module.tqdm = partial(open_musiclm_module.tqdm, disable=True)

    semantic_window_seconds = int(model_config.global_cfg.semantic_audio_length_seconds)
    coarse_window_seconds = int(model_config.global_cfg.coarse_audio_length_seconds)
    fine_window_seconds = int(model_config.global_cfg.fine_audio_length_seconds)
    semantic_steps_per_second = int(model_config.hubert_kmeans_cfg.output_hz)
    acoustic_steps_per_second = int(model_config.acoustic_output_hz)

    preprocessed_folder = Path(args.preprocessed_data_folder)
    _validate_memmap_folder(preprocessed_folder)

    memmap_store = MemmapTokenStore(preprocessed_folder)
    split_key = args.split.lower()

    id_map_stat = (preprocessed_folder / "id_mapping.npy").stat()
    metadata_stat = (preprocessed_folder / "metadata.json").stat()
    entry_cache_path = preprocessed_folder / (
        f"memmap_entries_{split_key}_{int(id_map_stat.st_mtime)}_{id_map_stat.st_size}_"
        f"{int(metadata_stat.st_mtime)}_{metadata_stat.st_size}.tsv"
    )

    log(f"scanning preprocessed memmap data: {preprocessed_folder}")
    entries = load_memmap_entries(memmap_store, split_name=split_key, cache_path=entry_cache_path)
    if not entries:
        raise RuntimeError("No matching rows found in memmap preprocessed data for current split/filter")

    track_to_genre = {}
    if args.tracks_csv:
        tracks_csv_path = Path(args.tracks_csv)
        if tracks_csv_path.exists():
            track_to_genre = load_track_to_genre_map(tracks_csv_path, genre_column=args.genre_column)
            log(f"loaded track->genre map: {len(track_to_genre)} rows from {tracks_csv_path}")
        elif args.genre_selective:
            raise FileNotFoundError(f"tracks csv not found: {tracks_csv_path}")
        else:
            log(f"warning: tracks csv not found (genre column will be empty in manifest): {tracks_csv_path}")

    if args.genre_selective:
        samples, genre_stats, missing_genre_count = select_genre_balanced_samples(
            entries,
            track_to_genre=track_to_genre,
            samples_per_genre=args.genre_samples_per_genre,
            seed=args.seed,
        )
        log(
            f"genre_selective enabled: genres={len(genre_stats)}, selected={len(samples)}, "
            f"missing_genre_rows={missing_genre_count}, per_genre_target={args.genre_samples_per_genre}"
        )
        preview = ", ".join([f"{g}:{picked}/{total}" for g, total, picked in genre_stats[:10]])
        if preview:
            log(f"genre_selective preview (first 10): {preview}")
    elif args.num_samples <= 0 or args.num_samples >= len(entries):
        samples = entries
    else:
        rng = random.Random(args.seed)
        samples = rng.sample(entries, k=args.num_samples)

    if args.num_shards > 1:
        full_selected = len(samples)
        samples = [row for i, row in enumerate(samples) if i % args.num_shards == args.shard_id]
        log(
            f"sharding enabled: shard {args.shard_id}/{args.num_shards}, "
            f"selected {len(samples)} of {full_selected} rows"
        )

    selection_manifest_path = output_dir / f"selected_samples_shard_{args.shard_id}_of_{args.num_shards}.tsv"
    write_selection_manifest(
        selection_manifest_path,
        samples,
        track_to_genre=track_to_genre,
        base_seed=args.seed,
    )
    log(f"wrote selection manifest -> {selection_manifest_path}")

    if not samples:
        log("no rows assigned to this shard; exiting")
        return

    log(
        f"found rows: {len(entries)}, selected: {len(samples)}, stage: {args.stage}, batch_size: {args.batch_size}, "
        f"split: {args.split}, decode_audio: {args.decode_audio}, token_format: {args.token_format}, "
        f"save_workers: {args.save_workers}, genre_selective: {args.genre_selective}"
    )

    save_executor = None
    pending_save_futures = []
    if args.save_workers > 1:
        log(f"enabling multiprocessing save pool with workers={args.save_workers}")
        save_executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.save_workers)

    total_batches = math.ceil(len(samples) / args.batch_size)

    progress = tqdm(
        range(0, len(samples), args.batch_size),
        total=total_batches,
        desc=f"generating ({args.stage})",
        dynamic_ncols=True,
    )
    for batch_idx, batch_start in enumerate(progress, start=1):
        batch_paths = samples[batch_start: batch_start + args.batch_size]
        log(f"[batch {batch_idx}/{total_batches}] preparing batch with {len(batch_paths)} items")
        batch_end = min(batch_start + len(batch_paths), len(samples))
        progress.set_postfix_str(f"{batch_start + 1}-{batch_end}/{len(samples)}")

        batch_items = []
        batch_sqlite_indices = []
        batch_seeds = []

        for j, path in enumerate(batch_paths):
            global_idx = batch_start + j
            sqlite_idx, track_id = path
            batch_items.append((global_idx, track_id, sqlite_idx))
            batch_sqlite_indices.append(sqlite_idx)
            batch_seeds.append(args.seed + global_idx)

        clap_token_ids_batch = None
        semantic_batch = None
        coarse_batch = None
        fine_batch = None

        t_prep_start = time.time()
        if args.stage in {"full", "semantic"}:
            batch_clap_tokens = memmap_store.load_clap_tokens_batch(
                sqlite_indices=batch_sqlite_indices,
                seeds=batch_seeds,
            )
            clap_token_ids_batch = torch.cat(batch_clap_tokens, dim=0).to(device)
            log(f"[batch {batch_idx}/{total_batches}] loaded CLAP tokens from memmap in {time.time() - t_prep_start:.2f}s")
        else:
            clap_list = []
            semantic_list = []
            coarse_list = []
            semantic_present_count = 0
            for _, track_id, sqlite_idx in batch_items:
                file_stem = f"{track_id}_{sqlite_idx}"
                payload = load_saved_token_payload(input_token_dir, file_stem, args.input_token_format, args.stage)

                if "clap" not in payload:
                    raise KeyError(f"Missing clap tokens in {file_stem}")
                clap_list.append(_token_array_to_batch_tensor("clap", payload["clap"]))

                if args.stage == "coarse":
                    if "semantic" not in payload:
                        raise KeyError(f"Missing semantic tokens in {file_stem} for coarse stage")
                    semantic_list.append(_token_array_to_batch_tensor("semantic", payload["semantic"]))

                if args.stage == "fine":
                    if "coarse" not in payload:
                        raise KeyError(f"Missing coarse tokens in {file_stem} for fine stage")
                    coarse_list.append(_token_array_to_batch_tensor("coarse", payload["coarse"]))
                    if "semantic" in payload:
                        semantic_list.append(_token_array_to_batch_tensor("semantic", payload["semantic"]))
                        semantic_present_count += 1

            clap_token_ids_batch = torch.cat(clap_list, dim=0).to(device)
            if semantic_list:
                if args.stage == "fine" and semantic_present_count != len(batch_items):
                    raise RuntimeError("Inconsistent semantic-token availability across fine-stage input files")
                semantic_batch = torch.cat(semantic_list, dim=0).to(device)
            if coarse_list:
                coarse_batch = torch.cat(coarse_list, dim=0).to(device)
            log(f"[batch {batch_idx}/{total_batches}] loaded prior-stage tokens in {time.time() - t_prep_start:.2f}s")

        t_gen_start = time.time()
        if args.stage == "full":
            semantic_batch, coarse_batch, fine_batch = generate_tokens_for_clap_batch(
                musiclm,
                clap_token_ids_batch=clap_token_ids_batch,
                output_seconds=args.output_seconds,
                semantic_window_seconds=semantic_window_seconds,
                coarse_window_seconds=coarse_window_seconds,
                fine_window_seconds=fine_window_seconds,
                semantic_steps_per_second=semantic_steps_per_second,
                acoustic_steps_per_second=acoustic_steps_per_second,
            )
        elif args.stage == "semantic":
            semantic_batch = generate_semantic_tokens_for_clap_batch(
                musiclm,
                clap_token_ids_batch=clap_token_ids_batch,
                output_seconds=args.output_seconds,
                semantic_window_seconds=semantic_window_seconds,
                semantic_steps_per_second=semantic_steps_per_second,
            )
        elif args.stage == "coarse":
            coarse_batch = generate_coarse_tokens_for_batch(
                musiclm,
                clap_token_ids_batch=clap_token_ids_batch,
                semantic_token_ids_batch=semantic_batch,
                coarse_window_seconds=coarse_window_seconds,
                semantic_steps_per_second=semantic_steps_per_second,
                acoustic_steps_per_second=acoustic_steps_per_second,
            )
        elif args.stage == "fine":
            fine_batch = generate_fine_tokens_for_batch(
                musiclm,
                clap_token_ids_batch=clap_token_ids_batch,
                coarse_token_ids_batch=coarse_batch,
                fine_window_seconds=fine_window_seconds,
                acoustic_steps_per_second=acoustic_steps_per_second,
            )
        else:
            raise RuntimeError(f"Unsupported stage: {args.stage}")

        log(f"[batch {batch_idx}/{total_batches}] token generation done in {time.time() - t_gen_start:.1f}s")

        clap_token_ids_batch = clap_token_ids_batch.detach().cpu()
        if semantic_batch is not None:
            semantic_batch = semantic_batch.detach().cpu()
        if coarse_batch is not None:
            coarse_batch = coarse_batch.detach().cpu()
        if fine_batch is not None:
            fine_batch = fine_batch.detach().cpu()

        wave_batch = None
        if args.decode_audio:
            log(f"[batch {batch_idx}/{total_batches}] decoding audio from generated tokens")
            with torch.no_grad():
                acoustic_batch = torch.cat([coarse_batch.to(device), fine_batch.to(device)], dim=-1)
                wave_batch = musiclm.neural_codec.decode_from_codebook_indices(acoustic_batch)
                wave_batch = rearrange(wave_batch, "b 1 n -> b n").detach().cpu()
            if wave_batch.dim() == 1:
                wave_batch = wave_batch.unsqueeze(0)

        t_save_start = time.time()
        submitted = 0
        for k, (global_idx, track_id, sqlite_idx) in enumerate(batch_items):
            file_stem = f"{track_id}_{sqlite_idx}"
            token_payload = {
                "clap": clap_token_ids_batch[k].numpy().astype(np.uint16),
            }
            if semantic_batch is not None:
                token_payload["semantic"] = semantic_batch[k].numpy().astype(np.uint16)
            if coarse_batch is not None:
                token_payload["coarse"] = coarse_batch[k].numpy().astype(np.uint16)
            if fine_batch is not None:
                token_payload["fine"] = fine_batch[k].numpy().astype(np.uint16)

            wave_np = None
            if args.decode_audio:
                wave_np = wave_batch[k:k+1].numpy().astype(np.float32)

            if save_executor is None:
                token_out_path, audio_out_path = _save_generated_output(
                    str(token_output_dir),
                    str(audio_output_dir),
                    file_stem,
                    args.stage,
                    args.token_format,
                    token_payload,
                    args.decode_audio,
                    wave_np,
                    int(musiclm.neural_codec.sample_rate),
                )
                log(f"saved tokens -> {token_out_path}")
                if audio_out_path is not None:
                    log(f"saved generated -> {audio_out_path}")
            else:
                fut = save_executor.submit(
                    _save_generated_output,
                    str(token_output_dir),
                    str(audio_output_dir),
                    file_stem,
                    args.stage,
                    args.token_format,
                    token_payload,
                    args.decode_audio,
                    wave_np,
                    int(musiclm.neural_codec.sample_rate),
                )
                pending_save_futures.append(fut)
                submitted += 1

                if len(pending_save_futures) >= args.max_pending_saves:
                    done, not_done = concurrent.futures.wait(
                        pending_save_futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for d in done:
                        _ = d.result()
                    pending_save_futures = list(not_done)

        if save_executor is None:
            log(f"[batch {batch_idx}/{total_batches}] save complete in {time.time() - t_save_start:.1f}s")
        else:
            log(
                f"[batch {batch_idx}/{total_batches}] queued save jobs={submitted}, "
                f"pending={len(pending_save_futures)}, enqueue_time={time.time() - t_save_start:.1f}s"
            )


    if save_executor is not None:
        log(f"waiting for pending save jobs: {len(pending_save_futures)}")
        for fut in concurrent.futures.as_completed(pending_save_futures):
            _ = fut.result()
        save_executor.shutdown(wait=True)

    log("all done")


if __name__ == "__main__":
    main()
