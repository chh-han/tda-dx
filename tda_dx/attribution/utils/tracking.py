"""
attributor/utils/tracking.py

Track ID normalization, run ID construction, and preprocessed-folder
validation utilities.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional


def normalize_track_id(track_id):
    try:
        return str(int(track_id))
    except (TypeError, ValueError):
        return str(track_id)


def build_run_id(args):
    parts = [
        str(args.condition),
        str(args.stage),
        str(args.method),
        f'seed{int(args.seed)}',
    ]
    return '__'.join(parts)


def _select_header_column(level0, level1, target):
    """Find a column index by matching against two-level CSV headers (FMA format)."""
    target = str(target).strip()
    if not target:
        return None
    target_lower = target.lower()
    target_parts = [p.strip().lower() for p in target.replace('.', '|').split('|') if p.strip()]
    for i, (h0, h1) in enumerate(zip(level0, level1)):
        h0_l = h0.strip().lower()
        h1_l = h1.strip().lower()
        if h1_l == target_lower:
            return i
        if f'{h0_l}|{h1_l}' == target_lower:
            return i
        if f'{h0_l}.{h1_l}' == target_lower:
            return i
        if len(target_parts) == 2 and [h0_l, h1_l] == target_parts:
            return i
    return None


def load_track_to_genre_map(tracks_csv_path, genre_column='genre_top'):
    """Load track_id -> genre mapping from FMA tracks.csv."""
    tracks_csv_path = Path(tracks_csv_path)
    if not tracks_csv_path.exists():
        raise FileNotFoundError(f'tracks csv not found: {tracks_csv_path}')
    with open(tracks_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            level0 = next(reader)
            level1 = next(reader)
            _ = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f'Invalid tracks csv format: {tracks_csv_path}') from exc
        if len(level0) != len(level1):
            raise RuntimeError(f'Unexpected tracks csv header width mismatch in {tracks_csv_path}')
        track_id_col = _select_header_column(level0, level1, 'track_id')
        if track_id_col is None:
            track_id_col = 0
        genre_col = _select_header_column(level0, level1, genre_column)
        if genre_col is None and str(genre_column).strip().lower() == 'genre_top':
            genre_col = _select_header_column(level0, level1, 'track|genre_top')
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
            if not genre or genre.lower() in {'nan', 'none', 'null'}:
                continue
            track_to_genre[norm_track] = genre
    return track_to_genre


def _extract_track_id_from_path(path_str: str):
    stem = Path(path_str).stem
    try:
        return str(int(stem))
    except ValueError:
        for part in reversed(stem.split('_')):
            try:
                return str(int(part))
            except ValueError:
                continue
    return stem


def _required_memmap_files(stage: str, folder: Path):
    base = [
        folder / 'metadata.json',
        folder / 'id_mapping.npy',
        folder / 'clap_tokens.npy',
    ]
    if stage == 'semantic':
        return base + [folder / 'semantic_tokens.npy']
    if stage == 'coarse':
        return base + [folder / 'semantic_tokens.npy', folder / 'coarse_tokens.npy']
    if stage == 'fine':
        return base + [folder / 'coarse_tokens.npy', folder / 'fine_tokens.npy']
    raise ValueError(f'invalid stage: {stage}')


def _validate_memmap_preprocessed_folder(stage: str, folder: Path):
    missing = [str(path) for path in _required_memmap_files(stage, folder) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f'memmap preprocessed data is required for stage={stage}, '
            f'but files are missing in {folder}: {missing}'
        )


def _load_idx_to_trackid_map(folder: Path):
    with open(folder / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    track_file = metadata.get('idx_to_track_id_file', 'idx_to_track_id.json')
    track_path = folder / track_file
    if track_path.exists():
        with open(track_path, 'r') as f:
            raw = json.load(f)
        return {int(k): normalize_track_id(v) for k, v in raw.items()}

    idx_to_path_file = metadata.get('idx_to_path_file', 'idx_to_path.json')
    idx_to_path = folder / idx_to_path_file
    if idx_to_path.exists():
        with open(idx_to_path, 'r') as f:
            raw = json.load(f)
        return {int(k): normalize_track_id(_extract_track_id_from_path(str(v))) for k, v in raw.items()}

    legacy = metadata.get('idx_to_track_id')
    if isinstance(legacy, dict):
        return {int(k): normalize_track_id(v) for k, v in legacy.items()}
    return {}


def load_track_id_to_subset_map(tracks_csv_path: str) -> Dict[str, str]:
    """Load track_id -> subset mapping from FMA tracks.csv (e.g. 'small', 'large').

    Reads the two-level FMA CSV header and finds the 'set'/'subset' column,
    matching the same logic used in SingleStageTrainer._load_track_id_to_subset_map.
    """
    tracks_path = Path(str(tracks_csv_path).strip())
    if not tracks_path.exists():
        raise FileNotFoundError(f'tracks.csv not found: {tracks_path}')

    mapping: Dict[str, str] = {}
    with open(tracks_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header1 = next(reader)
        header2 = next(reader)

        subset_idx = None
        for i, (a, b) in enumerate(zip(header1, header2)):
            if a.strip().lower() == 'set' and b.strip().lower() == 'subset':
                subset_idx = i
                break
        if subset_idx is None:
            raise ValueError(f"'set'/'subset' column not found in {tracks_path}")

        for row in reader:
            if len(row) <= subset_idx:
                continue
            try:
                track_id = str(int(row[0]))
            except ValueError:
                continue
            mapping[track_id] = str(row[subset_idx]).strip().lower()

    return mapping
