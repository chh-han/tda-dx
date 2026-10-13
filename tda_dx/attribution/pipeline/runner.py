"""
attributor/pipeline/runner.py

Main orchestration for data attribution experiments.

Handles setup, data loading, model loading, method dispatch, self-sanity
checks, and result saving.  Method-specific logic is delegated to
METHOD_REGISTRY classes (GradCosMethod, TRAKMethod).
"""

import argparse
import copy
import functools
import json
import os
import warnings

warnings.filterwarnings("ignore", message=".*pow_by_natural.*")
import random
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tda_dx.attribution.utils.attribution_utils import create_functional_loss_fn

from tda_dx.attribution.utils.tracking import (
    normalize_track_id,
    build_run_id,
    load_track_to_genre_map,
    load_track_id_to_subset_map,
    _load_idx_to_trackid_map,
)
from tda_dx.attribution.data.loading import (
    DS_FIELDS,
    collate_token_tuples,
    TokenExpandedGeneratedDataset,
    load_preprocessed_dataset,
    GeneratedTokenDataset,
    GeneratedTokenNpzDataset,
    DattriDatasetAdapter,
)
from tda_dx.attribution.model.setup import (
    create_transformer_from_checkpoint,
    create_stage_wrapper,
    resolve_default_trak_layer_names,
    resolve_last_representation_layer_names,
    configure_trainable_params,
)
from tda_dx.attribution.sanity.metrics import compute_self_sanity_metrics
from tda_dx.attribution.pipeline.saving import (
    normalize_score_shape,
    save_attribution_results,
    merge_sharded_results,
)
from tda_dx.attribution.methods import METHOD_REGISTRY
from tda_dx.config import load_model_config


def run_attribution(args):
    run_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_id = args.run_id or build_run_id(args)
    effective_self_sanity = bool(args.self_sanity_check or args.mode == 'merge_attribute')
    base_results_folder = Path(args.results_folder)
    results_folder = base_results_folder
    if args.condition != 'main':
        results_folder = results_folder / args.condition
    if args.self_sanity_check and args.mode == 'full':
        results_folder = results_folder / 'self_sanity'
    results_folder.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    num_gpus = torch.cuda.device_count() if device == 'cuda' else 0
    print('=' * 60)
    print(f'Data Attribution — {args.stage} stage — {args.method}')
    print('=' * 60)
    print(f'  Device:            {device} ({num_gpus} GPUs)')
    print(f'  CUDA_VISIBLE:      {os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")}')
    print(f'  Model config:      {args.model_config}')
    print(f'  Checkpoint:        {args.checkpoint_path}')
    print(f'  Data folder:       {args.preprocessed_data_folder}')
    if args.mode == 'cache_shard':
        print(f'  Generated tokens:  (skipped in cache_shard mode)')
    elif args.generated_tokens_dir:
        print(f'  Generated dir:     {args.generated_tokens_dir}')
        print(f'  Tokens manifest:   {args.generated_tokens_manifest or "(auto-discover)"}')
    else:
        print(f'  Generated tokens:  {args.generated_tokens_file}')
    print(f'  Tracks CSV:        {args.tracks_csv or "(not set)"}')
    print(f'  Genre column:      {args.genre_column}')
    print(f'  Results folder:    {results_folder}')
    print(f'  ---')
    print(f'  Method:            {args.method}')
    print(f'  Condition:         {args.condition}')
    print(f'  Run ID:            {run_id}')
    print(f'  Batch size:        {args.batch_size}')
    print(f'  Seed:              {args.seed}')
    print(f'  Self sanity check: {effective_self_sanity}')
    print(f'  Last represent:    {args.last_represent_layer}')
    print(f'  Use head layer:    {args.use_head_layer}')
    if args.method in ('trak', 'logra', 'tracin', 'factgrass'):
        print(f'  Projection dim:    {args.proj_dim}')
    if args.method == 'trak':
        print(f'  TRAK regularization: {args.regularization}')
        if args.trak_checkpoint_paths:
            print(f'  TRAK ckpts:         {args.trak_checkpoint_paths}')
    if args.method == 'logra':
        print(f'  LoGra hessian:     {args.logra_hessian}')
        print(f'  LoGra damping:     {args.logra_damping}')
        print(f'  LoGra offload:     {args.logra_offload}')
        print(f'  LoGra chunk size:  {args.logra_chunk_size}')
        if args.logra_cache_dir:
            print(f'  LoGra cache dir:   {args.logra_cache_dir}')
    if args.method == 'factgrass':
        print(f'  FactGraSS hessian: {args.factgrass_hessian}')
        print(f'  FactGraSS damping: {args.factgrass_damping}')
        print(f'  FactGraSS blowup:  {args.factgrass_blowup_factor}')
        print(f'  FactGraSS offload: {args.factgrass_offload}')
        print(f'  FactGraSS chunk:   {args.factgrass_chunk_size}')
        if args.factgrass_cache_dir:
            print(f'  FactGraSS cache:   {args.factgrass_cache_dir}')
    if args.method == 'tracin':
        print(f'  TracIn normalized: {args.tracin_normalized_grad}')
        print(f'  TracIn ckpts:      {args.tracin_checkpoint_paths or "(current stage checkpoint only)"}')
        if args.tracin_weights:
            print(f'  TracIn weights:    {args.tracin_weights}')
    if args.method == 'cg':
        print(f'  CG max_iter:       {args.cg_max_iter}')
        print(f'  CG tol:            {args.cg_tol}')
        print(f'  CG mode:           {args.cg_mode}')
        print(f'  CG regularization: {args.cg_regularization}')
    if args.method == 'lissa':
        print(f'  LiSSA batch_size:      {args.lissa_batch_size}')
        print(f'  LiSSA num_repeat:      {args.lissa_num_repeat}')
        print(f'  LiSSA recursion_depth: {args.lissa_recursion_depth}')
        print(f'  LiSSA damping:         {args.lissa_damping}')
        print(f'  LiSSA scaling:         {args.lissa_scaling}')
        print(f'  LiSSA mode:            {args.lissa_mode}')
    print(f'  Mode:              {args.mode}')
    if args.mode == 'cache_shard':
        print(f'  Shard:             {args.shard_id} / {args.num_shards}')
    elif args.mode == 'merge_attribute':
        print(f'  Num shards:        {args.num_shards}')
    print(f'  Num train samples: {args.num_train_samples or "all"}')
    print(f'  Num train tracks:  {args.num_train_tracks or "all"}')
    print(f'  Samples per track: {args.samples_per_track or "all"}')
    if args.num_train_tracks:
        print(f'  Track subset mode: {args.train_track_sampling}')
    print(f'  ---')
    print(f'  wandb:             {args.wandb}')
    if args.wandb:
        print(f'  wandb project:     {args.wandb_project}')
        print(f'  wandb group:       {args.wandb_group or f"attr_{args.stage}"}')
        print(f'  Alert every:       {args.alert_every}')
    print('=' * 60)

    # load config
    model_config = load_model_config(args.model_config)

    # train dataset from preprocessed memmap data
    preprocessed_data_folder = Path(args.preprocessed_data_folder)
    print(f'Loading train dataset from memmap folder {preprocessed_data_folder}...')
    train_full_dataset = load_preprocessed_dataset(preprocessed_data_folder, args.stage, model_config)
    idx_to_trackid = _load_idx_to_trackid_map(preprocessed_data_folder)

    total_train_all = len(train_full_dataset)
    print(f'Total train samples available: {total_train_all}')

    # load test set from generated tokens (not needed for cache_shard/self-sanity mode)
    generated_tokens_dataset = None
    num_test = 0
    if args.self_sanity_check:
        print('[self-sanity-check] Skipping generated token loading; '
              'test query set will be the same as training samples (same order).')
    elif args.mode not in ('cache_shard', 'hessian_shard', 'score_shard'):
        if args.generated_tokens_dir:
            print(f'Loading generated tokens from directory {args.generated_tokens_dir}...')
            generated_tokens_dataset = GeneratedTokenNpzDataset(
                tokens_dir=args.generated_tokens_dir,
                stage=args.stage,
                manifest_path=args.generated_tokens_manifest,
            )
        else:
            print(f'Loading generated tokens from file {args.generated_tokens_file}...')
            generated_tokens_dataset = GeneratedTokenDataset(args.generated_tokens_file)

        print(f'Generated tokens: {len(generated_tokens_dataset)} samples, stage={generated_tokens_dataset.stage}')
        print(f'  Track IDs: {generated_tokens_dataset.track_ids}')
        if generated_tokens_dataset.stage != args.stage:
            raise ValueError(
                f'Stage mismatch: generated tokens are from stage={generated_tokens_dataset.stage!r} '
                f'but --stage={args.stage!r}. Re-run token generation with the correct stage.'
            )
        num_test = len(generated_tokens_dataset)
    else:
        print('[cache_shard] Skipping test data loading (not needed for gradient caching)')

    # train split
    train_indices = np.arange(total_train_all)
    print('Using all train samples')

    # --- FMA subset filter ---------------------------------------------------
    fma_train_subsets = None
    if args.fma_train_subset:
        raw = str(args.fma_train_subset).strip()
        parsed = {p.strip().lower() for p in raw.split(',') if p.strip()}
        if parsed == {'full'}:
            print(f'Subset filter: "full" → no filtering, using all train samples')
        else:
            fma_train_subsets = parsed

    if fma_train_subsets:
        if not args.tracks_csv:
            raise ValueError(
                '--fma_train_subset requires --tracks_csv (path to FMA tracks.csv).'
            )
        print(f'Loading subset map from {args.tracks_csv} for subset filter {fma_train_subsets}...')
        track_id_to_subset = load_track_id_to_subset_map(args.tracks_csv)
        before_count = len(train_indices)
        filtered = []
        for ds_pos in train_indices.tolist():
            sid = train_full_dataset.get_sqlite_idx_for_sample(int(ds_pos))
            track_id = normalize_track_id(idx_to_trackid.get(int(sid), str(sid)))
            subset_val = track_id_to_subset.get(track_id, '').lower()
            if subset_val in fma_train_subsets:
                filtered.append(ds_pos)
        train_indices = np.array(filtered, dtype=np.int64)
        print(
            f'Subset filter ({args.fma_train_subset}): {before_count} → {len(train_indices)} samples'
        )
        if len(train_indices) == 0:
            raise RuntimeError(
                f'Subset filter {fma_train_subsets!r} produced 0 training samples. '
                f'Check that tracks.csv subset values match the requested subset name.'
            )
    # -------------------------------------------------------------------------

    selected_train_track_ids = None
    if args.num_train_tracks:
        candidate_sqlite_indices = [
            train_full_dataset.get_sqlite_idx_for_sample(int(ds_pos)) for ds_pos in train_indices
        ]
        candidate_track_ids = [
            normalize_track_id(idx_to_trackid.get(sid, str(sid))) for sid in candidate_sqlite_indices
        ]
        unique_track_ids = list(dict.fromkeys(candidate_track_ids))
        if args.num_train_tracks < len(unique_track_ids):
            if args.train_track_sampling == 'random':
                rng = np.random.default_rng(args.seed)
                selected_positions = np.sort(
                    rng.choice(len(unique_track_ids), size=args.num_train_tracks, replace=False)
                )
                selected_train_track_ids = [unique_track_ids[pos] for pos in selected_positions.tolist()]
            else:
                selected_train_track_ids = unique_track_ids[:args.num_train_tracks]
            selected_track_set = set(selected_train_track_ids)
            train_indices = np.array(
                [
                    ds_pos for ds_pos, track_id in zip(train_indices.tolist(), candidate_track_ids)
                    if track_id in selected_track_set
                ],
                dtype=np.int64,
            )
            print(
                f'Limited train pool to {len(selected_train_track_ids)} tracks '
                f'({len(train_indices)} samples) using {args.train_track_sampling} selection'
            )
        else:
            selected_train_track_ids = unique_track_ids
            print(
                f'--num_train_tracks={args.num_train_tracks} covers all available tracks '
                f'({len(unique_track_ids)} unique tracks); using full train pool'
            )

    if args.samples_per_track:
        # Build per-track sample list, then randomly pick up to N per track
        candidate_sqlite_indices = [
            train_full_dataset.get_sqlite_idx_for_sample(int(ds_pos)) for ds_pos in train_indices
        ]
        candidate_track_ids = [
            normalize_track_id(idx_to_trackid.get(sid, str(sid))) for sid in candidate_sqlite_indices
        ]
        from collections import defaultdict
        track_to_positions = defaultdict(list)
        for ds_pos, track_id in zip(train_indices.tolist(), candidate_track_ids):
            track_to_positions[track_id].append(ds_pos)
        rng_spt = np.random.default_rng(args.seed)
        sampled = []
        for track_id, positions in track_to_positions.items():
            if len(positions) <= args.samples_per_track:
                sampled.extend(positions)
            else:
                chosen = rng_spt.choice(len(positions), size=args.samples_per_track, replace=False)
                sampled.extend([positions[i] for i in sorted(chosen)])
        train_indices = np.array(sampled, dtype=np.int64)
        print(
            f'samples_per_track={args.samples_per_track}: '
            f'{len(track_to_positions)} tracks → {len(train_indices)} samples'
        )

    if args.num_train_samples and len(train_indices) > args.num_train_samples:
        train_indices = train_indices[:args.num_train_samples]
        print(f'Limited train pool to {args.num_train_samples} samples')

    # Shard train_indices for parallel multi-GPU execution in full mode.
    # (TRAK cache_shard mode has its own interleaved shard logic below.)
    _is_sharded = args.num_shards > 1 and args.mode in ('full', 'resume_after_grad')
    if _is_sharded:
        shard_size = (len(train_indices) + args.num_shards - 1) // args.num_shards
        _sh_start = args.shard_id * shard_size
        _sh_end   = min(_sh_start + shard_size, len(train_indices))
        train_indices = train_indices[_sh_start:_sh_end]
        print(f'[SHARD {args.shard_id}/{args.num_shards}] train [{_sh_start}:{_sh_end}] → {len(train_indices)} samples')

    if generated_tokens_dataset is not None:
        num_test = len(generated_tokens_dataset)

    num_train = len(train_indices)
    # Resolve per-sample track IDs via the dataset's exhaustive-traversal mapping
    train_sqlite_indices = [
        train_full_dataset.get_sqlite_idx_for_sample(int(ds_pos)) for ds_pos in train_indices
    ]
    train_track_ids = [
        normalize_track_id(idx_to_trackid.get(sid, str(sid))) for sid in train_sqlite_indices
    ]
    if args.self_sanity_check:
        num_test = num_train
    print(f'Train pool: {num_train}, Test set: {num_test}')

    # load genre metadata if tracks_csv is provided
    track_to_genre = {}
    if args.tracks_csv:
        print(f'Loading genre map from {args.tracks_csv}...')
        track_to_genre = load_track_to_genre_map(args.tracks_csv, genre_column=args.genre_column)
        print(f'  Loaded {len(track_to_genre)} track->genre mappings')

    def _resolve_genre(track_id):
        return track_to_genre.get(normalize_track_id(track_id), '')

    # save split info
    _shard_tag = f'_shard_{args.shard_id:03d}_of_{args.num_shards:03d}' if _is_sharded else ''
    np.save(str(results_folder / f'train_indices{_shard_tag}.npy'), train_indices)

    # save experiment config
    experiment_config = {
        'run_id': run_id,
        'condition': args.condition,
        'stage': args.stage,
        'method': args.method,
        'model_config': args.model_config,
        'checkpoint_path': args.checkpoint_path,
        'preprocessed_data_folder': args.preprocessed_data_folder,
        'generated_tokens_file': args.generated_tokens_file,
        'generated_tokens_dir': args.generated_tokens_dir,
        'generated_tokens_manifest': args.generated_tokens_manifest,
        'num_train_samples': num_train,
        'num_train_tracks_requested': args.num_train_tracks,
        'train_track_sampling': args.train_track_sampling,
        'num_train_tracks_selected': len(set(train_track_ids)),
        'num_test_samples': num_test,
        'batch_size': args.batch_size,
        'total_dataset_size': total_train_all,
        'data_format': 'memmap',
        'self_sanity_check': effective_self_sanity,
        'self_sanity_query_chunk_size': int(args.self_sanity_query_chunk_size),
        'last_represent_layer': bool(args.last_represent_layer),
        'use_head_layer': bool(args.use_head_layer),
    }
    if args.method in ('trak', 'grad_cos', 'logra', 'tracin', 'factgrass'):
        experiment_config['proj_dim'] = args.proj_dim
    if args.method == 'trak':
        experiment_config['regularization'] = args.regularization
        experiment_config['trak_checkpoint_paths'] = args.trak_checkpoint_paths
    if args.method == 'logra':
        experiment_config['logra_hessian'] = args.logra_hessian
        experiment_config['logra_damping'] = args.logra_damping
        experiment_config['logra_offload'] = args.logra_offload
        experiment_config['logra_chunk_size'] = args.logra_chunk_size
        experiment_config['logra_cache_dir'] = args.logra_cache_dir
    if args.method == 'factgrass':
        experiment_config['factgrass_hessian'] = args.factgrass_hessian
        experiment_config['factgrass_damping'] = args.factgrass_damping
        experiment_config['factgrass_blowup_factor'] = args.factgrass_blowup_factor
        experiment_config['factgrass_offload'] = args.factgrass_offload
        experiment_config['factgrass_chunk_size'] = args.factgrass_chunk_size
        experiment_config['factgrass_cache_dir'] = args.factgrass_cache_dir
    if args.method == 'tracin':
        experiment_config['tracin_checkpoint_paths'] = args.tracin_checkpoint_paths
        experiment_config['tracin_weights'] = args.tracin_weights
        experiment_config['tracin_normalized_grad'] = bool(args.tracin_normalized_grad)
    if args.method == 'cg':
        experiment_config['cg_max_iter'] = args.cg_max_iter
        experiment_config['cg_tol'] = args.cg_tol
        experiment_config['cg_mode'] = args.cg_mode
        experiment_config['cg_regularization'] = args.cg_regularization
    if args.method == 'lissa':
        experiment_config['lissa_batch_size'] = args.lissa_batch_size
        experiment_config['lissa_num_repeat'] = args.lissa_num_repeat
        experiment_config['lissa_recursion_depth'] = args.lissa_recursion_depth
        experiment_config['lissa_damping'] = args.lissa_damping
        experiment_config['lissa_scaling'] = args.lissa_scaling
        experiment_config['lissa_mode'] = args.lissa_mode
    if selected_train_track_ids is not None:
        experiment_config['selected_train_track_ids'] = selected_train_track_ids
    config_path = results_folder / 'experiment_config.json'
    with open(str(config_path), 'w') as f:
        json.dump(experiment_config, f, indent=2)

    # wandb init
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or f"attr_{args.stage}",
            name=run_id,
            config=vars(args),
            dir=str(results_folder),
            reinit=True,
            tags=[args.stage, args.method, args.condition],
        )

    # wrap dataset with DattriDatasetAdapter (squeezes leading batch dim)
    adapted_train_dataset = DattriDatasetAdapter(train_full_dataset)
    train_dataset = Subset(adapted_train_dataset, train_indices.tolist())
    if args.self_sanity_check and args.mode == 'full':
        test_dataset = train_dataset
    else:
        test_dataset = generated_tokens_dataset

    test_losses = np.array([], dtype=np.float32)
    token_scores_np = None
    token_axis_path = None
    self_sanity_precomputed = None
    scores_raw_np = None
    attr_start = time.time()

    if args.condition == 'random_baseline':
        print('Skipping model loading and generating random baseline scores...')
        rng = np.random.default_rng(args.seed)
        scores_raw_np = rng.standard_normal((num_train, num_test)).astype(np.float32, copy=False)
        attr_time = time.time() - attr_start
        print(f'Random baseline completed in {attr_time:.0f}s')
    else:
        # load transformer from checkpoint
        print(f'Loading transformer from {args.checkpoint_path}...')
        transformer = create_transformer_from_checkpoint(
            args.stage, model_config, args.checkpoint_path, device,
        )
        transformer.eval()

        if getattr(args, 'bf16', False):
            transformer = transformer.to(torch.bfloat16)
            print('Model converted to bfloat16')

        if getattr(args, 'gradient_checkpointing', False):
            # Walk the nesting: Wrapper → TokenConditionedTransformer → Transformer
            _ckpt_enabled = 0
            for _mod in transformer.modules():
                if hasattr(_mod, 'enable_gradient_checkpointing') and _mod is not transformer:
                    _mod.enable_gradient_checkpointing()
                    _ckpt_enabled += 1
            if _ckpt_enabled:
                print(f'Gradient checkpointing enabled on {_ckpt_enabled} Transformer module(s)')
            else:
                print('Warning: --gradient_checkpointing requested but no Transformer module with '
                      'enable_gradient_checkpointing() found; see open_musiclm/transformer.py')

        total_params = sum(p.numel() for p in transformer.parameters())
        print(f'Model params: {total_params:,}')
        if args.wandb:
            import wandb
            wandb.log({"model/total_params": total_params})


        # free eval cache before the memory-heavy attribution step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # create functional loss fn for dattri
        stage_wrapper = create_stage_wrapper(args.stage, transformer)
        # Extract wrapper reference before compile so inner_transformer points to the
        # actual nn.Module (compiled wrapper wraps it but shares the same parameters).
        wrapper = stage_wrapper.transformer_wrapper
        loss_fn, inner_transformer = create_functional_loss_fn(wrapper)
        if getattr(args, 'compile', False):
            stage_wrapper = torch.compile(stage_wrapper, mode='reduce-overhead')
            print('stage_wrapper compiled with torch.compile (mode=reduce-overhead); '
                  'first iteration will be slow due to compilation')
        layer_names_for_attr = None
        if args.last_represent_layer:
            layer_names_for_attr, last_layer_meta = resolve_last_representation_layer_names(inner_transformer)
            total_params_cnt, trainable_params_cnt = configure_trainable_params(
                inner_transformer,
                layer_names_for_attr,
            )
            print(
                'Using last representation layer before head: '
                f'block={last_layer_meta["last_transformer_block_index"]} '
                f'({last_layer_meta["last_transformer_block_prefix"]}*) + '
                f'{last_layer_meta["final_norm_prefix"]}*'
            )
            print(f'Last-layer parameter tensors selected: {len(layer_names_for_attr)}')
            if len(layer_names_for_attr) <= 12:
                print(f'Selected parameter names: {layer_names_for_attr}')
            else:
                print(
                    'Selected parameter names (first 12): '
                    f'{layer_names_for_attr[:12]} ... (+{len(layer_names_for_attr) - 12} more)'
                )
            print(
                f'Last-layer grad-enabled params: {trainable_params_cnt:,} / {total_params_cnt:,} '
                f'({100.0 * trainable_params_cnt / total_params_cnt:.4f}%)'
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif args.use_head_layer:
            # Force output-head-only attribution for any method.
            layer_names_for_attr = resolve_default_trak_layer_names(inner_transformer, stage_wrapper)
            total_params_cnt, trainable_params_cnt = configure_trainable_params(
                inner_transformer,
                layer_names_for_attr,
            )
            print(f'Using output-head params: {layer_names_for_attr}')
            print(
                f'Head-layer grad-enabled params: {trainable_params_cnt:,} / {total_params_cnt:,} '
                f'({100.0 * trainable_params_cnt / total_params_cnt:.4f}%)'
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Default: use all trainable parameters (no layer restriction).
            total_params_cnt = sum(p.numel() for p in inner_transformer.parameters())
            trainable_params_cnt = sum(p.numel() for p in inner_transformer.parameters() if p.requires_grad)
            print(f'Using all trainable params: {trainable_params_cnt:,} / {total_params_cnt:,} '
                  f'({100.0 * trainable_params_cnt / total_params_cnt:.4f}%)')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # set up data loaders for attribution
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_token_tuples)
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collate_token_tuples)

        # attribution task
        from dattri.task import AttributionTask

        checkpoint = inner_transformer.state_dict()
        task = AttributionTask(
            loss_func=loss_fn,
            model=inner_transformer,
            checkpoints=checkpoint,
        )

        # instantiate and setup method
        if args.method not in METHOD_REGISTRY:
            raise ValueError(f'Unknown method: {args.method}. Choose from: {list(METHOD_REGISTRY.keys())}')

        ds_fields = DS_FIELDS[args.stage]
        params = list(inner_transformer.parameters())
        method = METHOD_REGISTRY[args.method]()

        # Attach layer_names to args for method access
        args._layer_names_for_attr = layer_names_for_attr

        method.setup(
            stage_wrapper=stage_wrapper,
            params=params,
            ds_fields=ds_fields,
            device=device,
            args=args,
        )

        # Methods backed by dattri attributors need extra task/transformer references.
        if args.method in ('trak', 'logra', 'tracin', 'factgrass', 'cg', 'lissa'):
            method.set_task_and_transformer(task, inner_transformer, loss_fn)

        # ── METHOD DISPATCH ────────────────────────────────────────────────
        if args.mode == 'hessian_shard':
            _dl_workers = getattr(args, 'dataloader_workers', 0)
            shard_size = (num_train + args.num_shards - 1) // args.num_shards
            sh_start = args.shard_id * shard_size
            sh_end = min(sh_start + shard_size, num_train)
            shard_train_indices = train_indices[sh_start:sh_end]
            shard_dataset = Subset(adapted_train_dataset, shard_train_indices.tolist())
            shard_loader = DataLoader(shard_dataset, batch_size=args.batch_size,
                                      shuffle=False, collate_fn=collate_token_tuples,
                                      num_workers=_dl_workers)
            method.hessian_shard(shard_loader=shard_loader,
                                 shard_id=args.shard_id, num_shards=args.num_shards)
            return

        elif args.mode == 'precompute':
            # No training shard needed; test_loader already built above.
            method.precompute(test_loader=test_loader, num_shards=args.num_shards)
            return

        elif args.mode == 'score_shard':
            _dl_workers = getattr(args, 'dataloader_workers', 0)
            shard_size = (num_train + args.num_shards - 1) // args.num_shards
            sh_start = args.shard_id * shard_size
            sh_end = min(sh_start + shard_size, num_train)
            shard_train_indices = train_indices[sh_start:sh_end]
            shard_dataset = Subset(adapted_train_dataset, shard_train_indices.tolist())
            shard_loader = DataLoader(shard_dataset, batch_size=args.batch_size,
                                      shuffle=False, collate_fn=collate_token_tuples,
                                      num_workers=_dl_workers)
            # Save shard train_indices for later merge (score_merge needs them)
            _shard_tag = f'_shard_{args.shard_id:03d}_of_{args.num_shards:03d}'
            np.save(str(results_folder / f'train_indices{_shard_tag}.npy'), shard_train_indices)
            method.score_shard(shard_loader=shard_loader,
                               shard_id=args.shard_id, num_shards=args.num_shards)
            return

        elif args.mode == 'cache_shard':
            _dl_workers = getattr(args, 'dataloader_workers', 4)
            if args.method == 'trak':
                shard_positions = list(range(args.shard_id, num_train, args.num_shards))
                shard_train_indices = train_indices[shard_positions]
                shard_dataset = Subset(adapted_train_dataset, shard_train_indices.tolist())
                shard_loader = DataLoader(shard_dataset, batch_size=args.batch_size,
                                          shuffle=False, collate_fn=collate_token_tuples,
                                          num_workers=_dl_workers)
            else:
                # grad_cos: contiguous block sharding
                shard_size = (num_train + args.num_shards - 1) // args.num_shards
                sh_start = args.shard_id * shard_size
                sh_end = min(sh_start + shard_size, num_train)
                shard_positions = list(range(sh_start, sh_end))
                shard_train_indices = train_indices[shard_positions]
                shard_dataset = Subset(adapted_train_dataset, shard_train_indices.tolist())
                shard_loader = DataLoader(
                    shard_dataset, batch_size=args.batch_size,
                    shuffle=False, collate_fn=collate_token_tuples,
                    num_workers=_dl_workers,
                )

            method.cache_shard(
                shard_loader=shard_loader,
                shard_positions=shard_positions,
                results_folder=results_folder,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
            )
            return  # Early exit — no score computation or post-processing

        elif args.mode == 'merge_attribute':
            _shard_source = None
            if getattr(args, 'shard_source_base', None):
                _shard_source = Path(args.shard_source_base) / args.stage
                print(f'[merge_attribute] Reading shards from override: {_shard_source}')
            _merge_kwargs = dict(
                num_train=num_train,
                results_folder=results_folder,
                num_shards=args.num_shards,
                test_loader=test_loader,
            )
            # shard_source_folder is only supported by trak/factgrass merge_attribute.
            if _shard_source is not None:
                _merge_kwargs['shard_source_folder'] = _shard_source
            scores = method.merge_attribute(**_merge_kwargs)

            if getattr(args, 'skip_self_sanity', False):
                print('[merge_attribute] Skipping self-sanity (--skip_self_sanity set).')
            else:
                self_metrics, self_arrays = method.self_sanity(train_loader, num_train)
                sanity_folder = results_folder / 'self_sanity'
                sanity_folder.mkdir(parents=True, exist_ok=True)
                np.save(str(sanity_folder / f'{args.stage}_{args.method}_self_diag_percentiles.npy'),
                        self_arrays['diag_percentiles'])
                np.save(str(sanity_folder / f'{args.stage}_{args.method}_self_top1_hits_strict.npy'),
                        self_arrays['top1_hits_strict'])
                np.save(str(sanity_folder / f'{args.stage}_{args.method}_self_top1_hits_tie.npy'),
                        self_arrays['top1_hits_tie'])
                np.save(str(sanity_folder / f'{args.stage}_{args.method}_self_diag_ranks.npy'),
                        self_arrays['diag_ranks'])
                np.save(str(sanity_folder / f'{args.stage}_{args.method}_self_top1_indices.npy'),
                        self_arrays['top1_indices'])

                sanity_metrics_path = sanity_folder / f'{args.stage}_{args.method}_self_sanity_metrics.json'
                with open(str(sanity_metrics_path), 'w') as f:
                    json.dump(self_metrics, f, indent=2)

                train_genres_sanity = [_resolve_genre(tid) for tid in train_track_ids]
                sanity_axis_path = sanity_folder / f'{args.stage}_{args.method}_self_sanity_axis.tsv'
                import csv as _csv
                with open(str(sanity_axis_path), 'w', newline='') as f:
                    writer = _csv.writer(f, delimiter='\t')
                    writer.writerow([
                        'sample_index', 'train_ds_index', 'train_sqlite_idx',
                        'train_track_id', 'train_genre', 'diag_score',
                        'diag_rank', 'diag_percentile', 'top1_hit_strict',
                        'top1_hit_tie', 'top1_index',
                    ])
                    for i in range(num_train):
                        writer.writerow([
                            i, int(train_indices[i]), int(train_sqlite_indices[i]),
                            train_track_ids[i], train_genres_sanity[i],
                            float(self_arrays['diag_scores'][i]),
                            float(self_arrays['diag_ranks'][i]),
                            float(self_arrays['diag_percentiles'][i]),
                            bool(self_arrays['top1_hits_strict'][i]),
                            bool(self_arrays['top1_hits_tie'][i]),
                            int(self_arrays['top1_indices'][i]),
                        ])

                print(
                    f'[self-sanity] top1_hit_rate_strict='
                    f'{self_metrics["top1_hit_rate_strict"]:.4f}, '
                    f'mean_percentile={self_metrics["mean_percentile"]:.2f}'
                )
                print(f'  metrics: {sanity_metrics_path}')
                print(f'  axis:    {sanity_axis_path}')

                if args.wandb:
                    import wandb
                    wandb.log({
                        'self_sanity/top1_hit_rate_strict': self_metrics['top1_hit_rate_strict'],
                        'self_sanity/top1_hit_rate_tie': self_metrics['top1_hit_rate_tie'],
                        'self_sanity/mean_percentile': self_metrics['mean_percentile'],
                        'self_sanity/median_percentile': self_metrics['median_percentile'],
                        'self_sanity/p10_percentile': self_metrics['p10_percentile'],
                        'self_sanity/p90_percentile': self_metrics['p90_percentile'],
                    })

        else:
            # ===== FULL MODE =====
            if args.self_sanity_check:
                # Use method's self_sanity() for both grad_cos and TRAK;
                # avoids N×N score matrix (chunked, OOM-safe)
                self_sanity_precomputed = method.self_sanity(train_loader, num_train)
            else:
                if args.mode == 'resume_after_grad':
                    scores = method.resume_after_grad(
                        train_loader=train_loader, test_loader=test_loader,
                    )
                else:
                    scores = method.full(
                        train_loader=train_loader, test_loader=test_loader,
                    )

            if args.method == 'grad_cos':
                # grad_cos token-level attribution (only in full mode, non-self-sanity)
                if args.self_sanity_check:
                    print('[self-sanity-check] Skipping grad_cos token-level attribution '
                          '(self sanity is sample-level).')
                elif not args.skip_grad_cos_token_level:
                    print('Computing token-level attribution scores...')
                    token_start = time.time()
                    token_test_dataset = TokenExpandedGeneratedDataset(test_dataset)
                    token_test_loader = DataLoader(
                        token_test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=collate_token_tuples,
                    )
                    # Use dattri TracIn for token-level attribution
                    from dattri.algorithm.tracin import TracInAttributor
                    proj_type = "sjlt" if device == "cuda" else "normal"
                    token_attributor = TracInAttributor(
                        task=task,
                        weight_list=torch.ones(1),
                        normalized_grad=True,
                        device=device,
                        projector_kwargs={"proj_dim": args.proj_dim, "device": device, "proj_type": proj_type},
                        layer_name=layer_names_for_attr,
                    )
                    token_scores = token_attributor.attribute(train_loader, token_test_loader)
                    token_scores_np = normalize_score_shape(
                        token_scores.cpu().numpy(),
                        num_train,
                        len(token_test_dataset),
                        label='token-level score',
                    )
                    np.save(str(results_folder / f'{args.stage}_{args.method}_token_scores.npy'), token_scores_np)

                    import csv as _csv
                    token_axis_path = results_folder / f'{args.stage}_{args.method}_token_axis.tsv'
                    with open(str(token_axis_path), 'w', newline='') as f:
                        writer = _csv.writer(f, delimiter='\t')
                        writer.writerow(['token_test_index', 'test_index', 'test_track_id', 'token_position'])
                        for token_test_idx, (test_idx, token_pos) in enumerate(token_test_dataset.entries):
                            track_id = (
                                str(generated_tokens_dataset.track_ids[test_idx])
                                if len(generated_tokens_dataset.track_ids) == num_test
                                else str(test_idx)
                            )
                            writer.writerow([token_test_idx, test_idx, track_id, token_pos])

                    print(
                        f'Token-level attribution completed in {time.time() - token_start:.0f}s '
                        f'with shape {token_scores_np.shape}'
                    )

        attr_time = time.time() - attr_start
        print(f'Attribution completed in {attr_time:.0f}s')
        scores_raw_np = None if self_sanity_precomputed is not None else scores.cpu().numpy()

    # ── SELF-SANITY POST-PROCESSING ────────────────────────────────────────
    import csv as _csv

    if args.self_sanity_check and args.mode == 'full':
        if num_train != num_test:
            raise RuntimeError(
                f'self-sanity requires num_train == num_test, got {num_train} vs {num_test}'
            )
        if self_sanity_precomputed is not None:
            self_metrics, self_arrays = self_sanity_precomputed
        else:
            # In self-sanity mode, enforce dattri contract directly instead of
            # auto-transpose heuristics: score must be (num_train, num_test).
            if scores_raw_np.ndim != 2 or scores_raw_np.shape != (num_train, num_test):
                raise RuntimeError(
                    'self-sanity requires raw score shape (num_train, num_test); '
                    f'got {scores_raw_np.shape}, expected {(num_train, num_test)}'
                )
            scores_np = scores_raw_np
            self_metrics, self_arrays = compute_self_sanity_metrics(scores_np)
            np.save(str(results_folder / f'{args.stage}_{args.method}_self_sanity_scores.npy'), scores_np)

        np.save(str(results_folder / f'{args.stage}_{args.method}_self_diag_percentiles.npy'),
                self_arrays['diag_percentiles'])
        np.save(str(results_folder / f'{args.stage}_{args.method}_self_top1_hits_strict.npy'),
                self_arrays['top1_hits_strict'])
        np.save(str(results_folder / f'{args.stage}_{args.method}_self_top1_hits_tie.npy'),
                self_arrays['top1_hits_tie'])
        np.save(str(results_folder / f'{args.stage}_{args.method}_self_diag_ranks.npy'),
                self_arrays['diag_ranks'])
        np.save(str(results_folder / f'{args.stage}_{args.method}_self_top1_indices.npy'),
                self_arrays['top1_indices'])

        metrics_path = results_folder / f'{args.stage}_{args.method}_self_sanity_metrics.json'
        with open(str(metrics_path), 'w') as f:
            json.dump(self_metrics, f, indent=2)

        axis_path = results_folder / f'{args.stage}_{args.method}_self_sanity_axis.tsv'
        with open(str(axis_path), 'w', newline='') as f:
            writer = _csv.writer(f, delimiter='\t')
            writer.writerow([
                'sample_index',
                'train_ds_index',
                'train_sqlite_idx',
                'train_track_id',
                'train_genre',
                'diag_score',
                'diag_rank',
                'diag_percentile',
                'top1_hit_strict',
                'top1_hit_tie',
                'top1_index',
            ])
            train_genres = [_resolve_genre(tid) for tid in train_track_ids]
            for i in range(num_train):
                writer.writerow([
                    i,
                    int(train_indices[i]),
                    int(train_sqlite_indices[i]),
                    train_track_ids[i],
                    train_genres[i],
                    float(self_arrays['diag_scores'][i]),
                    float(self_arrays['diag_ranks'][i]),
                    float(self_arrays['diag_percentiles'][i]),
                    bool(self_arrays['top1_hits_strict'][i]),
                    bool(self_arrays['top1_hits_tie'][i]),
                    int(self_arrays['top1_indices'][i]),
                ])

        print('[self-sanity-check] Completed')
        print(
            f'  top1_hit_rate_strict={self_metrics["top1_hit_rate_strict"]:.4f}, '
            f'top1_hit_rate_tie={self_metrics["top1_hit_rate_tie"]:.4f}, '
            f'mean_percentile={self_metrics["mean_percentile"]:.2f}, '
            f'median_percentile={self_metrics["median_percentile"]:.2f}'
        )
        print(f'  metrics: {metrics_path}')
        print(f'  axis:    {axis_path}')

        if args.wandb:
            import wandb
            wandb.log({
                "self_sanity/top1_hit_rate_strict": self_metrics["top1_hit_rate_strict"],
                "self_sanity/top1_hit_rate_tie": self_metrics["top1_hit_rate_tie"],
                "self_sanity/mean_percentile": self_metrics["mean_percentile"],
                "self_sanity/median_percentile": self_metrics["median_percentile"],
                "self_sanity/p10_percentile": self_metrics["p10_percentile"],
                "self_sanity/p90_percentile": self_metrics["p90_percentile"],
                "attribution/time_seconds": attr_time,
                "attribution/time_minutes": attr_time / 60,
                "timing/total_seconds": time.time() - run_start,
            })
            if args.alert_every > 0:
                wandb.alert(
                    title=f"Self sanity complete: {args.method} on {args.stage}",
                    text=(f"N={num_train} top1_strict={self_metrics['top1_hit_rate_strict']:.4f} "
                          f"mean_pct={self_metrics['mean_percentile']:.2f}"),
                    level=wandb.AlertLevel.INFO,
                )
            wandb.finish()
        return

    # ── NORMAL POST-PROCESSING: save scores + rankings ─────────────────────
    scores_np = normalize_score_shape(scores_raw_np, num_train, num_test, label='score')

    if len(generated_tokens_dataset.track_ids) == num_test:
        test_track_ids = [str(tid) for tid in generated_tokens_dataset.track_ids]
    else:
        test_track_ids = [str(i) for i in range(num_test)]

    save_attribution_results(
        scores_np=scores_np,
        args=args,
        results_folder=results_folder,
        num_train=num_train,
        num_test=num_test,
        train_indices=train_indices,
        train_track_ids=train_track_ids,
        train_sqlite_indices=train_sqlite_indices,
        test_track_ids=test_track_ids,
        test_losses=test_losses,
        token_scores_np=token_scores_np,
        token_axis_path=token_axis_path,
        attr_time=attr_time,
        run_id=run_id,
        _shard_tag=_shard_tag,
        _resolve_genre=_resolve_genre,
        generated_tokens_dataset=generated_tokens_dataset,
        print_all_scores=getattr(args, 'print_all_scores', False),
    )


def _shard_worker(rank: int, args) -> None:
    """Worker for auto multi-GPU spawn: each process handles one shard."""
    args = copy.copy(args)
    args.shard_id = rank
    torch.cuda.set_device(rank)
    run_attribution(args)


def main():
    parser = argparse.ArgumentParser(description='Data attribution for MusicLM')

    # required
    parser.add_argument('--stage', type=str, required=True, choices=['semantic', 'coarse', 'fine', 'all'],
                        help='Stage to run. Use "all" to run semantic+coarse+fine sequentially.')

    # sharding mode
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'cache_shard', 'merge_attribute',
                                 'hessian_shard', 'precompute', 'score_shard',
                                 'resume_after_grad'],
                        help='full: original single-job mode. '
                             'cache_shard/merge_attribute: disk-offload shard/merge. '
                             'hessian_shard/precompute/score_shard: streaming compute-and-discard '
                             '(factgrass only, no gradient storage).')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Shard index (0-based) for cache_shard mode.')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of shards for cache_shard / merge_attribute mode.')
    parser.add_argument('--shard_source_base', type=str, default=None,
                        help='Override shard source root for merge_attribute mode. '
                             'If set, reads Phi/Q shard files from <shard_source_base>/<stage>/shards/ '
                             'instead of <results_folder>/shards/. '
                             'Useful for re-running merge (e.g. with different regularization) '
                             'without re-caching gradients.')

    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to trained transformer checkpoint (single-stage mode).')
    parser.add_argument('--preprocessed_data_folder', type=str, required=True,
                        help='Folder containing memmap outputs (metadata.json/id_mapping.npy/*_tokens.npy)')
    parser.add_argument('--generated_tokens_file', type=str, default=None,
                        help='.pt file with generated tokens (optional when --generated_tokens_dir is used).')
    parser.add_argument('--generated_tokens_dir', type=str, default=None,
                        help='Directory containing per-sample .npz generated token files.')
    parser.add_argument('--generated_tokens_manifest', type=str, default=None,
                        help='Optional TSV manifest (selected_samples_shard_*.tsv) to preserve test order.')

    # per-stage checkpoint / tokens (used when --stage all)
    parser.add_argument('--semantic_checkpoint', type=str, default=None)
    parser.add_argument('--coarse_checkpoint', type=str, default=None)
    parser.add_argument('--fine_checkpoint', type=str, default=None)
    parser.add_argument('--semantic_generated_tokens_file', type=str, default=None)
    parser.add_argument('--coarse_generated_tokens_file', type=str, default=None)
    parser.add_argument('--fine_generated_tokens_file', type=str, default=None)
    parser.add_argument('--semantic_generated_tokens_dir', type=str, default=None)
    parser.add_argument('--coarse_generated_tokens_dir', type=str, default=None)
    parser.add_argument('--fine_generated_tokens_dir', type=str, default=None)
    parser.add_argument('--semantic_generated_tokens_manifest', type=str, default=None)
    parser.add_argument('--coarse_generated_tokens_manifest', type=str, default=None)
    parser.add_argument('--fine_generated_tokens_manifest', type=str, default=None)

    # config
    parser.add_argument('--model_config', type=str, default='./configs/model/musiclm_large.json')
    parser.add_argument('--tracks_csv', type=str, default=None,
                        help='Path to FMA tracks.csv for genre metadata and/or subset filtering.')
    parser.add_argument('--genre_column', type=str, default='genre_top',
                        help='Genre column name in tracks CSV (default: genre_top).')
    parser.add_argument('--fma_train_subset', type=str, default=None,
                        help='Filter train pool to tracks belonging to this FMA subset '
                             '(e.g. small or small,medium). Requires --tracks_csv.')

    # attribution settings
    parser.add_argument('--method', type=str, default='trak', choices=['trak', 'grad_cos', 'logra', 'tracin', 'factgrass', 'cg', 'lissa', 'logix'])
    parser.add_argument('--condition', type=str, default='main',
                        choices=['main', 'random_baseline'],
                        help='Experimental condition label. random_baseline skips model attribution and '
                             'generates seeded random scores with matched train/test shape.')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Optional stable run identifier. Defaults to condition/stage/method/seed.')
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help='Limit number of train samples (default: use all from split)')
    parser.add_argument('--num_train_tracks', type=int, default=None,
                        help='Limit train pool to samples from at most this many unique track IDs.')
    parser.add_argument('--samples_per_track', type=int, default=None,
                        help='Randomly sample at most this many sequences per track (seeded by --seed).')
    parser.add_argument('--train_track_sampling', type=str, default='first',
                        choices=['first', 'random'],
                        help='How to choose tracks when --num_train_tracks is set. '
                             'first = dataset order, random = seeded unique-track sample.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for Python/NumPy/Torch (controls random crop reproducibility).')

    # method-specific (TRAK / grad_cos)
    parser.add_argument('--proj_dim', type=int, default=2048, help='Random projection dimension')
    parser.add_argument('--regularization', type=float, default=0.0,
                        help='Ridge regularization λ added to kernel diagonal before solve: (Φ^TΦ + λI)^{-1}. Default 0.0 (no regularization).')
    parser.add_argument('--trak_checkpoint_paths', type=str, default=None,
                        help='Optional comma-separated checkpoint paths for TRAK ensemble. '
                             'If unset, uses the current stage checkpoint only.')
    parser.add_argument('--logra_hessian', type=str, default='eFIM', choices=['Identity', 'eFIM'],
                        help='LoGra Hessian approximation. Default eFIM.')
    parser.add_argument('--logra_damping', type=float, default=0.1,
                        help='LoGra damping for inverse eFIM. Default 0.1 (paper: 0.1*mean(eig)).')
    parser.add_argument('--logra_offload', type=str, default='cpu', choices=['none', 'cpu', 'disk'],
                        help='LoGra cache/offload backend. Default cpu.')
    parser.add_argument('--logra_cache_dir', type=str, default=None,
                        help='Optional explicit cache directory for LoGra disk offload.')
    parser.add_argument('--logra_chunk_size', type=int, default=16,
                        help='Chunk size used by LoGra disk offload.')
    parser.add_argument('--factgrass_hessian', type=str, default='eFIM', choices=['Identity', 'eFIM'],
                        help='FactGraSS Hessian approximation. Default eFIM.')
    parser.add_argument('--factgrass_damping', type=float, default=None,
                        help='FactGraSS damping for inverse eFIM. Default None.')
    parser.add_argument('--factgrass_blowup_factor', type=int, default=4,
                        help='FactGraSS intermediate-dimension multiplier. Default 4.')
    parser.add_argument('--factgrass_offload', type=str, default='cpu', choices=['none', 'cpu', 'disk'],
                        help='FactGraSS cache/offload backend. Default cpu.')
    parser.add_argument('--factgrass_cache_dir', type=str, default=None,
                        help='Optional explicit cache directory for FactGraSS disk offload.')
    parser.add_argument('--factgrass_chunk_size', type=int, default=16,
                        help='Chunk size used by FactGraSS disk offload.')
    parser.add_argument('--factgrass_streaming', action='store_true',
                        help='Use streaming compute-and-discard mode for FactGraSS '
                             '(hessian_shard/precompute/score_shard). '
                             'No gradient files stored to disk.')
    parser.add_argument('--tracin_checkpoint_paths', type=str, default=None,
                        help='Optional comma-separated checkpoint paths for TracIn trajectory ensembling. '
                             'If unset, uses the current stage checkpoint only.')
    parser.add_argument('--tracin_weights', type=str, default=None,
                        help='Optional comma-separated checkpoint weights for TracIn. '
                             'Must match checkpoint count when provided.')
    parser.add_argument('--tracin_normalized_grad', action='store_true',
                        help='Use normalized gradients in dattri TracIn (CosIn-style). '
                             'Default False keeps paper-style gradient dot products.')
    parser.add_argument('--cg_max_iter', type=int, default=10,
                        help='CG IF: maximum conjugate-gradient iterations. Default 10.')
    parser.add_argument('--cg_tol', type=float, default=1e-7,
                        help='CG IF: convergence tolerance on residual norm. Default 1e-7.')
    parser.add_argument('--cg_mode', type=str, default='rev-rev', choices=['rev-rev', 'rev-fwd'],
                        help='CG IF autodiff mode. rev-rev = more compatible, rev-fwd = more memory-efficient.')
    parser.add_argument('--cg_regularization', type=float, default=0.0,
                        help='CG IF Hessian regularization added as lambda * I. Default 0.0.')
    parser.add_argument('--lissa_batch_size', type=int, default=1,
                        help='LiSSA: mini-batch size for HVP estimation inner loop. Default 1.')
    parser.add_argument('--lissa_num_repeat', type=int, default=1,
                        help='LiSSA: number of independent IHVP estimates to average. Default 1.')
    parser.add_argument('--lissa_recursion_depth', type=int, default=5000,
                        help='LiSSA: recursion depth (Taylor expansion terms) per estimate. Default 5000.')
    parser.add_argument('--lissa_damping', type=float, default=0.0,
                        help='LiSSA: damping factor for non-convexity. Default 0.0.')
    parser.add_argument('--lissa_scaling', type=float, default=50.0,
                        help='LiSSA: scaling factor for convergence. Default 50.0.')
    parser.add_argument('--lissa_mode', type=str, default='rev-rev', choices=['rev-rev', 'rev-fwd'],
                        help='LiSSA autodiff mode. rev-rev = more compatible, rev-fwd = more memory-efficient.')
    parser.add_argument('--custom_backward', action='store_true',
                        help='Use standard loss.backward() instead of functorch vmap for supported methods.')
    # LogIX arguments
    parser.add_argument('--logix_damping', type=float, default=1e-3,
                        help='LogIX: damping factor for influence computation numerical stability. Default 1e-3.')
    parser.add_argument('--logix_hessian', type=str, default='kfac',
                        choices=['kfac', 'ekfac', 'raw'],
                        help='LogIX: Hessian approximation type. Default kfac.')
    parser.add_argument('--logix_dir', type=str, default='./logix_logs',
                        help='LogIX: root directory for gradient log storage. Default ./logix_logs.')
    parser.add_argument('--logix_cpu_offload', action='store_true',
                        help='LogIX: offload gradient covariance matrices to CPU to save GPU memory.')
    parser.add_argument('--grad_cos_chunk_size', type=int, default=10000000,
                        help='grad_cos --custom_backward: parameter elements per projection chunk. '
                             'Larger = faster but uses more GPU memory (default: 50000 → ~100MB for proj_dim=512).')

    # output
    parser.add_argument('--results_folder', type=str, default='./results/attribution')
    parser.add_argument('--print_all_scores', action='store_true',
                        help='Print every attribution score value and train/test axis mappings.')
    parser.add_argument('--skip_grad_cos_token_level', action='store_true',
                        help='Skip token-level grad_cos attribution (default: compute track + token levels).')
    parser.add_argument('--skip_self_sanity', action='store_true',
                        help='Skip self-sanity computation in merge_attribute mode. '
                             'Useful when only attribution scores are needed.')
    parser.add_argument('--self-sanity-check', action='store_true',
                        help='Run sample-level self-influence sanity check only (test=query is train set '
                             'with same sample order).')
    parser.add_argument('--self-sanity-query-chunk-size', type=int, default=256,
                        help='Number of query samples per chunk for TRAK self-sanity streaming '
                             '(prevents full NxN materialization).')
    parser.add_argument('--last-represent-layer', dest='last_represent_layer', action='store_true',
                        help='Restrict attribution gradients to the final pre-head representation layer '
                             '(transformer.layers[-1] + transformer.norm) for any method.')
    parser.add_argument('--use-head-layer', action='store_true',
                        help='Restrict attribution gradients to output head parameters (logit_weights.*) '
                             'for any method.')
    # Backward compatibility alias
    parser.add_argument('--last_layer_only', dest='last_represent_layer', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--bf16', action='store_true',
                        help='Load model in bfloat16 to halve weight and gradient memory (~50%% reduction). '
                             'Results are numerically near-identical to float32.')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing in the transformer to trade compute for '
                             'activation memory. Mathematically identical; ~1.3x slower.')
    parser.add_argument('--compile', action='store_true',
                        help='Apply torch.compile to the stage wrapper for faster forward+backward. '
                             'First iteration is slow (compilation), subsequent ones are 1.5–2x faster.')
    parser.add_argument('--dataloader_workers', type=int, default=0,
                        help='Number of DataLoader worker processes for prefetching (default: 0). '
                             'Set >0 only if the HPC environment supports multiprocessing IPC sockets.')

    # wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str,
                        default=os.environ.get('WANDB_PROJECT', '<your-wandb-project>'),
                        help='wandb project name. Defaults to $WANDB_PROJECT or a placeholder; '
                             'do not hardcode a personal project name.')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='wandb group name (default: attr_{stage})')
    parser.add_argument('--alert_every', type=int, default=0,
                        help='Send wandb alert on completion (0 = no alerts)')

    args = parser.parse_args()

    # Validate sharding args
    if args.mode == 'cache_shard' and args.method not in ('trak', 'grad_cos', 'tracin', 'factgrass'):
        parser.error(
            f'--mode {args.mode} is only supported with --method trak, grad_cos, tracin, or factgrass '
            f'(got {args.method})'
        )
    if args.mode == 'merge_attribute' and args.method not in ('trak', 'grad_cos', 'tracin', 'logra', 'factgrass'):
        parser.error(
            f'--mode {args.mode} is only supported with --method trak, grad_cos, tracin, logra, or factgrass '
            f'(got {args.method})'
        )
    if args.mode == 'resume_after_grad' and args.method != 'logra':
        parser.error(
            f'--mode {args.mode} is only supported with --method logra '
            f'(got {args.method})'
        )
    if args.mode in ('hessian_shard', 'precompute', 'score_shard') and args.method != 'factgrass':
        parser.error(f'--mode {args.mode} is only supported with --method factgrass')
    if args.mode in ('cache_shard', 'merge_attribute') and args.method == 'grad_cos' and not args.custom_backward:
        parser.error(f'--mode {args.mode} with --method grad_cos requires --custom_backward')
    if args.self_sanity_check and args.mode not in ('full', 'merge_attribute'):
        parser.error('--self-sanity-check currently supports only --mode full or --mode merge_attribute')
    if args.condition != 'main' and args.mode != 'full':
        parser.error('--condition random_baseline currently supports only --mode full')
    if args.condition != 'main' and args.self_sanity_check:
        parser.error('--condition random_baseline does not support --self-sanity-check')
    if args.self_sanity_query_chunk_size <= 0:
        parser.error('--self-sanity-query-chunk-size must be > 0')
    if args.num_train_tracks is not None and args.num_train_tracks <= 0:
        parser.error('--num_train_tracks must be > 0')
    if args.method == 'logra' and args.proj_dim <= 0:
        parser.error('--proj_dim must be > 0 for LoGra')
    if args.method == 'logra':
        import math as _math
        _root = int(_math.isqrt(args.proj_dim))
        if _root * _root != args.proj_dim:
            parser.error(
                f'--proj_dim must be a perfect square for LoGra (factorized Kronecker projection). '
                f'Got {args.proj_dim}. Recommended values: 4096 (=64^2, paper LLM setting) or '
                f'16384 (=128^2, paper small-scale setting).'
            )
    if args.method == 'logra' and args.use_head_layer:
        parser.error('--method logra does not support --use-head-layer because LoGra hooks only nn.Linear modules')
    if args.method == 'logra' and args.logra_chunk_size <= 0:
        parser.error('--logra_chunk_size must be > 0')
    if args.method == 'factgrass' and args.proj_dim <= 0:
        parser.error('--proj_dim must be > 0 for FactGraSS')
    if args.method == 'factgrass' and args.use_head_layer:
        parser.error('--method factgrass does not support --use-head-layer because FactGraSS hooks only nn.Linear modules')
    if args.method == 'factgrass' and args.factgrass_chunk_size <= 0:
        parser.error('--factgrass_chunk_size must be > 0')
    if args.method == 'factgrass' and args.factgrass_blowup_factor <= 0:
        parser.error('--factgrass_blowup_factor must be > 0')
    if args.method == 'tracin' and args.tracin_weights is not None and args.tracin_checkpoint_paths is None:
        parser.error('--tracin_weights requires --tracin_checkpoint_paths')
    if args.last_represent_layer and args.use_head_layer:
        parser.error('--last-represent-layer and --use-head-layer are mutually exclusive')
    if args.mode in ('cache_shard', 'hessian_shard', 'score_shard'):
        if args.num_shards < 2:
            parser.error(f'--num_shards must be >= 2 in {args.mode} mode')
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            parser.error(f'--shard_id must be in [0, {args.num_shards - 1}]')
    elif args.mode == 'merge_attribute':
        if args.method == 'logra' and args.num_shards < 1:
            parser.error('--num_shards must be >= 1 in merge_attribute mode for LoGra')
        elif args.method != 'logra' and args.num_shards < 2:
            parser.error('--num_shards must be >= 2 in merge_attribute mode')
    if args.mode == 'full' and args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            parser.error(f'--shard_id must be in [0, {args.num_shards - 1}]')

    if args.stage == 'all':
        # Validate that per-stage args are provided
        for stage_name in ('semantic', 'coarse', 'fine'):
            ckpt = getattr(args, f'{stage_name}_checkpoint', None) or args.checkpoint_path
            stage_tokens_file = (
                getattr(args, f'{stage_name}_generated_tokens_file', None) or args.generated_tokens_file
            )
            stage_tokens_dir = (
                getattr(args, f'{stage_name}_generated_tokens_dir', None) or args.generated_tokens_dir
            )
            if args.condition == 'main' and not ckpt:
                parser.error(f'--{stage_name}_checkpoint (or --checkpoint_path) is required when --stage all')
            if (
                args.mode != 'cache_shard'
                and not args.self_sanity_check
                and not stage_tokens_file
                and not stage_tokens_dir
            ):
                parser.error(
                    f'Provide --{stage_name}_generated_tokens_file/--{stage_name}_generated_tokens_dir '
                    f'(or shared --generated_tokens_file/--generated_tokens_dir) when --stage all'
                )

        base_results_folder = args.results_folder
        for stage_name in ('semantic', 'coarse', 'fine'):
            stage_args = copy.copy(args)
            stage_args.stage = stage_name
            stage_args.checkpoint_path = (
                getattr(args, f'{stage_name}_checkpoint', None) or args.checkpoint_path
            )
            stage_args.generated_tokens_file = (
                getattr(args, f'{stage_name}_generated_tokens_file', None) or args.generated_tokens_file
            )
            stage_args.generated_tokens_dir = (
                getattr(args, f'{stage_name}_generated_tokens_dir', None) or args.generated_tokens_dir
            )
            stage_args.generated_tokens_manifest = (
                getattr(args, f'{stage_name}_generated_tokens_manifest', None) or args.generated_tokens_manifest
            )
            stage_args.results_folder = str(Path(base_results_folder) / stage_name)
            print(f'\n{"=" * 60}')
            print(f'  Running stage: {stage_name.upper()}')
            print(f'{"=" * 60}')
            run_attribution(stage_args)
    else:
        # Single-stage mode: append stage name to results_folder so each stage gets its own
        # subdirectory (mirrors the --stage all behaviour and prevents files being overwritten
        # when multiple stages are submitted as separate jobs to the same base folder).
        args.results_folder = str(Path(args.results_folder) / args.stage)
        # For single-stage mode, fall back to per-stage checkpoint if --checkpoint_path not given.
        if not args.checkpoint_path and args.stage in ('semantic', 'coarse', 'fine'):
            stage_ckpt = getattr(args, f'{args.stage}_checkpoint', None)
            if stage_ckpt:
                args.checkpoint_path = stage_ckpt
        if args.condition == 'main' and not args.checkpoint_path:
            parser.error('--checkpoint_path (or --{stage}_checkpoint) is required for single-stage mode')
        if (
            args.mode not in ('cache_shard', 'hessian_shard', 'score_shard')
            and not args.self_sanity_check
            and not args.generated_tokens_file
            and not args.generated_tokens_dir
        ):
            parser.error('provide --generated_tokens_file or --generated_tokens_dir for single-stage mode')

        num_gpus = torch.cuda.device_count()
        if args.mode == 'full' and args.num_shards == 1 and num_gpus > 1:
            print(f'Detected {num_gpus} GPUs — auto-sharding across {num_gpus} processes.')
            args.num_shards = num_gpus
            torch.multiprocessing.spawn(_shard_worker, args=(args,), nprocs=num_gpus, join=True)
            # All shards done — merge into final rankings + metadata
            print(f'\n[merge] Merging {num_gpus} shards ...')
            merge_sharded_results(
                Path(args.results_folder), args.stage, args.method, num_gpus,
            )
        elif args.mode == 'full' and args.num_shards > 1:
            # Manual multi-shard: this is shard args.shard_id of args.num_shards.
            run_attribution(args)
            # Only the last shard triggers the merge (all others must have finished first).
            if args.shard_id == args.num_shards - 1:
                print(f'\n[merge] Shard {args.shard_id} is last — merging {args.num_shards} shards ...')
                merge_sharded_results(
                    Path(args.results_folder), args.stage, args.method, args.num_shards,
                )
        else:
            run_attribution(args)


if __name__ == '__main__':
    main()
