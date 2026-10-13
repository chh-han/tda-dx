import argparse
import functools
import json
import os
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tda_dx.config import (
    create_clap_quantized_from_config,
    create_coarse_transformer_from_config,
    create_fine_transformer_from_config,
    create_hubert_kmeans_from_config,
    create_neural_codec_from_config,
    create_semantic_transformer_from_config,
    create_single_stage_trainer_from_config,
    load_model_config,
    load_training_config,
)
from tda_dx.training.train_utils import load_checkpoint_from_args, validate_train_args

print = functools.partial(print, flush=True)


STAGES = ('semantic', 'coarse', 'fine')
EXPECTED_CE_WEIGHTS = {
    'semantic': 2,
    'coarse': 3,
    'fine': 3,
}


def parse_bool(value):
    if value is None or isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ('true', '1', 'yes', 'y', 'on'):
        return True
    if v in ('false', '0', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected boolean (true/false), got: {value}')


def parse_alert_every(value):
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)

    v = str(value).strip().lower()
    if v in ('true', 'yes', 'y', 'on'):
        return 1
    if v in ('false', 'no', 'n', 'off'):
        return 0

    try:
        parsed = int(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'Expected non-negative integer or boolean for alert_every, got: {value}'
        ) from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(f'alert_every must be >= 0, got: {parsed}')
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='train transformer stage (semantic/coarse/fine)')

    parser.add_argument('--stage', required=True, choices=STAGES)

    # base paths
    parser.add_argument('--results_folder', default=None,
                        help='If omitted, defaults to ./results/{stage}')
    parser.add_argument('--continue_from_dir', default=None, type=str)
    parser.add_argument('--continue_from_step', default=None, type=int)
    parser.add_argument('--model_config', default='./configs/model/musiclm_large.json')
    parser.add_argument('--training_config', default='./configs/training/train_musiclm_fma.json')
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')
    parser.add_argument('--fine_tune_from', default=None, type=str)

    # runtime / distributed
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--expect_distributed', action='store_true',
                        help='Fail if WORLD_SIZE=1. Useful when a multi-GPU launch is required.')

    # logging
    parser.add_argument('--log_with', default='tensorboard',
                        choices=['none', 'tensorboard', 'wandb', 'all'],
                        help='Tracker backend for accelerate.')
    parser.add_argument('--logging_dir', default=None,
                        help='If omitted, defaults to ./logs/{stage}')
    parser.add_argument('--wandb_project',
                        default=os.environ.get('WANDB_PROJECT', '<your-wandb-project>'),
                        help='Used when wandb logging is enabled. Set via --wandb_project '
                             'or the WANDB_PROJECT env var; do not hardcode a personal project name.')
    parser.add_argument('--wandb_tags', nargs='*', default=None,
                        help='Optional wandb tags (space separated).')
    parser.add_argument('--alert_every', type=parse_alert_every, default=0,
                        help='Send wandb alert every N validation steps. bool true/false is also supported.')
    parser.add_argument('--test_mode', action='store_true',
                        help='Skip training and run test-split evaluation only. Requires --continue_from_dir or --fine_tune_from.')
    parser.add_argument('--run_test_after_train', type=parse_bool, default=True,
                        help='Run test-split evaluation after training completes (default: true).')
    parser.add_argument('--run_validation_before_test', type=parse_bool, default=False,
                        help='Run one full validation-split evaluation immediately before test evaluation.')
    parser.add_argument('--test_metrics_json', type=parse_bool, nargs='?', const=True, default=True,
                        help='Whether to save final test metrics JSON. If true, saves to {results_folder}/{stage}.test_metrics.final.json')
    parser.add_argument('--fma_test_split', type=str, default='test',
                        help='Official split name to use as test set when use_preprocessed_data=true.')
    parser.add_argument('--fma_tracks_csv_path', type=str, default=None,
                        help='Optional tracks.csv path. Required when filtering validation/test by subset.')
    parser.add_argument('--fma_valid_subset', type=str, default=None,
                        help='Optional subset filter for validation only (e.g., small or small,large).')
    parser.add_argument('--fma_test_subset', type=str, default=None,
                        help='Optional subset filter for test only (e.g., large).')

    # trainer overrides (common)
    parser.add_argument('--data_folder', default=None,
                        help='Override {stage}_trainer_cfg.folder')
    parser.add_argument('--use_preprocessed_data', type=parse_bool, default=None,
                        help='Override {stage}_trainer_cfg.use_preprocessed_data')
    parser.add_argument('--num_train_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--grad_accum_every', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_warmup', type=int, default=None)
    parser.add_argument('--lr_schedule', type=str, default=None, choices=['constant', 'cosine'],
                        help='LR schedule after warmup. constant=flat, cosine=cosine decay.')
    parser.add_argument('--lr_min_factor', type=float, default=None,
                        help='Minimum LR as fraction of peak LR for cosine schedule (default: 0.01).')
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--valid_frac', type=float, default=None)
    parser.add_argument('--valid_freq', type=float, default=None,
                        help='Validation frequency as fraction of one training epoch in [0, 1]. 1.0 means once per epoch.')
    parser.add_argument('--save_results_every', type=int, default=None)
    parser.add_argument('--save_model_every', type=int, default=None)
    parser.add_argument('--val_every_n_epochs', type=int, default=None)
    parser.add_argument('--val_every_n_steps', type=int, default=None)
    parser.add_argument('--save_top_k_checkpoints', type=int, default=None)
    parser.add_argument('--val_early_stop_patience_validations', type=int, default=None,
                        help='Early-stop patience in number of validation checks (preferred).')
    parser.add_argument('--val_early_stop_patience_epochs', type=int, default=None,
                        help='Early-stop patience in epochs without improvement (0 disables).')
    parser.add_argument('--val_early_stop_metric', type=str, default=None, choices=['loss', 'accuracy'],
                        help='Metric used for early-stop improvement tracking.')
    parser.add_argument('--val_top_k', type=int, nargs='*', default=None,
                        help='Validation top-k accuracy metrics to log, e.g. --val_top_k 1 5 10')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--prefetch_factor', type=int, default=None)
    parser.add_argument('--persistent_workers', type=parse_bool, default=None)
    parser.add_argument('--pin_memory', type=parse_bool, default=None)
    parser.add_argument('--save_predicted_tokens', type=parse_bool, default=None)
    parser.add_argument('--save_reconstructed_wave', type=parse_bool, default=None)
    parser.add_argument('--cross_entropy_loss_weights', type=float, nargs='*', default=None,
                        help='semantic expects 2 weights, coarse/fine expect 3')

    # data traversal
    parser.add_argument('--exhaustive_stride', type=int, default=None,
                        help='Stride in seconds between exhaustive traversal anchors (default: 1)')
    parser.add_argument('--jitter_seconds', type=float, default=None,
                        help='Random token-level jitter range in seconds, applied ± around anchors (default: 0)')

    return parser.parse_args()


def _stage_title(stage: str) -> str:
    return f'{stage.capitalize()} Stage Training'


def _override_if_not_none(obj, field_name: str, value) -> None:
    if value is not None:
        setattr(obj, field_name, value)


def _resolve_log_with(mode: str):
    if mode == 'none':
        return []
    if mode == 'all':
        return ['tensorboard', 'wandb']
    return [mode]


def _build_wandb_run_name(stage: str, trainer_cfg, model_config) -> str:
    # Get stage-specific model config
    stage_model_cfg = getattr(model_config, f'{stage}_cfg')
    
    lr_str = f'{trainer_cfg.lr:g}'
    return (
        f'{stage}_lr{lr_str}_wu{trainer_cfg.lr_warmup}'
        f'_bs{trainer_cfg.batch_size}_ga{trainer_cfg.grad_accum_every}'
        f'_d{stage_model_cfg.depth}_h{stage_model_cfg.heads}_dim{stage_model_cfg.dim}'
    )


def _resolve_device(args: argparse.Namespace) -> str:
    has_cuda = torch.cuda.is_available()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if args.device == 'cpu':
        return 'cpu'

    if args.device == 'cuda':
        if not has_cuda:
            raise RuntimeError('--device cuda was requested but CUDA is not available.')
        return f'cuda:{local_rank}' if int(os.environ.get('WORLD_SIZE', '1')) > 1 else 'cuda'

    # auto
    if has_cuda:
        return f'cuda:{local_rank}' if int(os.environ.get('WORLD_SIZE', '1')) > 1 else 'cuda'
    return 'cpu'


def _get_stage_cfg(training_config, stage: str):
    return getattr(training_config, f'{stage}_trainer_cfg')


def _apply_overrides(training_config, args: argparse.Namespace):
    cfg = _get_stage_cfg(training_config, args.stage)

    _override_if_not_none(cfg, 'folder', args.data_folder)
    _override_if_not_none(cfg, 'num_train_steps', args.num_train_steps)
    _override_if_not_none(cfg, 'batch_size', args.batch_size)
    _override_if_not_none(cfg, 'grad_accum_every', args.grad_accum_every)
    _override_if_not_none(cfg, 'lr', args.lr)
    _override_if_not_none(cfg, 'lr_warmup', args.lr_warmup)
    _override_if_not_none(cfg, 'lr_schedule', args.lr_schedule)
    _override_if_not_none(cfg, 'lr_min_factor', args.lr_min_factor)
    _override_if_not_none(cfg, 'wd', args.wd)
    _override_if_not_none(cfg, 'max_grad_norm', args.max_grad_norm)
    _override_if_not_none(cfg, 'valid_frac', args.valid_frac)
    _override_if_not_none(cfg, 'valid_freq', args.valid_freq)
    _override_if_not_none(cfg, 'save_results_every', args.save_results_every)
    _override_if_not_none(cfg, 'save_model_every', args.save_model_every)
    # Validation cadence is controlled by valid_freq; ignore step/epoch cadence knobs.
    cfg.val_every_n_epochs = 0
    cfg.val_every_n_steps = 0
    if args.val_every_n_epochs is not None or args.val_every_n_steps is not None:
        print('warning: --val_every_n_epochs/--val_every_n_steps are ignored; use --valid_freq (0~1).')
    _override_if_not_none(cfg, 'save_top_k_checkpoints', args.save_top_k_checkpoints)
    _override_if_not_none(cfg, 'val_early_stop_patience_validations', args.val_early_stop_patience_validations)
    _override_if_not_none(cfg, 'val_early_stop_patience_epochs', args.val_early_stop_patience_epochs)
    _override_if_not_none(cfg, 'val_early_stop_metric', args.val_early_stop_metric)
    _override_if_not_none(cfg, 'num_workers', args.num_workers)
    _override_if_not_none(cfg, 'prefetch_factor', args.prefetch_factor)

    if args.use_preprocessed_data is not None:
        cfg.use_preprocessed_data = args.use_preprocessed_data
    if args.save_predicted_tokens is not None:
        cfg.save_predicted_tokens = args.save_predicted_tokens
    if args.save_reconstructed_wave is not None:
        cfg.save_reconstructed_wave = args.save_reconstructed_wave
    if args.persistent_workers is not None:
        cfg.persistent_workers = args.persistent_workers
    if args.pin_memory is not None:
        cfg.pin_memory = args.pin_memory

    _override_if_not_none(cfg, 'exhaustive_stride', args.exhaustive_stride)
    _override_if_not_none(cfg, 'jitter_seconds', args.jitter_seconds)

    if args.cross_entropy_loss_weights is not None:
        expected = EXPECTED_CE_WEIGHTS[args.stage]
        if len(args.cross_entropy_loss_weights) != expected:
            raise ValueError(
                f'--cross_entropy_loss_weights for stage={args.stage} must have {expected} values, '
                f'got {len(args.cross_entropy_loss_weights)}'
            )
        cfg.cross_entropy_loss_weights = args.cross_entropy_loss_weights

    if cfg.valid_frac < 0 or cfg.valid_frac >= 1:
        raise ValueError(f'valid_frac must be in [0, 1). got {cfg.valid_frac}')
    if cfg.valid_freq < 0 or cfg.valid_freq > 1:
        raise ValueError(f'valid_freq must be in [0, 1]. got {cfg.valid_freq}')
    if cfg.batch_size <= 0 or cfg.num_train_steps <= 0:
        raise ValueError('batch_size and num_train_steps must be positive.')
    if getattr(cfg, 'stage', None) != args.stage:
        raise ValueError(
            f"training_config.{args.stage}_trainer_cfg.stage must be '{args.stage}', got {cfg.stage!r}"
        )

    return cfg


def _default_paths(args: argparse.Namespace) -> None:
    if args.results_folder is None:
        args.results_folder = f'./results/{args.stage}'
    if args.logging_dir is None:
        args.logging_dir = f'./logs/{args.stage}'


def _print_run_summary(args: argparse.Namespace, stage_cfg, device: str, log_with, wandb_run_name: str):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))

    print('=' * 60)
    print(_stage_title(args.stage))
    print('=' * 60)
    print(f'Host:                 {socket.gethostname()}')
    print(f'PID:                  {os.getpid()}')
    print(f'World size / rank:    {world_size} / {rank} (local_rank={local_rank})')
    print(f'Device:               {device} (cuda_count={num_gpus})')
    print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")}')
    print(f'Results folder:       {args.results_folder}')
    print(f'Model config:         {args.model_config}')
    print(f'Training config:      {args.training_config}')
    print(f'Continue from:        {args.continue_from_dir or "(none)"}')
    print(f'Fine tune from:       {args.fine_tune_from or "(none)"}')
    print(f'--- Effective {args.stage} trainer config ---')
    print(json.dumps(asdict(stage_cfg), indent=2))
    print('--- Logging ---')
    print(f'log_with:             {log_with}')
    if 'tensorboard' in log_with:
        print(f'logging_dir:          {args.logging_dir}')
    if 'wandb' in log_with:
        print(f'wandb_project:        {args.wandb_project}')
        print(f'wandb_run_name:       {wandb_run_name}')
        print(f'alert_every:          {args.alert_every}')
        if args.wandb_tags:
            print(f'wandb_tags:           {args.wandb_tags}')
    print(f'test_mode:            {args.test_mode}')
    print(f'run_test_after_train: {args.run_test_after_train}')
    print(f'run_val_before_test:  {args.run_validation_before_test}')
    print(f'fma_test_split:       {args.fma_test_split}')
    print(f'fma_tracks_csv_path:  {args.fma_tracks_csv_path or "(none)"}')
    print(f'fma_valid_subset:     {args.fma_valid_subset or "(none)"}')
    print(f'fma_test_subset:      {args.fma_test_subset or "(none)"}')
    print(f'test_metrics_json:    {args.test_metrics_json}')
    print('=' * 60)


def _required_memmap_files(stage: str, data_folder: Path):
    base = [
        data_folder / 'metadata.json',
        data_folder / 'id_mapping.npy',
        data_folder / 'clap_tokens.npy',
    ]
    if stage == 'semantic':
        return base + [data_folder / 'semantic_tokens.npy']
    if stage == 'coarse':
        return base + [data_folder / 'semantic_tokens.npy', data_folder / 'coarse_tokens.npy']
    if stage == 'fine':
        return base + [data_folder / 'coarse_tokens.npy', data_folder / 'fine_tokens.npy']
    raise ValueError(f'invalid stage: {stage}')


def _validate_mmap_preprocessed_folder(stage: str, data_folder: Path):
    memmap_files = _required_memmap_files(stage, data_folder)
    missing = [str(path) for path in memmap_files if not path.exists()]
    if len(missing) > 0:
        raise FileNotFoundError(
            f'memmap preprocessed data is required for stage={stage}, '
            f'but files are missing in {data_folder}: {missing}'
        )


def _create_transformer_for_stage(stage: str, model_config, checkpoint_path: str, device: str):
    if stage == 'semantic':
        print('loading semantic stage...')
        return create_semantic_transformer_from_config(model_config, checkpoint_path, device)
    if stage == 'coarse':
        print('loading coarse stage...')
        return create_coarse_transformer_from_config(model_config, checkpoint_path, device)
    if stage == 'fine':
        print('loading fine stage...')
        return create_fine_transformer_from_config(model_config, checkpoint_path, device)
    raise ValueError(f'invalid stage: {stage}')


def main() -> None:
    args = parse_args()
    _default_paths(args)
    validate_train_args(args)

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)
    stage_cfg = _apply_overrides(training_config, args)

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if args.expect_distributed and world_size <= 1:
        raise RuntimeError('--expect_distributed was set, but WORLD_SIZE=1. Launch with accelerate/torchrun.')

    device = _resolve_device(args)

    log_with = _resolve_log_with(args.log_with)
    accelerate_kwargs = {
        'log_with': log_with,
    }
    if 'tensorboard' in log_with:
        accelerate_kwargs['logging_dir'] = args.logging_dir

    wandb_run_name = _build_wandb_run_name(args.stage, stage_cfg, model_config)
    if 'wandb' in log_with:
        os.environ['WANDB_PROJECT'] = args.wandb_project
        os.environ['WANDB_NAME'] = wandb_run_name
        if args.wandb_tags:
            os.environ['WANDB_TAGS'] = ','.join(args.wandb_tags)

    _print_run_summary(args, stage_cfg, device, log_with, wandb_run_name)

    use_preprocessed_data = stage_cfg.use_preprocessed_data

    clap = None
    wav2vec = None
    encodec_wrapper = None

    if use_preprocessed_data:
        data_folder = Path(stage_cfg.folder)
        _validate_mmap_preprocessed_folder(args.stage, data_folder)

        print(f'training from preprocessed memmap data {stage_cfg.folder}')

        if args.stage == 'coarse':
            print('loading neural codec...')
            encodec_wrapper = create_neural_codec_from_config(model_config, device)
        elif args.stage == 'fine':
            print('loading neural codec...')
            encodec_wrapper = create_neural_codec_from_config(model_config, device)
    else:
        print('loading clap...')
        clap = create_clap_quantized_from_config(model_config, args.rvq_path, device)

        if args.stage in ('semantic', 'coarse'):
            print('loading wav2vec...')
            wav2vec = create_hubert_kmeans_from_config(model_config, args.kmeans_path, device)

        if args.stage in ('coarse', 'fine'):
            print('loading neural codec...')
            encodec_wrapper = create_neural_codec_from_config(model_config, device)

    transformer = _create_transformer_for_stage(args.stage, model_config, args.fine_tune_from, device)

    trainer = create_single_stage_trainer_from_config(
        model_config=model_config,
        training_config=training_config,
        stage=args.stage,
        results_folder=args.results_folder,
        transformer=transformer,
        clap=clap,
        wav2vec=wav2vec,
        encodec_wrapper=encodec_wrapper,
        device=device,
        alert_every=args.alert_every,
        accelerate_kwargs=accelerate_kwargs,
        config_paths=[args.model_config, args.training_config],
        use_mmap_data=use_preprocessed_data,
        use_fma_official_split=True,
        val_top_k=args.val_top_k,
        fma_test_split=args.fma_test_split,
        fma_tracks_csv_path=args.fma_tracks_csv_path,
        fma_valid_subset=args.fma_valid_subset,
        fma_test_subset=args.fma_test_subset,
    )

    load_checkpoint_from_args(trainer, args)

    if args.test_mode:
        if args.continue_from_dir is None and args.fine_tune_from is None:
            raise ValueError('--test_mode requires --continue_from_dir or --fine_tune_from to evaluate a trained checkpoint.')
        start = time.time()
        if args.run_validation_before_test:
            trainer.evaluate_validation(save_json=False, json_path=None)
        trainer.evaluate_test(save_json=args.test_metrics_json, json_path=None)
        trainer.accelerator.wait_for_everyone()
        if trainer.is_main:
            elapsed = time.time() - start
            print(f'Test-mode evaluation finished in {elapsed / 60:.1f} minutes.')
        return

    start = time.time()
    trainer.train()

    if args.run_test_after_train:
        if args.run_validation_before_test:
            trainer.evaluate_validation(save_json=False, json_path=None)
        trainer.evaluate_test(save_json=args.test_metrics_json, json_path=None)

    trainer.accelerator.wait_for_everyone()
    if trainer.is_main:
        elapsed = time.time() - start
        print(f'Training finished in {elapsed / 60:.1f} minutes.')


if __name__ == '__main__':
    main()
