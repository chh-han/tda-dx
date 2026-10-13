import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tda_dx.config import (create_clap_quantized_from_config,
                                 create_encodec_from_config,
                                 create_hubert_kmeans_from_config,
                                 load_model_config, load_training_config,
                                 create_data_preprocessor_from_config)



def main():
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--model_config', default='./configs/model/musiclm_large.json')
    parser.add_argument('--training_config', default='./configs/training/train_fma_preprocess.json')
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')
    parser.add_argument('--filter_fma', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--result_folder', default=None,
                        help='Optional override for output folder. If omitted, uses training config data_preprocessor_cfg.results_folder')
    parser.add_argument('--tracks_csv', default=None,
                        help='Optional path to FMA tracks.csv. If set, split metadata is saved with preprocessed outputs.')
    parser.add_argument('--input_audio_format', type=str, default='audio', choices=['audio', 'npy'],
                        help='audio: load wav/mp3/flac, npy: load pre-resampled waveform .npy files')
    parser.add_argument('--npy_sample_hz', type=int, default=48000,
                        help='sample rate of input .npy waveforms when --input_audio_format npy')
    parser.add_argument('--target_lufs', type=float, default=None,
                        help='target integrated loudness in LUFS for inline normalisation '
                             '(only applies to --input_audio_format audio). '
                             'e.g. --target_lufs -14.0. Requires pyloudnorm.')

    parser.add_argument('--resume_from_db', default=True, action=argparse.BooleanOptionalAction,
                        help='resume preprocessing from the last stored idx in preprocessed.db when possible')
    parser.add_argument('--replace_existing', default=False, action=argparse.BooleanOptionalAction,
                        help='overwrite existing outputs in result folder (db rows or memmap files)')
    parser.add_argument('--gc_every_n_steps', type=int, default=50,
                        help='run python gc and sqlite memory trim every N steps (0 disables)')
    parser.add_argument('--sqlite_cache_size_kib', type=int, default=65536,
                        help='sqlite page-cache size in KiB (negative PRAGMA cache_size form)')
    parser.add_argument('--output_format', type=str, default='memmap', choices=['db', 'memmap'],
                        help='db: write preprocessed.db, memmap: write *_tokens.npy + metadata.json/id_mapping.npy')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='number of DataLoader worker processes for data loading (0=synchronous). '
                             'Overrides the value in training config when specified.')

    args = parser.parse_args()

    print(f'using model config {args.model_config}, training config {args.training_config}, rvq checkpoint {args.rvq_path}, kmeans checkpoint {args.kmeans_path}')

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    clap = create_clap_quantized_from_config(model_config, args.rvq_path, device)

    print('loading wav2vec...')
    wav2vec = create_hubert_kmeans_from_config(model_config, args.kmeans_path, device)

    print('loading encodec...')
    encodec_wrapper = create_encodec_from_config(model_config, device)


    # get rid of some experimental tracks, see notebooks/analyze_fma.ipynb
    if args.filter_fma:
        try:
            import pandas as pd
            import ast
        except ImportError:
            pd = None

        assert pd is not None, 'pandas not found, please install pandas to filter fma'

        metadata_folder = training_config.data_preprocessor_cfg.metadata_folder

        tracks = pd.read_csv(os.path.join(metadata_folder, 'tracks.csv'), index_col=0, header=[0, 1])
        tracks_total = len(tracks)
        experimental_genre = 38
        experimental_tracks = tracks.loc[tracks['track', 'genres_all'].apply(lambda x: experimental_genre in ast.literal_eval(x))]
        ignore_track_ids = list(experimental_tracks.loc[
            (experimental_tracks['track', 'listens'] <= 1000) |
            (experimental_tracks['track', 'favorites'] <= 5)
        ].index)
        # Use extension-agnostic identifiers so filtering works for both audio and npy inputs.
        ignore_files = [f'{int(i):06d}' for i in ignore_track_ids]
        print(
            f'filter_fma enabled: tracks.csv total={tracks_total}, '
            f'experimental_genre_tracks={len(experimental_tracks)}'
        )
        print(f'filter_fma enabled: filtered_out={len(ignore_files)} low-quality experimental tracks')
        print(f'filter_fma enabled: tracks_after_filter={tracks_total - len(ignore_files)}')
    else:
        ignore_files = None

    if args.result_folder is not None:
        training_config.data_preprocessor_cfg.results_folder = args.result_folder

    preprocessor_kwargs = {
        'input_audio_format': args.input_audio_format,
        'npy_sample_hz': args.npy_sample_hz,
        'tracks_csv_path': args.tracks_csv,
        'ignore_files': ignore_files,
        'resume_from_db': args.resume_from_db,
        'replace_existing': args.replace_existing,
        'gc_every_n_steps': args.gc_every_n_steps,
        'sqlite_cache_size_kib': args.sqlite_cache_size_kib,
        'output_format': args.output_format,
    }
    if args.target_lufs is not None:
        preprocessor_kwargs['target_lufs'] = args.target_lufs
    if args.num_workers is not None:
        preprocessor_kwargs['num_workers'] = args.num_workers

    processor = create_data_preprocessor_from_config(
        model_config,
        training_config,
        clap,
        wav2vec,
        encodec_wrapper,
        device,
        **preprocessor_kwargs,
    )

    processor.process()


if __name__ == '__main__':
    main()
