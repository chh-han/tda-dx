#!/usr/bin/env python3
"""Stage 1: Convert raw audio files to normalized .npy format for fast loading."""
from tda_dx.preprocessing.audio_npy import main, parse_args
if __name__ == "__main__":
    main(parse_args())
