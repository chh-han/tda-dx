#!/usr/bin/env python3
"""Stage 6b helper: Extract CLAP/CLEWS/MERT embeddings from audio.

Pass --encoder {clap,clews,mert} to dispatch.
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=["clap", "clews", "mert"], required=True,
                        help="Which encoder to extract embeddings with.")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    if args.encoder == "clap":
        from tda_dx.attribution.retrieval.extract_clap import main as run
    elif args.encoder == "clews":
        from tda_dx.attribution.retrieval.extract_clews import main as run
    else:
        from tda_dx.attribution.retrieval.extract_mert import main as run
    run()


if __name__ == "__main__":
    main()
