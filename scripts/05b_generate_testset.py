#!/usr/bin/env python3
"""Stage 5b: Generate test-set audio + tokens by running base model on FMA test
conditioning. Produces the per-query token sequences and audio that stage 6
attribution and stage 7 analysis consume."""
from tda_dx.inference.generate_testset import main
if __name__ == "__main__":
    main()
