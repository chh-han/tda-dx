#!/usr/bin/env python3
"""Stage 5: Train hierarchical transformer (semantic / coarse / fine).

Single CLI for all three stages — pass `--stage {semantic,coarse,fine}`.
"""
from tda_dx.training.train import main
if __name__ == "__main__":
    main()
