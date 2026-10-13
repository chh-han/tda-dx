"""TDA-Dx: diagnostics for training data attribution."""

import os as _os
import sys as _sys

# Make vendored third-party packages (e.g. dattri, laion_clap) importable
# under their original namespace, so attribution code can keep using
# `from dattri.algorithm.block_projected_if.core.hook import HookManager`
# without rewrites.
_THIRD_PARTY_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "third_party")
if _THIRD_PARTY_DIR not in _sys.path:
    _sys.path.insert(0, _THIRD_PARTY_DIR)

__version__ = "0.1.0"
