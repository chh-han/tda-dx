"""Smoke tests: every public top-level subpackage imports without error."""
import importlib

import pytest


SUBMODULES = [
    "tda_dx.models",
    "tda_dx.data",
    "tda_dx.preprocessing",
    "tda_dx.tokenizers",
    "tda_dx.training",
    "tda_dx.inference",
    "tda_dx.features",
    "tda_dx.attribution.methods",
    "tda_dx.attribution.pipeline",
    "tda_dx.attribution.retrieval",
    "tda_dx.analysis",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports(name):
    mod = importlib.import_module(name)
    assert mod is not None
