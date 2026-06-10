"""Every module imports without the heavy stack (the lazy-import contract).

torch / transformers / whisper / parselmouth / sentence-transformers are imported
lazily inside functions, so the whole framework must import on a CPU box with only
numpy/pandas/scipy/sklearn/librosa/soundfile/nltk present.
"""
import importlib

import pytest

CORE = ["dataset", "degradations", "metrics", "model", "evaluate", "visualize",
        "transcribe", "nlp_features", "attack_profiling", "reconstruction",
        "prosody", "probes", "embeddings"]


@pytest.mark.parametrize("mod", CORE)
def test_src_module_imports(mod):
    importlib.import_module(f"src.{mod}")


def test_experiments_loao_imports():
    importlib.import_module("experiments.loao")


def test_score_convention_constant():
    # The bug-prone invariant: bona fide is index 1 for this baseline.
    from src import model
    assert (model.BONAFIDE_IDX, model.SPOOF_IDX) == (1, 0)
