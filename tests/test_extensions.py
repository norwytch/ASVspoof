import numpy as np
import pandas as pd
import pytest

from src.attack_profiling import (category_of, cluster_attack_agreement,
                                  eer_by_category, load_taxonomy)
from src.nlp_features import correlate_with_audio, repetition_rate

TAX_PATH = "data/attack_taxonomy.json"


# --- Extension 1: NLP signals ------------------------------------------------
def test_repetition_rate():
    assert repetition_rate("a b c d e") == 0.0
    assert repetition_rate("go go go go go") > 0.5
    assert repetition_rate("hi") == 0.0


def test_correlate_with_audio_sign():
    n = 30
    df = pd.DataFrame({
        "gpt2_perplexity": np.linspace(10, 100, n),
        "repetition_rate": np.zeros(n),
        "asr_avg_logprob": np.linspace(-0.1, -1.0, n),
        "asr_no_speech_prob": np.random.default_rng(0).random(n),
        "asr_compression_ratio": np.random.default_rng(1).random(n),
        "audio_score": np.linspace(2, -2, n),            # falls as perplexity rises
    })
    c = correlate_with_audio(df)
    assert c["gpt2_perplexity"][0] < -0.9


# --- Extension 2: attack taxonomy / profiling --------------------------------
def test_taxonomy_eval_attacks_are_A07_A19():
    tax = load_taxonomy(TAX_PATH)
    ev = {a for a in tax if tax[a].get("partition") == "eval"}
    assert ev == {f"A{n:02d}" for n in range(7, 20)}


def test_category_of():
    tax = load_taxonomy(TAX_PATH)
    assert category_of("-", tax) == "bonafide"
    assert category_of("A07", tax) in {"neural_tts", "concatenative_tts",
                                       "voice_conversion", "hybrid_tts_vc"}


def test_eer_by_category():
    tax = {"A07": {"category": "neural_tts"}, "A17": {"category": "voice_conversion"}}
    rng = np.random.default_rng(0)
    y = np.array([1] * 200 + [0] * 200)
    s = np.concatenate([rng.normal(1, 1, 200), rng.normal(-1, 1, 200)])
    aid = np.array(["-"] * 200 + ["A07"] * 100 + ["A17"] * 100)
    r = eer_by_category(y, s, aid, tax)
    assert set(r) == {"neural_tts", "voice_conversion"}


def test_cluster_attack_agreement_perfect():
    cl = np.array([0, 0, 1, 1, 2, 2])
    at = np.array(["A", "A", "B", "B", "C", "C"])
    assert cluster_attack_agreement(cl, at)["adjusted_rand"] > 0.99


# --- Extension 4: prosody ----------------------------------------------------
def test_timing_features_keys():
    from src.prosody import extract_timing_features
    sr = 16000
    a = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 2, 2 * sr))).astype("float32")
    f = extract_timing_features(a, sr)
    assert {"ioi_cv", "speaking_rate"} <= set(f)


def _cmudict_ready():
    try:
        from nltk.corpus import cmudict
        cmudict.dict()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cmudict_ready(), reason="nltk cmudict not downloaded")
def test_stress_pattern_coverage():
    from src.prosody import get_stress_pattern
    pattern, coverage = get_stress_pattern("the quick brown fox")
    assert coverage > 0 and len(pattern) > 0
