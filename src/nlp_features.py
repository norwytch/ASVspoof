"""Extension 1: Transcript-conditioned anomaly signals.

Hypothesis: synthetic speech yields atypical ASR behavior — higher LM
perplexity on the recovered transcript, more degenerate repetition, and lower
Whisper confidence — and these correlate with the audio model's errors.

Features per utterance:
    - gpt2_perplexity   : GPT-2 perplexity of the Whisper transcript (batch it!)
    - repetition_rate   : fraction of repeated n-grams in the transcript
    - asr_confidence    : Whisper avg_logprob (and no_speech_prob)

Output: results/nlp_scores.csv. Then correlate each signal with the audio
model's per-utterance score / error.
"""
from __future__ import annotations

import pandas as pd


def gpt2_perplexity(texts: list[str], batch_size: int = 16) -> list[float]:
    """Batched GPT-2 perplexity. TODO: load gpt2 once, batch, return per-text PPL."""
    raise NotImplementedError


def repetition_rate(text: str, n: int = 2) -> float:
    """Fraction of repeated n-grams (degenerate-decoding proxy)."""
    raise NotImplementedError


def build_nlp_table(transcripts: dict, audio_scores: dict) -> pd.DataFrame:
    """Assemble per-utterance NLP features + audio score for correlation."""
    raise NotImplementedError
