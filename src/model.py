"""Baseline spoofing-detector loading and inference wrapper.

Target: a pretrained SSL-based countermeasure (XLSR / wav2vec2 + AASIST-style
head). NOTE: verify the exact HuggingFace repo id before relying on it — the
`ntt-hilab-gensp/ssl_spoof` id in the proposal was not confirmed. The
wav2vec2-AASIST line from the ASVspoof 2021 baselines is a reliable fallback.

All scores follow the repo convention in metrics.py: higher == more bona fide.
"""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_MODEL_ID = "ntt-hilab-gensp/ssl_spoof"  # TODO: verify / replace
TARGET_SR = 16000


class SpoofDetector:
    """Thin wrapper around a pretrained countermeasure."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        # TODO: load model + feature extractor from HuggingFace.

    def predict(self, audio: np.ndarray, sr: int) -> float:
        """Return a single bona-fide score for a full utterance.

        TODO: resample to TARGET_SR, run the model, return the bona fide logit.
        """
        raise NotImplementedError

    def predict_batch(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        """Batched inference over utterances (pad/collate as needed)."""
        raise NotImplementedError

    def predict_streaming(self, audio: np.ndarray, sr: int, *,
                          chunk_ms: int | None, overlap_ms: int = 0,
                          agg: str = "mean") -> tuple[list[float], float]:
        """Chunked inference + score aggregation.

        Returns ``(per_chunk_scores, aggregated_score)``. ``chunk_ms=None``
        means score the full utterance. Aggregation: mean / min / max.
        Uses degradations.chunk_audio for windowing.
        """
        raise NotImplementedError
