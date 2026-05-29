"""Baseline spoofing-detector loading and inference wrapper.

Default target: ``lab260/AASIST3`` (wav2vec2 SSL encoder + graph-attention head).
The proposal's original id ``ntt-hilab-gensp/ssl_spoof`` is gated/unavailable
(HTTP 401), so it is not used.

AASIST3 specifics (from its model card):
    - input : 16 kHz mono, fixed length 64600 samples (~4 s); pad/truncate
    - output: 2-way logits where index 0 == bonafide, 1 == spoof
    - loaded via a custom ``from_pretrained`` (PyTorchModelHubMixin), NOT
      ``transformers.AutoModel``; its model code must be importable.

SCORE CONVENTION (matches metrics.py): we return ``logit[bonafide] - logit[spoof]``
so that HIGHER == MORE BONA FIDE.

torch / torchaudio are imported lazily inside the methods that need them, so the
rest of the framework (dataset parsing, degradations, metrics, the evaluate
orchestration) can be imported and unit-tested without a torch install.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL_ID = "lab260/AASIST3"
TARGET_SR = 16000
MAX_SAMPLES = 64600          # ~4.04 s at 16 kHz; AASIST fixed input length
BONAFIDE_IDX, SPOOF_IDX = 0, 1


def _to_mono_16k(audio: np.ndarray, sr: int):
    """numpy waveform -> (samples,) float32 tensor at 16 kHz mono."""
    import torch
    import torchaudio

    x = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
    if x.ndim > 1:                       # (channels, samples) or (samples, channels)
        x = x.mean(dim=0 if x.shape[0] < x.shape[-1] else -1)
    if sr != TARGET_SR:
        x = torchaudio.functional.resample(x, sr, TARGET_SR)
    return x


def _fix_length(x, length: int = MAX_SAMPLES):
    """Pad with zeros or truncate to exactly ``length`` samples."""
    import torch

    if x.numel() < length:
        x = torch.nn.functional.pad(x, (0, length - x.numel()))
    return x[:length]


class SpoofDetector:
    """Thin wrapper around a pretrained countermeasure.

    All public methods return scores under the repo convention
    (higher == more bona fide).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None,
                 lazy: bool = False):
        import torch

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        if not lazy:
            self.load()

    def load(self) -> "SpoofDetector":
        """Load and freeze the model.

        AASIST3 ships custom architecture code with a ``from_pretrained``
        classmethod. We import it lazily so the rest of the framework does not
        hard-depend on the model package being installed.
        """
        if self.model is not None:
            return self
        try:
            from model import aasist3  # provided by the lab260/AASIST3 repo
            model = aasist3.from_pretrained(self.model_id)
        except Exception as e:  # noqa: BLE001 - surface a precise setup hint
            raise RuntimeError(
                f"Could not load '{self.model_id}'. AASIST3 needs its custom "
                "model code on the import path (clone the HF repo or pip-install "
                "it), e.g. `from model import aasist3`. If you swap in a "
                "transformers-native checkpoint, override SpoofDetector.load(). "
                f"Original error: {e}"
            ) from e
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    # -- core forward ------------------------------------------------------- #
    def _forward(self, batch):
        """(B, MAX_SAMPLES) waveform tensor -> (B,) bona-fide scores."""
        import torch

        if self.model is None:
            self.load()
        with torch.inference_mode():
            logits = self.model(batch.to(self.device))
            if isinstance(logits, (tuple, list)):    # some heads return (emb, logits)
                logits = logits[-1]
            logits = logits.float()
            score = logits[:, BONAFIDE_IDX] - logits[:, SPOOF_IDX]
        return score.cpu()

    # -- public API --------------------------------------------------------- #
    def predict(self, audio: np.ndarray, sr: int) -> float:
        """Bona-fide score for a single full utterance."""
        x = _fix_length(_to_mono_16k(audio, sr)).unsqueeze(0)
        return float(self._forward(x)[0])

    def predict_batch(self, audios: list[np.ndarray], sr: int,
                      batch_size: int = 32) -> np.ndarray:
        """Batched inference over utterances. Returns (N,) scores."""
        import torch

        out: list[float] = []
        for i in range(0, len(audios), batch_size):
            chunk = audios[i:i + batch_size]
            batch = torch.stack([_fix_length(_to_mono_16k(a, sr)) for a in chunk])
            out.extend(self._forward(batch).tolist())
        return np.asarray(out, dtype=np.float32)

    def predict_streaming(self, audio: np.ndarray, sr: int, *,
                          chunk_ms: int | None, overlap_ms: int = 0,
                          agg: str = "mean") -> tuple[list[float], float]:
        """Chunked inference + score aggregation.

        ``chunk_ms=None`` scores the full utterance. Aggregation over chunks:
        ``mean`` (robust default), ``min`` (most-spoof-leaning chunk dominates),
        or ``max``. Returns ``(per_chunk_scores, aggregated_score)``.
        """
        from .degradations import chunk_audio

        if chunk_ms is None:
            s = self.predict(audio, sr)
            return [s], s

        pieces = chunk_audio(audio, sr, chunk_ms=chunk_ms, overlap_ms=overlap_ms)
        scores = self.predict_batch(pieces, sr).tolist()
        agg_fn = {"mean": np.mean, "min": np.min, "max": np.max}[agg]
        return scores, float(agg_fn(scores))
