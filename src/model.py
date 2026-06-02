"""Baseline spoofing-detector loading and inference wrapper.

Baseline = **SSL_Anti-spoofing** (wav2vec2 XLS-R 300M front-end + AASIST graph
back-end; TakHemlata et al., Interspeech 2022). It is loaded fairseq-free via
``src.ssl_aasist`` (the XLS-R encoder is rebuilt with a HuggingFace
``Wav2Vec2Model`` and the pretrained checkpoint's wav2vec keys are remapped onto
HF naming -- see that module). The original proposal id
``ntt-hilab-gensp/ssl_spoof`` is gated (HTTP 401), and ``lab260/AASIST3`` was
dropped because every public AASIST3 checkpoint is degenerate (scores all inputs
bona fide, ~63% EER) even in its own pinned environment.

Model specifics:
    - input : 16 kHz mono, fixed length 64600 samples (~4 s); pad/truncate, raw
      waveform (NO per-utterance normalisation)
    - output: 2-way logits where **index 1 == bona fide, index 0 == spoof**

SCORE CONVENTION (matches metrics.py): we return ``logit[bonafide] - logit[spoof]``
so that HIGHER == MORE BONA FIDE.

torch / torchaudio are imported lazily inside the methods that need them, so the
rest of the framework (dataset parsing, degradations, metrics, the evaluate
orchestration) can be imported and unit-tested without a torch install.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL_ID = "SSL_Anti-spoofing/LA_model"   # see src.ssl_aasist for paths
TARGET_SR = 16000
MAX_SAMPLES = 64600          # ~4.04 s at 16 kHz; AASIST fixed input length
BONAFIDE_IDX, SPOOF_IDX = 1, 0


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


def _fix_length(x, length: int = MAX_SAMPLES, mode: str = "repeat"):
    """Truncate, or pad short waveforms to exactly ``length`` samples.

    ``mode="repeat"`` (default) **tiles** the waveform, matching the original
    SSL_Anti-spoofing / AASIST ``data_utils.pad`` recipe the checkpoint was trained
    on. ``mode="zero"`` appends silence — convenient but a TRAIN/TEST MISMATCH for
    this model (a long silence tail it never saw), which inflates EER; kept only
    for ablation. Use "repeat" for any real evaluation.
    """
    import torch

    n = x.numel()
    if n >= length:
        return x[:length]
    if mode == "repeat":
        reps = length // n + 1
        return x.repeat(reps)[:length]
    if mode == "zero":
        return torch.nn.functional.pad(x, (0, length - n))
    raise ValueError(f"unknown pad mode {mode!r} (use 'repeat' or 'zero')")


class SpoofDetector:
    """Thin wrapper around a pretrained countermeasure.

    All public methods return scores under the repo convention
    (higher == more bona fide).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None,
                 lazy: bool = False, pad_mode: str = "repeat"):
        import torch

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_mode = pad_mode      # "repeat" matches the checkpoint's training recipe
        self.model = None
        if not lazy:
            self.load()

    def load(self) -> "SpoofDetector":
        """Load and freeze the SSL_Anti-spoofing (XLS-R + AASIST) baseline.

        Assembled fairseq-free by :func:`src.ssl_aasist.build_model`, which
        rebuilds the XLS-R encoder with a HuggingFace ``Wav2Vec2Model`` and loads
        the pretrained checkpoint's remapped weights. ``model_id`` may be a path
        to a ``.pth`` checkpoint; otherwise the packaged default is used.
        """
        if self.model is not None:
            return self
        from .ssl_aasist import DEFAULT_CKPT, build_model

        ckpt = self.model_id if (self.model_id and str(self.model_id).endswith(".pth")) else DEFAULT_CKPT
        try:
            model = build_model(ckpt_path=ckpt, device=self.device)
        except Exception as e:  # noqa: BLE001 - surface a precise setup hint
            raise RuntimeError(
                "Could not load the SSL_Anti-spoofing baseline. Ensure the repo is "
                "cloned to third_party/SSL_Anti-spoofing and the pretrained "
                "checkpoint is present (see src/ssl_aasist.py). "
                f"Original error: {e}"
            ) from e
        self.model = model
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
        x = _fix_length(_to_mono_16k(audio, sr), mode=self.pad_mode).unsqueeze(0)
        return float(self._forward(x)[0])

    def predict_batch(self, audios: list[np.ndarray], sr: int,
                      batch_size: int = 32) -> np.ndarray:
        """Batched inference over utterances. Returns (N,) scores."""
        import torch

        out: list[float] = []
        for i in range(0, len(audios), batch_size):
            chunk = audios[i:i + batch_size]
            batch = torch.stack([_fix_length(_to_mono_16k(a, sr), mode=self.pad_mode) for a in chunk])
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
