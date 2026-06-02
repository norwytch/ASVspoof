"""Part 2 — frozen XLS-R per-layer embedding cache (research-design.md §3.1).

The one expensive, one-time step. Extract utterance-level (mean-pooled) wav2vec2
-XLS-R hidden states for every trial, per layer, and cache to disk. Everything
downstream (LOAO heads, probes, the correlation) runs on these cached vectors —
cheap, CPU-friendly, cache-once-then-iterate.

Two frozen regimes (§3.1):
  (A) off-the-shelf pretrained XLS-R  — DEFAULT_ENCODER below.
  (B) XLS-R fine-tuned on the binary spoof task, then frozen — pass that
      checkpoint's model id / path to load_xlsr_encoder.

torch / transformers are imported lazily so dataset/metrics/probes import without them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_ENCODER = "facebook/wav2vec2-xls-r-300m"  # 24 transformer layers, 1024-d
TARGET_SR = 16000


def load_xlsr_encoder(model_id: str = DEFAULT_ENCODER, device: str | None = None):
    """Load a frozen wav2vec2/XLS-R encoder (all params requires_grad=False)."""
    import torch
    from transformers import Wav2Vec2Model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = Wav2Vec2Model.from_pretrained(model_id).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    enc._wfd_device = device  # stash for helpers
    return enc


def extract_layer_embeddings(encoder, audio: np.ndarray, sr: int,
                             layers: list[int] | None = None) -> dict[int, np.ndarray]:
    """Mean-pooled hidden state per requested layer. Returns {layer: (D,) float32}.

    layer 0 = CNN feature projection output; 1..N = transformer layers. ``layers
    =None`` returns all hidden states.
    """
    import torch
    import torchaudio

    device = getattr(encoder, "_wfd_device", "cpu")
    x = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
    if x.ndim > 1:
        x = x.mean(dim=0 if x.shape[0] < x.shape[-1] else -1)
    if sr != TARGET_SR:
        x = torchaudio.functional.resample(x, sr, TARGET_SR)
    with torch.inference_mode():
        hs = encoder(x.unsqueeze(0).to(device), output_hidden_states=True).hidden_states
    sel = range(len(hs)) if layers is None else layers
    return {L: hs[L][0].mean(dim=0).cpu().numpy().astype(np.float32) for L in sel}


def cache_embeddings(trials, *, model_id: str = DEFAULT_ENCODER,
                     layers: list[int] | None = None,
                     out_dir: str | Path = "results/embeddings",
                     device: str | None = None, encoder=None) -> Path:
    """Extract + cache per-layer embeddings for every trial.

    Writes ``<out_dir>/layer_<L>.npy`` (an (N, D) matrix aligned to ``utt_ids.npy``)
    plus ``meta.csv`` (utt_id, label, attack_id). Resumable-friendly: skip if the
    matrices already exist and match the trial count.

    ``encoder`` may be a pre-built frozen Wav2Vec2-style module (e.g. the
    fine-tuned XLS-R from ``ssl_aasist.load_finetuned_encoder`` for Regime B); if
    given, ``model_id`` is ignored.
    """
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if encoder is None:
        encoder = load_xlsr_encoder(model_id, device)

    import time

    n_total = len(trials)
    acc: dict[int, list[np.ndarray]] = {}
    t0 = time.time()
    for i, row in enumerate(trials.itertuples(index=False), 1):
        audio, sr = sf.read(row.path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        feats = extract_layer_embeddings(encoder, audio, sr, layers)
        for L, v in feats.items():
            acc.setdefault(L, []).append(v)
        if i % 500 == 0 or i == n_total:
            rate = i / (time.time() - t0)
            print(f"  [{i}/{n_total}] {rate:.0f} utt/s, "
                  f"eta {(n_total - i) / rate / 60:.1f} min", flush=True)

    for L, vecs in acc.items():
        np.save(out_dir / f"layer_{L}.npy", np.stack(vecs))
    np.save(out_dir / "utt_ids.npy", trials.utt_id.to_numpy())
    trials[["utt_id", "label", "attack_id"]].to_csv(out_dir / "meta.csv", index=False)
    return out_dir


def load_layer_matrix(out_dir: str | Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached ``(utt_ids, X)`` for one layer."""
    out_dir = Path(out_dir)
    return np.load(out_dir / "utt_ids.npy", allow_pickle=True), np.load(out_dir / f"layer_{layer}.npy")
