"""Extension 3: Reconstruction-error detection (AeroBlade analog).

Train a lightweight decoder to reconstruct frozen-HuBERT features of BONA FIDE
speech ONLY. At test time, reconstruction error is the anomaly score: real
speech reconstructs well, synthetic speech does not. The decoder must never see
spoofed examples — that is the entire premise.

Headline experiment: zero-shot transfer to ASVspoof 5 — does the generative
detector generalize to unseen attacks better than the discriminative baseline?

Score convention note: reconstruction error is HIGHER for spoof, but the repo
convention is "higher == more bona fide". ``score_dataset`` therefore returns
the NEGATED error so it plugs directly into metrics.py / visualize.py.

torch / transformers are imported lazily so the module imports without them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ENCODER_ID = "facebook/hubert-base-ls960"
TARGET_SR = 16000
FEATURE_DIM = 768


def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_decoder(input_dim: int = FEATURE_DIM, hidden_dim: int = 512):
    """MLP decoder over frame-level HuBERT features. Returns an nn.Module."""
    torch, nn = _import_torch()

    class HuBERTDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            return self.decoder(x)

    return HuBERTDecoder()


def load_encoder(device: str = "cpu"):
    """Load a frozen HuBERT encoder (all params requires_grad=False)."""
    torch, _ = _import_torch()
    from transformers import HubertModel

    enc = HubertModel.from_pretrained(ENCODER_ID).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _features(encoder, audio: np.ndarray, sr: int, device: str):
    """Waveform -> frozen HuBERT frame features (T, 768) tensor on ``device``."""
    torch, _ = _import_torch()
    import torchaudio

    x = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
    if x.ndim > 1:
        x = x.mean(dim=0 if x.shape[0] < x.shape[-1] else -1)
    if sr != TARGET_SR:
        x = torchaudio.functional.resample(x, sr, TARGET_SR)
    with torch.inference_mode():
        out = encoder(x.unsqueeze(0).to(device)).last_hidden_state[0]
    return out  # (T, 768)


def train_decoder(bonafide_trials, *, epochs: int = 10, lr: float = 1e-3,
                  device: str | None = None,
                  ckpt: str | Path = "checkpoints/hubert_decoder.pt"):
    """Train the decoder on bona-fide TRAIN utterances ONLY; save checkpoint.

    ``bonafide_trials`` is a DataFrame filtered to label==1 with a ``path``
    column. Loss is per-frame MSE in HuBERT feature space.
    """
    torch, nn = _import_torch()
    import soundfile as sf

    assert (bonafide_trials["label"] == 1).all(), \
        "train_decoder must receive bona-fide-only trials (the whole premise)."

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(device)
    decoder = build_decoder().to(device).train()
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    paths = bonafide_trials["path"].tolist()
    for epoch in range(epochs):
        np.random.shuffle(paths)
        total = 0.0
        for p in paths:
            audio, sr = sf.read(p, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            feats = _features(encoder, audio, sr, device)  # (T,768), no grad
            opt.zero_grad()
            loss = loss_fn(decoder(feats), feats)
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{epochs}  mean MSE {total / len(paths):.5f}")

    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), ckpt)
    return decoder


def load_decoder(ckpt: str | Path = "checkpoints/hubert_decoder.pt", device: str = "cpu"):
    """Load a trained decoder checkpoint."""
    torch, _ = _import_torch()
    decoder = build_decoder().to(device)
    decoder.load_state_dict(torch.load(ckpt, map_location=device))
    return decoder.eval()


def reconstruction_error(encoder, decoder, audio: np.ndarray, sr: int,
                         device: str = "cpu", return_frames: bool = False):
    """Mean frame-level MSE in HuBERT feature space (higher == more synthetic).

    With ``return_frames``, also returns the per-frame error track (for the
    artifact-localization heatmap).
    """
    torch, _ = _import_torch()
    feats = _features(encoder, audio, sr, device)
    with torch.inference_mode():
        recon = decoder(feats)
        frame_err = torch.mean((feats - recon) ** 2, dim=-1)  # (T,)
    mean_err = float(frame_err.mean())
    return (mean_err, frame_err.cpu().numpy()) if return_frames else mean_err


def score_dataset(trials, *, ckpt: str | Path = "checkpoints/hubert_decoder.pt",
                  device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Score every trial; returns (labels, scores) with HIGHER == more bona fide.

    Scores are NEGATED reconstruction error so they feed metrics.py directly.
    """
    torch, _ = _import_torch()
    import soundfile as sf

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(device)
    decoder = load_decoder(ckpt, device)

    scores = np.empty(len(trials), dtype=np.float32)
    for i, row in enumerate(trials.itertuples(index=False)):
        audio, sr = sf.read(row.path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        scores[i] = -reconstruction_error(encoder, decoder, audio, sr, device)
    return trials["label"].to_numpy(), scores
