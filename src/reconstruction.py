"""Extension 3: Reconstruction-error detection (AeroBlade analog).

Train a lightweight decoder to reconstruct frozen-HuBERT features of BONA FIDE
speech ONLY. At test time, reconstruction error is the anomaly score: real
speech reconstructs well, synthetic speech does not. The decoder must never see
spoofed examples — that is the entire premise.

Headline experiment: zero-shot transfer to ASVspoof 5 — does the generative
detector generalize to unseen attacks better than the discriminative baseline?
"""
from __future__ import annotations

import torch
import torch.nn as nn

ENCODER_ID = "facebook/hubert-base-ls960"


class HuBERTDecoder(nn.Module):
    """MLP decoder over frame-level HuBERT features (768-d)."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


def load_encoder(device: str = "cpu"):
    """Load frozen HuBERT encoder (requires_grad=False on all params)."""
    raise NotImplementedError


def train_decoder(bonafide_trials, epochs: int = 10,
                  ckpt: str = "checkpoints/hubert_decoder.pt"):
    """Train decoder on bona fide TRAIN split only; save checkpoint. MSE loss."""
    raise NotImplementedError


@torch.no_grad()
def reconstruction_error(encoder, decoder, audio, sr) -> float:
    """Mean frame-level MSE in HuBERT feature space (higher == more synthetic)."""
    raise NotImplementedError
