"""Plot generation. All figures saved as PNG to results/figures/.

Plots:
    1. ROC curves   — clean vs each degradation family
    2. DET curves   — log-scale miss vs false-alarm
    3. EER vs bitrate (MP3), EER vs SNR (noise), EER vs chunk size (streaming)
    4. Extension plots: t-SNE/UMAP of transcript embeddings, EER heatmap by
       attack category x condition, reconstruction score histograms, prosody
       box plots, F0 trajectory overlays.
"""
from __future__ import annotations

from pathlib import Path

FIGDIR = Path("results/figures")


def plot_roc(curves: dict, title: str, out: str | Path) -> None:
    """Overlay ROC curves. ``curves`` maps label -> (labels, scores)."""
    raise NotImplementedError


def plot_det(curves: dict, title: str, out: str | Path) -> None:
    """DET curves (norm-deviate axes, log-style)."""
    raise NotImplementedError


def plot_eer_sweep(x, eers, xlabel: str, out: str | Path) -> None:
    """Line plot of EER vs a swept parameter (bitrate / SNR / chunk size)."""
    raise NotImplementedError
