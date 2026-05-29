"""Plot generation. All figures saved as PNG to results/figures/.

Most functions take cached score arrays (from results/scores/*.npz) or the
results.csv table. Use ``load_scores`` to pull a condition's labels+scores.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve

from .metrics import compute_eer

FIGDIR = Path("results/figures")
SCORES_DIR = Path("results/scores")


def load_scores(slug: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (labels, scores) for a cached condition by its slug (e.g. 'clean')."""
    z = np.load(SCORES_DIR / f"{slug}.npz", allow_pickle=True)
    return z["label"], z["score"]


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str,
             out: str | Path) -> Path:
    """Overlay ROC curves. ``curves`` maps label -> (labels, scores)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (labels, scores) in curves.items():
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer, _ = compute_eer(labels, scores)
        ax.plot(fpr, tpr, label=f"{name} (EER {eer * 100:.1f}%)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(fontsize=8)
    return _save(fig, out)


def plot_det(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str,
             out: str | Path) -> Path:
    """DET curves on normal-deviate axes (standard ASVspoof presentation)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ticks = [0.01, 0.05, 0.1, 0.2, 0.4]
    for name, (labels, scores) in curves.items():
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        m = (fpr > 0) & (fnr > 0)
        ax.plot(norm.ppf(fpr[m]), norm.ppf(fnr[m]), label=name)
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t*100:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks)); ax.set_yticklabels([f"{t*100:g}" for t in ticks])
    ax.set(xlabel="False Alarm (%)", ylabel="Miss (%)", title=title)
    ax.legend(fontsize=8)
    return _save(fig, out)


def plot_eer_sweep(x, eers, xlabel: str, out: str | Path,
                   baseline_eer: float | None = None) -> Path:
    """Line plot of EER (%) vs a swept parameter (bitrate / SNR / chunk size)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [e * 100 for e in eers], "o-")
    if baseline_eer is not None:
        ax.axhline(baseline_eer * 100, ls="--", c="gray", label="clean baseline")
        ax.legend(fontsize=8)
    ax.set(xlabel=xlabel, ylabel="EER (%)", title=f"EER vs {xlabel}")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def plot_attack_heatmap(per_attack_df, out: str | Path) -> Path:
    """Heatmap of EER (%) over attack_id (rows) x condition/param (cols).

    ``per_attack_df`` is the results/per_attack_eer.csv table.
    """
    pivot = per_attack_df.pivot_table(
        index="attack_id", columns=["condition", "param"], values="eer_pct")
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.6), 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{c}/{p}" for c, p in pivot.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="EER (%)")
    ax.set_title("Per-attack EER by condition")
    return _save(fig, out)


def plot_score_hist(labels: np.ndarray, scores: np.ndarray, title: str,
                    out: str | Path) -> Path:
    """Bona fide vs spoof score distributions (used by reconstruction/prosody too)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores[labels == 1], bins=50, alpha=0.6, label="bona fide", density=True)
    ax.hist(scores[labels == 0], bins=50, alpha=0.6, label="spoof", density=True)
    ax.set(xlabel="score", ylabel="density", title=title)
    ax.legend(fontsize=8)
    return _save(fig, out)
