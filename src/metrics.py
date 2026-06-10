"""Detection metrics for spoofing countermeasures.

SCORE/LABEL CONVENTION (used everywhere in this repo):
    label == 1  ->  bona fide (genuine / target)
    label == 0  ->  spoof
    score: higher == more bona-fide-like (a bona fide log-likelihood)

All metrics below assume this convention. If your model emits a "spoof
probability", negate it before passing it in.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate. Returns ``(eer, threshold)``.

    Uses the standard linear-interpolation crossing of FPR and FNR rather than
    the nearest-point approximation, which can be biased on coarse score grids.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # index just before the FPR/FNR crossing
    diff = fpr - fnr
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        i = int(np.nanargmin(np.abs(diff)))
        return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])
    i = idx[0]
    # linear interpolation between i and i+1
    x0, x1 = diff[i], diff[i + 1]
    alpha = x0 / (x0 - x1) if (x0 - x1) != 0 else 0.0
    eer = float(fpr[i] + alpha * (fpr[i + 1] - fpr[i]))
    # sklearn prepends a non-finite "inf" threshold; interpolating against it -> nan.
    t0, t1 = thr[i], thr[i + 1]
    if not np.isfinite(t0):
        threshold = float(t1)
    elif not np.isfinite(t1):
        threshold = float(t0)
    else:
        threshold = float(t0 + alpha * (t1 - t0))
    return eer, threshold


def compute_min_dcf(labels: np.ndarray, scores: np.ndarray,
                    p_target: float = 0.05, c_miss: float = 1.0,
                    c_fa: float = 1.0) -> float:
    """Normalized minimum Detection Cost Function (ASVspoof-style).

    Normalizes by the default-cost so the value sits in a comparable range,
    matching the ASVspoof convention (the un-normalized version in the original
    proposal is not directly comparable across operating points).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr  # P(miss) = P(reject bona fide)
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(dcf) / default_cost)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(labels, scores))


def per_attack_eer(labels: np.ndarray, scores: np.ndarray,
                   attack_ids: np.ndarray) -> dict[str, float]:
    """EER broken down by attack type.

    Each attack subset is scored against the SAME bona fide pool (bona fide
    utterances carry no attack id), which is the standard ASVspoof per-attack
    evaluation. ``attack_ids`` should mark bona fide rows with a sentinel
    (e.g. '-' or 'bonafide').
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    attack_ids = np.asarray(attack_ids)
    bona = labels == 1
    results: dict[str, float] = {}
    for attack in sorted(set(attack_ids[labels == 0])):
        mask = bona | (attack_ids == attack)
        eer, _ = compute_eer(labels[mask], scores[mask])
        results[str(attack)] = eer
    return results


def summarize(labels: np.ndarray, scores: np.ndarray,
              baseline_eer: float | None = None) -> dict[str, float]:
    """Compute the full metric row for one condition."""
    eer, thr = compute_eer(labels, scores)
    row = {
        "eer": eer,
        "eer_pct": eer * 100.0,
        "min_dcf": compute_min_dcf(labels, scores),
        "auc": compute_auc(labels, scores),
        "threshold": thr,
    }
    if baseline_eer is not None:
        row["delta_eer_pct"] = (eer - baseline_eer) * 100.0
    return row


# --------------------------------------------------------------------------- #
# Part 2 — statistical rigor (research-design.md §5)
# --------------------------------------------------------------------------- #
def bootstrap_eer_ci(labels: np.ndarray, scores: np.ndarray, *,
                     n_boot: int = 1000, alpha: float = 0.05,
                     seed: int = 0) -> dict[str, float]:
    """Utterance-level bootstrap CI for EER (Bisani & Ney 2004).

    Resamples trials with replacement; returns point EER plus the (alpha/2,
    1-alpha/2) percentile interval. This is the eval-set *sampling* CI — distinct
    from the seed-variance reported by retraining the head.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(labels)
    point, _ = compute_eer(labels, scores)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # guard against a resample with only one class
        if labels[idx].min() == labels[idx].max():
            boots[i] = np.nan
            continue
        boots[i], _ = compute_eer(labels[idx], scores[idx])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"eer": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "eer_pct": float(point * 100), "ci_low_pct": float(lo * 100),
            "ci_high_pct": float(hi * 100)}


def seed_variance(eers: list[float] | np.ndarray) -> dict[str, float]:
    """Mean ± std (and range) of EER across retraining seeds (§5)."""
    a = np.asarray(eers, dtype=float)
    return {"mean_eer_pct": float(a.mean() * 100), "std_eer_pct": float(a.std(ddof=1) * 100),
            "min_eer_pct": float(a.min() * 100), "max_eer_pct": float(a.max() * 100),
            "n_seeds": int(a.size)}


def spearman_with_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Spearman rank correlation with a bootstrap CI — the core H1 test (§3.5).

    n is small (≈4 families / ≈13 systems), so the rank statistic and a
    bootstrap CI are reported instead of a single point estimate; the directional
    prediction should be pre-registered.
    """
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho, pval = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            boots[i] = np.nan
            continue
        boots[i] = spearmanr(x[idx], y[idx]).correlation
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"rho": float(rho), "pval": float(pval),
            "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}
