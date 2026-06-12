"""Split-conformal calibration for spoofing countermeasure scores.

SCORE/LABEL CONVENTION (same as src/metrics.py, used everywhere in this repo):
    label == 1  ->  bona fide
    label == 0  ->  spoof
    score: higher == more bona-fide-like

THE GUARANTEE BEING SOLD. We calibrate a decision threshold tau on held-out
*spoof* scores such that, for a new spoof exchangeable with the calibration
set, P(score > tau) <= alpha — i.e. the spoof miss rate (a spoof accepted as
bona fide) is controlled at alpha. This is the deployment-relevant direction:
a missed spoof is the costly error in authentication.

THE QUESTION THE PAPER ASKS. Exchangeability is exactly what a novel attack
violates. Calibrate on seen families, evaluate empirical miss rate on a
held-out family: if it exceeds alpha, the guarantee silently failed under
attack shift. ``weighted_conformal_threshold`` is the repair (Tibshirani et
al. 2019, covariate-shift weighted conformal), with weights derived from the
retrieval novelty score (src/retrieval.py) — the same quantity that predicts
non-transfer in the LOAO study (H2, rho=-0.67), which is what makes it the
principled choice of shift covariate rather than an arbitrary one.

Ties: with continuous CM scores, ties have measure zero; we use strict ">"
for a miss and document rather than randomize.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "conformal_threshold",
    "weighted_conformal_threshold",
    "miss_rate",
    "novelty_to_weights",
    "family_coverage_table",
]


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal threshold from calibration *spoof* scores.

    Returns tau = the ceil((n+1)(1-alpha))-th smallest calibration score, so
    that under exchangeability P(test score > tau) <= alpha. If the required
    rank exceeds n (alpha too small for the calibration size), returns +inf —
    the vacuous threshold — rather than silently under-covering.
    """
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration set")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def weighted_conformal_threshold(cal_scores: np.ndarray,
                                 cal_weights: np.ndarray,
                                 test_weight: float,
                                 alpha: float) -> float:
    """Covariate-shift-weighted conformal threshold (Tibshirani et al. 2019).

    ``cal_weights[i]`` and ``test_weight`` are (unnormalized) likelihood
    ratios dP_test/dP_cal evaluated at each point's covariate. The test point
    contributes mass at +inf (its score is unknown), which is what restores
    validity under the shift. With all weights equal this reduces exactly to
    ``conformal_threshold`` (tested).
    """
    s = np.asarray(cal_scores, dtype=np.float64)
    w = np.asarray(cal_weights, dtype=np.float64)
    if s.shape != w.shape:
        raise ValueError("cal_scores and cal_weights must align")
    if np.any(w < 0) or test_weight < 0:
        raise ValueError("weights must be non-negative")
    order = np.argsort(s)
    s, w = s[order], w[order]
    total = w.sum() + test_weight
    if total <= 0:
        raise ValueError("all weights are zero")
    # cumulative normalized mass over sorted calibration scores; the test
    # point's mass sits at +inf and is never crossed by the cumsum.
    cum = np.cumsum(w) / total
    idx = np.searchsorted(cum, 1.0 - alpha, side="left")
    if idx >= len(s):
        return float("inf")
    return float(s[idx])


def miss_rate(spoof_scores: np.ndarray, tau: float) -> float:
    """Empirical spoof miss rate at threshold tau (score > tau == accepted)."""
    s = np.asarray(spoof_scores)
    if len(s) == 0:
        return float("nan")
    return float(np.mean(s > tau))


def novelty_to_weights(cal_novelty: np.ndarray,
                       test_novelty: np.ndarray,
                       *, C: float = 1.0, clip: float = 50.0
                       ) -> tuple[np.ndarray, float]:
    """Density-ratio weights from the retrieval novelty covariate.

    Probabilistic-classification estimator: fit logistic regression to
    distinguish calibration from test novelty values; the odds ratio
    p(test|z)/p(cal|z), rescaled by the class prior ratio, estimates
    dP_test/dP_cal at z. Returns (weights at each calibration point,
    weight at the median test novelty) ready for
    ``weighted_conformal_threshold``. Ratios are clipped at ``clip`` —
    unbounded weights make the threshold degenerate (a known failure mode of
    weighted conformal worth *reporting*, not hiding).
    """
    from sklearn.linear_model import LogisticRegression

    zc = np.asarray(cal_novelty, dtype=np.float64).reshape(-1, 1)
    zt = np.asarray(test_novelty, dtype=np.float64).reshape(-1, 1)
    X = np.vstack([zc, zt])
    y = np.concatenate([np.zeros(len(zc)), np.ones(len(zt))])
    clf = LogisticRegression(C=C).fit(X, y)

    def ratio(z: np.ndarray) -> np.ndarray:
        p = clf.predict_proba(z)[:, 1]
        prior = len(zt) / len(zc)
        r = (p / np.clip(1.0 - p, 1e-12, None)) / prior
        return np.clip(r, 0.0, clip)

    w_cal = ratio(zc)
    w_test = float(ratio(np.median(zt, keepdims=True).reshape(1, 1))[0])
    return w_cal, w_test


def family_coverage_table(scores: np.ndarray, labels: np.ndarray,
                          groups: np.ndarray, *, alpha: float = 0.05,
                          seed: int = 0) -> list[dict]:
    """Hold-one-group-out conformal coverage. The paper's central table.

    For each group g (attack system or family): calibrate tau on spoof scores
    from every *other* group, report the empirical miss rate on g. The
    ``within`` column is the sanity control — calibrate and test on a random
    split of g itself, where exchangeability holds by construction and the
    miss rate should sit at ~alpha. ``gap`` = held-out miss minus alpha:
    positive means the guarantee broke.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    spoof = labels == 0
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for g in sorted(set(groups[spoof])):
        held = spoof & (groups == g)
        rest = spoof & (groups != g)
        tau = conformal_threshold(scores[rest], alpha)
        held_miss = miss_rate(scores[held], tau)
        # within-group control split
        idx = np.flatnonzero(held)
        perm = rng.permutation(idx)
        half = len(perm) // 2
        tau_w = conformal_threshold(scores[perm[:half]], alpha)
        within = miss_rate(scores[perm[half:]], tau_w)
        rows.append({"group": g, "n": int(held.sum()), "tau": tau,
                     "held_out_miss": held_miss, "within_miss": within,
                     "gap": held_miss - alpha})
    return rows
