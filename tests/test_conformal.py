"""Tests for src/conformal.py — analytic cases, no model or data needed."""
from __future__ import annotations

import numpy as np
import pytest

from src.conformal import (conformal_threshold, family_coverage_table,
                           miss_rate, weighted_conformal_threshold)


def test_threshold_is_correct_order_statistic():
    # n=9, alpha=0.1 -> k = ceil(10*0.9) = 9 -> the 9th smallest = max
    cal = np.arange(1.0, 10.0)
    assert conformal_threshold(cal, 0.1) == 9.0
    # n=9, alpha=0.5 -> k = ceil(10*0.5) = 5 -> 5th smallest
    assert conformal_threshold(cal, 0.5) == 5.0


def test_vacuous_when_alpha_too_small_for_n():
    # n=5, alpha=0.05 -> k = ceil(6*0.95) = 6 > 5 -> +inf
    assert conformal_threshold(np.arange(5.0), 0.05) == float("inf")
    assert miss_rate(np.array([1e9]), float("inf")) == 0.0


def test_marginal_coverage_under_exchangeability():
    rng = np.random.default_rng(0)
    alpha, misses = 0.1, []
    for _ in range(500):
        pool = rng.standard_normal(201)
        tau = conformal_threshold(pool[:200], alpha)
        misses.append(pool[200] > tau)
    # P(miss) <= alpha; with 500 reps allow 3-sigma slack above alpha
    assert np.mean(misses) <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / 500)


def test_weighted_reduces_to_unweighted_with_equal_weights():
    rng = np.random.default_rng(1)
    cal = rng.standard_normal(100)
    for alpha in (0.05, 0.1, 0.3):
        tau_u = conformal_threshold(cal, alpha)
        tau_w = weighted_conformal_threshold(cal, np.ones(100), 1.0, alpha)
        assert tau_w == pytest.approx(tau_u)


def test_weighted_threshold_rises_when_shift_upweights_high_scores():
    cal = np.linspace(0.0, 1.0, 100)
    w_flat = np.ones(100)
    w_high = np.linspace(0.1, 5.0, 100)  # test dist favors high-score region
    tau_flat = weighted_conformal_threshold(cal, w_flat, 1.0, 0.1)
    tau_high = weighted_conformal_threshold(cal, w_high, 1.0, 0.1)
    assert tau_high >= tau_flat


def test_family_coverage_table_within_control_near_alpha():
    rng = np.random.default_rng(2)
    n = 4000
    scores = rng.standard_normal(n)
    labels = np.zeros(n, dtype=int)  # all spoof
    groups = np.repeat(["g1", "g2"], n // 2)
    rows = family_coverage_table(scores, labels, groups, alpha=0.1)
    for r in rows:
        # exchangeable within a group: control should sit near alpha
        assert r["within_miss"] <= 0.1 + 0.03
        # both groups drawn from the same dist: held-out gap should be small
        assert abs(r["gap"]) < 0.05
