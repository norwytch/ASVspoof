import numpy as np

from src.metrics import (bootstrap_eer_ci, compute_eer, compute_min_dcf,
                         per_attack_eer, seed_variance, spearman_with_ci)


def _scores(seed=0, sep=2.0, n=400):
    rng = np.random.default_rng(seed)
    y = np.array([1] * n + [0] * n)
    s = np.concatenate([rng.normal(sep, 1, n), rng.normal(-sep, 1, n)])
    return y, s


def test_eer_perfect_separation():
    y = np.array([1] * 50 + [0] * 50)
    s = np.array([1.0] * 50 + [0.0] * 50)
    eer, _ = compute_eer(y, s)
    assert eer < 1e-6


def test_eer_separated_is_low():
    y, s = _scores(sep=2.0)
    eer, thr = compute_eer(y, s)
    assert 0.0 < eer < 0.15
    assert np.isfinite(thr)


def test_min_dcf_normalized_range():
    y, s = _scores(sep=1.0)
    d = compute_min_dcf(y, s)
    assert 0.0 <= d <= 1.0


def test_per_attack_keys():
    y, s = _scores(sep=1.0, n=100)
    aid = np.array(["-"] * 100 + ["A07"] * 50 + ["A10"] * 50)
    r = per_attack_eer(y, s, aid)
    assert set(r) == {"A07", "A10"}


def test_bootstrap_ci_brackets_point():
    y, s = _scores(sep=1.0, n=300)
    r = bootstrap_eer_ci(y, s, n_boot=200, seed=0)
    assert r["ci_low"] <= r["eer"] <= r["ci_high"]


def test_spearman_sign_and_ci():
    x = np.arange(10.0)
    r = spearman_with_ci(x, x[::-1], n_boot=300, seed=0)
    assert r["rho"] < -0.9 and r["n"] == 10


def test_seed_variance_summary():
    r = seed_variance([0.05, 0.06, 0.055, 0.07, 0.05])
    assert r["n_seeds"] == 5 and r["std_eer_pct"] >= 0 and r["max_eer_pct"] >= r["min_eer_pct"]
