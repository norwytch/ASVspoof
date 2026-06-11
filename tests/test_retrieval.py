import numpy as np
import pytest

from src.retrieval import (BruteForceIndex, LSHIndex, attribute, knn_detect,
                           novelty_scores, recall_at_k)


def _clusters(rng, n_clusters=5, per=100, D=32, scale=3.0):
    cents = rng.normal(0, 1, (n_clusters, D)) * scale
    X = np.vstack([cents[c] + rng.normal(0, 1, (per, D)) for c in range(n_clusters)])
    return cents, X.astype("float32")


def _queries(rng, cents, which, per=10, D=32):
    return np.vstack([cents[c] + rng.normal(0, 1, (per, D)) for c in which]).astype("float32")


def test_recall_at_k_identity():
    a = np.array([[0, 1, 2], [3, 4, 5]])
    assert recall_at_k(a, a) == 1.0


def test_lsh_recall_is_high():
    rng = np.random.default_rng(0)
    cents, X = _clusters(rng)
    Q = _queries(rng, cents, range(5))
    exact, _ = BruteForceIndex().build(X).query(Q, k=10)
    approx, _ = LSHIndex(n_bits=14, n_tables=8, seed=0).build(X).query(Q, k=10, probe_radius=2)
    assert recall_at_k(approx, exact) > 0.85


def test_lsh_recall_monotone_in_probe():
    rng = np.random.default_rng(1)
    cents, X = _clusters(rng)
    Q = _queries(rng, cents, range(5))
    exact, _ = BruteForceIndex().build(X).query(Q, k=10)
    lsh = LSHIndex(n_bits=16, n_tables=6, seed=0).build(X)
    r0 = recall_at_k(lsh.query(Q, k=10, probe_radius=0)[0], exact)
    r2 = recall_at_k(lsh.query(Q, k=10, probe_radius=2)[0], exact)
    assert r2 >= r0


def test_lsh_rerank_matches_brute_top1_when_recalled():
    # among the candidates it gathers, LSH ranks by exact cosine -> top-1 agrees with brute force
    rng = np.random.default_rng(5)
    cents, X = _clusters(rng)
    Q = _queries(rng, cents, range(5))
    e_idx, _ = BruteForceIndex().build(X).query(Q, k=1)
    a_idx, _ = LSHIndex(n_bits=12, n_tables=10, seed=0).build(X).query(Q, k=1, probe_radius=2)
    agree = np.mean([a[0] == e[0] for a, e in zip(a_idx, e_idx) if a[0] >= 0])
    assert agree > 0.9


def test_knn_detect_convention():
    rng = np.random.default_rng(2)
    cents, X = _clusters(rng, n_clusters=4)
    labels = np.array([1] * 200 + [0] * 200)              # clusters 0,1 bona; 2,3 spoof
    idx = BruteForceIndex().build(X)
    qb = _queries(rng, cents, (0, 1), per=20)
    qs = _queries(rng, cents, (2, 3), per=20)
    assert knn_detect(idx, labels, qb, k=15).mean() > knn_detect(idx, labels, qs, k=15).mean()


def test_attribute_to_nearest_generator():
    rng = np.random.default_rng(3)
    cents, X = _clusters(rng, n_clusters=4)
    attack = np.array(["-"] * 100 + ["-"] * 100 + ["A07"] * 100 + ["A19"] * 100)  # cl3 -> A19
    idx = BruteForceIndex().build(X)
    q = _queries(rng, cents, (3,), per=20)
    assert (attribute(idx, attack, q, k=15) == "A19").mean() > 0.8


def test_novelty_separates_seen_from_novel():
    rng = np.random.default_rng(4)
    cents, X = _clusters(rng, n_clusters=5)
    idx = BruteForceIndex().build(X)
    q_seen = _queries(rng, cents, (0,), per=20)
    q_novel = (cents.mean(0) + rng.normal(0, 1, (20, 32)) * 8).astype("float32")
    assert novelty_scores(idx, q_novel).mean() > novelty_scores(idx, q_seen).mean() + 0.2


def test_faiss_backend_matches_brute_force_and_drives_heads():
    pytest.importorskip("faiss")          # skips in CI (faiss not installed)
    from src.retrieval import FaissIndex
    rng = np.random.default_rng(7)
    cents, X = _clusters(rng)             # 5 clusters x 100
    Q = _queries(rng, cents, range(5))
    exact, _ = BruteForceIndex().build(X).query(Q, k=10)
    faiss_idx, _ = FaissIndex().build(X).query(Q, k=10)      # exact IndexFlatIP
    assert recall_at_k(faiss_idx, exact) > 0.99             # exact == brute force
    labels = np.array([1] * 300 + [0] * 200)
    assert knn_detect(FaissIndex().build(X), labels, Q, k=10).shape == (len(Q),)
