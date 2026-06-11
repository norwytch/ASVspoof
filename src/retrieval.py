"""Vector search + retrieval-based detection over the cached SSL embeddings.

Two layers, both pure numpy (no torch), operating on the cached XLS-R embeddings
(`src/embeddings.py`):

1. **A from-scratch ANN index** — random-hyperplane (cosine) LSH with multi-probe —
   plus a brute-force exact reference, an optional `FaissIndex` (the production
   library, exact `IndexFlatIP` or approximate `IndexIVFFlat`), and a recall@k /
   speedup benchmark that pits the hand-rolled LSH against FAISS IVF.
2. **Retrieval heads** on top:
   - `knn_detect`  — a non-parametric k-NN spoof detector (higher score = more bona fide);
   - `attribute`   — generator (attack-id) prediction by nearest-neighbour vote;
   - `novelty_scores` — open-set novelty = distance to everything in the index, i.e. the
     retrieval view of leave-one-attack-out (a generator whose neighbours are all bona
     fide gets a *low* novelty score and evades — the A19 story).

LSH recap: project each vector onto `n_bits` random hyperplanes; the sign pattern is a
bit-signature, and similar (small-angle) vectors collide in the same bucket with high
probability. `n_tables` independent hashes + Hamming-radius multi-probe trade recall for
speed. Candidates are exact-cosine reranked, so results are approximate only in *which*
candidates are considered, never in the ranking of those candidates.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


class BruteForceIndex:
    """Exact cosine top-k by full scan — the recall ground truth / small-N default."""

    def build(self, X: np.ndarray) -> "BruteForceIndex":
        self.Xn = _normalize(X)
        return self

    def query(self, q: np.ndarray, k: int = 10, probe_radius: int = 0):
        """Returns (idx, sim), each (n_queries, k). ``probe_radius`` is ignored."""
        qn = _normalize(np.atleast_2d(q))
        sims = qn @ self.Xn.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return idx, np.take_along_axis(sims, idx, axis=1)


class LSHIndex:
    """Random-hyperplane cosine LSH, from scratch, with Hamming multi-probe."""

    def __init__(self, n_bits: int = 16, n_tables: int = 8, seed: int = 0):
        assert n_bits <= 30, "bit-packing uses int; keep n_bits small"
        self.n_bits, self.n_tables, self.seed = n_bits, n_tables, seed

    def build(self, X: np.ndarray) -> "LSHIndex":
        self.Xn = _normalize(X)
        self.n, self.dim = self.Xn.shape
        rng = np.random.default_rng(self.seed)
        self._pow = (1 << np.arange(self.n_bits)).astype(np.int64)
        self.planes = [rng.standard_normal((self.n_bits, self.dim)).astype(np.float32)
                       for _ in range(self.n_tables)]
        self.tables = []
        for R in self.planes:
            codes = self._codes(self.Xn, R)
            buckets: dict[int, list[int]] = defaultdict(list)
            for i, c in enumerate(codes):
                buckets[int(c)].append(i)
            self.tables.append(buckets)
        return self

    def _codes(self, X: np.ndarray, R: np.ndarray) -> np.ndarray:
        return ((X @ R.T) > 0) @ self._pow          # (n,) int signature per row

    def _probe_codes(self, code: int, radius: int) -> list[int]:
        codes = [code]
        if radius >= 1:
            codes += [code ^ int(self._pow[b]) for b in range(self.n_bits)]
        if radius >= 2:
            codes += [code ^ int(self._pow[a]) ^ int(self._pow[b])
                      for a in range(self.n_bits) for b in range(a + 1, self.n_bits)]
        return codes

    def query(self, q: np.ndarray, k: int = 10, probe_radius: int = 1):
        """Approximate cosine top-k. Returns (idx, sim), padding with -1 / 0 when a
        query gathers fewer than k candidates (which is itself the recall signal)."""
        qn = _normalize(np.atleast_2d(q))
        out_idx, out_sim = [], []
        for v in qn:
            cand: set[int] = set()
            for R, buckets in zip(self.planes, self.tables):
                code = int(self._codes(v[None, :], R)[0])
                for c in self._probe_codes(code, probe_radius):
                    hit = buckets.get(c)
                    if hit:
                        cand.update(hit)
            if not cand:
                out_idx.append(np.full(k, -1)); out_sim.append(np.zeros(k)); continue
            ca = np.fromiter(cand, dtype=int)
            sims = self.Xn[ca] @ v
            order = np.argsort(-sims)[:k]
            top, ts = ca[order], sims[order]
            if len(top) < k:                          # pad short candidate sets
                top = np.concatenate([top, np.full(k - len(top), -1)])
                ts = np.concatenate([ts, np.zeros(k - len(ts))])
            out_idx.append(top); out_sim.append(ts)
        return np.asarray(out_idx), np.asarray(out_sim)


class FaissIndex:
    """FAISS backend — the production reference, same (idx, sim) interface.

    Inner product on L2-normalized vectors == cosine, matching the from-scratch
    indexes. ``nlist=0`` -> exact ``IndexFlatIP`` (the industry equivalent of
    BruteForceIndex); ``nlist>0`` -> approximate ``IndexIVFFlat`` (the scale-out
    path), with ``nprobe`` cells probed per query. faiss is imported lazily so the
    rest of the framework (and CI) does not depend on it.
    """

    def __init__(self, nlist: int = 0, nprobe: int = 8, seed: int = 0):
        self.nlist, self.nprobe, self.seed = nlist, nprobe, seed

    def build(self, X: np.ndarray) -> "FaissIndex":
        import faiss

        # faiss + numba/librosa can ship conflicting OpenMP runtimes (segfaults on
        # macOS/conda); single-threaded faiss avoids the clash and is fine at this scale.
        faiss.omp_set_num_threads(1)
        Xn = np.ascontiguousarray(_normalize(X), dtype=np.float32)
        self.dim = Xn.shape[1]
        if self.nlist and self.nlist > 0:
            quant = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quant, self.dim, self.nlist,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(Xn)
            self.index.add(Xn)
            self.index.nprobe = self.nprobe
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(Xn)
        return self

    def query(self, q: np.ndarray, k: int = 10, probe_radius: int = 0):
        """Returns (idx, sim). ``probe_radius`` is ignored (FAISS uses ``nprobe``)."""
        qn = np.ascontiguousarray(_normalize(np.atleast_2d(q)), dtype=np.float32)
        sim, idx = self.index.search(qn, k)        # faiss returns (distances, indices)
        return idx, sim


def recall_at_k(approx_idx: np.ndarray, exact_idx: np.ndarray) -> float:
    """Mean fraction of each query's exact top-k that the approx index also returned."""
    recs = []
    for a, e in zip(approx_idx, exact_idx):
        a = {int(x) for x in a if x >= 0}
        e = {int(x) for x in e}
        recs.append(len(a & e) / max(len(e), 1))
    return float(np.mean(recs))


# --------------------------------------------------------------------------- #
# Retrieval heads (label/score convention: higher == more bona fide; label 1 = bona)
# --------------------------------------------------------------------------- #
def knn_detect(index, train_labels: np.ndarray, query_X: np.ndarray, *,
               k: int = 20, probe_radius: int = 2) -> np.ndarray:
    """Non-parametric detector: similarity-weighted bona-vs-spoof vote over k NN.

    Score in [-1, 1], higher == more bona fide. Plugs into metrics.compute_eer.
    """
    train_labels = np.asarray(train_labels)
    idx, sim = index.query(query_X, k=k, probe_radius=probe_radius)
    scores = np.zeros(len(idx))
    for j, (ii, ss) in enumerate(zip(idx, sim)):
        m = ii >= 0
        if not m.any():
            continue
        lab = train_labels[ii[m]]
        w = np.clip(ss[m], 0, None)
        bona, spoof = w[lab == 1].sum(), w[lab == 0].sum()
        scores[j] = (bona - spoof) / (bona + spoof + 1e-9)
    return scores


def attribute(index, train_attack: np.ndarray, query_X: np.ndarray, *,
              k: int = 20, probe_radius: int = 2) -> np.ndarray:
    """Predict each query's generator by similarity-weighted vote among nearest
    *spoof* neighbours (bona neighbours, attack_id '-', are abstained on)."""
    train_attack = np.asarray(train_attack)
    idx, sim = index.query(query_X, k=k, probe_radius=probe_radius)
    preds = []
    for ii, ss in zip(idx, sim):
        m = ii >= 0
        cand, w = train_attack[ii[m]], np.clip(ss[m], 0, None)
        spoof = cand != "-"
        if not spoof.any():
            preds.append("-"); continue
        votes: dict[str, float] = defaultdict(float)
        for a, wt in zip(cand[spoof], w[spoof]):
            votes[a] += wt
        preds.append(max(votes, key=votes.get))
    return np.asarray(preds)


def novelty_scores(index, query_X: np.ndarray, *, k: int = 1,
                   probe_radius: int = 2) -> np.ndarray:
    """Open-set novelty = 1 - max cosine similarity to anything in the index.

    High == far from everything known (a candidate novel generator). The retrieval
    view of LOAO: build the index on {bona + seen spoof}, query a held-out family.
    """
    idx, sim = index.query(query_X, k=k, probe_radius=probe_radius)
    nov = []
    for ss, ii in zip(sim, idx):
        m = ii >= 0
        nov.append(1.0 - float(ss[m].max()) if m.any() else 1.0)
    return np.asarray(nov)
