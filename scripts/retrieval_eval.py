"""Retrieval evaluation on the cached XLS-R embeddings (Part 2 retrieval extension).

1. **ANN benchmark** — from-scratch LSH recall@k + speedup vs brute force.
2. **k-NN detector** — non-parametric spoof EER on a held-out split (compare to the
   linear-probe / AASIST numbers in report.md).
3. **Attribution** — predict a held-out spoof's generator by nearest-neighbour vote; top-1 acc.
4. **Open-set (LOAO)** — for each attack f, index {bona + seen-spoof}, and ask whether a
   *novelty* score (distance to everything indexed) flags the truly-unseen f apart from
   held-out instances of *seen* families (AUROC). A generator near the bona manifold (A19)
   gets LOW novelty and evades — the retrieval view of its leave-one-attack-out gap.

Run: python -m scripts.retrieval_eval --emb-dir results/embeddings --layer 9
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.metrics import compute_eer
from src.retrieval import (BruteForceIndex, FaissIndex, LSHIndex, attribute,
                           knn_detect, novelty_scores, recall_at_k)


def _stratified_test_mask(meta, *, seed=0, test_frac=0.3):
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(meta), dtype=bool)
    for _, g in meta.groupby(["label", "attack_id"]):
        idx = g.index.to_numpy()
        k = max(1, int(round(len(idx) * test_frac)))
        is_test[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
    return is_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="results/embeddings")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default="results/retrieval_eval.csv")
    args = ap.parse_args()

    d = Path(args.emb_dir)
    meta = pd.read_csv(d / "meta.csv").reset_index(drop=True)
    X = np.load(d / f"layer_{args.layer}.npy")
    y = meta.label.to_numpy()
    attack = meta.attack_id.to_numpy()
    bona = y == 1

    # --- 1. ANN benchmark ----------------------------------------------------
    print(f"=== 1. ANN benchmark on {len(X)} vectors (LSH vs brute force) ===")
    bf = BruteForceIndex().build(X)
    qi = np.random.default_rng(0).choice(len(X), size=min(500, len(X)), replace=False)
    t = time.time(); exact, _ = bf.query(X[qi], k=10); bf_ms = 1000 * (time.time() - t) / len(qi)
    print(f"  brute force: {bf_ms:.2f} ms/query")
    for nb, nt, pr in [(16, 8, 1), (18, 10, 2), (20, 12, 2)]:
        lsh = LSHIndex(n_bits=nb, n_tables=nt, seed=0).build(X)
        t = time.time(); approx, _ = lsh.query(X[qi], k=10, probe_radius=pr)
        ms = 1000 * (time.time() - t) / len(qi)
        print(f"  LSH(scratch) bits={nb} tables={nt} probe={pr}: recall@10={recall_at_k(approx, exact):.3f}  "
              f"{ms:.2f} ms/query ({bf_ms / ms:.1f}x)")

    try:
        import faiss  # noqa: F401
        nlist = max(1, int(4 * np.sqrt(len(X))))
        t = time.time(); FaissIndex().build(X).query(X[qi], k=10); fms = 1000 * (time.time() - t) / len(qi)
        print(f"  FAISS IndexFlatIP (exact):           recall@10=1.000  {fms:.2f} ms/query ({bf_ms / fms:.1f}x)")
        for nprobe in (4, 16):
            ivf = FaissIndex(nlist=nlist, nprobe=nprobe).build(X)
            t = time.time(); ai, _ = ivf.query(X[qi], k=10); ms = 1000 * (time.time() - t) / len(qi)
            print(f"  FAISS IVF nlist={nlist} nprobe={nprobe}: recall@10={recall_at_k(ai, exact):.3f}  "
                  f"{ms:.2f} ms/query ({bf_ms / ms:.1f}x)")
    except ImportError:
        print("  (faiss not installed — skipping FAISS backends; `pip install faiss-cpu`)")

    rows = []
    is_test = _stratified_test_mask(meta)

    # --- 2. k-NN detector EER ------------------------------------------------
    idx_tr = BruteForceIndex().build(X[~is_test])
    scores = knn_detect(idx_tr, y[~is_test], X[is_test], k=args.k)
    eer, _ = compute_eer(y[is_test], scores)
    print(f"\n=== 2. k-NN detector EER (k={args.k}) = {eer * 100:.2f}% on the held-out split ===")
    rows.append({"metric": "knn_detector_eer_pct", "value": round(eer * 100, 3)})

    # --- 3. generator attribution --------------------------------------------
    sp_tr = (~bona) & ~is_test
    sp_te = (~bona) & is_test
    idx_sp = BruteForceIndex().build(X[sp_tr])
    pred = attribute(idx_sp, attack[sp_tr], X[sp_te], k=args.k)
    acc = float((pred == attack[sp_te]).mean())
    print(f"=== 3. generator attribution top-1 acc = {acc * 100:.1f}% over {int(sp_te.sum())} spoof queries ===")
    rows.append({"metric": "attribution_top1_acc_pct", "value": round(acc * 100, 2)})

    # --- 4. open-set novelty (LOAO) ------------------------------------------
    print("\n=== 4. open-set novelty per held-out attack (LOAO) ===")
    attacks = sorted(set(attack[~bona]))
    nov_rows = []
    for f in attacks:
        held = attack == f
        index_mask = (bona & ~is_test) | ((~bona) & ~is_test & ~held)   # train, f excluded
        idx_open = BruteForceIndex().build(X[index_mask])
        nov_f = novelty_scores(idx_open, X[held], k=1)                  # truly unseen family
        seen_te = (~bona) & is_test & ~held                            # seen families, held-out instances
        nov_seen = novelty_scores(idx_open, X[seen_te], k=1)
        labels = np.r_[np.ones(len(nov_f)), np.zeros(len(nov_seen))]
        auroc = roc_auc_score(labels, np.r_[nov_f, nov_seen])
        nov_rows.append({"attack": f, "novelty_mean": round(float(nov_f.mean()), 4),
                         "auroc_novel_vs_seen": round(float(auroc), 3)})
    nv = pd.DataFrame(nov_rows).sort_values("novelty_mean")
    print(nv.to_string(index=False))
    print("\nLower novelty / AUROC ~0.5 => retrieval can't flag that generator as novel; "
          "expect the bona-closest generator (A19) near the bottom — it evades, mirroring "
          "its leave-one-attack-out gap.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    nv.to_csv(args.out.replace(".csv", "_novelty.csv"), index=False)
    print(f"\nwrote {args.out} + {args.out.replace('.csv', '_novelty.csv')}")


if __name__ == "__main__":
    main()
