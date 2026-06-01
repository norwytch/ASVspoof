"""Leave-one-attack-out generalization study — the core H1 test (research-design.md).

For each spoof family f, measure the NON-TRANSFER GAP:
    gap(f) = EER_loao(f) - EER_seen(f)
where EER_loao uses a detector trained on all families EXCEPT f, and EER_seen
uses a detector that DID train on f (both evaluated on a held-out slice of f vs.
a held-out bona-fide pool). Then test whether the family's linear
probe-selectivity predicts its gap (Spearman, §3.5).

Runs entirely on cached embeddings (src/embeddings.py) — CPU-friendly.

Usage:
    python -m src.embeddings ...          # cache embeddings first (needs torch)
    python -m experiments.loao --emb-dir results/embeddings --layer 9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import probes
from src.attack_profiling import category_of, load_taxonomy
from src.metrics import compute_eer, spearman_with_ci


def _stratified_split(meta: pd.DataFrame, *, test_frac: float, seed: int) -> np.ndarray:
    """Boolean is_test mask, stratified by (label, family). Deterministic by seed."""
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(meta), dtype=bool)
    for _, grp in meta.groupby(["label", "family"]):
        idx = grp.index.to_numpy()
        k = max(1, int(round(len(idx) * test_frac)))
        is_test[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
    return is_test


def run_loao(meta: pd.DataFrame, X: np.ndarray, *, seed: int = 0,
             test_frac: float = 0.3, C: float = 1.0) -> dict:
    """Core computation on embeddings ``X`` aligned row-wise to ``meta``.

    ``meta`` needs columns label (1=bonafide), family. Returns per-family rows
    and the Spearman result for selectivity-vs-gap.
    """
    meta = meta.reset_index(drop=True)
    is_test = _stratified_split(meta, test_frac=test_frac, seed=seed)
    y = meta.label.to_numpy()
    fam = meta.family.to_numpy()

    bona = fam == "bonafide"
    families = sorted(f for f in set(fam[~bona]))

    bona_tr = bona & ~is_test
    bona_te = bona & is_test
    spoof_tr = (~bona) & ~is_test
    spoof_te = (~bona) & is_test

    rows = []
    for f in families:
        held = fam == f
        # detectors: SEEN trains on all spoof; LOAO trains on all spoof except f
        tr_seen = bona_tr | spoof_tr
        tr_loao = bona_tr | (spoof_tr & ~held)
        te = bona_te | (spoof_te & held)               # bona-fide + held-out f
        if te.sum() < 4 or (spoof_te & held).sum() < 2:
            continue

        s_seen = probes.detector_scores(X[tr_seen], y[tr_seen], X[te], C=C, seed=seed)
        s_loao = probes.detector_scores(X[tr_loao], y[tr_loao], X[te], C=C, seed=seed)
        eer_seen, _ = compute_eer(y[te], s_seen)
        eer_loao, _ = compute_eer(y[te], s_loao)

        # selectivity: how separable is f's identity among spoof (f IS in-domain here)
        sp = (~bona)
        sel = probes.probe_selectivity(X[sp], (fam[sp] == f).astype(int), C=C, seed=seed)

        rows.append({
            "family": f,
            "n_held_spoof": int((spoof_te & held).sum()),
            "eer_seen_pct": eer_seen * 100,
            "eer_loao_pct": eer_loao * 100,
            "gap_pct": (eer_loao - eer_seen) * 100,
            "probe_accuracy": sel["accuracy"],
            "probe_selectivity": sel["selectivity"],
        })

    df = pd.DataFrame(rows)
    corr = (spearman_with_ci(df.probe_selectivity.to_numpy(), df.gap_pct.to_numpy())
            if len(df) >= 3 else {"rho": float("nan"), "note": "need >=3 families"})
    return {"per_family": df, "correlation": corr}


def _load_cache(emb_dir: str | Path, layer: int, taxonomy: dict):
    emb_dir = Path(emb_dir)
    meta = pd.read_csv(emb_dir / "meta.csv")
    meta["family"] = meta["attack_id"].map(lambda a: category_of(a, taxonomy))
    X = np.load(emb_dir / f"layer_{layer}.npy")
    assert len(meta) == len(X), "meta.csv and layer matrix are misaligned"
    return meta, X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", default="results/embeddings", help="cache from src.embeddings")
    p.add_argument("--layer", type=int, default=9, help="which hidden layer to probe")
    p.add_argument("--taxonomy", default="data/attack_taxonomy.json")
    p.add_argument("--seeds", type=int, default=5, help="retrain seeds for variance")
    p.add_argument("--out", default="results/loao.csv")
    args = p.parse_args()

    tax = load_taxonomy(args.taxonomy)
    meta, X = _load_cache(args.emb_dir, args.layer, tax)

    per_seed = [run_loao(meta, X, seed=s) for s in range(args.seeds)]
    # average per-family across seeds
    cat = pd.concat([r["per_family"].assign(seed=s) for s, r in enumerate(per_seed)])
    agg = (cat.groupby("family")
              .agg(gap_pct=("gap_pct", "mean"), gap_std=("gap_pct", "std"),
                   probe_selectivity=("probe_selectivity", "mean"),
                   eer_loao_pct=("eer_loao_pct", "mean"))
              .reset_index())
    corr = spearman_with_ci(agg.probe_selectivity.to_numpy(), agg.gap_pct.to_numpy())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(agg.to_string(index=False))
    print(f"\nH1 Spearman(selectivity, gap): rho={corr['rho']:.3f} "
          f"CI=[{corr.get('ci_low', float('nan')):.3f}, {corr.get('ci_high', float('nan')):.3f}] "
          f"p={corr.get('pval', float('nan')):.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
