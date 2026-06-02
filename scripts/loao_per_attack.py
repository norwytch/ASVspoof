"""Per-attack (n=13) LOAO non-transfer gaps for a given embedding cache.

Generalizes experiments.loao to per-attack granularity (family = attack_id), the
properly-powered version of the H1 test, and lets you point it at any emb-dir
(Regime A: results/embeddings ; Regime B: results/embeddings_ft).
"""
import argparse
import numpy as np, pandas as pd
from experiments.loao import run_loao
from src.metrics import spearman_with_ci

p = argparse.ArgumentParser()
p.add_argument("--emb-dir", default="results/embeddings")
p.add_argument("--layer", type=int, default=9)
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--out", default="results/loao_per_attack.csv")
args = p.parse_args()

meta = pd.read_csv(f"{args.emb_dir}/meta.csv")
X = np.load(f"{args.emb_dir}/layer_{args.layer}.npy")
meta["family"] = np.where(meta.label == 1, "bonafide", meta.attack_id)

per_seed = [run_loao(meta, X, seed=s) for s in range(args.seeds)]
cat = pd.concat([r["per_family"].assign(seed=s) for s, r in enumerate(per_seed)])
agg = (cat.groupby("family")
          .agg(gap_pct=("gap_pct", "mean"), gap_std=("gap_pct", "std"),
               eer_seen_pct=("eer_seen_pct", "mean"), eer_loao_pct=("eer_loao_pct", "mean"),
               probe_selectivity=("probe_selectivity", "mean"))
          .reset_index().sort_values("gap_pct", ascending=False))
corr = spearman_with_ci(agg.probe_selectivity.to_numpy(), agg.gap_pct.to_numpy())
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}")
print(agg.to_string(index=False))
print(f"\nn={len(agg)}  H1 Spearman(selectivity, gap): rho={corr['rho']:+.3f} "
      f"CI=[{corr.get('ci_low', float('nan')):+.3f},{corr.get('ci_high', float('nan')):+.3f}] "
      f"p={corr.get('pval', float('nan')):.3f}")
agg.to_csv(args.out, index=False)
print("wrote", args.out)
