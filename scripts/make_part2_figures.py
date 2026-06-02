"""Part 2 figures: the generalization-study arc.

  1. loao_gap_per_attack.png    — non-transfer gap per generator (the finding)
  2. selectivity_layer_ceiling.png — identity selectivity vs layer (H1 falsified:
     near-perfect at every depth, flat across attacks -> can't explain the gap)
  3. geometry_gap_scatter.png   — bona-fide proximity vs gap (H2 supported)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.metrics import spearman_with_ci

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)

# 1. LOAO gap per attack -------------------------------------------------------
g = pd.read_csv("results/loao_per_attack.csv").sort_values("gap_pct")
fig, ax = plt.subplots(figsize=(7, 4.2))
colors = ["#c0392b" if v > 3 else ("#e67e22" if v > 1 else "#7f8c8d") for v in g.gap_pct]
ax.barh(g.family, g.gap_pct, color=colors)
if "gap_std" in g:
    ax.errorbar(g.gap_pct, g.family, xerr=g.gap_std, fmt="none", ecolor="#2c3e50",
                elinewidth=0.8, capsize=2)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("LOAO non-transfer gap  (EER$_{loao}$ − EER$_{seen}$, pp)")
ax.set_title("Leave-one-attack-out: which unseen generators evade the detector")
fig.tight_layout(); fig.savefig(OUT / "loao_gap_per_attack.png", dpi=150); plt.close(fig)

# 2. Selectivity vs layer (the ceiling) ---------------------------------------
ls = pd.read_csv("results/layer_sweep_selectivity.csv")
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.fill_between(ls.layer, ls.sel_min, ls.sel_max, alpha=0.2, color="#2980b9",
                label="min–max across 13 generators")
ax.plot(ls.layer, ls.sel_mean, "-o", color="#2980b9", ms=3, label="mean selectivity")
ax.axhline(0.5, ls="--", color="#c0392b", lw=1, label="ceiling (perfect identity)")
ax.set_ylim(0, 0.55)
ax.set_xlabel("XLS-R layer"); ax.set_ylabel("balanced identity selectivity")
ax.set_title("H1 falsified: generator identity is decodable to ceiling at every layer")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "selectivity_layer_ceiling.png", dpi=150); plt.close(fig)

# 3. Geometry vs gap (H2) ------------------------------------------------------
h2 = pd.read_csv("results/geometry_h2.csv")
corr = spearman_with_ci(h2.cos_bona.to_numpy(), h2.gap_pct.to_numpy())
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(h2.cos_bona, h2.gap_pct, s=45, color="#8e44ad", zorder=3)
for _, r in h2.iterrows():
    ax.annotate(r.family, (r.cos_bona, r.gap_pct), fontsize=7,
                xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("cosine distance of generator centroid to bona-fide centroid\n(smaller = more bona-fide-like)")
ax.set_ylabel("LOAO non-transfer gap (pp)")
ax.set_title(f"H2 supported: bona-fide proximity predicts non-transfer "
             f"(ρ={corr['rho']:+.2f}, p={corr.get('pval', float('nan')):.3f})")
fig.tight_layout(); fig.savefig(OUT / "geometry_gap_scatter.png", dpi=150); plt.close(fig)

print("wrote:", *(p.name for p in sorted(OUT.glob("*.png")) if p.name in {
    "loao_gap_per_attack.png", "selectivity_layer_ceiling.png", "geometry_gap_scatter.png"}))
