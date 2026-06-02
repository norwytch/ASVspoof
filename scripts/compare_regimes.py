"""Regime A vs B: the (near-)causal capstone for H2.

A = off-the-shelf XLS-R, B = fine-tuned XLS-R. If non-transfer is a boundary-
geometry effect, fine-tuning should help most for the generators it pushes
furthest off the bona-fide manifold. Test: across attacks, does Δ(distance from
bona) predict Δ(non-transfer gap)?
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.metrics import spearman_with_ci

gA = pd.read_csv("results/loao_per_attack.csv")[["family", "gap_pct", "eer_loao_pct"]]
gB = pd.read_csv("results/loao_per_attack_ft.csv")[["family", "gap_pct", "eer_loao_pct"]]
geoA = pd.read_csv("results/geometry_h2.csv")[["family", "cos_bona"]]
geoB = pd.read_csv("results/geometry_h2_ft.csv")[["family", "cos_bona"]]

m = (gA.merge(gB, on="family", suffixes=("_A", "_B"))
       .merge(geoA, on="family").merge(geoB, on="family", suffixes=("_A", "_B")))
m["d_gap"] = m.gap_pct_B - m.gap_pct_A          # negative = gap shrank
m["d_cos"] = m.cos_bona_B - m.cos_bona_A        # positive = pushed off bona
m = m.sort_values("gap_pct_A", ascending=False)

pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.3f}")
print(m[["family", "gap_pct_A", "gap_pct_B", "d_gap",
         "cos_bona_A", "cos_bona_B", "d_cos"]].to_string(index=False))

c = spearman_with_ci(m.d_cos.to_numpy(), m.d_gap.to_numpy())
print(f"\nCross-regime causal test  Spearman(Δcos_bona, Δgap), n={len(m)}: "
      f"rho={c['rho']:+.3f} CI=[{c.get('ci_low', float('nan')):+.3f},"
      f"{c.get('ci_high', float('nan')):+.3f}] p={c.get('pval', float('nan')):.3f}")
print("(H2-causal expects NEG: pushed further off bona => gap shrinks more)")
print(f"\nmean gap  A={m.gap_pct_A.mean():.2f}pp  B={m.gap_pct_B.mean():.2f}pp"
      f"   |   A19 gap {m.loc[m.family=='A19','gap_pct_A'].item():.1f} -> "
      f"{m.loc[m.family=='A19','gap_pct_B'].item():.1f}pp")

# --- figure: slopegraph of gaps A->B -----------------------------------------
OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6.5, 5))
for _, r in m.iterrows():
    hot = r.family in ("A19", "A10")
    ax.plot([0, 1], [r.gap_pct_A, r.gap_pct_B], "-o",
            color="#c0392b" if hot else "#95a5a6", lw=2 if hot else 1,
            ms=6 if hot else 4, zorder=3 if hot else 1)
    if hot or abs(r.gap_pct_A) > 2:
        ax.annotate(r.family, (0, r.gap_pct_A), xytext=(-26, -2),
                    textcoords="offset points", fontsize=8,
                    color="#c0392b" if hot else "#555")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Regime A\n(off-the-shelf XLS-R)",
                                           "Regime B\n(fine-tuned XLS-R)"])
ax.set_ylabel("LOAO non-transfer gap (pp)")
ax.set_title("Fine-tuning collapses non-transfer for the\nworst-case generators (A19, A10)")
ax.axhline(0, color="k", lw=0.5)
fig.tight_layout(); fig.savefig(OUT / "regime_gap_slopegraph.png", dpi=150); plt.close(fig)

m.to_csv("results/regime_comparison.csv", index=False)
print("\nwrote results/regime_comparison.csv + results/figures/regime_gap_slopegraph.png")
