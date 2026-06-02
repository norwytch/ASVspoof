"""Part 2, H2: does boundary GEOMETRY explain LOAO non-transfer?

H1 falsified (generator identity is uniformly, near-perfectly linearly decodable
at every layer, so its variation can't explain the gap). H2: a detector trained
on {bonafide + seen spoof} fails on an unseen generator f to the extent that f's
manifold sits close to bona fide RELATIVE to the other (seen) spoof generators.

Per-attack geometric predictors at the detector layer (default 9), on
StandardScaler'd embeddings, correlated (Spearman, n=13) with the per-attack
LOAO gap. Pure-geometry predictors first; a kNN transfer proxy last (flagged
semi-circular since it imitates the LOAO detector).
"""
import argparse
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from src.metrics import spearman_with_ci

p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, default=9)
p.add_argument("--emb-dir", default="results/embeddings")
p.add_argument("--gaps", default="results/loao_per_attack.csv")
p.add_argument("--out", default="results/geometry_h2.csv")
args = p.parse_args()

d = Path(args.emb_dir)
meta = pd.read_csv(d / "meta.csv")
X = np.load(d / f"layer_{args.layer}.npy")
X = StandardScaler().fit_transform(X)               # whiten per-dim scale
y = meta.label.to_numpy()                            # 1 = bona fide
fam = meta.attack_id.to_numpy()
bona = y == 1
attacks = sorted(set(fam[~bona]))
gaps = pd.read_csv(args.gaps).set_index("family")["gap_pct"]

c_bona = X[bona].mean(0)
cent = {g: X[fam == g].mean(0) for g in attacks}

def cos_dist(a, b):
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

rows = []
for f in attacks:
    cf = cent[f]
    d_bona = float(np.linalg.norm(cf - c_bona))
    cos_bona = cos_dist(cf, c_bona)
    others = [g for g in attacks if g != f]
    d_near_spoof = min(float(np.linalg.norm(cf - cent[g])) for g in others)
    d_mean_spoof = float(np.mean([np.linalg.norm(cf - cent[g]) for g in others]))
    # relative position: <1 => closer to bona than to nearest other spoof => expect high gap
    rel = d_bona / d_near_spoof
    # signed margin along the bona<->seen-spoof axis (seen = all spoof except f)
    seen_spoof_centroid = X[(~bona) & (fam != f)].mean(0)
    axis = c_bona - seen_spoof_centroid
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    proj = float((cf - seen_spoof_centroid) @ axis)   # higher => more toward bona side
    # kNN transfer proxy (semi-circular): train on bona + seen spoof, predict held f
    tr = bona | ((~bona) & (fam != f))
    knn = KNeighborsClassifier(n_neighbors=15).fit(X[tr], y[tr])
    knn_bona_frac = float((knn.predict(X[fam == f]) == 1).mean())  # fraction called bona
    rows.append(dict(family=f, gap_pct=float(gaps[f]), d_bona=d_bona, cos_bona=cos_bona,
                     d_near_spoof=d_near_spoof, d_mean_spoof=d_mean_spoof, rel=rel,
                     bona_axis_proj=proj, knn_bona_frac=knn_bona_frac))

df = pd.DataFrame(rows).sort_values("gap_pct", ascending=False)
pd.set_option("display.width", 220, "display.float_format", lambda x: f"{x:.3f}")
print(df.to_string(index=False))

print(f"\nSpearman vs gap (n={len(df)}); H2 predicts: closer-to-bona => higher gap")
preds = {
    "d_bona (lower=>closer bona)": ("neg", df.d_bona),
    "cos_bona (lower=>closer bona)": ("neg", df.cos_bona),
    "d_near_spoof (higher=>isolated)": ("pos", df.d_near_spoof),
    "rel=d_bona/d_near_spoof (lower=>bona-side)": ("neg", df.rel),
    "bona_axis_proj (higher=>bona-side)": ("pos", df.bona_axis_proj),
    "knn_bona_frac [semi-circular]": ("pos", df.knn_bona_frac),
}
for name, (exp_sign, vals) in preds.items():
    c = spearman_with_ci(vals.to_numpy(), df.gap_pct.to_numpy())
    print(f"  {name:48s} rho={c['rho']:+.3f} "
          f"CI=[{c.get('ci_low', float('nan')):+.3f},{c.get('ci_high', float('nan')):+.3f}] "
          f"p={c.get('pval', float('nan')):.3f}  (H2 expects {exp_sign})")

df.to_csv(args.out, index=False)
print(f"\nwrote {args.out}")
