"""Part 2 rescue: is generator-identity selectivity graded at ANY layer, or
ceiling everywhere? Per-attack balanced selectivity across all 25 layers, then
Spearman(selectivity@L, gap) using the layer-9 LOAO gaps.

Fans the (layer x attack x seed) grid across all cores (joblib); each task does
its own 3-fold CV serially so the parallelism is at the grid level.
"""
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.metrics import spearman_with_ci

d = Path("results/embeddings")
meta = pd.read_csv(d / "meta.csv")
sp = meta.label.to_numpy() == 0
fam = meta.attack_id.to_numpy()[sp]
attacks = sorted(set(fam))
gaps = pd.read_csv("results/loao_per_attack.csv").set_index("family")["gap_pct"]

# preload all spoof-layer matrices once
Xs = {L: np.load(d / f"layer_{L}.npy")[sp] for L in range(25)}


def bal_sel(L, f, seed):
    """balanced-accuracy selectivity = real - shuffled, subsampled to balance."""
    X = Xs[L]
    y = (fam == f).astype(int)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    neg = rng.choice(neg, size=len(pos), replace=False)
    idx = np.concatenate([pos, neg]); Xb, yb = X[idx], y[idx]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    real = cross_val_score(pipe, Xb, yb, cv=cv, scoring="balanced_accuracy").mean()
    ctrl = cross_val_score(pipe, Xb, rng.permutation(yb), cv=cv,
                           scoring="balanced_accuracy").mean()
    return (L, f, real - ctrl)


grid = [(L, f, s) for L in range(25) for f in attacks for s in range(3)]
print(f"running {len(grid)} (layer,attack,seed) tasks across cores...", flush=True)
res = Parallel(n_jobs=-1, verbose=5)(delayed(bal_sel)(L, f, s) for L, f, s in grid)

# average selectivity over seeds -> sel[(L, f)]
acc = {}
for L, f, v in res:
    acc.setdefault((L, f), []).append(v)
sel = {(L, f): float(np.mean(vs)) for (L, f), vs in acc.items()}

rows = []
for L in range(25):
    s_arr = np.array([sel[(L, f)] for f in attacks])
    g_arr = np.array([gaps[f] for f in attacks])
    corr = spearman_with_ci(s_arr, g_arr)
    rows.append({"layer": L, "sel_mean": s_arr.mean(), "sel_std": s_arr.std(),
                 "sel_min": s_arr.min(), "sel_max": s_arr.max(),
                 "rho_vs_gap": corr["rho"], "p": corr.get("pval", float("nan"))})
    print(f"L{L:2d} sel {s_arr.mean():.3f}±{s_arr.std():.3f} "
          f"[{s_arr.min():.3f},{s_arr.max():.3f}]  "
          f"rho_vs_gap={corr['rho']:+.3f} p={corr.get('pval', float('nan')):.3f}",
          flush=True)

# also dump the full per-attack-per-layer selectivity matrix for downstream use
mat = pd.DataFrame({f: [sel[(L, f)] for L in range(25)] for f in attacks})
mat.insert(0, "layer", range(25))
mat.to_csv("results/selectivity_by_layer_attack.csv", index=False)
pd.DataFrame(rows).to_csv("results/layer_sweep_selectivity.csv", index=False)
print("\nwrote results/layer_sweep_selectivity.csv + selectivity_by_layer_attack.csv")
