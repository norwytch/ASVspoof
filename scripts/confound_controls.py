"""Confound controls for Part 2 (research-design.md §3.4).

Does the generator-identity selectivity (H1) and the bona-proximity geometry (H2)
survive the two most dangerous confounds in ASVspoof 2021 LA — transmission
**codec** and **speaker**? If "generator identity" is really *codec* identity, or
"proximity to bona" is really "same codec/speaker as the bona pool," the Part 2
story collapses. This script tests that, on the cached eval-only embeddings.

Controls
--------
C1  Codec/speaker decodability — how linearly recoverable are codec and speaker
    from the embedding (balanced selectivity)? A high number flags a candidate
    confound; it does not by itself invalidate anything.
C2  Identity selectivity WITHIN a single codec — recompute per-attack one-vs-rest
    selectivity using only utterances sharing one codec. If it stays at ceiling,
    generator identity is not a codec artifact.
C3  Composition check — per-attack codec mix vs bona, and whether A19 (the H2
    leverage point) is anomalous; plus bona/spoof speaker overlap.
C4  H2 with a codec-matched bona reference — recompute distance-to-bona using a
    bona subset resampled to each attack's codec distribution, then re-test
    Spearman(gap, d_bona). If rho holds, bona-proximity is not codec-proximity.

Usage
-----
    python -m scripts.confound_controls \
        --emb-dir results/embeddings --layer 9 \
        --protocol data/asvspoof2021_LA/keys/CM/trial_metadata.txt \
        --gaps results/loao_per_attack.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_codec_speaker(protocol: str | Path) -> pd.DataFrame:
    """utt_id -> (speaker, codec) from the CM key (cols: speaker utt codec ... )."""
    rows = []
    with open(protocol) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                rows.append({"utt_id": p[1], "speaker": p[0], "codec": p[2]})
    return pd.DataFrame(rows)


def _bal_selectivity(X, y, *, seed=0, n_splits=3, balance=True):
    """balanced-accuracy selectivity = real - shuffled, optional class balancing."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if balance:
        # subsample each class to the rarest class size (multiclass-safe)
        classes, counts = np.unique(y, return_counts=True)
        k = counts.min()
        idx = np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False)
                              for c in classes])
        X, y = X[idx], y[idx]
    if len(np.unique(y)) < 2 or np.min(np.unique(y, return_counts=True)[1]) < n_splits:
        return float("nan")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    real = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy").mean()
    ctrl = cross_val_score(pipe, X, rng.permutation(y), cv=cv,
                           scoring="balanced_accuracy").mean()
    return float(real - ctrl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="results/embeddings")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--protocol", default="data/asvspoof2021_LA/keys/CM/trial_metadata.txt")
    ap.add_argument("--gaps", default="results/loao_per_attack.csv")
    ap.add_argument("--out", default="results/confound_controls.csv")
    args = ap.parse_args()

    d = Path(args.emb_dir)
    meta = pd.read_csv(d / "meta.csv")
    X = np.load(d / f"layer_{args.layer}.npy")
    meta = meta.merge(parse_codec_speaker(args.protocol), on="utt_id", how="left")
    assert len(meta) == len(X), "meta/emb misalignment"
    bona = meta.label.to_numpy() == 1
    fam = meta.attack_id.to_numpy()
    codec = meta.codec.fillna("none").to_numpy()
    spk = meta.speaker.fillna("?").to_numpy()
    attacks = sorted(set(fam[~bona]))
    gaps = pd.read_csv(args.gaps).set_index("family")["gap_pct"]

    # ---- C1: how decodable are codec / speaker themselves? --------------------
    print("=== C1: confound decodability (balanced selectivity, 0=chance .5=max) ===")
    print(f"  codec   : {_bal_selectivity(X, codec):.3f}")
    print(f"  speaker : {_bal_selectivity(X, spk):.3f}")

    # ---- C2: generator-identity selectivity WITHIN one codec ------------------
    sp = ~bona
    top_codec = pd.Series(codec[sp]).value_counts().idxmax()
    m = sp & (codec == top_codec)
    print(f"\n=== C2: per-attack identity selectivity within codec '{top_codec}' "
          f"(n={m.sum()}) vs overall ===")
    rows = []
    for f in attacks:
        all_sel = _bal_selectivity(X[sp], (fam[sp] == f).astype(int))
        within = _bal_selectivity(X[m], (fam[m] == f).astype(int)) if (fam[m] == f).sum() >= 6 else float("nan")
        rows.append({"attack": f, "sel_all": all_sel, "sel_within_codec": within})
    c2 = pd.DataFrame(rows)
    print(c2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- C3: per-attack codec composition vs bona + speaker overlap ----------
    print("\n=== C3: codec composition (fraction) per attack vs bona ===")
    comp = (pd.crosstab(fam, codec, normalize="index").round(3))
    print(comp.to_string())
    print(f"\n  speakers: bona={len(set(spk[bona]))}, spoof={len(set(spk[~bona]))}, "
          f"shared={len(set(spk[bona]) & set(spk[~bona]))}  "
          f"(high overlap => speaker can't trivially separate bona vs spoof)")

    # ---- C4: H2 with a codec-matched bona reference --------------------------
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(0)
    bona_idx = np.where(bona)[0]
    bona_codec = codec[bona_idx]
    rows = []
    for f in attacks:
        fmask = np.where(fam == f)[0]
        # resample bona to match attack f's codec histogram
        want = pd.Series(codec[fmask]).value_counts(normalize=True)
        picks = []
        for cdc, frac in want.items():
            pool = bona_idx[bona_codec == cdc]
            if len(pool):
                picks.append(rng.choice(pool, max(1, int(frac * 400)), replace=True))
        if not picks:
            continue
        bref = np.concatenate(picks)
        d_raw = np.linalg.norm(Xs[fmask].mean(0) - Xs[bona_idx].mean(0))
        d_matched = np.linalg.norm(Xs[fmask].mean(0) - Xs[bref].mean(0))
        rows.append({"family": f, "gap_pct": float(gaps.get(f, np.nan)),
                     "d_bona_raw": d_raw, "d_bona_codecmatched": d_matched})
    c4 = pd.DataFrame(rows).dropna()
    rho_raw, p_raw = spearmanr(c4.gap_pct, c4.d_bona_raw)
    rho_cm, p_cm = spearmanr(c4.gap_pct, c4.d_bona_codecmatched)
    print("\n=== C4: H2 Spearman(gap, distance-to-bona), raw vs codec-matched bona ===")
    print(f"  raw           rho={rho_raw:+.3f} p={p_raw:.3f}")
    print(f"  codec-matched rho={rho_cm:+.3f} p={p_cm:.3f}  "
          f"(if it holds, bona-proximity is NOT codec-proximity)")

    c4.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("\nINTERPRET: C1 high + C2 within-codec selectivity still high => identity "
          "is genuine, not codec. C4 codec-matched rho ~ raw rho => H2 survives the "
          "codec confound. If C2 collapses or C4 goes null, the confound is real.")


if __name__ == "__main__":
    main()
