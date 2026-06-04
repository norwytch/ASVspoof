"""Recompute Part-1 metrics on the official `eval` phase from cached score npz.

The degradation sweep caches per-utterance scores in results/scores/<slug>.npz for
*every* trial it was run on. The published ASVspoof 2021 LA EER (0.82%) is scored on
the `eval` phase only; this filters each cache to eval and rewrites the summary,
per-attack, and codec CSVs — reusing all the (expensive) scoring. Run after a sweep:

    python -m scripts.eval_phase_metrics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import compute_eer, per_attack_eer, summarize

SCORES_DIR = Path("results/scores")
META = "data/asvspoof2021_LA/keys/CM/trial_metadata.txt"
KEEP_PHASE = "eval"


def _load_phase_codec():
    phase, codec = {}, {}
    with open(META) as f:
        for line in f:
            c = line.split()
            if len(c) >= 8:
                phase[c[1]] = c[7]
                codec[c[1]] = c[2]
    return phase, codec


def _slug_to_row(slug: str):
    if "_" not in slug:
        return slug, "—"
    fam, body = slug.split("_", 1)
    return fam, body


def main():
    phase, codec = _load_phase_codec()
    files = sorted(SCORES_DIR.glob("*.npz"))
    if not files:
        raise SystemExit("no cached scores in results/scores/")

    # baseline = clean, eval-only
    cz = np.load(SCORES_DIR / "clean.npz", allow_pickle=True)
    cids = cz["utt_id"].astype(str)
    cmask = np.array([phase.get(u) == KEEP_PHASE for u in cids])
    base_eer, _ = compute_eer(cz["label"][cmask], cz["score"][cmask])
    print(f"clean eval EER = {base_eer*100:.3f}%  (n={int(cmask.sum())})")

    rows, per_attack_rows = [], []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        ids = z["utt_id"].astype(str)
        m = np.array([phase.get(u) == KEEP_PHASE for u in ids])
        labels, scores, attacks = z["label"][m], z["score"][m], z["attack_id"][m]
        fam, param = _slug_to_row(fp.stem)
        row = {"condition": fam, "param": param}
        row.update(summarize(labels, scores, baseline_eer=base_eer))
        rows.append(row)
        for attack, eer in per_attack_eer(labels, scores, attacks).items():
            per_attack_rows.append({"condition": fam, "param": param,
                                    "attack_id": attack, "eer_pct": eer * 100.0})

    # codec stratification on clean/eval
    codec_rows = []
    cc = np.array([codec.get(u, "-") for u in cids])[cmask]
    cl, cs = cz["label"][cmask], cz["score"][cmask]
    for c in sorted(set(cc.tolist())):
        cm = cc == c
        if cm.sum() < 2 or len(set(cl[cm].tolist())) < 2:
            continue
        eer_c, _ = compute_eer(cl[cm], cs[cm])
        codec_rows.append({"codec": c, "n": int(cm.sum()),
                           "n_bonafide": int((cl[cm] == 1).sum()),
                           "eer_pct": round(eer_c * 100.0, 3)})

    pd.DataFrame(rows).to_csv("results/results_full.csv", index=False)
    pd.DataFrame(per_attack_rows).to_csv("results/per_attack_eer_full.csv", index=False)
    pd.DataFrame(codec_rows).to_csv("results/codec_eer_full.csv", index=False)
    print(f"wrote results_full.csv ({len(rows)} conditions), per_attack_eer_full.csv, "
          f"codec_eer_full.csv  — all eval-only")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
