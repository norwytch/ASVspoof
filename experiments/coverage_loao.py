"""Conformal coverage under attack shift — runnable TODAY on cached scores.

Calibrates a spoof-miss threshold (target miss rate alpha) on all attacks
except one, then measures the empirical miss rate on the held-out attack.
``within_miss`` is the exchangeability control (split of the same attack);
``gap`` > 0 means the conformal guarantee silently failed under the shift.

This is the deployed-model arm of the study: it needs no embeddings, no GPU,
no retraining — just results/scores/clean.npz (or any degradation condition,
which gives the coverage-under-degradation table for free). The frozen-probe
arm and the weighted repair (novelty weights from src/retrieval.py over the
cached embeddings) plug into the same family_coverage_table call.

Usage:
    python -m experiments.coverage_loao --scores results/scores/clean.npz --alpha 0.05
    python -m experiments.coverage_loao --scores results/scores/clean.npz --by family
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.conformal import family_coverage_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/scores/clean.npz"))
    ap.add_argument("--taxonomy", type=Path, default=Path("data/attack_taxonomy.json"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--by", choices=["attack", "family"], default="attack")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.scores, allow_pickle=True)
    scores, labels, attacks = d["score"], d["label"], d["attack_id"]

    if args.by == "family":
        tax = json.loads(args.taxonomy.read_text())["attacks"]
        groups = np.array([tax[a]["category"] if a in tax else "bonafide"
                           for a in attacks], dtype=object)
    else:
        groups = attacks

    rows = family_coverage_table(scores, labels, groups, alpha=args.alpha)
    df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"target miss rate alpha = {args.alpha}  |  scores = {args.scores.name}")
    print(df.to_string(index=False))

    out = args.out or Path("results") / f"coverage_{args.by}_{args.scores.stem}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
