"""Main evaluation loop: sweep all degradation conditions and write results.

Outputs:
    results/results.csv        full metric table (one row per condition/param)
    results/scores/*.npz       cached per-utterance scores per condition

Usage:
    python -m src.evaluate --protocol <key> --flac-dir <dir> [--full] [--subset 5000]
"""
from __future__ import annotations

import argparse


def run_condition(detector, trials, family: str, params: dict):
    """Apply one degradation config to every trial, score, return (labels, scores, attack_ids).

    TODO: load audio -> apply degradation (or streaming) -> detector.predict
    -> collect arrays. Cache scores to disk so metric recomputation is free.
    """
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True, help="ASVspoof CM key/protocol file")
    p.add_argument("--flac-dir", required=True, help="Directory of eval .flac files")
    p.add_argument("--model-id", default=None)
    p.add_argument("--subset", type=int, default=5000, help="Stratified subset size")
    p.add_argument("--full", action="store_true", help="Use the entire eval set")
    p.add_argument("--out", default="results/results.csv")
    args = p.parse_args()

    # TODO:
    #  1. parse_protocol + select_eval_subset
    #  2. load SpoofDetector
    #  3. clean baseline -> baseline_eer
    #  4. for family, configs in DEGRADATIONS: run_condition + summarize
    #  5. also emit per_attack_eer per condition for failure analysis
    #  6. write results.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
