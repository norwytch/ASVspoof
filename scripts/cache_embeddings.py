"""Part 2 driver: cache frozen XLS-R per-layer embeddings for a stratified subset.

embeddings.py has no __main__ by design (the heavy step); this wraps it.
Regime A (off-the-shelf XLS-R) by default; pass --model-id for a fine-tuned encoder.
"""
import argparse

from src.dataset import load_trials
from src.embeddings import cache_embeddings

p = argparse.ArgumentParser()
p.add_argument("--protocol", default="data/asvspoof2021_LA/keys/CM/trial_metadata.txt")
p.add_argument("--flac-dir", default="data/asvspoof2021_LA/flac")
p.add_argument("--subset", type=int, default=8000)
p.add_argument("--out-dir", default="results/embeddings")
p.add_argument("--model-id", default=None, help="encoder id/path; default = off-the-shelf XLS-R")
args = p.parse_args()

trials = load_trials(args.protocol, args.flac_dir, n=args.subset)
print(f"caching XLS-R embeddings for {len(trials)} trials "
      f"({int((trials.label==1).sum())} bona fide / {int((trials.label==0).sum())} spoof)")
kw = {} if args.model_id is None else {"model_id": args.model_id}
cache_embeddings(trials, out_dir=args.out_dir, **kw)
print("done ->", args.out_dir)
