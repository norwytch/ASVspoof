"""Option 4 driver: cache AASIST's penultimate (pre-logit) embedding.

The detector's OWN time-aware utterance representation, captured via a hook on
the 2-class head. Same subset (seed/n) as scripts.cache_embeddings so the LOAO /
geometry comparison is apples-to-apples. Output drops straight into the Part 2
scripts: `--emb-dir results/embeddings_aasist --layer 0`.
"""
import argparse

from src.dataset import load_trials
from src.embeddings import cache_aasist_embeddings

p = argparse.ArgumentParser()
p.add_argument("--protocol", default="data/asvspoof2021_LA/keys/CM/trial_metadata.txt")
p.add_argument("--flac-dir", default="data/asvspoof2021_LA/flac")
p.add_argument("--subset", type=int, default=8000)
p.add_argument("--out-dir", default="results/embeddings_aasist")
p.add_argument("--ckpt", default=None, help="path to LA_model.pth; default packaged")
args = p.parse_args()

trials = load_trials(args.protocol, args.flac_dir, n=args.subset)
print(f"caching AASIST penultimate embeddings for {len(trials)} trials "
      f"({int((trials.label==1).sum())} bona fide / {int((trials.label==0).sum())} spoof)")
cache_aasist_embeddings(trials, ckpt=args.ckpt, out_dir=args.out_dir)
print("done ->", args.out_dir)
