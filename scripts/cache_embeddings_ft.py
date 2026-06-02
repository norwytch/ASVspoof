"""Part 2 / Regime B: cache per-layer embeddings from the FINE-TUNED XLS-R
(the SSL_Anti-spoofing front-end), to contrast with Regime A's off-the-shelf
XLS-R. Same stratified subset, same layout -> loao.py / geometry_h2.py run on it
unchanged by pointing --emb-dir / --gaps at the Regime-B outputs.
"""
import argparse

from src.dataset import load_trials
from src.embeddings import cache_embeddings
from src.ssl_aasist import load_finetuned_encoder

p = argparse.ArgumentParser()
p.add_argument("--protocol", default="data/asvspoof2021_LA/keys/CM/trial_metadata.txt")
p.add_argument("--flac-dir", default="data/asvspoof2021_LA/flac")
p.add_argument("--subset", type=int, default=8000)
p.add_argument("--out-dir", default="results/embeddings_ft")
args = p.parse_args()

trials = load_trials(args.protocol, args.flac_dir, n=args.subset)
print(f"[Regime B] caching FINE-TUNED XLS-R embeddings for {len(trials)} trials "
      f"({int((trials.label==1).sum())} bona fide / {int((trials.label==0).sum())} spoof)",
      flush=True)
enc = load_finetuned_encoder()
cache_embeddings(trials, out_dir=args.out_dir, encoder=enc)
print("done ->", args.out_dir)
