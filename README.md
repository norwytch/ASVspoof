# ASVspoof 2021 Stress-Testing with Degraded Channels and Unseen Generators

[![tests](https://github.com/norwytch/ASVspoof/actions/workflows/ci.yml/badge.svg)](https://github.com/norwytch/ASVspoof/actions/workflows/ci.yml)

Two questions about a pretrained audio deepfake detector — SSL_Anti-spoofing (wav2vec2
XLS-R 300M + AASIST) — on ASVspoof 2021 LA:

- **Part 1: where does it break?** EER under real-world audio degradation, with a
  per-attack breakdown.
- **Part 2: why does it fail to generalize?** A leave-one-attack-out study on the frozen
  XLS-R embedding, asking what makes a detector miss an unseen generator.

The detector is loaded fairseq-free via an exact weight remap of the published checkpoint
(`src/ssl_aasist.py`). Two extensions reuse the same frozen embeddings and detector scores:
a **retrieval/search** layer over the embeddings and a **conformal coverage** analysis of the
scores under attack shift (sections below).

## Part 1 — Robustness under degradation

Re-scores the eval set under MP3 compression (8–128 kbps), telephony (300–3400 Hz bandpass
+ G.711 mu-law), additive noise (0–30 dB SNR), and chunked/streaming inference, plus a
per-attack (A07–A19) and native-codec breakdown. Code in `src/degradations.py` and
`src/evaluate.py`.

## Part 2 — Why it fails to generalize

A leave-one-attack-out study on the frozen embedding. The pre-registered hypothesis (H1)
was that how strongly the embedding encodes a generator's identity predicts non-transfer
to that held-out generator. H1 was falsified: identity is linearly decodable at every
layer, so it can't explain why only some attacks fail. The replacement (H2) — non-transfer
tracks how close a generator sits to the bona-fide manifold — held up. Full design,
controls, and references in [research-design.md](research-design.md).

## Retrieval — search over the embeddings

A nearest-neighbour layer on the same frozen embeddings. `src/retrieval.py` has a
from-scratch random-hyperplane LSH index (with a brute-force reference and an optional
FAISS backend) and three heads: a non-parametric k-NN detector, generator attribution by
neighbour vote, and an open-set novelty score (distance to everything known). The novelty
score is the retrieval view of Part 2 — a generator near the bona-fide manifold (A19) sits
close to indexed bona fide, gets a low novelty score, and evades, which is its non-transfer
made geometric. `scripts/retrieval_eval.py` runs the recall@k / latency benchmark (the
hand-rolled LSH against FAISS), the k-NN detector EER, attribution accuracy, and per-attack
open-set novelty. Audio has no lexical channel, so this is dense-only — there's no
sparse/BM25 side to add.

## Conformal coverage under attack shift

A split-conformal layer on the deployed detector's scores. `src/conformal.py` calibrates a
spoof-miss threshold on seen attacks so that, under exchangeability, the miss rate (a spoof
accepted as bona fide) stays at α — then asks what a novel attack, which breaks
exchangeability, does to it. `experiments/coverage_loao.py` runs this hold-one-attack-out
on the cached scores: the within-attack control sits at α for every group, but the
guarantee breaks on the near-bona generators — A10/A18/A19 miss 18–20% of held-out spoofs
at α=0.05 (`results/coverage_attack_clean.csv`), exactly where H2 predicted non-transfer.
Degradation *reshuffles* which attacks evade (under noise A10's failure vanishes but A18/A17
blow up). The weighted variant (Tibshirani 2019), with weights from the bona-proximity
covariate, repairs the voice-conversion failures (A17 back to α) but *backfires* on A10 —
whose failure isn't geometric — so the repair doubles as a diagnostic separating the two
mechanisms. Runs on the cached scores, no GPU; full write-up in [report.md](report.md) Part 3.

## Key findings

- Clean EER 0.82%, AUC 0.998 on the 148,176-trial eval set, matching the published
  SSL_Anti-spoofing baseline.
- Noise, not compression, is the failure mode. MP3 is roughly free (~0.7% at 32–128 kbps);
  additive noise pushes EER to 9.8% at 0 dB. Streaming needs about 4 s of context (2.7% at
  2 s, 13.8% at 0.5 s). Native codecs are all under 1%.
- No seen-attack blind spot: every eval attack is at or below 2.6% EER.
- H1 falsified, H2 supported. Bona-fide proximity predicts the leave-one-attack-out gap
  (distance-to-bona vs. gap ρ=−0.67, p=0.013; negative at all 25 layers, p<0.05 at 19/25).
  A19, the generator closest to real speech, has the largest gap (+14.8 pp); fine-tuning the
  encoder moves it off the bona manifold and the gap drops to +1.7 pp.
- Conformal coverage breaks under attack shift. A spoof-miss guarantee calibrated on seen
  attacks (α=0.05) holds within-attack (~0.05) but fails on the near-bona generators
  (A10/A18/A19 miss 18–20%). Degradation reshuffles which attacks evade, and a
  bona-proximity-weighted repair fixes the voice-conversion failures but backfires on A10 —
  separating the two failure mechanisms.

Full write-up and figures in [report.md](report.md).

## Caveats

- An earlier version reported a 9.73% clean EER. That was two bugs: zero-padding instead of
  the recipe's repeat-padding, and a protocol-parser leak that scored 16,926 out-of-spec
  trials. Both are fixed; clean EER is now 0.82%. The earlier "A10 blind spot" was an
  artifact of those bugs (A10 is 0.55% once corrected).
- Part 2 is correlational, n=13 attacks, one corpus. On the detector's own AASIST
  representation the gap nearly vanishes (A19 +0.13 pp), so the effect is a property of the
  frozen-SSL probe rather than the deployed model, and task-tuning removes it. Cross-dataset
  validation is the main next step.
- The four detection extensions (NLP, attack profiling, reconstruction, prosody) are
  implemented and unit-tested but not yet run at scale.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

Use Python ≤ 3.12 (the G.711 path uses stdlib `audioop`, removed in 3.13). Download data
per [data/README.md](data/README.md). The proposal's `lab260/AASIST3` checkpoint is
degenerate across every public mirror (~63% EER), which is why this uses SSL_Anti-spoofing.

## Data and artifacts

The corpus is gitignored. Cached embeddings and the model weights are on Hugging Face:

- Embeddings: [`sempertemper/asvspoof-xlsr-embeddings`](https://huggingface.co/datasets/sempertemper/asvspoof-xlsr-embeddings) — one tar with the eval-only caches (`embeddings/`, `embeddings_ft/`, `embeddings_meanstd/`, `embeddings_aasist/`).
- Weights: [`sempertemper/ssl-antispoofing-weights`](https://huggingface.co/sempertemper/ssl-antispoofing-weights) — `LA_model.pth`.

```bash
hf download sempertemper/asvspoof-xlsr-embeddings asvspoof_xlsr_embeddings.tar --repo-type dataset --local-dir results/
tar -xf results/asvspoof_xlsr_embeddings.tar -C results/
hf download sempertemper/ssl-antispoofing-weights ssl_antispoofing_weights.tar --local-dir third_party/weights/
tar -xf third_party/weights/ssl_antispoofing_weights.tar -C third_party/weights/
```

Both also regenerate from scratch — weights from the original SSL_Anti-spoofing repo,
embeddings from `scripts/cache_embeddings.py`.

## Reproduce

```bash
# Part 1 — degradation sweep + figures
python -m src.evaluate --protocol data/asvspoof2021_LA/keys/CM/trial_metadata.txt \
                       --flac-dir data/asvspoof2021_LA/flac --full
python scripts/make_figures.py results/per_attack_eer_full.csv

# Part 2 — cache embeddings, then LOAO / H1 / H2 / Regime B
python -m scripts.cache_embeddings --subset 8000
python -m scripts.loao_per_attack --emb-dir results/embeddings --out results/loao_per_attack.csv
python -m scripts.layer_sweep_selectivity                       # H1
python -m scripts.geometry_h2                                   # H2
python -m scripts.cache_embeddings_ft --subset 8000 && python -m scripts.compare_regimes
python -m scripts.make_part2_figures

# Retrieval — ANN benchmark (LSH vs FAISS) + k-NN detector / attribution / open-set novelty
python -m scripts.retrieval_eval --emb-dir results/embeddings --layer 9

# Conformal coverage under attack shift (CPU, on cached scores)
python -m experiments.coverage_loao --scores results/scores/clean.npz --by attack
```

## Roadmap

1. Confound controls. `scripts/confound_controls.py` checks whether H1 and H2 survive the
   codec and speaker confounds. (Layer-robustness is done — H2 is negative at all 25 layers,
   p<0.05 at 19/25; see `results/geometry_layer_sweep.csv`.)
2. Cross-dataset generalization — the biggest upgrade. The current unseen-generator test is
   leave-one-out within one 2019-era corpus; re-running it training on 2019/2021 and testing
   on In-the-Wild and ASVspoof 5 would cover genuinely novel generators and more attacks.
3. Realistic degradation: MUSAN babble and reverb instead of white noise, and a
   noise/codec-augmentation baseline to see whether augmenting recovers the lost robustness.
4. Run the four detection extensions at scale.
5. Part 2 causality: a targeted band-mask causal test, and whether bona-proximity
   can flag novel-attack risk from embedding geometry before attack samples exist.

## Layout

```
src/          dataset, degradations, metrics, ssl_aasist loader, model, evaluate, visualize,
              + embeddings, probes, retrieval, conformal, hooks (Part 2 + extensions),
              + the four detection extensions
experiments/  loao.py (leave-one-attack-out), coverage_loao.py (conformal coverage)
scripts/      cache_embeddings[_ft], loao_per_attack, layer_sweep_selectivity, geometry_h2,
              compare_regimes, confound_controls, retrieval_eval, make_figures
tests/        pytest suite for the dep-free logic (run in CI)
data/         download instructions + attack_taxonomy.json (corpora gitignored)
results/      figures + summary CSVs (embeddings/scores gitignored; on Hugging Face)
report.md            full write-up — robustness, generalization, conformal (Parts 1–3), with figures
research-design.md   Part 2 design + verified references
SECURITY.md          security policy (defensive research scope + reporting)
```
