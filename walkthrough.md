# Experiment Walkthrough

A step-by-step tour of the whole experiment — each step's **what / how / why / result**.
Companion to [report.md](report.md) (findings) and [research-design.md](research-design.md)
(the Part 2 design + literature). Numbers are from the full run on the H100.

## The one-sentence frame

Audio deepfake detectors will encounter degraded and unseen audio during deployment. 

A deployed audio-deepfake detector faces two things training never showed it —
**degraded channels** (Part 1) and **unseen generators** (Part 2). Part 1 measures
*where* it breaks; Part 2 dissects *why* it fails to generalize.

---

## Setup — the ingredients

**The baseline detector** (`src/ssl_aasist.py`). The detector is **SSL_Anti-spoofing**
(XLS-R 300M front-end + AASIST graph-attention back-end, Tak et al., Interspeech 2022).
The catch: the original loads its front-end through `fairseq`, which won't build on
modern Python. So it's rebuilt **fairseq-free** — instantiate a HuggingFace
`Wav2Vec2Model` from the `facebook/wav2vec2-xls-r-300m` config, then **remap the
checkpoint's fairseq weight keys onto HF names** (`post_extract_proj →
feature_projection.projection`, `fc1 → feed_forward.intermediate_dense`, the
weight-norm `pos_conv` parametrization, etc.). The remap is **exact — 0 missing /
0 unexpected keys** — which is the proof the architecture matches. Output: 2 logits,
**index 1 = bona fide**; scored as `logit[1] − logit[0]` so higher = more genuine.

*Why it matters:* the proposal's `lab260/AASIST3` was degenerate (~63% EER, scored
everything bona fide), so choosing a baseline that actually works — and proving the
port is faithful — is itself a result.

---

## PART 1 — Robustness: *where* does it break?

### Step 1 — Parse the protocol
`src/dataset.py` reads the ASVspoof 2021 LA CM key → **165,102 scored trials**, attacks
**A07–A19**, label 1 = bona fide. A token-scan parser, so it survives both the 2021
8-column and 2019 5-column layouts.

### Step 2 — Clean baseline
Run the detector on every untouched trial; compute **EER, normalized min-DCF, AUC**.
→ **EER 9.73%, AUC 0.967.** EER (equal error rate) is the threshold where false-accept
rate equals false-reject rate — the single number anti-spoofing is judged on.
Everything downstream is measured *relative* to this credible, non-degenerate point.

### Step 3 — Degradation sweep (`src/evaluate.py`, batched bf16)
Each degradation is a function `(waveform, sr) → waveform`; the whole eval set is
re-scored under each and EER recomputed:

- **MP3** (8–128 kbps, ffmpeg round-trip) → essentially free; EER even *drops* to 8.5% @ 32 kbps.
- **Additive noise** (0–30 dB SNR) → the failure axis: 10.2% → **25.7% @ 0 dB**.
- **Streaming** (chunk audio + aggregate scores) → needs ≥4 s of context; 12.5% by 2 s.
- **Native codec** — not synthetic; EER is *stratified* by LA's own transmission codec
  column (a-law / µ-law / Opus / GSM / PSTN). PSTN worst at 8.2%.

*Why:* this is the deployment question — production audio is compressed, noisy,
band-limited, chunked. The headline ("noise, not compression, breaks it") is
actionable and non-obvious.

### Step 4 — Per-attack failure analysis (`results/per_attack_eer_full.csv`)
Compute EER **per attack type vs the shared bona-fide pool**.
→ **A10 (Tacotron2+WaveRNN) is a 27.5% blind spot even on clean audio**, while
A09/A13 sit ~0.5%. A *seen-attack* weakness — a specific generator the model can't catch.

### Step 5 — Figures (`scripts/make_figures.py`)
ROC/DET overlays, EER-vs-{bitrate, SNR, chunk} sweeps, the per-attack heatmap.

---

## PART 2 — Generalization: *why* does it fail?

The pivot: Part 1 uses the full nonlinear detector. Part 2 asks a **representational**
question on the *frozen* embedding — *what property of an unseen generator makes a
detector fail to flag it?*

### Step 6 — Cache frozen embeddings (`scripts/cache_embeddings.py`)
Take a stratified **8k subset**, run frozen XLS-R with `output_hidden_states=True`,
**mean-pool over time** → one 1024-d vector per utterance, **per layer, all 25 layers**.
Saved as `layer_*.npy` + `meta.csv` + `utt_ids.npy`. ~3 min on the H100. *This is the
one expensive step; everything after is cheap linear algebra on these vectors — the
"cache once, iterate" design.*

### Step 7 — Leave-one-attack-out (LOAO) gaps (`scripts/loao_per_attack.py` → `run_loao`)
For each attack `f` (n=13): train a **linear** head on `{bona + all spoof}` (the
"**seen**" detector) and another on `{bona + all spoof except f}` (the "**loao**"
detector), both evaluated on held-out `f` + a disjoint bona-fide test split. The
**non-transfer gap = EER_loao − EER_seen** = how much *not having seen f* hurts catching
f. Averaged over 5 seeds.
→ Most generators transfer fine (gap ≈ 0); **A19 gap +13.9 pp (3.4% → 17.3%)**,
A10 +6.3 pp. These two *are* the generalization failure.

### Step 8 — H1 test: is it *generator identity*? (`scripts/layer_sweep_selectivity.py`)
**The pre-registered hypothesis:** generators whose *identity* is more strongly encoded
in the embedding are the ones that fail to transfer. Test it with **probing +
control-task selectivity**: a linear probe predicts "is this generator f?"
(one-vs-rest); **selectivity = balanced-accuracy(real labels) −
balanced-accuracy(shuffled labels)**. The shuffled-label control is the capacity floor —
it stops you mistaking "the probe is powerful" for "the info is there." Run across **all
25 layers × 13 attacks × 3 seeds**.
→ Selectivity is at **ceiling (~0.49 of a 0.5 max) at every layer**, with near-zero
variance across generators, and **no layer's selectivity correlates with the gap (all
p ≥ 0.12)**. **A constant cannot predict a variable → H1 falsified, robustly.**

> Nuance worth keeping: raw-*accuracy* selectivity first *looked* flat at ~0.12 — that
> was a class-imbalance artifact. **Balanced** accuracy is the correct metric and reveals
> the ceiling. The fix changed the interpretation, not just the number.

### Step 9 — H2 test: is it *boundary geometry*? (`scripts/geometry_h2.py`)
**The replacement hypothesis:** a detector trained on `{bona + seen spoof}` draws a
decision boundary; an unseen generator transfers *iff* it lands on the spoof side — so
non-transfer should track **how close f sits to the bona-fide manifold** relative to the
other generators. On StandardScaler'd layer-9 embeddings, per attack: cosine/Euclidean
distance from f's centroid to the bona centroid; distance to the nearest *other* spoof
(isolation); projection onto the bona↔seen-spoof axis; and a kNN "what fraction of f gets
called bona" proxy. Spearman vs gap (n=13):

| Predictor | ρ | p |
|---|---|---|
| cosine distance to bona (closer ⇒ higher gap) | **−0.60** | **0.029** |
| bona-axis projection (bona-side ⇒ higher gap) | +0.59 | 0.033 |
| Euclidean distance to bona | −0.54 | 0.055 |
| isolation among spoofs (nearest other spoof) | +0.03 | 0.91 (null) |

→ **Every bona-proximity measure points the same way; isolation does nothing — exactly
what the boundary account predicts. H2 supported.** A19 is the exemplar: its centroid
sits ~3.5× closer to bona than the next-closest generator (~14× the typical), and a kNN
that never saw it labels **51% of its samples bona fide**.

### Step 10 — Regime A vs B: the (near-)causal capstone (`scripts/cache_embeddings_ft.py` + `compare_regimes.py`)
Re-cache embeddings from the **fine-tuned** XLS-R (Regime B) instead of off-the-shelf
(Regime A), and re-run the study. If geometry is the mechanism, fine-tuning should help
most for the generators it pushes furthest off the bona-fide manifold.
→ **A19's gap collapses 13.9 → 4.6 pp** as its cosine distance moves 0.095 → 1.17 (off
the manifold). A clean mechanism case study.

> **But stay honest:** the *population-level* cross-regime test
> `Spearman(Δcos, Δgap)` is **null** (ρ = −0.21, p = 0.49) — most attacks had ~0 gap to
> move (no dynamic range), and **A10 is a counter-example** (fine-tuning helped it while
> moving it *toward* bona, because A10's failure is the neural-TTS blind spot — a
> *different* mechanism). So Regime A→B is a **case study, not population proof**, and
> the report says exactly that.

---

## Why this is good science (the interview point)

The arc is **falsify your own hypothesis**: H1 (identity → non-transfer) was
pre-registered, the probe was built to test it, and the data killed it cleanly — then
H2 (geometry) survived four consistent predictors plus a null control, and a fine-tuning
intervention moved the worst case in the predicted direction. "I was wrong in a specific,
measurable way, and here is what's actually going on" is what separates research from a
leaderboard submission.

## Known weak points (own them before an interviewer does)

- **n = 13 attacks.** Part 2 correlations are on 13 points — strong-suggestive, not
  definitive; reported with bootstrap CIs and a pre-registered direction.
- **Two different lenses.** Part 1's per-attack EER is the *nonlinear* AASIST detector;
  Part 2's LOAO uses a *linear* head on mean-pooled frozen embeddings. A10 is hard in
  both; A19 is easy-when-seen but the worst *non-transfer* — different measurements, not
  a contradiction.
- **The Regime A→B contrast is a case study**, not population proof (the cross-regime
  test is null; A10 is a counter-example).
- **Mean-pooling** discards temporal structure — a deliberate simplification for a
  fixed-length linear probe, not a claim that time doesn't matter.

---

## The pipeline as commands

```bash
# Part 1
python -m src.evaluate --protocol data/asvspoof2021_LA/keys/CM/trial_metadata.txt \
                       --flac-dir data/asvspoof2021_LA/flac --full     # -> results/*_full.csv
python scripts/make_figures.py results/per_attack_eer_full.csv

# Part 2
python -m scripts.cache_embeddings --subset 8000           # Step 6: frozen XLS-R cache
python -m scripts.loao_per_attack --emb-dir results/embeddings   # Step 7: non-transfer gaps
python -m scripts.layer_sweep_selectivity                  # Step 8: H1 (identity ceiling)
python -m scripts.geometry_h2                              # Step 9: H2 (boundary geometry)
python -m scripts.cache_embeddings_ft --subset 8000        # Step 10: Regime B encoder
python -m scripts.compare_regimes
python -m scripts.make_part2_figures
```
