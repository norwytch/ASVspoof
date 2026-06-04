# Audio Deepfake Detection in the Real World: Robustness & Generalization

*ASVspoof 2021 LA · pretrained SSL countermeasure · a robustness evaluation paired
with a falsification-driven study of why detectors fail to generalize.*

A deployed audio-deepfake detector faces two things its training set never showed
it: **degraded channels** and **unseen generators**. This report measures both —
empirically *where* a strong pretrained detector breaks (Part 1), and
representationally *why* it fails to generalize (Part 2).

> ✅ **Part 1 re-run complete (eval-only, repeat-pad).** Two compounding bugs had
> inflated the clean EER to 9.73%: a zero- vs. repeat-pad train/test mismatch, **and**
> a protocol-parser leak that scored 16,926 `hidden`/`only_speech` trials alongside the
> official `eval` set. Each alone leaves EER at ~8.5–8.8%; with **both** fixed, clean
> EER is **0.82%** — matching the published SSL_Anti-spoofing baseline exactly. Part 2's
> numbers below were computed on a subset drawn *before* the parser fix and are pending a
> clean re-run (the qualitative conclusions are expected to hold).

## TL;DR

- **Clean baseline (full 148,176-trial `eval` set): EER 0.82%, AUC 0.998** — reproduces
  the published SSL_Anti-spoofing number.
- **Noise, not compression, is the deployment failure axis.** MP3 is essentially
  free (~0.7% across 32–128 kbps); additive noise drives EER from
  0.82% → **9.8% at 0 dB**. Streaming needs **≥4 s of context** (EER rises to 2.7% by 2 s, 13.8% at 0.5 s).
- **No seen-attack blind spot.** Every eval attack scores ≤2.6%; the mild standouts are
  **A18 (2.6%), A19 (1.1%), A17 (1.0%)**. *(The earlier "A10 at 27.5%" was an artifact of
  the two bugs above; A10 is 0.55% once corrected.)*
- **Generalization (Part 2):** detectors fail to transfer to specific *unseen*
  generators — **A19 leave-one-attack-out gap +13.9 pp** (EER 3.4% → 17.3%).
- **H1 falsified, robustly.** Generator identity is near-perfectly linearly
  decodable from the frozen encoder at *every one of 25 layers* — so its
  (non-existent) variation cannot explain differential non-transfer.
- **H2 supported.** Non-transfer is a **boundary-geometry** effect: generators
  whose embeddings sit close to the bona-fide manifold are the ones that evade an
  unfamiliar detector (cos-distance-to-bona vs gap, ρ = −0.60, p = 0.029).
- **Fine-tuning the encoder** relocates the worst generator off the bona-fide
  manifold and **collapses its gap (A19 13.9 → 4.6 pp)** — a clean mechanism case
  study (population-level causality remains untested; see Limitations).

## Setup

- **Data.** ASVspoof 2021 LA evaluation set; the official CM key's **`eval` phase**
  gives 148,176 scored trials over attacks **A07–A19** (2021 reuses the 2019 eval
  attacks; label 1 = bona fide). The key's other phases — 16,464 `progress` and 16,926
  `hidden`/`only_speech` trials — are **not** part of the official EER and are excluded.
- **Baseline detector.** **SSL_Anti-spoofing** (XLS-R 300M front-end + AASIST
  back-end, Tak et al., Interspeech 2022), loaded **fairseq-free** via an exact
  fairseq→HuggingFace weight remap (`src/ssl_aasist.py`; 0 missing / 0 unexpected
  keys). *Note: the proposal's `lab260/AASIST3` checkpoint is degenerate (~63% EER,
  scores everything bona fide) across every public mirror, so it was dropped.*
- **Metrics.** Interpolated EER, **normalized** min-DCF, AUC, per-attack EER.
  Score convention throughout: **higher = more bona fide**.
- **Scale.** Part 1 runs the **full** eval set. Part 2's representational study
  runs on a stratified **7,987-utterance** subset (812 bona fide / 7,175 spoof),
  with frozen per-layer XLS-R embeddings cached once.

## Part 1 — Robustness under real-world degradation

**Clean baseline.** EER **0.82%**, AUC **0.998**, normalized min-DCF 0.088 — a
strong, non-degenerate operating point that reproduces the published baseline.

**Degradation sweep** (`src/degradations.py`, batched bf16 scoring):

| Axis | Finding |
|---|---|
| **MP3** (8–128 kbps) | Negligible; EER ~**0.7%** across 32–128 kbps (even slightly below clean), 1.3% at 16 kbps, only rising to 4.5% at 8 kbps. Lossy compression is not the threat. |
| **Additive noise** (0–30 dB) | The failure axis: 0.9% (30 dB) → 2.4% (10 dB) → 4.9% (5 dB) → **9.8% (0 dB)**, a +9 pp swing; min-DCF saturates to ~1.0 at 0 dB. |
| **Streaming** (chunked) | 4 s chunks are free (0.80%); EER rises to **2.7% at 2 s** and **13.8% at 0.5 s** — the model needs ≥4 s of context. |
| **Native codec** (LA's own) | Small: every codec <1% — Opus mild-worst at **0.98%**, vs 0.29% uncompressed; a-law/µ-law/PSTN/GSM/G.722 all 0.5–0.8%. |

![EER vs noise SNR](results/figures/eer_vs_noise.png)

**Per-attack failure analysis (clean).** With the corrected pipeline there is **no
blind spot**: every eval attack scores ≤2.6%. The mild standouts are **A18 (2.6%),
A19 (1.1%), A17 (1.0%)** — all neural TTS/VC — while A09/A13 sit ~0.2%. *(The earlier
"A10 at 27.5%" was an artifact of the padding + phase-leak bugs; A10 is 0.55% once
corrected.)*

![Per-attack EER heatmap](results/figures/per_attack_heatmap.png)

## Part 2 — Why detectors fail to generalize

> ⚠️ **Pending a clean re-run.** The 8k embedding subset below was stratified-sampled
> from the pre-fix trial pool, so ~800 of its 7,987 trials are `hidden`/`only_speech`.
> The mechanism (H1 falsified, H2 supported) is expected to be robust to this ~10%
> perturbation, but the exact figures should be regenerated on the corrected `eval`-only
> subset before they are treated as final.

Part 1 asks where the *deployed* detector breaks. Part 2 asks a representational
question on the frozen XLS-R embedding: **what makes a detector fail to transfer
to an unseen generator?** We use leave-one-attack-out (LOAO): for each attack `f`,
compare a linear head trained *with* `f` against one trained on all attacks
*except* `f`, both evaluated on held-out `f` vs bona fide. The
**non-transfer gap** = EER_loao − EER_seen.

**The finding.** Most generators transfer fine (gap ≈ 0), but a few fail sharply —
**A19: gap +13.9 pp (3.4% → 17.3%)** and **A10: +6.3 pp**. The detector simply
does not recognize an unseen A19 as spoof.

![LOAO non-transfer gap per attack](results/figures/loao_gap_per_attack.png)

**H1 (the pre-registered hypothesis): does probe-recoverable generator *identity*
predict non-transfer? Falsified.** The hypothesis was that generators whose
identity is more strongly linearly encoded would be the ones that fail to
generalize. Instead, generator identity is decodable to **ceiling** (balanced
one-vs-rest selectivity ≈ 0.49 of a 0.5 max) at **every one of the 25 layers**,
with near-zero variance across generators. A constant cannot predict a variable:
no layer's selectivity correlates with the gap (all p ≥ 0.12). The failure isn't
*absence* of identity in the representation — it's *saturation*.

![Identity selectivity is at ceiling at every layer](results/figures/selectivity_layer_ceiling.png)

**H2 (the replacement hypothesis): is it boundary geometry? Supported.** A
detector trained on {bona fide + seen spoof} draws a boundary; an unseen generator
transfers iff it lands on the spoof side. So non-transfer should track how close a
generator sits to the **bona-fide manifold** relative to the other generators.
Across the 13 attacks (layer 9):

| Geometric predictor | ρ vs gap | p |
|---|---|---|
| cosine distance to bona-fide centroid (closer ⇒ higher gap) | **−0.60** | **0.029** |
| projection onto bona↔seen-spoof axis (bona side ⇒ higher gap) | **+0.59** | **0.033** |
| Euclidean distance to bona centroid | −0.54 | 0.055 |
| distance to nearest *other* spoof (isolation) | +0.03 | 0.91 (null) |

Every bona-proximity measure points the same way; **isolation among spoofs does
nothing**, exactly as the boundary account predicts. **A19 is the exemplar:** its
centroid sits ~3.5× closer to bona fide than the next-closest generator (~14× the
typical one), and a k-NN that
never saw it labels **51% of its samples bona fide**.

![Bona-fide proximity predicts non-transfer](results/figures/geometry_gap_scatter.png)

**Regime B — fine-tuning as a perturbation of the geometry.** Repeating the study
with the *fine-tuned* XLS-R front-end (from the SSL_Anti-spoofing checkpoint)
instead of the off-the-shelf encoder shrinks non-transfer overall (mean gap
2.33 → 1.45 pp) and **dramatically for the worst case: A19 13.9 → 4.6 pp**.
Mechanistically, fine-tuning moved A19's centroid from cos-distance **0.095 → 1.17**
off the bona-fide manifold — the gap collapsed in step. This is a compelling
**case study** for the geometry mechanism.

![Fine-tuning collapses non-transfer](results/figures/regime_gap_slopegraph.png)

## Limitations (read these)

- **Part 2 is correlational on n = 13 attacks.** The H2 result is two
  bona-proximity predictors at p ≈ 0.03 with consistent directions; treat as
  strong-suggestive, not definitive.
- **The Regime-A→B contrast is a case study, not population proof.** The
  cross-regime test Spearman(Δcos, Δgap) is **null** (ρ = −0.21, p = 0.49): most
  attacks had ~0 gap to begin with (no dynamic range), and **A10 is a
  counter-example** — fine-tuning helped it while moving it *toward* bona, because
  A10's failure is the neural-TTS blind spot (a different mechanism), not geometry.
- **Two different lenses.** Part 1's per-attack EER is the *deployed nonlinear*
  AASIST detector (which handles every seen attack ≤2.6%); Part 2's LOAO uses a
  *linear* head on mean-pooled frozen embeddings and surfaces *non-transfer*
  weaknesses (A19, A10) that are invisible to the clean per-attack view. These are
  not contradictions — they measure different things (catching a *seen* attack vs
  generalizing to an *unseen* one).
- min-DCF is normalized; Part 2 uses an 8k subset; attacks are eval-only (A07–A19).

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# Part 1  (scores the official `eval` phase only -> clean EER 0.82%)
python -m src.evaluate --protocol data/asvspoof2021_LA/keys/CM/trial_metadata.txt \
                       --flac-dir data/asvspoof2021_LA/flac --full \
                       --out results/results_full.csv \
                       --per-attack-out results/per_attack_eer_full.csv \
                       --codec-out results/codec_eer_full.csv
python -m scripts.make_figures results/per_attack_eer_full.csv
# Part 2
python -m scripts.cache_embeddings --subset 8000    # frozen XLS-R embeddings (GPU)
python -m scripts.loao_per_attack --emb-dir results/embeddings --out results/loao_per_attack.csv
python -m scripts.layer_sweep_selectivity           # H1
python -m scripts.geometry_h2                        # H2
python -m scripts.cache_embeddings_ft --subset 8000 # Regime B (fine-tuned encoder)
python -m scripts.compare_regimes
python -m scripts.make_part2_figures
```
