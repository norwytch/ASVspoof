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
- **Generalization (Part 2):** a *linear probe on frozen XLS-R* fails to transfer to
  a specific *unseen* generator — **A19 leave-one-attack-out gap +14.8 pp** (EER
  1.7% → 16.5%). *(On the deployed detector's own representation this nearly vanishes —
  see "lens caveat" — so it's a property of the probe, not the production model.)*
- **H1 falsified, robustly.** Generator identity is near-perfectly linearly
  decodable from the frozen encoder at *every one of 25 layers* — so its
  (non-existent) variation cannot explain differential non-transfer.
- **H2 supported.** Non-transfer is a **boundary-geometry** effect: generators
  whose embeddings sit close to the bona-fide manifold are the ones that evade an
  unfamiliar detector (d-to-bona vs gap, ρ = −0.67, p = 0.013; cos ρ = −0.56, p = 0.047),
  and it *strengthens* under mean+std pooling (ρ = −0.75) — not a temporal-pooling artifact.
- **Fine-tuning the encoder** relocates the worst generator off the bona-fide
  manifold and **collapses its gap (A19 14.8 → 1.7 pp)** — a clean mechanism case
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

*All Part-2 numbers below are on the corrected `eval`-only 8k subset (7,988 trials,
799 bona fide / 7,189 spoof).*

Part 1 asks where the *deployed* detector breaks. Part 2 asks a representational
question on the frozen XLS-R embedding: **what makes a detector fail to transfer
to an unseen generator?** We use leave-one-attack-out (LOAO): for each attack `f`,
compare a linear head trained *with* `f` against one trained on all attacks
*except* `f`, both evaluated on held-out `f` vs bona fide. The
**non-transfer gap** = EER_loao − EER_seen.

**The finding.** Almost every generator transfers fine (gap ≤ 1.7 pp), but one
fails sharply — **A19: gap +14.8 pp (1.7% → 16.5%)**. The detector simply does not
recognize an unseen A19 as spoof (next-largest is A10 at +1.7 pp).

![LOAO non-transfer gap per attack](results/figures/loao_gap_per_attack.png)

**H1 (the pre-registered hypothesis): does probe-recoverable generator *identity*
predict non-transfer? Falsified.** The hypothesis was that generators whose
identity is more strongly linearly encoded would be the ones that fail to
generalize. Instead, generator identity is decodable to **ceiling** (balanced
one-vs-rest selectivity ≈ 0.47–0.50 of a 0.5 max) at **every one of the 25 layers**,
with near-zero variance across generators. A constant cannot predict a variable:
no layer's selectivity correlates with the gap (all p ≥ 0.10; the headline
Spearman(selectivity, gap) is ρ = −0.03, p = 0.94). The failure isn't *absence* of
identity in the representation — it's *saturation*.

![Identity selectivity is at ceiling at every layer](results/figures/selectivity_layer_ceiling.png)

**H2 (the replacement hypothesis): is it boundary geometry? Supported.** A
detector trained on {bona fide + seen spoof} draws a boundary; an unseen generator
transfers iff it lands on the spoof side. So non-transfer should track how close a
generator sits to the **bona-fide manifold** relative to the other generators.
Across the 13 attacks (layer 9):

| Geometric predictor | ρ vs gap | p |
|---|---|---|
| Euclidean distance to bona centroid (closer ⇒ higher gap) | **−0.67** | **0.013** |
| k-NN bona-fide fraction (more bona-called ⇒ higher gap) | **+0.68** | **0.010** |
| cosine distance to bona-fide centroid (closer ⇒ higher gap) | **−0.56** | **0.047** |
| projection onto bona↔seen-spoof axis (bona side ⇒ higher gap) | +0.52 | 0.069 |
| distance to nearest *other* spoof (isolation) | −0.04 | 0.89 (null) |

Every bona-proximity measure points the same way; **isolation among spoofs does
nothing**, exactly as the boundary account predicts. **A19 is the exemplar:** its
centroid sits ~10× closer to bona fide than the typical generator (cosine distance
**0.08**), and a k-NN that never saw it labels **55% of its samples bona fide**.

![Bona-fide proximity predicts non-transfer](results/figures/geometry_gap_scatter.png)

**Regime B — fine-tuning as a perturbation of the geometry.** Repeating the study
with the *fine-tuned* XLS-R front-end (from the SSL_Anti-spoofing checkpoint)
instead of the off-the-shelf encoder shrinks non-transfer overall (mean gap
1.59 → 0.30 pp) and **dramatically for the worst case: A19 14.8 → 1.7 pp**.
Mechanistically, fine-tuning moved A19's centroid from cos-distance **0.08 → 1.07**
off the bona-fide manifold (its k-NN bona fraction fell 55% → 1%) — the gap
collapsed in step. This is a compelling **case study** for the geometry mechanism.

![Fine-tuning collapses non-transfer](results/figures/regime_gap_slopegraph.png)

**Temporal robustness (mean+std pooling).** A natural objection: mean-pooling the
frozen embedding throws away the temporal structure AASIST is built on. Concatenating
the per-utterance **standard deviation** over time (a parameter-free 2048-d
mean+std feature) does not weaken the result — it **sharpens** it: A19 still fails
(gap +15.9 pp) and every H2 predictor strengthens (d_bona ρ = −0.75, p = 0.003;
bona-axis projection ρ = +0.73, p = 0.005; cos_bona ρ = −0.58, p = 0.039). The
geometry account is not an artifact of discarding temporal variability.

**The lens caveat (AASIST's own representation).** Probing the **detector's own**
penultimate (time-aware) embedding — the input to AASIST's 2-class `out_layer`, i.e.
the *same function class* as the deployed head — tells a different and important
story: LOAO non-transfer nearly **vanishes** (A19 gap +0.13 pp; max +0.43 pp; mean
0.04 pp), and the geometry signal largely washes out (only k-NN bona fraction
survives, ρ = +0.56, p = 0.048). So the dramatic A19 non-transfer is substantially a
property of the **frozen-SSL-probe lens**, not the production detector — which
generalizes to unseen attacks almost perfectly. H2 explains why a *linear probe on
frozen SSL features* fails to transfer; it is not a claim that the deployed model does.

## Limitations (read these)

- **Part 2 is correlational on n = 13 attacks.** H2 rests on three bona-proximity
  predictors significant at p ≤ 0.05 (d_bona, k-NN, cos_bona) with consistent
  directions, but the effect is leveraged by A19; treat as strong-suggestive, not
  definitive.
- **The Regime-A→B contrast is a case study, not population proof.** The
  cross-regime test Spearman(Δcos, Δgap) is **null** (ρ = −0.08, p = 0.79): most
  attacks had ~0 gap to begin with (no dynamic range), and **A10 is a
  counter-example** — fine-tuning helped it while moving it *toward* bona (Δcos
  −1.08, Δgap −1.43), a different mechanism than the boundary-geometry account.
- **The non-transfer is a property of the probe lens, not the deployed model.** On
  AASIST's own penultimate embedding the LOAO gap nearly vanishes (A19 +0.13 pp), so
  Part 2 characterizes how a *linear probe on frozen SSL features* generalizes — a
  legitimate representational question — but does **not** show the production detector
  failing on unseen A19. Part 1's per-attack view (every seen attack ≤2.6%) is the
  deployed-model lens; the two are complementary, not contradictory.
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
# Part 2 follow-ups
python -m scripts.cache_embeddings --subset 8000 --pool meanstd --out-dir results/embeddings_meanstd      # Opt 1: mean+std
python -m scripts.cache_aasist_embeddings --subset 8000 --out-dir results/embeddings_aasist               # Opt 4: AASIST penultimate
# (each followed by loao_per_attack + geometry_h2 on its --emb-dir; Opt 4 uses --layer 0)
```
