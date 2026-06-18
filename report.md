# Audio Deepfake Detection in the Real World: Robustness & Generalization

*ASVspoof 2021 LA · pretrained SSL countermeasure · a robustness evaluation paired
with a falsification-driven study of why detectors fail to generalize.*

A deployed audio-deepfake detector faces two things its training set never showed it:
degraded channels and unseen generators. This report measures both. Part 1 asks
empirically where a strong pretrained detector breaks. Part 2 asks representationally why
it fails to generalize. Part 3 asks whether a calibrated deployment guarantee survives
attack shift.

This is the **results** document: what was run and what it showed. The pre-registered
hypotheses, the experimental protocol and controls, and the literature behind Part 2 are in
[research-design.md](research-design.md). That is the plan; this is the outcome.

> ✅ **Part 1 re-run complete (eval-only, repeat-pad).** Two compounding bugs had inflated
> the clean EER to 9.73%. The first was a zero- vs. repeat-pad train/test mismatch. The
> second was a protocol-parser leak that scored 16,926 `hidden`/`only_speech` trials
> alongside the official `eval` set. Each bug alone leaves EER at ~8.5–8.8%. With both
> fixed, clean EER is 0.82%, matching the published SSL_Anti-spoofing baseline.
> Part 2 has been re-run on the corrected `eval`-only subset: H1 falsified and H2
> supported both reproduce, and H2 strengthens under mean+std pooling. See its section for
> the lens caveat.

## TL;DR

- Clean baseline (full 148,176-trial `eval` set): EER 0.82%, AUC 0.998. Reproduces the
  published SSL_Anti-spoofing number.
- Noise, not compression, is the deployment failure axis. MP3 costs almost nothing
  (~0.7% across 32–128 kbps). Additive noise drives EER from 0.82% to 9.8% at 0 dB.
  Streaming needs at least 4 s of context (EER rises to 2.7% by 2 s, 13.8% at 0.5 s).
- No seen-attack blind spot. Every eval attack scores at or below 2.6%. The mild standouts
  are A18 (2.6%), A19 (1.1%), A17 (1.0%). The earlier "A10 at 27.5%" was an artifact of the
  two bugs above; A10 is 0.55% once corrected.
- Generalization (Part 2): a linear probe on frozen XLS-R fails to transfer to one
  specific unseen generator. A19 leave-one-attack-out gap is +14.8 pp (EER 1.7% to 16.5%).
  On the deployed detector's own representation this nearly vanishes (see the lens caveat),
  so it is a property of the probe, not the production model.
- H1 falsified. Generator identity is near-perfectly linearly decodable from
  the frozen encoder at every one of 25 layers. Its variation is non-existent, so it cannot
  explain differential non-transfer.
- H2 supported. Non-transfer is a boundary-geometry effect. Generators whose
  embeddings sit close to the bona-fide manifold are the ones that evade an unfamiliar
  detector (d-to-bona vs gap, ρ = −0.67, p = 0.013; cos ρ = −0.56, p = 0.047). It
  strengthens under mean+std pooling (ρ = −0.75), so it is not a temporal-pooling artifact.
- Fine-tuning the encoder relocates the worst generator off the bona-fide manifold and
  collapses its gap (A19 14.8 to 1.7 pp). This is a single mechanism case study;
  population-level causality remains untested (see Limitations).
- Conformal coverage breaks under attack shift (Part 3). A split-conformal spoof-miss
  guarantee calibrated on seen attacks (α=0.05) holds within-attack (~0.05) but fails on the
  near-bona generators (A10/A18/A19 miss 18–20%). Degradation reshuffles which attacks evade.
  A bona-proximity-weighted repair fixes the voice-conversion failures but backfires on A10,
  separating the two failure mechanisms.

## Status

Every finding in this report is on real ASVspoof 2021 LA data, and Parts 1–3 are complete:

- Part 1 (degradation robustness): complete, on the full 148,176-trial eval set.
- Part 2 (generalization: LOAO, H1, H2, Regime A/B, mean+std pooling, the AASIST-lens
  check): complete, on a stratified 8k eval-only subset with frozen XLS-R embeddings cached
  once.
- Part 3 (conformal coverage): complete, on the full eval scores, with the weighted-repair
  analysis on the 8k subset where embeddings and scores align.

The following are implemented and unit-tested but not yet run on real data, so nothing above
depends on them:

- Mechanistic-interpretability probing (`src/hooks.py`): activation patching of the
  SSL+AASIST stack. Written, not yet run against the model.
- Confound controls (`scripts/confound_controls.py`): codec and speaker checks on H1/H2,
  awaiting the protocol metadata key.
- A targeted band-mask causal intervention (`src/degradations.py`): the masking primitive
  exists, but the train-time experiment was not run. The Part 3 geometry and coverage
  analyses served as the causal handle instead.
- The four auxiliary detection extensions (NLP, attack profiling, reconstruction, prosody),
  not yet run at scale.
- Cross-dataset validation (In-the-Wild, ASVspoof 5), the main next step.

Synthetic data was used only to validate pipeline logic: the dep-free unit-test suite, and a
planted-effect check of the LOAO runner that recovered ρ ≈ 0.8 on constructed embeddings. No
reported result rests on synthetic data.

## Setup

- Data. ASVspoof 2021 LA evaluation set. The official CM key's `eval` phase gives
  148,176 scored trials over attacks A07–A19 (2021 reuses the 2019 eval attacks; label
  1 = bona fide). The key's other phases, 16,464 `progress` and 16,926 `hidden`/`only_speech`
  trials, are not part of the official EER and are excluded.
- Baseline detector. SSL_Anti-spoofing (XLS-R 300M front-end + AASIST back-end, Tak
  et al., Interspeech 2022). Loaded fairseq-free via an exact fairseq→HuggingFace weight
  remap (`src/ssl_aasist.py`; 0 missing / 0 unexpected keys). The proposal's `lab260/AASIST3`
  checkpoint is degenerate (~63% EER, scores everything bona fide) across every public
  mirror, so it was dropped.
- Metrics. Interpolated EER, normalized min-DCF, AUC, per-attack EER. Score
  convention throughout: higher = more bona fide.
- Scale. Part 1 runs the full eval set. Part 2's representational study runs on a
  stratified 7,987-utterance subset (812 bona fide / 7,175 spoof), with frozen per-layer
  XLS-R embeddings cached once.

## Part 1 — Robustness under real-world degradation

Clean baseline: EER 0.82%, AUC 0.998, normalized min-DCF 0.088. A strong,
non-degenerate operating point that reproduces the published baseline.

Degradation sweep (`src/degradations.py`, batched bf16 scoring):

| Axis | Finding |
|---|---|
| MP3 (8–128 kbps) | Negligible. EER ~0.7% across 32–128 kbps (even slightly below clean), 1.3% at 16 kbps, rising to only 4.5% at 8 kbps. Lossy compression is not the threat. |
| Additive noise (0–30 dB) | The failure axis: 0.9% (30 dB), 2.4% (10 dB), 4.9% (5 dB), 9.8% (0 dB), a +9 pp swing. min-DCF saturates to ~1.0 at 0 dB. |
| Streaming (chunked) | 4 s chunks cost almost nothing (0.80%). EER rises to 2.7% at 2 s and 13.8% at 0.5 s; the model needs at least 4 s of context. |
| Native codec (LA's own) | Small. Every codec <1%: Opus mild-worst at 0.98% vs 0.29% uncompressed; a-law/µ-law/PSTN/GSM/G.722 all 0.5–0.8%. |

![EER vs noise SNR](results/figures/eer_vs_noise.png)

Per-attack failure analysis (clean). With the corrected pipeline there is no blind
spot: every eval attack scores at or below 2.6%. The mild standouts are A18 (2.6%), A19 (1.1%),
A17 (1.0%), all neural TTS/VC, while A09/A13 sit around 0.2%. The earlier "A10 at 27.5%" was an
artifact of the padding and phase-leak bugs; A10 is 0.55% once corrected.

![Per-attack EER heatmap](results/figures/per_attack_heatmap.png)

## Part 2 — Why detectors fail to generalize

*All Part-2 numbers below are on the corrected `eval`-only 8k subset (7,988 trials,
799 bona fide / 7,189 spoof).*

Part 1 asks where the deployed detector breaks. Part 2 asks a representational question on
the frozen XLS-R embedding: what makes a detector fail to transfer to an unseen generator?

We use leave-one-attack-out (LOAO). For each attack `f`, we compare a linear head trained
*with* `f` against one trained on all attacks *except* `f`, both evaluated on held-out `f`
vs bona fide. The non-transfer gap = EER_loao − EER_seen.

Almost every generator transfers fine (gap at or below 1.7 pp), but one fails sharply:
A19, gap +14.8 pp (1.7% to 16.5%). The detector does not recognize an unseen A19
as spoof. The next-largest gap is A10 at +1.7 pp.

![LOAO non-transfer gap per attack](results/figures/loao_gap_per_attack.png)

H1, the pre-registered hypothesis, was falsified. It asked whether probe-recoverable
generator *identity* predicts non-transfer: generators whose identity is more strongly
linearly encoded would be the ones that fail to generalize. Instead, generator identity is
decodable to ceiling (balanced one-vs-rest selectivity ≈ 0.47–0.50 of a 0.5 max) at
every one of the 25 layers, with near-zero variance across generators. A constant cannot
predict a variable: no layer's selectivity correlates with the gap (all p ≥ 0.10; the
headline Spearman(selectivity, gap) is ρ = −0.03, p = 0.94). Identity is present in the
representation everywhere, but it is saturated, so it carries no signal about the gap.

![Identity selectivity is at ceiling at every layer](results/figures/selectivity_layer_ceiling.png)

H2, the replacement hypothesis, asked whether the effect is boundary geometry, and it is
supported. A detector trained on {bona fide + seen spoof} draws a boundary, and an unseen
generator transfers only if it lands on the spoof side. So non-transfer should track how
close a generator sits to the bona-fide manifold relative to the other generators. Across
the 13 attacks (layer 9):

| Geometric predictor | ρ vs gap | p |
|---|---|---|
| Euclidean distance to bona centroid (closer ⇒ higher gap) | −0.67 | 0.013 |
| k-NN bona-fide fraction (more bona-called ⇒ higher gap) | +0.68 | 0.010 |
| cosine distance to bona-fide centroid (closer ⇒ higher gap) | −0.56 | 0.047 |
| projection onto bona↔seen-spoof axis (bona side ⇒ higher gap) | +0.52 | 0.069 |
| distance to nearest *other* spoof (isolation) | −0.04 | 0.89 (null) |

Every bona-proximity measure points the same way, and isolation among spoofs does nothing,
as the boundary account predicts. A19 is the exemplar. Its centroid sits about 10×
closer to bona fide than the typical generator (cosine distance 0.08), and a k-NN that
never saw it labels 55% of its samples bona fide.

This is not a layer-9 artifact. Sweeping all 25 layers, the d_bona↔gap correlation is
negative at every layer (ρ ∈ [−0.73, −0.41]) and significant at 19/25. The reported
layer 9 is mid-pack, not the strongest (`results/geometry_layer_sweep.csv`).

![Bona-fide proximity predicts non-transfer](results/figures/geometry_gap_scatter.png)

Regime B treats fine-tuning as a perturbation of the geometry. We repeat the study with the
fine-tuned XLS-R front-end (from the SSL_Anti-spoofing checkpoint) instead of the
off-the-shelf encoder. This shrinks non-transfer overall (mean gap 1.59 to 0.30 pp), and
most for the worst case (A19 14.8 to 1.7 pp). Mechanistically, fine-tuning moved
A19's centroid from cos-distance 0.08 to 1.07 off the bona-fide manifold, and its k-NN
bona fraction fell 55% to 1%; the gap collapsed in step. This is a case study
for the geometry mechanism.

![Fine-tuning collapses non-transfer](results/figures/regime_gap_slopegraph.png)

Temporal robustness (mean+std pooling). A natural objection is that mean-pooling the frozen
embedding throws away the temporal structure AASIST is built on. We address it by
concatenating the per-utterance standard deviation over time, a parameter-free 2048-d
mean+std feature. This does not weaken the result; it strengthens it. A19 still fails (gap
+15.9 pp) and every H2 predictor strengthens (d_bona ρ = −0.75, p = 0.003; bona-axis
projection ρ = +0.73, p = 0.005; cos_bona ρ = −0.58, p = 0.039). The geometry account is not
an artifact of discarding temporal variability.

The lens caveat (AASIST's own representation). We also probe the detector's own
penultimate (time-aware) embedding, the input to AASIST's 2-class `out_layer`, which is the
same function class as the deployed head. This tells a different story. LOAO
non-transfer nearly vanishes (A19 gap +0.13 pp; max +0.43 pp; mean 0.04 pp), and the
geometry signal largely washes out (only k-NN bona fraction survives, ρ = +0.56, p = 0.048).

So the A19 non-transfer is substantially a property of the frozen-SSL-probe lens,
not the production detector, which generalizes to unseen attacks almost perfectly. H2 explains
why a *linear probe on frozen SSL features* fails to transfer. It is not a claim that the
deployed model does.

## Part 3 — Conformal coverage under attack shift

Parts 1–2 ask where and why the detector fails. This arm asks a deployment question on the
detector's *scores*. Split-conformal calibration sets a threshold so that, under
exchangeability, the spoof-miss rate (a spoof accepted as bona fide, the costly error in
authentication) stays at α. A novel attack is what breaks exchangeability.

We calibrate the threshold on every attack but one (α=0.05) and measure the held-out miss
rate. A within-attack control (calibrate and test on a split of the same attack) sits at
~0.05 for every group, but the guarantee breaks on the same near-bona generators H2 flagged.

| held-out attack | within-attack control | held-out miss (α=0.05) |
|---|---|---|
| A10 | 0.053 | 0.202 |
| A18 | 0.049 | 0.200 |
| A19 | 0.046 | 0.179 |
| A17 | 0.047 | 0.064 |
| every other attack | ~0.05 | ≤ α |

The control rules out a broken conformal procedure and isolates the failure to attack shift.
(`experiments/coverage_loao.py`, on the full 148k eval scores.)

Degradation reshuffles which attacks evade. Sweeping the same hold-one-out coverage
across the degradation conditions is not monotone; the deployment condition changes the
identity of the evaders. Under additive noise A10's failure vanishes (0.20 to ~0.005 by
10 dB) while A18 blows up (0.20 to 0.41 at 0 dB) and A17 becomes a new failure
(0.06 to 0.30 at 10 dB). MP3 barely moves the pattern, consistent with Part 1. So the
guarantee does not degrade uniformly; the set of attacks that slip past is condition-dependent.
(`results/coverage_degradation_sweep.csv`.)

The weighted repair dissociates the two mechanisms. Covariate-shift weighted conformal
(Tibshirani et al. 2019), with weights from the bona-proximity covariate (the H2 axis),
should repair a failure that is actually a covariate shift. It does, but only for the right
attacks. The three voice-conversion attacks sit close to bona (cosine distance to the bona
centroid 0.03–0.04 vs ~0.06–0.09 typical), and reweighting repairs them: A17 returns to α
(0.060 to 0.043), and A19 and A18 improve (0.180 to 0.165, 0.207 to 0.197).

A10 sits far from bona (0.086). Its failure is not geometric. It is the neural-TTS
weakness, the Regime-B counter-example, so the geometric covariate mis-models it and pushes
the threshold the wrong way: A10 gets worse (0.194 to 0.241). Net mean |miss − α| is
unchanged (0.061 to 0.062), insensitive to weight-clipping.

The repair is therefore not a blanket fix but a diagnostic. It separates the VC
bona-proximity failures (repairable) from the A10 neural-TTS blind spot (not), a third
independent line of evidence for H2 and the A10 counter-example. A residual *support* shift
remains even for the VC attacks (their high-score/bona-close region is sparsely populated by
seen spoofs), so the repair is partial. (On the 8k subset where embeddings and scores align.)

## Limitations

- Part 2 is correlational on n = 13 attacks. H2 rests on three bona-proximity predictors
  significant at p ≤ 0.05 (d_bona, k-NN, cos_bona) with consistent directions, but the effect
  is leveraged by A19. Treat it as strong-suggestive, not definitive.
- The Regime-A→B contrast is a case study, not population proof. The cross-regime test
  Spearman(Δcos, Δgap) is null (ρ = −0.08, p = 0.79): most attacks had ~0 gap to begin
  with (no dynamic range). And A10 is a counter-example: fine-tuning helped it while
  moving it *toward* bona (Δcos −1.08, Δgap −1.43), a different mechanism than the
  boundary-geometry account.
- The non-transfer is a property of the probe lens, not the deployed model. On AASIST's
  own penultimate embedding the LOAO gap nearly vanishes (A19 +0.13 pp). So Part 2
  characterizes how a *linear probe on frozen SSL features* generalizes, a legitimate
  representational question, but does not show the production detector failing on unseen
  A19. Part 1's per-attack view (every seen attack at or below 2.6%) is the deployed-model
  lens; the two are complementary, not contradictory.
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
# Part 3 — conformal coverage under attack shift (CPU, on cached scores; no GPU)
python -m experiments.coverage_loao --scores results/scores/clean.npz --by attack
```
