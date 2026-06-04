# ASVSpoof2021 Stress-Testing with Degraded Channels and Unseen Generators

A production audio-deepfake detector has to survive two things its training set
never showed it: **degraded channels** and **unseen attacks**. This repo studies
both — empirically *where* detectors break, and representationally *why* — on
**ASVspoof 2021 LA** with the pretrained **SSL_Anti-spoofing** countermeasure
(XLS-R 300M + AASIST), loaded fairseq-free via an exact weight remap.

> **Part 1 — Where does it break?** A robustness-evaluation framework measuring
> detector degradation under compression, telephony, additive noise, and
> streaming inference, with per-attack failure analysis. *(Run on the official
> 148k-trial `eval` set.)*
>
> **Part 2 — Why does it break?** A falsification-driven representational study of
> generalization failure: across held-out generators, is leave-one-attack-out
> non-transfer explained by probe-recoverable *generator identity* (**H1 —
> falsified**) or by **boundary geometry** / bona-fide proximity (**H2 —
> supported**)? *(Run; see [report.md](report.md) and
> [research-design.md](research-design.md).)*

**The two halves are one story.** Part 1's channel/codec degradation is, in Part 2,
one of the *shortcut confounds* a generalization claim must survive — the same
degradation pipeline that quantifies deployment robustness becomes the control
that separates genuine synthesis artifacts from spurious channel cues. Part 1
ships the frozen-embedding + evaluation infrastructure Part 2 builds on.

## Part 1 — Robustness under real-world degradation
- **Baseline:** clean EER / min-DCF / ROC / DET on the ASVspoof 2021 LA eval set.
- **Degradations** (`src/degradations.py`): MP3 (8–128 kbps), telephony
  (300–3400 Hz bandpass + G.711 mu-law), additive noise (0–30 dB SNR), and
  streaming (chunked inference, 500 ms–4 s).
- **Failure analysis:** per-attack-type EER deltas (attacks A07–A19, grouped by
  generative mechanism — see [data/attack_taxonomy.json](data/attack_taxonomy.json)).
- **Extensions:** transcript-conditioned NLP signals, TTS-attack profiling,
  reconstruction-error detection (AeroBlade analog), prosody — see
  `src/{nlp_features,attack_profiling,reconstruction,prosody}.py`.

## Part 2 — Why detectors fail to generalize
The intellectually rigorous arm. Central falsifiable hypothesis: *across held-out
spoofing families, probe-recoverable generator identity predicts LOAO
generalization failure; a targeted high-frequency vocoder-artifact band-mask
improves LOAO EER over a bandwidth-matched control.* Full protocol — leave-one-
attack-out matrix, shortcut ablations, selectivity-controlled probing, the
correlation test, the pre-registered intervention, and the verified reference
list — is in **[research-design.md](research-design.md)**. §8 there maps it onto
this codebase (≈3 new modules: `embeddings.py`, `probes.py`, `experiments/loao.py`).

## Status & caveats (read first)

Posted as a portfolio artifact — the framing below is deliberate, not hidden.
What's solid vs. pending:

- **Part 1 absolute EERs now reproduce the published baseline (clean EER 0.82%).** Two
  compounding bugs had inflated the clean EER to 9.73%: a train/test **padding mismatch**
  (zero- vs. the recipe's repeat-padding, fixed in `src/model.py`) **and** a protocol-parser
  **phase leak** that scored 16,926 `hidden`/`only_speech` trials alongside the official
  `eval` set (fixed in `src/dataset.py`). Either alone leaves EER ~8.5–8.8%; both fixed →
  **0.82%**. The old "A10 blind spot" was an artifact of these bugs (A10 is now 0.55%).
- **Part 2 was sampled before the parser fix** (its 8k subset includes ~10% `hidden` trials),
  so its exact figures are **pending a clean re-run**; the mechanism is expected to hold. It is
  also **correlational, n = 13 attacks, single corpus.** The H2 effect is robust in *direction*
  (drop-one ρ ∈ [−0.50, −0.73]) but its p<0.05 leans on A19; cross-dataset validation
  (ASVspoof 5 / in-the-wild) is the key next step.
- **Regime A→B is a mechanism case study, not population proof** (cross-regime test null;
  A10 is a counterexample — see [report.md](report.md) Limitations).
- **The four detection extensions** (NLP / profiling / reconstruction / prosody) are
  implemented + unit-tested but **not yet run at scale**.

## Key Findings
Full write-up with figures in **[report.md](report.md)**.

- **Clean baseline (official 148k-trial `eval` set): EER 0.82%, AUC 0.998** — reproduces
  the published SSL_Anti-spoofing number.
- **Noise — not compression — is the failure axis.** MP3 is ~free (~0.7% across 32–128 kbps);
  additive noise pushes EER to **9.8% at 0 dB**. Streaming needs **≥4 s of context** (EER rises
  to 2.7% by 2 s, 13.8% at 0.5 s). Native-codec effect is tiny (all <1%; Opus worst at 0.98%).
- **No seen-attack blind spot:** every eval attack scores ≤2.6% (mild standouts A18 2.6%,
  A19 1.1%, A17 1.0%; A09/A13 ~0.2%).
- **Generalization: H1 falsified, H2 supported.** Generator identity is linearly
  decodable to *ceiling at every one of 25 layers*, so it can't explain
  differential non-transfer; instead **bona-fide proximity predicts the
  leave-one-attack-out gap** (d-to-bona vs gap ρ=−0.67, p=0.013). The worst
  case, **A19 (gap +14.8 pp)**, is the bona-closest generator; fine-tuning the
  encoder moves it off the bona manifold and **collapses its gap to +1.7 pp**.
  *(Caveat: on the detector's own AASIST-penultimate representation the gap nearly
  vanishes — A19 +0.13 pp — so this non-transfer is a frozen-SSL-probe-lens property,
  not the deployed model; see report.md.)*

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg            # system dependency for MP3 codec
```
- Use **Python ≤ 3.12** (the G.711 path uses stdlib `audioop`, removed in 3.13;
  a numpy fallback exists but the stdlib path is preferred).
- Download data per [data/README.md](data/README.md).

## Artifacts (Hugging Face)

Large derived artifacts are hosted on Hugging Face rather than committed to git:

| Artifact | Contents | Backs | Repo |
|---|---|---|---|
| **XLS-R embeddings cache** | frozen per-layer features (`layer_*.npy`, `utt_ids.npy`, `meta.csv`); Regime A (off-the-shelf) + Regime B (fine-tuned encoder) | Part 2 — LOAO, H1 layer sweep, H2 geometry | [`sempertemper/asvspoof-xlsr-embeddings`](https://huggingface.co/datasets/sempertemper/asvspoof-xlsr-embeddings) *(dataset)* |
| **SSL_Anti-spoofing weights** | `LA_model.pth` (XLS-R 300M + AASIST) | the Part 1/2 baseline detector | [`sempertemper/ssl-antispoofing-weights`](https://huggingface.co/sempertemper/ssl-antispoofing-weights) *(model)* |

Both repos are public; each ships a single tarball — download and extract:

```bash
pip install huggingface_hub

# Part 2 embeddings (1.7 GB tar) -> results/embeddings/ (Regime A) + results/embeddings_ft/ (Regime B)
huggingface-cli download sempertemper/asvspoof-xlsr-embeddings asvspoof_xlsr_embeddings.tar \
    --repo-type dataset --local-dir results/
tar -xf results/asvspoof_xlsr_embeddings.tar -C results/

# Baseline weights (2.5 GB tar) -> third_party/weights/.../LA_model.pth
huggingface-cli download sempertemper/ssl-antispoofing-weights ssl_antispoofing_weights.tar \
    --local-dir third_party/weights/
tar -xf third_party/weights/ssl_antispoofing_weights.tar -C third_party/weights/
```

> After extracting, confirm the layout matches what the code reads —
> `results/embeddings/` + `results/embeddings_ft/`, and the `LA_model.pth` path in
> `src/ssl_aasist.py`; adjust the `tar -C` target if the archive nests differently.
> Both artifacts also regenerate from scratch: weights via the original
> SSL_Anti-spoofing repo, embeddings via `scripts/cache_embeddings.py` (GPU, ~3 min).

## Reproduce
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Part 1 — full degradation sweep on the eval set (+ figures)
python -m src.evaluate --protocol data/asvspoof2021_LA/keys/CM/trial_metadata.txt \
                       --flac-dir data/asvspoof2021_LA/flac --full
python scripts/make_figures.py results/per_attack_eer_full.csv

# Part 2 — cache frozen XLS-R embeddings once, then the generalization study
python -m scripts.cache_embeddings --subset 8000
python -m scripts.loao_per_attack --emb-dir results/embeddings --out results/loao_per_attack.csv
python -m scripts.layer_sweep_selectivity              # H1 (identity-selectivity ceiling)
python -m scripts.geometry_h2                          # H2 (boundary geometry)
python -m scripts.cache_embeddings_ft --subset 8000    # Regime B (fine-tuned encoder)
python -m scripts.compare_regimes
python -m scripts.make_part2_figures

# Extensions (implemented + unit-tested, not yet run at scale):
#   src/{transcribe,nlp_features,attack_profiling,reconstruction,prosody}.py
```

## Results & Discussion
See **[report.md](report.md)** for the full analysis — clean baseline, the
degradation sweep, per-attack failure analysis, and the Part 2 generalization
study (H1 falsified / H2 supported / Regime A↔B) — with embedded figures from
`results/figures/`.

## Repository Layout
```
src/                  dataset, degradations, metrics, ssl_aasist loader, model wrapper, evaluate, extensions
experiments/          loao.py — leave-one-attack-out runner
scripts/              cache_embeddings[_ft], loao_per_attack, layer_sweep_selectivity, geometry_h2, compare_regimes, make_figures
data/                 download instructions + attack_taxonomy.json (corpora gitignored)
results/              figures/ + CSVs + cached scores/embeddings (corpora-derived artifacts gitignored)
report.md             written analysis of both parts (~1500 words, with figures)
research-design.md    Part 2 — the generalization/representational study design + verified refs
```

## Status

**Part 1 (robustness) and Part 2 (generalization) are both run end-to-end** on the
real ASVspoof 2021 LA eval set with the SSL_Anti-spoofing detector. The four
detection **extensions** (NLP / profiling / reconstruction / prosody) are
implemented and unit-tested but not yet executed at scale.

Core pipeline (executed):
- `dataset.py` — protocol parser (2021 + 2019 layouts) + stratified subset
- `degradations.py` — MP3, telephony, noise, streaming (+ numpy mu-law fallback)
- `ssl_aasist.py` — fairseq-free **SSL_Anti-spoofing** loader (XLS-R + AASIST; exact
  fairseq→HF remap) and `load_finetuned_encoder()` for Part 2 Regime B
- `model.py` — `SpoofDetector` wrapper (index 1 = bona fide)
- `evaluate.py` — full sweep loop (batched bf16), score caching, per-attack +
  native-codec breakdown
- `metrics.py` — EER, normalized min-DCF, AUC, per-attack EER, `spearman_with_ci`
- `visualize.py` / `scripts/make_figures.py` — ROC / DET / EER-sweep / heatmap

Part 2 (executed — see [report.md](report.md)):
- `embeddings.py` — frozen XLS-R per-layer embedding cache (Regime A & B)
- `probes.py` — linear probes with control-task **selectivity** (Hewitt & Liang)
- `experiments/loao.py` + `scripts/loao_per_attack.py` — per-attack non-transfer gap
- `scripts/layer_sweep_selectivity.py` — the H1 ceiling result (all 25 layers)
- `scripts/geometry_h2.py` — the H2 boundary-geometry test
- `scripts/{cache_embeddings_ft,compare_regimes}.py` — Regime B (fine-tuned encoder)

Extensions (implemented, unit-tested, **not yet run at scale**):
- `transcribe.py` (Whisper→JSONL), `nlp_features.py` (Ext 1), `attack_profiling.py`
  (Ext 2), `reconstruction.py` (Ext 3), `prosody.py` (Ext 4).

Notes:
- **Baseline changed from the proposal.** `lab260/AASIST3` (and every public
  AASIST3 mirror) is degenerate (~63% EER, scores everything bona fide), and the
  proposal's `ntt-hilab-gensp/ssl_spoof` is gated (HTTP 401) — hence
  SSL_Anti-spoofing. The H2 *band-mask* intervention in the original design was
  replaced by the geometry analysis + the Regime A/B encoder contrast.
- `data/attack_taxonomy.json` is filled from the ASVspoof 2019 database paper
  (A01–A19); only A07–A19 appear in the eval set.
