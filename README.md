# ASVSpoof2021 Stress-Testing with Degraded Channels and Unseen Generators

A production audio-deepfake detector has to survive two things its training set
never showed it: **degraded channels** and **unseen attacks**. This repo studies
both — empirically *where* detectors break, and representationally *why* — on
**ASVspoof 2021 LA** with the pretrained **SSL_Anti-spoofing** countermeasure
(XLS-R 300M + AASIST), loaded fairseq-free via an exact weight remap.

> **Part 1 — Where does it break?** A robustness-evaluation framework measuring
> detector degradation under compression, telephony, additive noise, and
> streaming inference, with per-attack failure analysis. *(Run on the full
> 165k-trial eval set.)*
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

## Key Findings
Full write-up with figures in **[report.md](report.md)**.

- **Clean baseline (full 165k-trial eval): EER 9.73%, AUC 0.967.**
- **Noise — not compression — is the failure axis.** MP3 is ~free (EER *drops* to
  8.5% at 32 kbps); additive noise pushes EER to **25.7% at 0 dB**. Streaming needs
  **≥4 s of context** (EER rises to 12.5% by 2 s). Native-codec effect is modest (PSTN worst, 8.2%).
- **A10 (Tacotron2+WaveRNN) is the standing blind spot:** 27.5% EER even on clean
  audio, while A09/A13 sit near 0.5%.
- **Generalization: H1 falsified, H2 supported.** Generator identity is linearly
  decodable to *ceiling at every one of 25 layers*, so it can't explain
  differential non-transfer; instead **bona-fide proximity predicts the
  leave-one-attack-out gap** (cos-distance vs gap ρ=−0.60, p=0.029). The worst
  case, **A19 (gap +13.9 pp)**, is the bona-closest generator; fine-tuning the
  encoder moves it off the bona manifold and **collapses its gap to +4.6 pp**.

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
