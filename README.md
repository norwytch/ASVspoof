# Audio Deepfake Detection in the Real World: Robustness & Generalization

A production audio-deepfake detector has to survive two things its training set
never showed it: **degraded channels** and **unseen attacks**. This repo studies
both — empirically *where* detectors break, and representationally *why* — on
**ASVspoof 2021 LA** with a pretrained SSL countermeasure (`lab260/AASIST3`).

> **Part 1 — Where does it break?** A robustness-evaluation framework measuring
> detector degradation under compression, telephony, additive noise, and
> streaming inference, with per-attack failure analysis and four detection
> extensions. *(Implemented and unit-tested; runnable.)*
>
> **Part 2 — Why does it break?** A falsification-driven representational study
> of generalization failure: does linearly probe-recoverable *generator identity*
> in the frozen embedding predict *leave-one-attack-out* non-transfer, and does a
> targeted vocoder-artifact band-mask causally improve it? *(Design complete, all
> 30 citations literature-verified — see [research-design.md](research-design.md).)*

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
*(fill in after running — 3–4 bullets with real numbers)*

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg            # system dependency for MP3 codec
```
- Use **Python ≤ 3.12** (the G.711 path uses stdlib `audioop`, removed in 3.13;
  a numpy fallback exists but the stdlib path is preferred).
- Download data per [data/README.md](data/README.md).

## Reproduce
```bash
# Clean baseline + full degradation sweep on a stratified 5k subset
python -m src.evaluate --protocol data/asvspoof2021_LA/keys/CM/trial_metadata.txt \
                       --flac-dir data/asvspoof2021_LA/flac --subset 5000

# Headline clean baseline on the entire eval set
python -m src.evaluate --protocol ... --flac-dir ... --full

# Extensions (run after the main sweep)
python -m src.transcribe        # Whisper -> results/transcripts.jsonl (slow, resumable)
# ... then attack_profiling / reconstruction / prosody analyses
```

## Results
*(embed results/results.csv table + key plots from results/figures/ here)*

## Discussion
*(which conditions degraded most, which attacks were most affected, mitigations)*

## Repository Layout
```
src/                  degradations, metrics, model wrapper, evaluate loop, extensions
data/                 download instructions + attack_taxonomy.json (corpora gitignored)
results/              figures/, scores, results.csv (gitignored except figures/.gitkeep)
report.md             Part 1 written analysis (~1200–1500 words)
research-design.md    Part 2 — the generalization/representational study design + verified refs
```

## Status
**All modules implemented and unit-tested** (synthetic data; heavy deps —
torch/transformers/whisper/parselmouth/sentence-transformers — are lazily
imported so the rest of the framework imports and tests without them).

Core pipeline:
- `dataset.py` — protocol parser (2021 + 2019 layouts) + stratified subset
- `degradations.py` — MP3, telephony, noise, streaming (+ numpy mu-law fallback)
- `model.py` — `SpoofDetector` wrapper for `lab260/AASIST3`
- `evaluate.py` — full sweep loop with score caching + per-attack breakdown
- `metrics.py` — EER, min-DCF, AUC, per-attack EER
- `visualize.py` — ROC / DET / EER-sweep / heatmap / score-hist

Extensions:
- `transcribe.py` — resumable Whisper → JSONL
- `nlp_features.py` (Ext 1) — GPT-2 perplexity, repetition rate, ASR confidence + correlation
- `attack_profiling.py` (Ext 2) — taxonomy, transcript clustering, EER by attack category
- `reconstruction.py` (Ext 3) — frozen HuBERT + decoder trained on bona fide only
- `prosody.py` (Ext 4) — F0/timing/energy features, CMU-stress correlation, LR classifier

Not yet executed end-to-end (need the real weights / corpus / GPU): the AASIST3
model load, decoder training, Whisper/GPT-2 passes. The orchestration around
them is tested with mocks + synthetic audio.

Note: `model.py` defaults to `lab260/AASIST3` — the proposal's
`ntt-hilab-gensp/ssl_spoof` returns HTTP 401 (gated/unavailable). AASIST3 loads
via its own custom code, which must be on the import path.

`data/attack_taxonomy.json` is filled in from the ASVspoof 2019 database paper
(A01–A19). Only A07–A19 appear in the eval set.

**Part 2 status:** implemented and unit-tested (synthetic embeddings; heavy deps
lazy). Design + verified citations in [research-design.md](research-design.md).
- `embeddings.py` — frozen XLS-R per-layer embedding cache (the one-time cost)
- `probes.py` — linear probes with control-task **selectivity** (Hewitt & Liang)
- `dataset.leave_one_attack_out` — LOAO splits with disjoint bona-fide pool
- `degradations.{trim_silence, equalize_energy, fix_duration, apply_band_mask, …}` — shortcut ablations + the band-mask intervention
- `metrics.{bootstrap_eer_ci, seed_variance, spearman_with_ci}` — §5 rigor
- `experiments/loao.py` — the runner: per-family non-transfer gap + the H1 Spearman test

The `run_loao` core was validated on synthetic embeddings with a *planted* effect
(distinct families → higher selectivity **and** higher LOAO gap → ρ≈0.8 recovered).
Not yet run for real (needs the XLS-R embedding cache over downloaded audio):
```bash
python -m src.embeddings   ...                       # cache XLS-R features (needs torch+data)
python -m experiments.loao --emb-dir results/embeddings --layer 9 --seeds 5
```
