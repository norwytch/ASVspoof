# Audio Deepfake Detection Under Real-World Degradation

A robustness-evaluation framework for pretrained audio deepfake / spoofing
detectors. Evaluates a pretrained SSL countermeasure against compression,
telephony, additive noise, and streaming-inference degradation on **ASVspoof
2021 LA**, plus four extensions: transcript-conditioned anomaly signals,
TTS-attack profiling, reconstruction-error detection (AeroBlade analog), and
prosody-based detection.

## Motivation
Spoofing detectors trained on clean studio audio routinely fail in production,
where audio arrives compressed, band-limited, noisy, and chunked. This repo
quantifies that gap and characterizes *which* attacks break down under *which*
conditions.

## Approach
- **Baseline:** clean EER / min-DCF / ROC / DET on the ASVspoof 2021 LA eval set.
- **Degradations** (`src/degradations.py`): MP3 (8–128 kbps), telephony
  (300–3400 Hz bandpass + G.711 mu-law), additive noise (0–30 dB SNR), and
  streaming (chunked inference, 500 ms–4 s).
- **Failure analysis:** per-attack-type EER deltas.
- **Extensions:** see `src/{nlp_features,attack_profiling,reconstruction,prosody}.py`.

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
src/          degradations, metrics, model wrapper, evaluate loop, extensions
data/         download instructions + attack_taxonomy.json (corpora gitignored)
results/      figures/, scores, results.csv (gitignored except figures/.gitkeep)
report.md     written analysis (~1200–1500 words)
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
