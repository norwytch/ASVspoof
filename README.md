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
Scaffold. `src/degradations.py` and `src/metrics.py` are implemented; all other
modules are typed stubs with `NotImplementedError` bodies and TODOs. See the
implementation-order checklist in the original proposal.
