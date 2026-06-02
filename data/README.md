# Data

The corpora are **not** committed (see `.gitignore`). Download them here.

> **Derived artifacts** (XLS-R embeddings cache, SSL_Anti-spoofing weights) live on
> Hugging Face, not in git — see [Artifacts](../README.md#artifacts-hugging-face)
> in the top-level README for the repos and download commands.

## ASVspoof 2021 LA (primary)

- Challenge page: https://www.asvspoof.org/index2021.html
- Protocol files & baselines: https://github.com/asvspoof-challenge/2021
- Use the **eval** split for all degradation experiments.

Expected layout after download:

```
data/asvspoof2021_LA/
├── flac/                         # eval .flac files (~181k files / 165k scored trials, ~7.8 GB)
└── keys/
    └── CM/trial_metadata.txt     # space-delimited; gives label + attack_id per utt
```

> **Compute note.** The full LA eval set is large. `src/dataset.py:select_eval_subset`
> defaults to a stratified ~5k-utterance subset for the sweeps; pass `--full`
> to `evaluate.py` to run the entire set (reserve that for the headline clean
> baseline). Whisper + HuBERT passes over the full set are many GPU-hours.

## ASVspoof 5 (secondary, zero-shot)

- https://www.asvspoof.org/  ·  download: https://zenodo.org/records/14498691
- Used only for the zero-shot generalization experiments (Extension 3) and a
  supplementary clean-baseline gap measurement.

## MUSAN (optional, for babble noise)

- https://www.openslr.org/17/  → `data/musan/`

## attack_taxonomy.json

`attack_taxonomy.json` maps each attack id (A01–A19) to its generative
mechanism, transcribed from Table 1 / Section 3 of the ASVspoof 2019 database
paper (arXiv:1911.01601). Categories: `neural_tts`, `concatenative_tts`,
`voice_conversion`, `hybrid_tts_vc`, with the vocoder kept as a separate
`waveform_generator` field.

Note: the LA **eval** set (reused by ASVspoof 2021) contains only **A07–A19**;
A01–A06 are train/dev. The eval category balance is skewed — 6 neural_tts, 3
voice_conversion, 3 hybrid_tts_vc, but only **1 concatenative** (A16) — so
per-category EER for concatenative rests on a single system; interpret with care.
