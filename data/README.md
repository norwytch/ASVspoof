# Data

The corpora are **not** committed (see `.gitignore`). Download them here.

## ASVspoof 2021 LA (primary)

- Challenge page: https://www.asvspoof.org/index2021.html
- Protocol files & baselines: https://github.com/asvspoof-challenge/2021
- Use the **eval** split for all degradation experiments.

Expected layout after download:

```
data/asvspoof2021_LA/
├── flac/                         # eval .flac files (~181k utterances, ~25GB+)
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

`attack_taxonomy.json` maps each attack id to its generative mechanism. The
entries are **placeholders** — fill them in by reading the ASVspoof 2021 / 2019
evaluation-plan papers (arXiv:2109.00535, arXiv:2210.02437). This cannot be
inferred from the audio.
