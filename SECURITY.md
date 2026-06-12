# Security Policy

## Scope and intent

This is a research / portfolio project: a **defensive** evaluation framework for audio
deepfake (spoofing) detection on the public ASVspoof 2021 LA benchmark. It ships no
deployed service or endpoint. Its purpose is to measure *where and why* a pretrained
detector fails — for defenders — not to generate spoofed audio or to help evade detection.
The findings in `report.md` (which attacks evade, where conformal coverage breaks) are
evaluation results on a public benchmark, not an evasion recipe.

## Reporting a vulnerability

If you find a security issue in this code — e.g. unsafe deserialization, a path-traversal
in the data loaders, or a dependency CVE that affects how it runs:

- **Preferred:** open a private report via GitHub → **Security → Report a vulnerability**
  (private vulnerability reporting / Security Advisories).
- For non-sensitive bugs, a regular GitHub issue is fine.

Please don't open a public issue for anything exploitable until it has been addressed.
There are no versioned releases; `main` is the reference.

## In scope

- The Python code under `src/`, `scripts/`, `experiments/`, `tests/`.
- Dependency vulnerabilities that affect how this code runs.

## Out of scope

- **Adversarial robustness of the detector itself** — that a spoofing system can fool the
  countermeasure is the *research subject* of this repo (documented in `report.md`), not a
  vulnerability in this code.
- The pretrained model weights (SSL_Anti-spoofing) and the ASVspoof corpus — third-party
  artifacts; report issues to their maintainers.

## Untrusted-input note (real, and worth knowing)

This code loads model checkpoints and embedding caches with `torch.load(...)` and
`numpy.load(..., allow_pickle=True)`. **Both formats can execute arbitrary code on load.**
Only load `.pth` / `.npz` artifacts from sources you trust — i.e. the linked Hugging Face
repos or caches you generated yourself — never an untrusted file. No secrets or credentials
are committed; large/regenerable artifacts and corpora are gitignored, and weights/embeddings
are hosted externally.
