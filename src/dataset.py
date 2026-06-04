"""ASVspoof protocol parsing and evaluation-subset selection.

ASVspoof 2021 LA key/metadata files (``keys/CM/trial_metadata.txt``) are
space-delimited. The documented column layout (confirmed against the official
eval package) is::

    speaker_id  utt_id  codec  transmission  attack_id  key  trim  phase
    LA_0009     LA_E_9332881  alaw  ita_tx   A07        spoof  notrim  eval

- ``attack_id``  : A07..A19 for spoof, "-" for bona fide
- ``key``        : "bonafide" | "spoof"
- ``phase``      : "eval" | "progress" | "hidden" (official EER scores "eval" only)

The ASVspoof 2019 LA protocol is shorter::

    speaker_id  utt_id  -  attack_id  key

To survive both layouts (and minor column drift), the parser does not hardcode
indices for ``key``/``attack_id``: it takes ``utt_id`` from column 2 and then
*scans* each row for the bona-fide/spoof token and the A## attack token.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_ATTACK_RE = re.compile(r"^A\d{2}$")
_LABELS = {"bonafide": 1, "spoof": 0}


def parse_protocol(protocol_path: str | Path, flac_dir: str | Path,
                   *, phase: str | None = "eval", ext: str = ".flac") -> pd.DataFrame:
    """Parse an ASVspoof CM protocol/key file into a DataFrame of trials.

    Returns columns: ``utt_id, path, label (1=bonafide), attack_id, codec, phase``.

    Args:
        phase: if set, keep only rows whose row contains this phase token
            (e.g. "eval"). Pass ``None`` to keep all rows (2019 protocols have
            no phase column, in which case this filter is skipped automatically).
    """
    flac_dir = Path(flac_dir)
    rows = []
    with open(protocol_path) as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if len(parts) < 4:
                continue  # blank / malformed

            utt_id = parts[1]

            # label: scan for the bonafide/spoof token
            label = next((_LABELS[p] for p in parts if p in _LABELS), None)
            if label is None:
                raise ValueError(
                    f"{protocol_path}:{lineno}: no bonafide/spoof token in {parts!r}"
                )

            # attack id: A## token, else '-' (bona fide)
            attack_id = next((p for p in parts if _ATTACK_RE.match(p)), "-")

            # phase filter (only if a phase token system is present in the row).
            # NB: the 2021 LA key uses the bare token "hidden" (not "hidden_track");
            # matching the wrong token silently let the 16,926 hidden/only_speech
            # trials leak into the eval pool, inflating EER (0.82% -> 8.53%).
            row_phase = next((p for p in ("eval", "progress", "hidden")
                              if p in parts), None)
            if phase is not None and row_phase is not None and row_phase != phase:
                continue

            # codec is column 3 in the 2021 layout; '-' if absent
            codec = parts[2] if len(parts) >= 6 else "-"

            rows.append({
                "utt_id": utt_id,
                "path": str(flac_dir / f"{utt_id}{ext}"),
                "label": label,
                "attack_id": attack_id,
                "codec": codec,
                "phase": row_phase or "",
            })

    if not rows:
        raise ValueError(f"No trials parsed from {protocol_path} (phase={phase!r}).")
    df = pd.DataFrame(rows)
    df.attrs["n_bonafide"] = int((df.label == 1).sum())
    df.attrs["n_spoof"] = int((df.label == 0).sum())
    return df


def select_eval_subset(trials: pd.DataFrame, n: int = 5000, *,
                       full: bool = False, seed: int = 42,
                       min_per_group: int = 50) -> pd.DataFrame:
    """Stratified subset for cheap iteration; pass-through when ``full``.

    Stratifies by (label, attack_id) so every attack type and the bona-fide pool
    are represented. Sampling is proportional to each group's size but every
    group keeps at least ``min_per_group`` trials (capped at the group size) so
    rare attacks don't vanish. See README 'Compute' note for rationale.
    """
    if full or len(trials) <= n:
        return trials.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    groups = list(trials.groupby(["label", "attack_id"], sort=True))
    total = len(trials)

    # First pass: proportional allocation with a per-group floor.
    alloc: list[int] = []
    for _, g in groups:
        target = max(min_per_group, round(n * len(g) / total))
        alloc.append(min(target, len(g)))

    # Rescale toward n if the floors pushed us over/under budget.
    cur = sum(alloc)
    if cur > n:
        scale = n / cur
        alloc = [min(len(g), max(min_per_group if len(g) >= min_per_group else len(g),
                                 int(a * scale)))
                 for a, (_, g) in zip(alloc, groups)]

    picks = []
    for a, (_, g) in zip(alloc, groups):
        idx = rng.choice(g.index.to_numpy(), size=min(a, len(g)), replace=False)
        picks.append(trials.loc[idx])

    out = pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)
    out.attrs["n_bonafide"] = int((out.label == 1).sum())
    out.attrs["n_spoof"] = int((out.label == 0).sum())
    return out


def load_trials(protocol_path: str | Path, flac_dir: str | Path, *,
                n: int = 5000, full: bool = False, phase: str | None = "eval",
                seed: int = 42) -> pd.DataFrame:
    """Convenience: parse + (optionally) subset in one call."""
    return select_eval_subset(
        parse_protocol(protocol_path, flac_dir, phase=phase),
        n=n, full=full, seed=seed,
    )


def attach_family(trials: pd.DataFrame, taxonomy: dict) -> pd.DataFrame:
    """Add a 'family' column mapping attack_id -> generative-mechanism category.

    Bona fide rows ('-') map to 'bonafide'. Uses the same taxonomy as
    attack_profiling so Part 1 and Part 2 share one grouping.
    """
    from .attack_profiling import category_of

    out = trials.copy()
    out["family"] = out["attack_id"].map(lambda a: category_of(a, taxonomy))
    return out


def leave_one_attack_out(trials: pd.DataFrame, taxonomy: dict, *,
                         seed: int = 42, bonafide_test_frac: float = 0.3):
    """Yield ``(held_family, train_df, test_df)`` for each spoof family (Part 2 §3.2).

    For each spoof family *f*: train = (bona-fide train split) + all spoof
    families EXCEPT *f*; test = (bona-fide test split) + family *f*. The bona-fide
    pool is split once into disjoint train/test halves (deterministic by seed) so
    the held-out evaluation has no bona-fide leakage.
    """
    df = attach_family(trials, taxonomy)
    rng = np.random.default_rng(seed)

    bona = df[df.label == 1]
    n_test = int(round(len(bona) * bonafide_test_frac))
    test_idx = set(rng.choice(bona.index.to_numpy(), size=n_test, replace=False).tolist())
    bona_test = bona.loc[bona.index.isin(test_idx)]
    bona_train = bona.loc[~bona.index.isin(test_idx)]

    spoof = df[df.label == 0]
    families = sorted(f for f in spoof.family.unique() if f != "bonafide")
    for f in families:
        spoof_train = spoof[spoof.family != f]
        spoof_test = spoof[spoof.family == f]
        train = pd.concat([bona_train, spoof_train]).reset_index(drop=True)
        test = pd.concat([bona_test, spoof_test]).reset_index(drop=True)
        yield f, train, test
