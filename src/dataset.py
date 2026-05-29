"""ASVspoof protocol parsing and evaluation-subset selection.

The ASVspoof 2021 LA protocol files are space-delimited. The trial metadata
(CM key) columns are roughly:

    speaker_id  utterance_id  codec  ...  attack_id  label  ...

where ``label`` is "bonafide" or "spoof" and ``attack_id`` is one of A01..A19
(or "-" for bona fide). Verify column order against the keys you downloaded
before trusting this parser.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trial:
    utt_id: str
    path: Path
    label: int          # 1 = bona fide, 0 = spoof
    attack_id: str      # 'A01'..'A19' or '-' for bona fide


def parse_protocol(protocol_path: str | Path, flac_dir: str | Path) -> pd.DataFrame:
    """Parse an ASVspoof CM protocol/key file into a DataFrame of trials.

    Returns columns: utt_id, path, label (1=bonafide), attack_id.
    TODO: confirm column indices against the specific key file format.
    """
    raise NotImplementedError


def select_eval_subset(trials: pd.DataFrame, n: int = 5000, *,
                       full: bool = False, seed: int = 42) -> pd.DataFrame:
    """Stratified subset for cheap iteration; pass-through when ``full``.

    Stratifies by (label, attack_id) so every attack type and the bona fide
    pool are represented. See README 'Compute' note for why we default to a
    subset rather than the full ~181k-utterance eval set.
    """
    if full:
        return trials
    rng = np.random.default_rng(seed)
    # TODO: proportional stratified sample across (label, attack_id) groups.
    raise NotImplementedError
