"""Extension 4: Prosody-based artifact detection.

Hypothesis: neural TTS produces unnaturally smooth prosody (low F0 variability,
uniform timing, compressed energy dynamics). Extract low-level prosodic
features and test whether they discriminate bona fide from spoof, independent
of the SSL detector.

Also: the "poetry experiment" — does the F0 contour align with expected lexical
stress (CMU dict) more for bona fide than spoof?

``parselmouth`` (Praat) is imported lazily so the timing / stress / classifier
code can be tested without it installed.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

PROSODY_FEATURE_COLS = [
    "f0_mean", "f0_std", "f0_range", "f0_slope_mean", "f0_slope_std",
    "energy_std", "energy_range", "voiced_fraction",
    "ioi_mean", "ioi_std", "ioi_cv", "speaking_rate",
]


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_prosody_features(audio_path: str) -> dict:
    """F0 / energy / voicing statistics via parselmouth (Praat).

    Returns f0_mean, f0_std, f0_range, f0_slope_mean, f0_slope_std,
    energy_std, energy_range, voiced_fraction. Returns the raw voiced F0 track
    under ``_f0_voiced`` for the stress-correlation step (not a summary stat).
    """
    import parselmouth

    snd = parselmouth.Sound(audio_path)
    f0 = snd.to_pitch().selected_array["frequency"]
    f0_voiced = f0[f0 > 0]
    energy = snd.to_intensity().values.T.flatten()

    def _stat(fn, arr, n_min=1, default=0.0):
        return float(fn(arr)) if len(arr) >= n_min else default

    return {
        "f0_mean": _stat(np.mean, f0_voiced),
        "f0_std": _stat(np.std, f0_voiced),
        "f0_range": _stat(np.ptp, f0_voiced),
        "f0_slope_mean": _stat(np.mean, np.diff(f0_voiced), n_min=1),
        "f0_slope_std": _stat(np.std, np.diff(f0_voiced), n_min=1),
        "energy_std": _stat(np.std, energy),
        "energy_range": _stat(np.ptp, energy),
        "voiced_fraction": len(f0_voiced) / max(len(f0), 1),
        "_f0_voiced": f0_voiced,
    }


def extract_timing_features(audio: np.ndarray, sr: int) -> dict:
    """Inter-onset-interval timing stats via librosa onset detection."""
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    ioi = np.diff(onset_times)
    return {
        "ioi_mean": float(np.mean(ioi)) if len(ioi) else 0.0,
        "ioi_std": float(np.std(ioi)) if len(ioi) else 0.0,
        "ioi_cv": float(np.std(ioi) / np.mean(ioi)) if len(ioi) and np.mean(ioi) > 0 else 0.0,
        "speaking_rate": len(onset_frames) / (len(audio) / sr) if len(audio) else 0.0,
    }


def build_feature_table(trials: pd.DataFrame,
                        out: str | Path = "results/prosody_features.csv") -> pd.DataFrame:
    """Extract the full prosody + timing feature table over a trials DataFrame.

    Requires ``trials`` with columns utt_id, path, label (and optionally
    attack_id). Saves to ``out`` and returns the table (without the raw F0 track).
    """
    import soundfile as sf

    rows = []
    for row in trials.itertuples(index=False):
        feats = extract_prosody_features(row.path)
        feats.pop("_f0_voiced", None)
        audio, sr = sf.read(row.path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        feats.update(extract_timing_features(audio, sr))
        feats.update(utt_id=row.utt_id, label=int(row.label),
                     attack_id=getattr(row, "attack_id", "-"))
        rows.append(feats)
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


# --------------------------------------------------------------------------- #
# Stress / F0 "poetry experiment"
# --------------------------------------------------------------------------- #
_CMU = None


def _cmudict():
    global _CMU
    if _CMU is None:
        import nltk
        try:
            from nltk.corpus import cmudict
            _CMU = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict
            _CMU = cmudict.dict()
    return _CMU


def get_stress_pattern(text: str) -> tuple[list[int], float]:
    """Expected per-syllable stress from CMU dict (1=stressed, 0=unstressed).

    Returns ``(stress_pattern, coverage)`` where coverage is the fraction of
    words found in the dictionary. OOV words are skipped, not errored.
    """
    cmu = _cmudict()
    words = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return [], 0.0
    stress, found = [], 0
    for word in words:
        if word in cmu:
            found += 1
            for phone in cmu[word][0]:
                if phone[-1].isdigit():
                    stress.append(1 if int(phone[-1]) > 0 else 0)
    return stress, found / len(words)


def stress_f0_correlation(f0_voiced: np.ndarray, stress_pattern: list[int]) -> float | None:
    """Correlate syllable-chunked F0 with the expected stress pattern.

    Chunks the voiced F0 track into ``len(stress_pattern)`` segments and
    correlates segment means with the binary stress sequence. Returns None when
    there is insufficient data or no variance to correlate.
    """
    n = len(stress_pattern)
    if n < 2 or len(f0_voiced) < n:
        return None
    chunk = len(f0_voiced) // n
    f0_chunked = np.array([f0_voiced[i * chunk:(i + 1) * chunk].mean() for i in range(n)])
    if np.std(f0_chunked) == 0 or np.std(stress_pattern) == 0:
        return None
    return float(np.corrcoef(f0_chunked, stress_pattern)[0, 1])


# --------------------------------------------------------------------------- #
# Discriminative analysis
# --------------------------------------------------------------------------- #
def prosody_eer(features: str | Path | pd.DataFrame = "results/prosody_features.csv",
                *, seed: int = 42) -> dict:
    """Standardized logistic regression on prosody features; cross-validated EER.

    Returns ``{eer, auc, coefficients, feature_importance}``. Features are
    standardized (F0 in Hz and IOI in seconds are on very different scales).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .metrics import compute_auc, compute_eer

    df = pd.read_csv(features) if not isinstance(features, pd.DataFrame) else features
    cols = [c for c in PROSODY_FEATURE_COLS if c in df.columns]
    X = df[cols].to_numpy()
    y = df["label"].to_numpy()  # 1 = bona fide

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, random_state=seed))
    # P(bona fide) as the bona-fide score (higher == more genuine)
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    eer, _ = compute_eer(y, proba)

    clf.fit(X, y)
    coefs = clf.named_steps["logisticregression"].coef_[0]
    importance = sorted(zip(cols, np.abs(coefs)), key=lambda t: -t[1])
    return {
        "eer": eer,
        "auc": compute_auc(y, proba),
        "coefficients": dict(zip(cols, coefs.tolist())),
        "feature_importance": importance,
    }
