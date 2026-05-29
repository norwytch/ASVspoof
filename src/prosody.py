"""Extension 4: Prosody-based artifact detection.

Hypothesis: neural TTS produces unnaturally smooth prosody (low F0 variability,
uniform timing, compressed energy dynamics). Extract low-level prosodic
features and test whether they discriminate bona fide from spoof, independent
of the SSL detector.

Also: the "poetry experiment" — does the F0 contour align with expected lexical
stress (CMU dict) more for bona fide than spoof?
"""
from __future__ import annotations

import librosa
import numpy as np


def extract_prosody_features(audio_path: str) -> dict:
    """F0 / energy / voicing statistics via parselmouth (Praat).

    Returns: f0_mean, f0_std, f0_range, f0_slope_mean, f0_slope_std,
    energy_std, energy_range, voiced_fraction.
    TODO: parselmouth.Sound(audio_path) -> to_pitch / to_intensity.
    """
    raise NotImplementedError


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


def get_stress_pattern(text: str) -> list[int]:
    """Expected stress pattern from the CMU dict (1=stressed, 0=unstressed).

    Skip words not in the dictionary rather than erroring; callers should track
    coverage. TODO: nltk cmudict lookup over phones with digit stress markers.
    """
    raise NotImplementedError


def stress_f0_correlation(f0_voiced: np.ndarray, stress_pattern: list[int]) -> float | None:
    """Correlate syllable-chunked F0 with the expected stress pattern."""
    raise NotImplementedError


def prosody_eer(features_csv: str = "results/prosody_features.csv") -> float:
    """Train standardized logistic regression on prosody features; return EER.

    TODO: StandardScaler + LogisticRegression; F0 (Hz) and IOI (s) need scaling.
    """
    raise NotImplementedError
