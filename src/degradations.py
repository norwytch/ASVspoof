"""Real-world audio degradation functions.

Each degradation is a callable taking ``(audio: np.ndarray, sr: int)`` and
returning a degraded waveform at the same sample rate (unless documented
otherwise). Functions are parameterized so they can be swept by ``evaluate.py``.

Convention: ``audio`` is a mono float32 array in roughly [-1, 1].
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


def ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH. Call once at startup."""
    return shutil.which("ffmpeg") is not None


# --------------------------------------------------------------------------- #
# 1. MP3 compression
# --------------------------------------------------------------------------- #
def apply_mp3_compression(audio: np.ndarray, sr: int, bitrate_kbps: int = 32) -> np.ndarray:
    """Encode to MP3 at ``bitrate_kbps`` and decode back to PCM via ffmpeg.

    Bitrates to sweep: 8, 16, 32, 64, 128 kbps.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH; required for MP3 degradation.")

    with tempfile.TemporaryDirectory() as d:
        wav_in = Path(d) / "in.wav"
        mp3 = Path(d) / "enc.mp3"
        wav_out = Path(d) / "out.wav"
        sf.write(wav_in, audio, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "quiet", "-i", str(wav_in),
             "-b:a", f"{bitrate_kbps}k", str(mp3)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "quiet", "-i", str(mp3), str(wav_out)],
            check=True,
        )
        out, out_sr = sf.read(wav_out)
    if out.ndim > 1:
        out = out.mean(axis=1)
    if out_sr != sr:
        out = librosa.resample(out.astype(np.float32), orig_sr=out_sr, target_sr=sr)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Telephony simulation
# --------------------------------------------------------------------------- #
def apply_bandpass(audio: np.ndarray, sr: int, low_hz: int = 300, high_hz: int = 3400) -> np.ndarray:
    """Bandpass filter (300-3400 Hz) simulating telephone frequency response."""
    high_hz = min(high_hz, int(sr / 2) - 1)  # guard against Nyquist
    sos = butter(6, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _mulaw_encode_decode_numpy(x_int16: np.ndarray) -> np.ndarray:
    """G.711 mu-law round-trip on int16 samples, pure numpy (audioop fallback).

    Used when stdlib ``audioop`` is unavailable (Python >= 3.13).
    """
    MU = 255
    x = np.clip(x_int16.astype(np.float32) / 32768.0, -1.0, 1.0)
    # encode
    mag = np.log1p(MU * np.abs(x)) / np.log1p(MU)
    encoded = np.sign(x) * mag
    quantized = np.round((encoded + 1) / 2 * MU)  # 0..255
    # decode
    y = 2 * (quantized / MU) - 1
    decoded = np.sign(y) * (1.0 / MU) * (np.power(1 + MU, np.abs(y)) - 1)
    return decoded.astype(np.float32)


def apply_g711_mulaw(audio: np.ndarray, sr: int) -> np.ndarray:
    """G.711 mu-law: resample to 8 kHz, mu-law encode/decode, resample back.

    Prefers stdlib ``audioop``; falls back to numpy on Python 3.13+.
    """
    audio_8k = librosa.resample(audio, orig_sr=sr, target_sr=8000)
    try:
        import audioop  # deprecated 3.11, removed 3.13

        pcm = (np.clip(audio_8k, -1, 1) * 32767).astype(np.int16).tobytes()
        ulaw = audioop.lin2ulaw(pcm, 2)
        decoded = audioop.ulaw2lin(ulaw, 2)
        out = np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32767.0
    except ImportError:
        x_int16 = (np.clip(audio_8k, -1, 1) * 32767).astype(np.int16)
        out = _mulaw_encode_decode_numpy(x_int16)
    return librosa.resample(out, orig_sr=8000, target_sr=sr).astype(np.float32)


def apply_telephony(audio: np.ndarray, sr: int) -> np.ndarray:
    """Combined telephony condition: bandpass + G.711 mu-law."""
    return apply_g711_mulaw(apply_bandpass(audio, sr), sr)


# --------------------------------------------------------------------------- #
# 3. Additive noise
# --------------------------------------------------------------------------- #
def apply_noise(audio: np.ndarray, sr: int | None = None, snr_db: float = 10.0,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white Gaussian noise at the specified SNR (dB).

    SNR levels to sweep: 0, 5, 10, 20, 30 dB. ``sr`` accepted for a uniform
    degradation signature but unused for white noise.
    """
    rng = rng or np.random.default_rng(42)
    signal_power = float(np.mean(audio ** 2)) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def apply_babble_noise(audio: np.ndarray, sr: int, snr_db: float = 10.0,
                       musan_dir: str | None = None,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Mix a random MUSAN noise clip at the target SNR.

    TODO: load a random clip from ``musan_dir``, loop/trim to len(audio),
    scale to achieve ``snr_db``, and mix. Falls back to white noise if MUSAN
    is not configured.
    """
    if musan_dir is None:
        return apply_noise(audio, sr, snr_db, rng)
    raise NotImplementedError("MUSAN babble mixing not yet implemented.")


# --------------------------------------------------------------------------- #
# 4. Streaming / chunked inference
# --------------------------------------------------------------------------- #
def chunk_audio(audio: np.ndarray, sr: int, chunk_ms: int = 2000,
                overlap_ms: int = 0) -> list[np.ndarray]:
    """Split audio into (possibly overlapping) fixed-size chunks.

    Note: inference + aggregation lives in ``model.py`` / ``evaluate.py``; this
    helper only does the windowing so it is testable in isolation.
    """
    chunk_samples = int(sr * chunk_ms / 1000)
    hop = chunk_samples - int(sr * overlap_ms / 1000)
    if hop <= 0:
        raise ValueError("overlap_ms must be smaller than chunk_ms")
    if len(audio) < chunk_samples:
        return [audio]
    return [audio[s:s + chunk_samples]
            for s in range(0, len(audio) - chunk_samples + 1, hop)]


# --------------------------------------------------------------------------- #
# Registry — used by evaluate.py to sweep conditions
# --------------------------------------------------------------------------- #
DEGRADATIONS = {
    "clean":     [{}],
    "mp3":       [{"bitrate_kbps": b} for b in (128, 64, 32, 16, 8)],
    "noise":     [{"snr_db": s} for s in (30, 20, 10, 5, 0)],
    "telephony": [{"mode": "bandpass"}, {"mode": "g711"}],
    "streaming": [
        {"chunk_ms": None},          # full utterance
        {"chunk_ms": 4000, "overlap_ms": 0},
        {"chunk_ms": 2000, "overlap_ms": 0},
        {"chunk_ms": 2000, "overlap_ms": 200},
        {"chunk_ms": 500, "overlap_ms": 0},
    ],
}
