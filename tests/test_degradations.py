import numpy as np
import pytest

from src import degradations as deg

SR = 16000


def tone(f=220, dur=1.0, sr=SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * f * t)).astype("float32")


def test_noise_hits_target_snr():
    a = tone()
    n = deg.apply_noise(a, snr_db=10)
    meas = 10 * np.log10(np.mean(a ** 2) / np.mean((n - a) ** 2))
    assert abs(meas - 10) < 0.5


def test_bandpass_preserves_shape():
    a = tone()
    b = deg.apply_bandpass(a, SR)
    assert b.shape == a.shape


def test_g711_roundtrip_length():
    a = tone()
    g = deg.apply_g711_mulaw(a, SR)
    assert abs(len(g) - len(a)) <= 2


def test_chunk_audio_count():
    a = tone(dur=1.0)
    assert len(deg.chunk_audio(a, SR, chunk_ms=500, overlap_ms=0)) == 2


def test_band_mask_targeted_vs_control():
    a = tone(f=6000)
    e0 = np.sum(a ** 2)
    targeted = np.sum(deg.apply_band_mask(a, SR, 5000, 7000) ** 2) / e0
    lo, hi = deg.matched_control_band(5000, 7000, SR)
    control = np.sum(deg.apply_band_mask(a, SR, lo, hi) ** 2) / e0
    assert targeted < 0.10           # masking the 6 kHz band removes the tone
    assert control > 0.80            # the matched control band leaves it intact


def test_matched_control_band_same_width():
    lo, hi = deg.matched_control_band(5000, 7000, SR)
    assert abs((hi - lo) - 2000) < 1e-6


def test_equalize_energy_rms():
    eq = deg.equalize_energy(tone(), target_rms=0.05)
    assert abs(float(np.sqrt(np.mean(eq ** 2))) - 0.05) < 1e-3


def test_fix_duration_length():
    assert len(deg.fix_duration(tone(dur=1.0), SR, 4.0)) == SR * 4


def test_trim_silence_shortens():
    sig = np.concatenate([np.zeros(4000, "float32"), tone(dur=1.0), np.zeros(2000, "float32")])
    assert len(deg.trim_silence(sig, SR)) < len(sig)


@pytest.mark.skipif(not deg.ffmpeg_available(), reason="ffmpeg not installed")
def test_mp3_roundtrip():
    m = deg.apply_mp3_compression(tone(), SR, bitrate_kbps=32)
    assert m.dtype == np.float32 and len(m) > 0
