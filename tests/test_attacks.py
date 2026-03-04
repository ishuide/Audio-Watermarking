"""Robustness tests: watermark survival under signal-processing attacks."""

import numpy as np
import pytest

from audio_warp.config import WatermarkConfig
from audio_warp import crypto, embedder, detector, attacks


CONFIG = WatermarkConfig()
OWNER_ID = b"\x01\x02\x03\x04\x05\x06\x07\x08"


def _watermarked_audio() -> tuple[np.ndarray, object]:
    """Return (watermarked_audio, public_key) with broadband test signal."""
    rng = np.random.default_rng(42)
    n = int(CONFIG.sample_rate * 7)
    t = np.linspace(0, 7.0, n, dtype=np.float64)
    tones = (
        0.25 * np.sin(2 * np.pi * 440 * t)
        + 0.20 * np.sin(2 * np.pi * 1000 * t)
        + 0.15 * np.sin(2 * np.pi * 2500 * t)
        + 0.10 * np.sin(2 * np.pi * 5000 * t)
    )
    noise = 0.30 * rng.normal(0, 1, n)
    audio = (tones + noise).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-10
    priv, pub = crypto.generate_keypair()
    wm = embedder.embed(audio, OWNER_ID, priv, CONFIG)
    return wm, pub


class TestAttacks:
    """Each test applies an attack and checks whether detection still passes.

    Mild attacks are expected to pass; severe attacks may legitimately fail
    so we mark those as xfail.
    """

    def test_noise_30db(self):
        wm, pub = _watermarked_audio()
        attacked = attacks.add_noise(wm, snr_db=30.0)
        result = detector.detect(attacked, OWNER_ID, pub, CONFIG)
        assert result.found, f"Failed under 30 dB noise: {result.reason}"

    def test_lowpass_8khz(self):
        wm, pub = _watermarked_audio()
        attacked = attacks.lowpass_filter(wm, cutoff_hz=8000.0)
        result = detector.detect(attacked, OWNER_ID, pub, CONFIG)
        assert result.found, f"Failed under 8 kHz LP: {result.reason}"

    def test_amplitude_scale(self):
        wm, pub = _watermarked_audio()
        attacked = attacks.amplitude_scale(wm, factor=0.5)
        result = detector.detect(attacked, OWNER_ID, pub, CONFIG)
        assert result.found, f"Failed under 0.5x scale: {result.reason}"

    def test_resample_22k(self):
        wm, pub = _watermarked_audio()
        attacked = attacks.resample_attack(wm, intermediate_sr=22050)
        result = detector.detect(attacked, OWNER_ID, pub, CONFIG)
        assert result.found

    def test_compression_sim(self):
        wm, pub = _watermarked_audio()
        attacked = attacks.lossy_compression_sim(wm, bit_depth=12)
        result = detector.detect(attacked, OWNER_ID, pub, CONFIG)
        assert result.found

    def test_apply_attack_registry(self):
        """Smoke test: apply_attack dispatches without errors."""
        wm, _ = _watermarked_audio()
        for name in attacks.ATTACK_REGISTRY:
            out = attacks.apply_attack(name, wm)
            assert out.shape == wm.shape
