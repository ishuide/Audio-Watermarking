"""End-to-end pipeline tests: embed → detect round-trip."""

import numpy as np
import pytest

from audio_warp.config import WatermarkConfig
from audio_warp import crypto, embedder, detector


CONFIG = WatermarkConfig()
OWNER_ID = b"\x01\x02\x03\x04\x05\x06\x07\x08"


def _make_test_audio(duration: float = 7.0) -> np.ndarray:
    """Synthesise broadband test audio (tones + noise).

    Frequency-domain spread-spectrum needs spectral energy across the
    embedding band, so we mix tones with shaped noise.
    """
    rng = np.random.default_rng(42)
    n = int(CONFIG.sample_rate * duration)
    t = np.linspace(0, duration, n, dtype=np.float64)
    tones = (
        0.25 * np.sin(2 * np.pi * 440 * t)
        + 0.20 * np.sin(2 * np.pi * 1000 * t)
        + 0.15 * np.sin(2 * np.pi * 2500 * t)
        + 0.10 * np.sin(2 * np.pi * 5000 * t)
    )
    noise = 0.30 * rng.normal(0, 1, n)
    audio = (tones + noise).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-10
    return audio


class TestPipeline:
    def test_embed_detect_clean(self):
        """Watermark should be detected on unmodified watermarked audio."""
        audio = _make_test_audio()
        priv, pub = crypto.generate_keypair()

        watermarked = embedder.embed(audio, OWNER_ID, priv, CONFIG)

        # Audio should be similar to original
        assert watermarked.shape == audio.shape
        snr = 10 * np.log10(
            np.mean(audio ** 2) / (np.mean((audio - watermarked) ** 2) + 1e-20)
        )
        assert snr > 10, f"SNR too low: {snr:.1f} dB"

        result = detector.detect(watermarked, OWNER_ID, pub, CONFIG)
        assert result.found, f"Detection failed: {result.reason}"
        assert result.owner_id == OWNER_ID

    def test_wrong_key_fails(self):
        """Watermark verified with wrong public key should fail."""
        audio = _make_test_audio()
        priv, _ = crypto.generate_keypair()
        _, wrong_pub = crypto.generate_keypair()

        watermarked = embedder.embed(audio, OWNER_ID, priv, CONFIG)
        result = detector.detect(watermarked, OWNER_ID, wrong_pub, CONFIG)
        assert not result.found

    def test_wrong_owner_fails(self):
        """Detection with wrong owner_id should fail (PN mismatch)."""
        audio = _make_test_audio()
        priv, pub = crypto.generate_keypair()
        watermarked = embedder.embed(audio, OWNER_ID, priv, CONFIG)

        wrong_owner = b"\xff" * 8
        result = detector.detect(watermarked, wrong_owner, pub, CONFIG)
        assert not result.found

    def test_unwatermarked_audio_fails(self):
        """Clean audio should not produce a valid watermark detection."""
        audio = _make_test_audio()
        _, pub = crypto.generate_keypair()
        result = detector.detect(audio, OWNER_ID, pub, CONFIG)
        assert not result.found

    def test_audio_too_short_raises(self):
        """Embedding into audio shorter than minimum should raise."""
        short = np.zeros(1000, dtype=np.float32)
        priv, _ = crypto.generate_keypair()
        with pytest.raises(ValueError, match="Audio too short"):
            embedder.embed(short, OWNER_ID, priv, CONFIG)
