"""Unit tests for payload, ECC, crypto, and spread-spectrum modules."""

import numpy as np
import pytest

from audio_warp.config import WatermarkConfig
from audio_warp import crypto, payload, ecc, spread_spectrum


CONFIG = WatermarkConfig()
OWNER_ID = b"\x01\x02\x03\x04\x05\x06\x07\x08"


# ---- Config ----------------------------------------------------------------

class TestConfig:
    def test_payload_size(self):
        assert CONFIG.payload_size == 109

    def test_encoded_size(self):
        assert CONFIG.encoded_size == 109 + CONFIG.rs_nsym

    def test_bits_per_frame_positive(self):
        assert CONFIG.bits_per_frame > 0

    def test_min_audio_seconds(self):
        assert CONFIG.min_audio_seconds < 10.0  # should be a few seconds


# ---- Crypto ----------------------------------------------------------------

class TestCrypto:
    def test_keypair_generation(self):
        priv, pub = crypto.generate_keypair()
        assert priv is not None
        assert pub is not None

    def test_sign_verify(self):
        priv, pub = crypto.generate_keypair()
        data = b"test message"
        sig = crypto.sign(priv, data)
        assert len(sig) == 64
        assert crypto.verify(pub, data, sig)

    def test_verify_wrong_data(self):
        priv, pub = crypto.generate_keypair()
        sig = crypto.sign(priv, b"original")
        assert not crypto.verify(pub, b"tampered", sig)

    def test_audio_hash_deterministic(self):
        audio = np.random.default_rng(0).random(1000).astype(np.float32)
        h1 = crypto.audio_hash(audio)
        h2 = crypto.audio_hash(audio)
        assert h1 == h2
        assert len(h1) == 32

    def test_save_load_keys(self, tmp_path):
        priv, pub = crypto.generate_keypair()
        priv_path = str(tmp_path / "priv.pem")
        pub_path = str(tmp_path / "pub.pem")
        crypto.save_private_key(priv, priv_path)
        crypto.save_public_key(pub, pub_path)

        priv2 = crypto.load_private_key(priv_path)
        pub2 = crypto.load_public_key(pub_path)

        data = b"round-trip"
        sig = crypto.sign(priv2, data)
        assert crypto.verify(pub2, data, sig)


# ---- Payload ---------------------------------------------------------------

class TestPayload:
    def test_build_parse_roundtrip(self):
        audio_h = b"\xaa" * 32
        sig = b"\xbb" * 64
        raw = payload.build_payload(OWNER_ID, audio_h, sig, CONFIG)
        assert len(raw) == CONFIG.payload_size

        parsed = payload.parse_payload(raw, CONFIG)
        assert parsed["magic"] == CONFIG.magic
        assert parsed["version"] == CONFIG.version
        assert parsed["owner_id"] == OWNER_ID
        assert parsed["audio_hash"] == audio_h
        assert parsed["signature"] == sig

    def test_bad_magic_raises(self):
        raw = b"XXXX" + b"\x01" + OWNER_ID + b"\x00" * 32 + b"\x00" * 64
        with pytest.raises(ValueError, match="Bad magic"):
            payload.parse_payload(raw, CONFIG)

    def test_signable_data(self):
        sd = payload.signable_data(b"AWMK", 1, OWNER_ID, b"\xcc" * 32)
        assert sd.startswith(b"AWMK")
        assert len(sd) == 4 + 1 + 8 + 32


# ---- ECC -------------------------------------------------------------------

class TestECC:
    def test_encode_decode_roundtrip(self):
        data = bytes(range(109))
        encoded = ecc.encode(data, CONFIG)
        assert len(encoded) == CONFIG.encoded_size
        decoded = ecc.decode(encoded, CONFIG)
        assert decoded == data

    def test_error_correction(self):
        data = bytes(range(109))
        encoded = bytearray(ecc.encode(data, CONFIG))
        # Corrupt up to nsym//2 bytes
        rng = np.random.default_rng(99)
        n_errors = CONFIG.rs_nsym // 2
        positions = rng.choice(len(encoded), size=n_errors, replace=False)
        for pos in positions:
            encoded[pos] ^= 0xFF
        decoded = ecc.decode(bytes(encoded), CONFIG)
        assert decoded == data

    def test_too_many_errors_raises(self):
        data = bytes(range(109))
        encoded = bytearray(ecc.encode(data, CONFIG))
        # Corrupt more than nsym//2 bytes
        for i in range(CONFIG.rs_nsym):
            encoded[i] ^= 0xFF
        with pytest.raises(Exception):
            ecc.decode(bytes(encoded), CONFIG)


# ---- Spread-spectrum -------------------------------------------------------

class TestSpreadSpectrum:
    def test_pn_deterministic(self):
        pn1 = spread_spectrum.generate_pn_sequence(OWNER_ID, 0, 31)
        pn2 = spread_spectrum.generate_pn_sequence(OWNER_ID, 0, 31)
        np.testing.assert_array_equal(pn1, pn2)

    def test_pn_different_indices(self):
        pn0 = spread_spectrum.generate_pn_sequence(OWNER_ID, 0, 31)
        pn1 = spread_spectrum.generate_pn_sequence(OWNER_ID, 1, 31)
        assert not np.array_equal(pn0, pn1)

    def test_pn_values(self):
        pn = spread_spectrum.generate_pn_sequence(OWNER_ID, 42, 100)
        assert set(np.unique(pn)) == {-1.0, 1.0}

    def test_embed_extract_single_frame(self):
        """Embed bits in one frame and extract them back."""
        rng = np.random.default_rng(7)
        frame = rng.normal(0, 1, CONFIG.frame_size).astype(np.float64)
        # No window (non-overlapping frames)
        fft_data = np.fft.rfft(frame)

        bits_in = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
        n = len(bits_in)

        modified = spread_spectrum.embed_bits_in_frame(
            fft_data, bits_in, OWNER_ID, 0, CONFIG,
        )
        extracted, confs = spread_spectrum.extract_bits_from_frame(
            modified, n, OWNER_ID, 0, CONFIG,
        )
        assert extracted == bits_in
