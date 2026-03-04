"""High-level watermark embedding pipeline.

    canonical audio
        → SHA-256 hash
        → Ed25519 sign(magic‖ver‖owner‖hash)
        → build payload (109 B)
        → RS encode (159 B / 1 272 bits)
        → STFT → spread-spectrum embed → ISTFT overlap-add
        → watermarked audio
"""

from __future__ import annotations

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from audio_warp.config import WatermarkConfig
from audio_warp import crypto, payload, ecc, framing, spread_spectrum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to a list of bit values (MSB first)."""
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(
    audio: np.ndarray,
    owner_id: bytes,
    private_key: Ed25519PrivateKey,
    config: WatermarkConfig | None = None,
) -> np.ndarray:
    """Embed a cryptographic watermark into *audio* (canonical float32 mono).

    Parameters
    ----------
    audio      : canonical audio (mono, 44.1 kHz, float32, normalised)
    owner_id   : 8-byte owner identifier
    private_key: Ed25519 private key for signing
    config     : optional WatermarkConfig override

    Returns
    -------
    Watermarked audio as float32 ndarray (same length as input).

    Raises
    ------
    ValueError if audio is too short to hold the watermark.
    """
    if config is None:
        config = WatermarkConfig()

    if len(owner_id) != config.owner_id_size:
        raise ValueError(
            f"owner_id must be {config.owner_id_size} bytes, got {len(owner_id)}"
        )

    if len(audio) < config.min_audio_samples:
        raise ValueError(
            f"Audio too short: {len(audio)} samples "
            f"(need >= {config.min_audio_samples}, ~{config.min_audio_seconds:.1f} s)"
        )

    # 1. Hash the *original* canonical audio
    audio_h = crypto.audio_hash(audio)

    # 2. Sign (magic ‖ version ‖ owner_id ‖ hash)
    signable = payload.signable_data(config.magic, config.version, owner_id, audio_h)
    signature = crypto.sign(private_key, signable)

    # 3. Build and RS-encode the payload
    raw_payload = payload.build_payload(owner_id, audio_h, signature, config)
    encoded = ecc.encode(raw_payload, config)
    bits = _bytes_to_bits(encoded)

    # 4. STFT analysis
    frames, starts = framing.analyze_frames(audio, config)

    # 5. Embed bits across frames
    modified_frames: list[np.ndarray] = []
    bit_idx = 0
    total = len(bits)

    for frame in frames:
        fft_data = np.fft.rfft(frame)

        remaining = total - bit_idx
        n_bits = min(config.bits_per_frame, remaining)

        if n_bits > 0:
            frame_bits = bits[bit_idx:bit_idx + n_bits]
            fft_data = spread_spectrum.embed_bits_in_frame(
                fft_data, frame_bits, owner_id, bit_idx, config,
            )
            bit_idx += n_bits

        modified_frames.append(np.fft.irfft(fft_data, n=config.frame_size))

    # 6. Overlap-add synthesis (pass original audio for edge blending)
    watermarked = framing.synthesize(
        modified_frames, starts, config, len(audio), original_audio=audio,
    )
    return watermarked
