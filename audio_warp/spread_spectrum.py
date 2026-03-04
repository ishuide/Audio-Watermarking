"""Spread-spectrum embedding and extraction in the FFT magnitude domain.

Each watermark bit occupies a contiguous sub-band of *chips_per_bit*
frequency bins inside the mid-band [bin_low, bin_high).  A pseudo-random ±1
chip sequence (seeded from the owner ID and global bit index) determines the
polarity of multiplicative magnitude changes during embedding, and serves as
the matched-filter template during extraction.
"""

from __future__ import annotations

import hashlib

import numpy as np

from audio_warp.config import WatermarkConfig


# ---------------------------------------------------------------------------
# PN sequence generation
# ---------------------------------------------------------------------------

def generate_pn_sequence(owner_id: bytes, bit_index: int, length: int) -> np.ndarray:
    """Deterministic ±1 chip sequence for a given (owner, bit) pair.

    Seed is derived from SHA-256(owner_id ‖ bit_index) for reproducibility
    and cross-owner orthogonality.
    """
    seed_material = owner_id + bit_index.to_bytes(4, "big")
    seed_hash = hashlib.sha256(seed_material).digest()[:8]
    seed = int.from_bytes(seed_hash, "big")
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=length)


# ---------------------------------------------------------------------------
# Per-frame embedding
# ---------------------------------------------------------------------------

def embed_bits_in_frame(
    frame_fft: np.ndarray,
    bits: list[int] | np.ndarray,
    owner_id: bytes,
    bit_offset: int,
    config: WatermarkConfig,
) -> np.ndarray:
    """Embed watermark bits into a single frame's complex FFT spectrum.

    Modifies magnitude multiplicatively:
        |X'[k]| = |X[k]| * (1 + alpha * b * pn[k])
    where *b* ∈ {-1, +1} is the mapped bit value and *pn[k]* is the chip.

    Parameters
    ----------
    frame_fft : complex ndarray from ``np.fft.rfft``
    bits      : bit values (0 / 1) to embed in this frame
    owner_id  : 8-byte owner identifier
    bit_offset: global index of the first bit in *bits*
    config    : watermark configuration

    Returns
    -------
    Modified complex FFT array (same shape as *frame_fft*).
    """
    modified = frame_fft.copy()
    magnitude = np.abs(modified)
    phase = np.angle(modified)

    alpha = config.embed_strength

    for i, bit in enumerate(bits):
        global_idx = bit_offset + i
        start = config.bin_low + i * config.chips_per_bit
        end = start + config.chips_per_bit
        if end > config.bin_high:
            break

        pn = generate_pn_sequence(owner_id, global_idx, config.chips_per_bit)
        b = 2.0 * float(bit) - 1.0  # 0 → -1, 1 → +1

        magnitude[start:end] *= (1.0 + alpha * b * pn)

    return magnitude * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# Per-frame extraction
# ---------------------------------------------------------------------------

def extract_bits_from_frame(
    frame_fft: np.ndarray,
    num_bits: int,
    owner_id: bytes,
    bit_offset: int,
    config: WatermarkConfig,
) -> tuple[list[int], list[float]]:
    """Extract watermark bits from a single frame's complex FFT spectrum.

    Uses log-magnitude correlation with the PN sequences.

    Returns
    -------
    (bits, confidences) — detected bit values and absolute correlation scores.
    """
    magnitude = np.abs(frame_fft)
    log_mag = np.log(magnitude + 1e-10)

    bits: list[int] = []
    confidences: list[float] = []

    for i in range(num_bits):
        global_idx = bit_offset + i
        start = config.bin_low + i * config.chips_per_bit
        end = start + config.chips_per_bit
        if end > config.bin_high:
            break

        pn = generate_pn_sequence(owner_id, global_idx, config.chips_per_bit)

        local = log_mag[start:end]
        # Remove mean to suppress host-signal bias
        centred = local - np.mean(local)
        correlation = float(np.dot(centred, pn)) / config.chips_per_bit

        bits.append(1 if correlation > 0 else 0)
        confidences.append(abs(correlation))

    return bits, confidences
