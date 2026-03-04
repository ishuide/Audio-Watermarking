"""Reed-Solomon error correction coding.

Uses the ``reedsolo`` library over GF(2^8).  Default parameters add 50 parity
symbols to the 109-byte payload, producing 159 encoded bytes and tolerating up
to 25 corrupted symbols.
"""

from __future__ import annotations

from reedsolo import RSCodec, ReedSolomonError

from audio_warp.config import WatermarkConfig


def _codec(config: WatermarkConfig | None = None) -> RSCodec:
    if config is None:
        config = WatermarkConfig()
    return RSCodec(config.rs_nsym)


def encode(data: bytes, config: WatermarkConfig | None = None) -> bytes:
    """RS-encode *data*.  Returns data + parity bytes."""
    return bytes(_codec(config).encode(data))


def decode(data: bytes, config: WatermarkConfig | None = None) -> bytes:
    """RS-decode *data*, correcting errors.

    Returns the original message bytes.
    Raises ``reedsolo.ReedSolomonError`` if too many errors.
    """
    result = _codec(config).decode(data)
    # reedsolo >=1.7 returns (decoded_message, decoded_msgecc, errata_pos)
    if isinstance(result, tuple):
        return bytes(result[0])
    return bytes(result)
