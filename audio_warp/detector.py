"""High-level watermark detection and verification pipeline.

    (possibly attacked) audio
        → STFT → correlate PN sequences → extract bits
        → RS decode
        → parse payload (check magic / version)
        → Ed25519 verify signature
        → DetectionResult
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from audio_warp.config import WatermarkConfig
from audio_warp import crypto, payload, ecc, framing, spread_spectrum


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Outcome of a watermark detection attempt."""
    found: bool
    reason: str = ""
    owner_id: bytes = b""
    audio_hash: bytes = b""
    payload: bytes = b""
    signature_valid: bool = False
    ecc_stats: dict = field(default_factory=dict)
    mean_confidence: float = 0.0
    bit_error_estimate: float = 0.0
    correlations: list[float] = field(default_factory=list)
    threshold: float = 0.1 # Default threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bits_to_bytes(bits: list[int]) -> bytes:
    """Convert a list of bit values (MSB first) back to bytes."""
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
            else:
                byte <<= 1
        out.append(byte)
    return bytes(out)


def _extract_all_bits(
    audio: np.ndarray,
    owner_id: bytes,
    config: WatermarkConfig,
    sample_offset: int = 0,
) -> tuple[list[int], list[float]]:
    """Run STFT extraction at a given sample offset."""
    shifted = audio[sample_offset:]
    frames, _ = framing.analyze_frames(shifted, config)

    all_bits: list[int] = []
    all_conf: list[float] = []
    bit_idx = 0
    total = config.total_bits

    for frame in frames:
        remaining = total - bit_idx
        if remaining <= 0:
            break
        n_bits = min(config.bits_per_frame, remaining)

        fft_data = np.fft.rfft(frame)
        bits, confs = spread_spectrum.extract_bits_from_frame(
            fft_data, n_bits, owner_id, bit_idx, config,
        )
        all_bits.extend(bits)
        all_conf.extend(confs)
        bit_idx += n_bits

    return all_bits, all_conf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    audio: np.ndarray,
    owner_id: bytes,
    public_key: Ed25519PublicKey,
    config: WatermarkConfig | None = None,
) -> DetectionResult:
    """Attempt to detect and verify a watermark in *audio*.

    Parameters
    ----------
    audio      : canonical audio (mono, 44.1 kHz, float32)
    owner_id   : 8-byte owner identifier (needed to regenerate PN sequences)
    public_key : Ed25519 public key for signature verification
    config     : optional WatermarkConfig override

    Returns
    -------
    DetectionResult with *found=True* if watermark is present and verified.
    """
    if config is None:
        config = WatermarkConfig()

    # Try several small sample offsets to handle minor time-shifts
    best_result: DetectionResult | None = None

    offsets = [0]
    step = config.hop_size // config.sync_search_steps
    for s in range(1, config.sync_search_steps):
        offsets.append(s * step)

    for offset in offsets:
        if offset + config.min_audio_samples > len(audio):
            continue

        bits, confidences = _extract_all_bits(audio, owner_id, config, offset)

        if len(bits) < config.total_bits:
            continue

        encoded_bytes = _bits_to_bytes(bits[:config.total_bits])

        # RS decode
        try:
            decoded = ecc.decode(encoded_bytes, config)
            # ECC stats would normally come from the library, 
            # for now we just track success.
            ecc_stats = {"corrected": 0} 
        except Exception:
            # Track best failed attempt by confidence
            mc = float(np.mean(confidences)) if confidences else 0.0
            if best_result is None or mc > best_result.mean_confidence:
                best_result = DetectionResult(
                    found=False,
                    reason=f"RS decode failed (offset={offset})",
                    mean_confidence=mc,
                    correlations=confidences
                )
            continue

        # Parse payload
        try:
            parsed = payload.parse_payload(decoded, config)
        except ValueError as exc:
            best_result = DetectionResult(
                found=False, reason=f"Payload parse error: {exc}",
                mean_confidence=float(np.mean(confidences)),
                correlations=confidences
            )
            continue

        # Verify Ed25519 signature
        signable = payload.signable_data(
            parsed["magic"], parsed["version"],
            parsed["owner_id"], parsed["audio_hash"],
        )
        sig_valid = crypto.verify(public_key, signable, parsed["signature"])
        
        res = DetectionResult(
            found=True,
            owner_id=parsed["owner_id"],
            audio_hash=parsed["audio_hash"],
            payload=decoded,
            signature_valid=sig_valid,
            ecc_stats=ecc_stats,
            mean_confidence=float(np.mean(confidences)),
            correlations=confidences
        )
        
        if sig_valid:
            return res
        else:
            if best_result is None or res.mean_confidence > best_result.mean_confidence:
                best_result = res
                best_result.reason = "Signature verification failed"

    if best_result is not None:
        return best_result

    return DetectionResult(found=False, reason="Audio too short or no watermark found")
