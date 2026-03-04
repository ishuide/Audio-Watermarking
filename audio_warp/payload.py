"""Watermark payload construction and parsing.

Payload layout (109 bytes):
    MAGIC       4 B   b"AWMK"
    VERSION     1 B   0x01
    OWNER_ID    8 B   arbitrary identifier
    AUDIO_HASH 32 B   SHA-256 of canonical audio
    SIGNATURE  64 B   Ed25519(private_key, MAGIC‖VERSION‖OWNER_ID‖AUDIO_HASH)
"""

from __future__ import annotations

from audio_warp.config import WatermarkConfig


def build_payload(owner_id: bytes, audio_hash: bytes, signature: bytes,
                  config: WatermarkConfig | None = None) -> bytes:
    """Assemble the 109-byte watermark payload."""
    if config is None:
        config = WatermarkConfig()

    if len(owner_id) != config.owner_id_size:
        raise ValueError(f"owner_id must be {config.owner_id_size} bytes, got {len(owner_id)}")
    if len(audio_hash) != config.hash_size:
        raise ValueError(f"audio_hash must be {config.hash_size} bytes, got {len(audio_hash)}")
    if len(signature) != config.signature_size:
        raise ValueError(f"signature must be {config.signature_size} bytes, got {len(signature)}")

    buf = bytearray()
    buf.extend(config.magic)
    buf.append(config.version)
    buf.extend(owner_id)
    buf.extend(audio_hash)
    buf.extend(signature)
    assert len(buf) == config.payload_size
    return bytes(buf)


def parse_payload(data: bytes,
                  config: WatermarkConfig | None = None) -> dict:
    """Parse a raw payload and return its fields as a dict.

    Raises ValueError on structural problems (wrong size, bad magic, etc.).
    """
    if config is None:
        config = WatermarkConfig()

    if len(data) != config.payload_size:
        raise ValueError(
            f"Payload size mismatch: got {len(data)}, expected {config.payload_size}"
        )

    off = 0
    magic = data[off:off + 4]; off += 4
    version = data[off]; off += 1
    owner_id = data[off:off + config.owner_id_size]; off += config.owner_id_size
    audio_hash = data[off:off + config.hash_size]; off += config.hash_size
    signature = data[off:off + config.signature_size]; off += config.signature_size

    if magic != config.magic:
        raise ValueError(f"Bad magic: {magic!r}")
    if version != config.version:
        raise ValueError(f"Unsupported version: {version}")

    return {
        "magic": magic,
        "version": version,
        "owner_id": owner_id,
        "audio_hash": audio_hash,
        "signature": signature,
    }


def signable_data(magic: bytes, version: int, owner_id: bytes,
                  audio_hash: bytes) -> bytes:
    """Return the byte string that is signed / verified."""
    return magic + bytes([version]) + owner_id + audio_hash
