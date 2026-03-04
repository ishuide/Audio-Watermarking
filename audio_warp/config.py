"""Central configuration for the audio watermarking system."""

import math
from dataclasses import dataclass


@dataclass
class WatermarkConfig:
    """All tunable parameters for watermark embedding and detection."""

    # --- Audio canonical format ---
    sample_rate: int = 44100
    channels: int = 1  # mono

    # --- FFT / framing ---
    frame_size: int = 4096
    hop_size: int = 4096  # non-overlapping (avoids OLA double-window artefacts)

    # --- Frequency band for embedding (Hz) ---
    freq_low: float = 1000.0
    freq_high: float = 8000.0

    # --- Spread-spectrum ---
    chips_per_bit: int = 31
    embed_strength: float = 0.35  # base alpha

    # --- Reed-Solomon ---
    rs_nsym: int = 50  # parity symbols (can correct nsym // 2 byte errors)

    # --- Payload structure ---
    magic: bytes = b"AWMK"
    version: int = 1
    owner_id_size: int = 8
    hash_size: int = 32   # SHA-256
    signature_size: int = 64  # Ed25519

    # --- Detection ---
    sync_search_steps: int = 8  # number of sample-offset steps to try

    # ----- derived properties -----

    @property
    def payload_size(self) -> int:
        """Raw payload size in bytes (109)."""
        return len(self.magic) + 1 + self.owner_id_size + self.hash_size + self.signature_size

    @property
    def encoded_size(self) -> int:
        """RS-encoded payload size in bytes."""
        return self.payload_size + self.rs_nsym

    @property
    def total_bits(self) -> int:
        """Total watermark bits to embed."""
        return self.encoded_size * 8

    @property
    def bin_low(self) -> int:
        """Lowest FFT bin index in the embedding band."""
        return int(self.freq_low * self.frame_size / self.sample_rate)

    @property
    def bin_high(self) -> int:
        """Highest FFT bin index in the embedding band."""
        return int(self.freq_high * self.frame_size / self.sample_rate)

    @property
    def num_bins(self) -> int:
        """Number of usable frequency bins."""
        return self.bin_high - self.bin_low

    @property
    def bits_per_frame(self) -> int:
        """Watermark bits that fit in one frame."""
        return self.num_bins // self.chips_per_bit

    @property
    def frames_needed(self) -> int:
        """Minimum number of STFT frames required."""
        return math.ceil(self.total_bits / self.bits_per_frame)

    @property
    def min_audio_samples(self) -> int:
        """Minimum audio length in samples."""
        return (self.frames_needed - 1) * self.hop_size + self.frame_size

    @property
    def min_audio_seconds(self) -> float:
        """Minimum audio length in seconds."""
        return self.min_audio_samples / self.sample_rate
