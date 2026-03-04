"""Audio file I/O and canonical format conversion."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample as sp_resample

from audio_warp.config import WatermarkConfig


def read_audio(path: str) -> tuple[np.ndarray, int]:
    """Read an audio file and return (samples, sample_rate).

    Samples are returned as float32, shape (N,) or (N, C).
    """
    data, sr = sf.read(path, dtype="float32")
    return data, sr


def to_canonical(audio: np.ndarray, sr: int,
                 config: WatermarkConfig | None = None) -> np.ndarray:
    """Convert audio to canonical format: mono, 44.1 kHz, float32, peak-normalised."""
    if config is None:
        config = WatermarkConfig()

    # Stereo / multi-channel → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample to target sample rate
    if sr != config.sample_rate:
        num_samples = int(len(audio) * config.sample_rate / sr)
        audio = sp_resample(audio, num_samples).astype(np.float32)

    # Peak-normalise to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


def write_audio(path: str, audio: np.ndarray,
                sr: int = 44100) -> None:
    """Write audio samples to a WAV file (float32 sub-type)."""
    sf.write(path, audio, sr, subtype="FLOAT")


def load_and_canonicalize(path: str,
                          config: WatermarkConfig | None = None) -> np.ndarray:
    """Convenience: load audio and convert to canonical format in one step."""
    audio, sr = read_audio(path)
    return to_canonical(audio, sr, config)
