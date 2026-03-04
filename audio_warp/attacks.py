"""Signal-processing attacks for watermark robustness evaluation.

Each function takes audio (float32 ndarray) and returns attacked audio.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, resample


# ---------------------------------------------------------------------------
# Individual attacks
# ---------------------------------------------------------------------------

def add_noise(audio: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """Additive white Gaussian noise at the specified SNR (dB)."""
    signal_power = float(np.mean(audio ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.default_rng(42).normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def lowpass_filter(audio: np.ndarray, cutoff_hz: float = 8000.0,
                   sr: int = 44100, order: int = 5) -> np.ndarray:
    """Butterworth low-pass filter."""
    nyq = sr / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, audio).astype(np.float32)


def resample_attack(audio: np.ndarray, sr: int = 44100,
                    intermediate_sr: int = 22050) -> np.ndarray:
    """Down-sample then up-sample back to the original rate."""
    n_down = max(1, int(len(audio) * intermediate_sr / sr))
    down = resample(audio, n_down)
    up = resample(down, len(audio))
    return up.astype(np.float32)


def amplitude_scale(audio: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Uniform amplitude scaling."""
    return (audio * factor).astype(np.float32)


def time_shift(audio: np.ndarray, shift_samples: int = 100) -> np.ndarray:
    """Circular time-shift (positive = prepend silence, crop tail)."""
    result = np.zeros_like(audio)
    if shift_samples > 0:
        result[shift_samples:] = audio[:-shift_samples]
    elif shift_samples < 0:
        s = -shift_samples
        result[:len(audio) - s] = audio[s:]
    else:
        result[:] = audio
    return result


def lossy_compression_sim(audio: np.ndarray, sr: int = 44100,
                          bit_depth: int = 12) -> np.ndarray:
    """Simulate lossy codec effects: band-limit (20 Hz – 16 kHz) + quantise.

    A true MP3 round-trip requires ffmpeg; this approximation captures the
    dominant artefacts (bandwidth reduction + quantisation noise).
    """
    nyq = sr / 2.0
    low = max(20.0 / nyq, 1e-5)
    high = min(16000.0 / nyq, 1.0 - 1e-5)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, audio)
    levels = 2 ** bit_depth
    quantised = np.round(filtered * levels) / levels
    return quantised.astype(np.float32)


# ---------------------------------------------------------------------------
# Registry for CLI / batch testing
# ---------------------------------------------------------------------------

ATTACK_REGISTRY: dict[str, dict] = {
    "noise":    {"fn": add_noise,             "param": "snr_db",          "default": 30.0},
    "lowpass":  {"fn": lowpass_filter,         "param": "cutoff_hz",      "default": 8000.0},
    "resample": {"fn": resample_attack,        "param": "intermediate_sr","default": 22050},
    "scale":    {"fn": amplitude_scale,        "param": "factor",         "default": 0.5},
    "timeshift":{"fn": time_shift,             "param": "shift_samples",  "default": 100},
    "compress": {"fn": lossy_compression_sim,  "param": "bit_depth",      "default": 12},
}


def apply_attack(name: str, audio: np.ndarray, sev: float | None = 0.2,
                 sr: int = 44100) -> np.ndarray:
    """Look up an attack by name and map 0-1 severity to its parameters."""
    entry = ATTACK_REGISTRY[name]
    fn = entry["fn"]
    s = sev if sev is not None else 0.2
    
    # Map 0-1 severity to real-world ranges
    kw: dict = {}
    if name == "noise":
        # 0.0=80dB (clean), 1.0=5dB (destroyed)
        kw["snr_db"] = 80.0 - (s * 75.0)
    elif name == "lowpass":
        # 0.0=20kHz, 1.0=500Hz
        kw["cutoff_hz"] = 20000.0 - (s * 19500.0)
    elif name == "resample":
        # 0.0=44.1kHz, 1.0=4kHz
        kw["intermediate_sr"] = int(sr - (s * (sr - 4000)))
    elif name == "scale":
        # 0.0=1.0x, 1.0=0.0x
        kw["factor"] = 1.0 - s
    elif name == "timeshift":
        # 0.0=0 samples, 1.0=500 samples
        kw["shift_samples"] = int(s * 500)
    elif name == "compress":
        # 0.0=16bit, 1.0=4bit
        kw["bit_depth"] = int(16 - (s * 12))

    if "sr" in fn.__code__.co_varnames:
        kw["sr"] = sr
        
    # Apply and sanitize
    out = fn(audio, **kw)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
