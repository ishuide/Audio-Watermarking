"""Audio framing for FFT-based watermarking.

Uses non-overlapping rectangular frames.  This avoids the double-windowing
artefact that occurs when overlapping STFT frames are synthesised via OLA and
then re-analysed for detection: the second window application convolves the
spectrum with the window transform, smearing the embedded watermark.

With non-overlapping frames the detection FFT sees *exactly* the spectrum
that was modified during embedding, giving zero-noise extraction on clean
audio (aside from the irreducible host-signal interference in the
correlation).
"""

from __future__ import annotations

import numpy as np

from audio_warp.config import WatermarkConfig


def analyze_frames(audio: np.ndarray,
                   config: WatermarkConfig) -> tuple[list[np.ndarray], list[int]]:
    """Slice the audio into non-overlapping frames (no window).

    Returns
    -------
    frames : list of float64 arrays, each of length *frame_size*
    starts : corresponding sample-offset of each frame
    """
    frames: list[np.ndarray] = []
    starts: list[int] = []

    for start in range(0, len(audio) - config.frame_size + 1, config.hop_size):
        frame = audio[start:start + config.frame_size].astype(np.float64)
        frames.append(frame)
        starts.append(start)

    return frames, starts


def synthesize(modified_frames: list[np.ndarray],
               starts: list[int],
               config: WatermarkConfig,
               original_length: int,
               original_audio: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct audio by replacing frame regions in-place.

    For non-overlapping frames this is a simple copy-back.
    """
    if original_audio is not None:
        output = original_audio.astype(np.float64).copy()
    else:
        output = np.zeros(original_length, dtype=np.float64)

    for frame, start in zip(modified_frames, starts):
        end = min(start + config.frame_size, original_length)
        length = end - start
        output[start:end] = frame[:length]

    return output.astype(np.float32)
