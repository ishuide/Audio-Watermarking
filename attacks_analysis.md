# Attacks and Robustness Analysis

This document explains the "attacks" implemented in the project, why they were specifically chosen, and how they test the limits of the watermarking algorithm.

## 1. Why these specific attacks?
In the real world, audio files undergo various transformations during distribution, editing, and consumption. A watermark is only useful if it survives these standard processes. The attacks chosen for this project represent the most common "non-malicious" and "malicious" distortions.

---

## 2. Attack Breakdown

### A. Additive White Gaussian Noise (AWGN)
-   **Method:** Injection of random thermal noise at a specific Signal-to-Noise Ratio (SNR).
-   **Why:** Simulates transmission noise (e.g., over a radio or poor cable) and general audio degradation. It tests if the "spread" signal is strong enough to be recovered from a noisy background.

### B. Low-Pass Filtering (LPF)
-   **Method:** Removing all frequencies above a certain cutoff (e.g., 8kHz).
-   **Why:** Many distribution platforms (like legacy phone systems or low-bitrate streaming) limit high frequencies to save bandwidth. Since our watermark is spread across frequencies, we test if the *lower* frequency components of the watermark are sufficient for detection.

### C. Resampling (Downsampling/Upsampling)
-   **Method:** Converting 44.1kHz audio to 22.05kHz and then back to 44.1kHz.
-   **Why:** This is a very common transformation (e.g., CD audio to mobile-friendly formats). It introduces aliasing and removes high-frequency content. It tests the system's ability to handle loss of spectral resolution.

### D. Amplitude Scaling
-   **Method:** Reducing the volume of the audio (e.g., by 50%).
-   **Why:** A simple volume change should *never* destroy a watermark. Since our embedding is **multiplicative** (it scales with the host audio), it is mathematically invariant to uniform volume changes. This attack proves that "Gain" adjustments don't affect detection.

### E. Lossy Compression Simulation
-   **Method:** Band-limiting the signal (20Hz-16kHz) combined with bit-depth quantization (e.g., 12-bit).
-   **Why:** Simulates the effects of MP3 or AAC encoding. These codecs intentionally discard "perceptually irrelevant" information. Our watermark must be placed in a way that the codec considers it "relevant" or "impossible to remove" without destroying the audio quality.

### F. Time Shifting
-   **Method:** Circularly shifting the audio by a small number of samples.
-   **Why:** Tests for **Synchronisation**. Spread spectrum detection usually requires the detector to be perfectly aligned with the frames. Even a small shift can break the FFT alignment. (Note: Robustness to large shifts usually requires special "Sync Patterns" which this project explores).

---

## 3. How Robustness is Achieved
The algorithm survives these attacks because:
1.  **DSSS Gain:** The correlation process effectively "collects" the energy of the bit from many frequency bins. Even if an attack destroys 50% of the bins, the remaining 50% may still provide a positive correlation result.
2.  **ECC Backup:** If an attack causes a few bits to be detected incorrectly (0 instead of 1), the **Reed-Solomon** decoder can fix those errors automatically before the final payload is read.
