# Audio Watermarking: Problem Statement and Proposed Solution

## 1. Problem Statement
The proliferation of digital audio distribution has made copyright protection and ownership verification increasingly difficult. Unauthorized copying, distribution, and modification of audio content can occur without any audit trail. Traditional metadata (like ID3 tags) can be easily stripped or modified.

The goal of this project is to develop a **robust, imperceptible, and secure audio watermarking system** that can:
1.  **Embed Ownership Data:** Permanently attach an Owner ID and a cryptographic signature to the audio signal.
2.  **Ensure Integrity:** Link the watermark to the specific content of the audio using hashing, so the watermark becomes invalid if the audio is significantly altered.
3.  **Survive Attacks:** Remain detectable even after common signal processing operations (noise, filtering, resampling, compression).
4.  **Maintain Transparency:** Ensure the watermark is psychoacoustically imperceptible to human listeners.

---

## 2. Proposed Solution: FFT Spread-Spectrum (DSSS)
The project utilizes **Direct Sequence Spread Spectrum (DSSS)** embedding in the **FFT (Fast Fourier Transform) Magnitude Domain**. 

### Why Spread Spectrum?
Spread spectrum techniques distribute the watermark information across a wide range of frequencies. This makes the watermark:
-   **Robust:** Even if some frequency bands are filtered out, the information can be recovered from others.
-   **Statistically Secure:** Without the Pseudo-Noise (PN) sequence "key," the watermark looks like small, random noise that is difficult to detect or remove.

---

## 3. Mathematical Formulation

### 3.1 Embedding Process
The audio signal $x[n]$ is processed in frames. For each frame, the FFT is computed:
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} kn}$$

We modify the **magnitude** $|X[k]|$ while keeping the **phase** $\angle X[k]$ unchanged. A bit $b \in \{-1, +1\}$ is embedded using a PN sequence $p[k] \in \{-1, +1\}$:

$$|X'[k]| = |X[k]| \cdot (1 + \alpha \cdot b \cdot p[k])$$

Where:
-   $|X'[k]|$ is the modified magnitude.
-   $\alpha$ is the **embedding strength** (typically a small value like 0.01 to ensure imperceptibility).
-   $b$ is the data bit ($0 \rightarrow -1, 1 \rightarrow +1$).
-   $p[k]$ is the pseudo-random chip value for frequency bin $k$.

The modified complex spectrum is reconstructed:
$$X'[k] = |X'[k]| \cdot e^{j \angle X[k]}$$

The final watermarked signal is obtained via the Inverse FFT (IFFT).

### 3.2 Detection and Extraction
Detection is performed using **Matched Filter Correlation** in the log-magnitude domain. For a received frame $Y[k]$, we compute the correlation with the known PN sequence:

$$C = \frac{1}{M} \sum_{k=start}^{end} (\ln |Y[k]| - \mu_{log}) \cdot p[k]$$

Where:
-   $M$ is the number of chips per bit.
-   $\mu_{log}$ is the mean log-magnitude of the band (used to center the signal and remove host-audio bias).
-   $C$ is the correlation score.

**Decision Rule:**
-   If $C > 0$, the detected bit is **1**.
-   If $C \le 0$, the detected bit is **0**.
-   The **Confidence** of the detection is $|C|$.

---

## 4. Watermark Presence Scenarios

| Scenario | Detection Outcome | Explanation |
| :--- | :--- | :--- |
| **Watermark Present** | High Correlation ($C$) | The signal aligns with the PN sequence, resulting in a strong positive or negative sum. |
| **No Watermark** | Low Correlation ($C \approx 0$) | The signal and the PN sequence are orthogonal (uncorrelated), resulting in a sum close to zero. |
| **Wrong Owner ID** | Low Correlation ($C \approx 0$) | Since the PN sequence is seeded by the Owner ID, using the wrong ID generates the wrong PN sequence, which will not match the embedded watermark. |
| **Altered Audio** | Reduced Correlation | Attacks like compression or noise degrade the signal, lowering $|C|$. However, as long as $|C| > 0$ and the ECC can correct bit errors, the watermark is recovered. |
