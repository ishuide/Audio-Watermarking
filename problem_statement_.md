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
The audio signal <img src="https://latex.codecogs.com/gif.latex?x[n]"/> is processed in frames. For each frame, the FFT is computed:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X[k]%20=%20\sum_{n=0}^{N-1}%20x[n]%20e^{-j%20\frac{2\pi}{N}%20kn}"/></p>  
  
  
We modify the **magnitude** <img src="https://latex.codecogs.com/gif.latex?|X[k]|"/> while keeping the **phase** <img src="https://latex.codecogs.com/gif.latex?\angle%20X[k]"/> unchanged. A bit <img src="https://latex.codecogs.com/gif.latex?b%20\in%20\{-1,%20+1\}"/> is embedded using a PN sequence <img src="https://latex.codecogs.com/gif.latex?p[k]%20\in%20\{-1,%20+1\}"/>:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?|X&#39;[k]|%20=%20|X[k]|%20\cdot%20(1%20+%20\alpha%20\cdot%20b%20\cdot%20p[k])"/></p>  
  
  
Where:
-   <img src="https://latex.codecogs.com/gif.latex?|X&#39;[k]|"/> is the modified magnitude.
-   <img src="https://latex.codecogs.com/gif.latex?\alpha"/> is the **embedding strength** (typically a small value like 0.01 to ensure imperceptibility).
-   <img src="https://latex.codecogs.com/gif.latex?b"/> is the data bit (<img src="https://latex.codecogs.com/gif.latex?0%20\rightarrow%20-1,%201%20\rightarrow%20+1"/>).
-   <img src="https://latex.codecogs.com/gif.latex?p[k]"/> is the pseudo-random chip value for frequency bin <img src="https://latex.codecogs.com/gif.latex?k"/>.
  
The modified complex spectrum is reconstructed:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X&#39;[k]%20=%20|X&#39;[k]|%20\cdot%20e^{j%20\angle%20X[k]}"/></p>  
  
  
The final watermarked signal is obtained via the Inverse FFT (IFFT).
  
### 3.2 Detection and Extraction
Detection is performed using **Matched Filter Correlation** in the log-magnitude domain. For a received frame <img src="https://latex.codecogs.com/gif.latex?Y[k]"/>, we compute the correlation with the known PN sequence:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?C%20=%20\frac{1}{M}%20\sum_{k=start}^{end}%20(\ln%20|Y[k]|%20-%20\mu_{log})%20\cdot%20p[k]"/></p>  
  
  
Where:
-   <img src="https://latex.codecogs.com/gif.latex?M"/> is the number of chips per bit.
-   <img src="https://latex.codecogs.com/gif.latex?\mu_{log}"/> is the mean log-magnitude of the band (used to center the signal and remove host-audio bias).
-   <img src="https://latex.codecogs.com/gif.latex?C"/> is the correlation score.
  
**Decision Rule:**
-   If <img src="https://latex.codecogs.com/gif.latex?C%20&gt;%200"/>, the detected bit is **1**.
-   If <img src="https://latex.codecogs.com/gif.latex?C%20\le%200"/>, the detected bit is **0**.
-   The **Confidence** of the detection is <img src="https://latex.codecogs.com/gif.latex?|C|"/>.
  
---
  
## 4. Watermark Presence Scenarios
  
| Scenario | Detection Outcome | Explanation |
| :--- | :--- | :--- |
| **Watermark Present** | High Correlation (<img src="https://latex.codecogs.com/gif.latex?C"/>) | The signal aligns with the PN sequence, resulting in a strong positive or negative sum. |
| **No Watermark** | Low Correlation (<img src="https://latex.codecogs.com/gif.latex?C%20\approx%200"/>) | The signal and the PN sequence are orthogonal (uncorrelated), resulting in a sum close to zero. |
| **Wrong Owner ID** | Low Correlation (<img src="https://latex.codecogs.com/gif.latex?C%20\approx%200"/>) | Since the PN sequence is seeded by the Owner ID, using the wrong ID generates the wrong PN sequence, which will not match the embedded watermark. |
| **Altered Audio** | Reduced Correlation | Attacks like compression or noise degrade the signal, lowering <img src="https://latex.codecogs.com/gif.latex?|C|"/>. However, as long as <img src="https://latex.codecogs.com/gif.latex?|C|%20&gt;%200"/> and the ECC can correct bit errors, the watermark is recovered. |
  