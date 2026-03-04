# Project Architecture and Workflow

This document explains the organization of the `audio_warp` project, the role of each file, and the logical flow of the watermarking pipeline.

## 1. File Responsibilities

### Core Logic (`audio_warp/`)
-   **`spread_spectrum.py`**: The mathematical engine. Implements PN sequence generation, multiplicative embedding in the FFT domain, and log-magnitude correlation for extraction.
-   **`embedder.py`**: Coordinates the embedding process. Splits audio into frames and applies the spread-spectrum logic to each frame.
-   **`detector.py`**: Coordinates the detection process. Extracts bits from audio frames, applies Error Correction (ECC), and verifies cryptographic signatures.
-   **`attacks.py`**: Contains a suite of signal processing functions (noise, low-pass, resampling, etc.) to test the robustness of the watermark.
-   **`payload.py`**: Handles the construction of the binary payload (Magic bytes, Version, Owner ID, Hash, Signature).
-   **`crypto.py`**: Provides Ed25519 digital signatures and SHA-256 hashing to ensure the watermark is secure and linked to the audio content.
-   **`ecc.py`**: Implements Reed-Solomon error correction to handle bit errors caused by audio degradation.
-   **`framing.py`**: Handles the splitting of audio into overlapping windows (frames) for spectral analysis.
-   **`config.py`**: Centralizes parameters like sample rate, FFT size, embedding frequency bands, and ECC overhead.
-   **`audio_io.py`**: Handles reading/writing WAV files and converting them to a canonical format (mono, 44.1kHz, float32).

### Execution Strategy
-   **`run_pipeline.py`**: The main entry point for a full demonstration. It processes audio through all 7 stages and generates comprehensive reports and graphs in the `output/` folder.
-   **`cli.py`**: Provides a command-line interface for individual embed/detect/attack operations.

---

## 2. Project Execution Flow (The 7 Stages)

To run the full demonstration, you execute:
```bash
python run_pipeline.py
```

The project follows these sequential stages:

### Stage 1: Canonicalization
Audio is converted to a standard format (44.1kHz, Mono) to ensure the FFT bins align correctly during embedding and detection.

### Stage 2: Cryptography
-   An **Owner ID** is assigned.
-   A **SHA-256 hash** of the audio is calculated.
-   An **Ed25519 signature** is generated using a private key, signing the Owner ID and Audio Hash. This proves the audio hasn't been tampered with and identifies the owner.

### Stage 3: Payload Construction
The metadata (Magic, Ver, Owner ID, Hash, Signature) is packed into a compact binary format.

### Stage 4: Error Correction (ECC)
**Reed-Solomon** parity bytes are added to the payload. This allows the system to recover the full watermark even if some bits are flipped or lost during audio compression or noise.

### Stage 5: Embedding (FFT Spread-Spectrum)
The binary bits are converted into a wide-band noise-like signal and multiplied into the FFT magnitude of the audio frames. The watermarked audio is then saved.

### Stage 6: Detection & Verification
The detector:
1.  Extracts bits via correlation.
2.  Fixes errors using Reed-Solomon.
3.  Parses the payload.
4.  **Verifies the signature** using the owner's public key.
5.  **Compares the embedded hash** with the current audio hash to check for tampering.

### Stage 7: Attack Testing
The watermarked audio is subjected to various "attacks" (Noise, Low-pass filtering, Resampling). The system then tries to detect the watermark again to prove its **robustness**.

---

## 3. Robustness Methodology
The algorithm achieves robustness through three layers:
1.  **Redundancy (Spread Spectrum):** Each bit is "spread" across many frequency chips. Even if a few frequencies are destroyed, the bit survives.
2.  **Mathematical Resilience (ECC):** Reed-Solomon coding provides a buffer against bit-level errors.
3.  **Spectral Placement:** The watermark is placed in the mid-frequency bands (approx 1kHz - 4kHz) where audio energy is usually highest (masking the noise) and where standard filters are less likely to remove essential content.
