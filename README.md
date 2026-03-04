# Audio Warp — Audio Watermarking & Authentication Framework

## Quick Demo for Presentation

### Prerequisites (one-time setup)

```powershell
cd C:\Desktop\audio_warp
pip install -e ".[dev]"
pip install matplotlib
```

Verify everything works:

```powershell
python -m pytest tests -v
```

Expected: **30 passed**. (Ignore any Windows temp-path warning at the end — all tests pass.)

---

### Demo Option 1: One-Command Full Pipeline (recommended)

This processes all your audio files, generates graphs at every stage, and saves everything:

```powershell
python run_pipeline.py
```

- **Runtime**: ~2 minutes
- **Input**: Reads WAV files from `C:\Desktop\audio_watermark_project - Copy\data\raw\`
- **Output**: Everything saved to `C:\Desktop\audio_warp\output\`

After it finishes, open these to show results:

```powershell
# Final comparison graph (shows all files side by side)
start output\final_comparison\final_comparison.png

# Stage-by-stage graphs for any file (e.g., harvard)
start output\stage1_canonicalization\harvard_waveform_spectrum.png
start output\stage2_crypto\harvard_crypto.png
start output\stage3_payload\harvard_payload.png
start output\stage4_ecc\harvard_ecc.png
start output\stage5_embedding\harvard_embedding.png
start output\stage6_detection\harvard_detection.png
start output\stage7_attacks\harvard_attacks.png
```

---

### Demo Option 2: Step-by-Step CLI (interactive, good for Q&A)

#### Step 1 — Generate cryptographic keys

```powershell
python -m audio_warp keygen --output keys/
```

Output: `keys/private.pem` (signing key) and `keys/public.pem` (verification key).

#### Step 2 — Embed watermark into an audio file

```powershell
python -m audio_warp embed -i "C:\Desktop\audio_watermark_project - Copy\data\raw\harvard.wav" -o harvard_watermarked.wav -k keys/private.pem --owner 0102030405060708
```

Output shows SNR (~24 dB) and duration. The watermarked file sounds identical to the original.

#### Step 3 — Detect and verify ownership

```powershell
python -m audio_warp detect -i harvard_watermarked.wav -k keys/public.pem --owner 0102030405060708
```

Expected output: **WATERMARK VERIFIED** with owner ID and audio hash.

#### Step 4 — Prove it rejects fakes

Wrong key:
```powershell
python -m audio_warp keygen --output fake_keys/
python -m audio_warp detect -i harvard_watermarked.wav -k fake_keys/public.pem --owner 0102030405060708
```

Expected: **WATERMARK NOT VERIFIED** — proves you can't claim ownership without the original private key.

Wrong owner ID:
```powershell
python -m audio_warp detect -i harvard_watermarked.wav -k keys/public.pem --owner ffffffffffffffff
```

Expected: **WATERMARK NOT VERIFIED** — PN sequences don't match.

#### Step 5 — Attack the watermarked audio, then re-detect

```powershell
# Add noise
python -m audio_warp attack -i harvard_watermarked.wav -o harvard_noisy.wav --type noise --param 30
python -m audio_warp detect -i harvard_noisy.wav -k keys/public.pem --owner 0102030405060708

# Low-pass filter
python -m audio_warp attack -i harvard_watermarked.wav -o harvard_lowpass.wav --type lowpass --param 8000
python -m audio_warp detect -i harvard_lowpass.wav -k keys/public.pem --owner 0102030405060708

# Resample (44.1kHz -> 22kHz -> 44.1kHz)
python -m audio_warp attack -i harvard_watermarked.wav -o harvard_resample.wav --type resample
python -m audio_warp detect -i harvard_resample.wav -k keys/public.pem --owner 0102030405060708

# Amplitude scaling (50%)
python -m audio_warp attack -i harvard_watermarked.wav -o harvard_scaled.wav --type scale --param 0.5
python -m audio_warp detect -i harvard_scaled.wav -k keys/public.pem --owner 0102030405060708
```

#### Step 6 — Run the built-in demo (no files needed at all)

```powershell
python -m audio_warp demo
```

Generates synthetic audio in memory, embeds, detects, tests attacks — all in ~10 seconds.

---

### Demo Option 3: Just Run Tests

```powershell
python -m pytest tests -v
```

Shows 30 tests passing across components, pipeline, and attack robustness.

---

## What Each Graph Shows (when professor asks)

| Stage | Graph | What to explain |
|---|---|---|
| 1 | Waveform + spectrum | "Any audio (stereo, different sample rate) gets normalised to a standard format" |
| 2 | Hash + signature bytes | "SHA-256 fingerprints the audio, Ed25519 signs it — unforgeable without private key" |
| 3 | Payload heatmap | "109 bytes: 4B magic + 1B version + 8B owner + 32B hash + 64B signature" |
| 4 | ECC capacity | "Reed-Solomon adds 50 parity bytes, can fix up to 25 corrupted bytes" |
| 5 | Spectrum comparison | "Watermark hides in 1-8 kHz band using spread-spectrum — SNR ~24 dB, inaudible" |
| 6 | Confidence + identity | "Blind extraction + crypto verification = proven ownership" |
| 7 | Attack survival | "Survives noise, filtering, resampling, scaling — RS error correction saves it" |

---

## Things to Know Before the Demo

### What if the professor asks to test a random audio file?

```powershell
python -m audio_warp keygen --output keys/
python -m audio_warp embed -i "any_file.wav" -o watermarked.wav -k keys/private.pem --owner 0102030405060708
python -m audio_warp detect -i watermarked.wav -k keys/public.pem --owner 0102030405060708
```

It works with any WAV/FLAC/OGG file that is at least **6 seconds** long. Shorter files get rejected with a clear error.

### What if the professor asks "can you hear the difference?"

The watermark is inaudible. SNR is 21-26 dB, and the changes are spread across hundreds of frequency bins using pseudo-random sequences — no single frequency is noticeably altered. You can play both files side by side to prove it.

### What if the professor asks "why does harvard fail some attacks?"

Harvard.wav is speech (18 seconds). Speech has long silent gaps where there is almost no spectral energy in the 1-8 kHz embedding band. Those silent-gap frames have very weak watermark signals that noise/compression can overwhelm. Longer files with more consistent energy (like collectathon at 85s) survive all attacks.

### What if the professor asks "why not just hash the file?"

A hash changes if even one bit of the file changes (adding noise, compressing, etc.). Our watermark is embedded *inside* the audio signal itself using spread-spectrum techniques and survives signal processing. Plus the Ed25519 signature cryptographically binds the owner's identity — you can't forge it without the private key.

### What if the professor asks "what is spread spectrum?"

Each bit of the watermark is spread across 31 frequency bins using a pseudo-random +/-1 chip sequence. During detection, correlating with the same sequence concentrates the watermark energy while averaging out the host signal interference. This is the same principle used in GPS and CDMA cellular networks.

### What if the professor asks about the "adaptive embedding strength"?

The embedding strength alpha (0.35) modifies FFT magnitudes multiplicatively: `|X'[k]| = |X[k]| * (1 + 0.35 * bit * pn[k])`. Because it's multiplicative, loud frames get stronger watermarks (more energy to hide in) and quiet frames get weaker ones (less distortion). This naturally adapts to signal content.

### What if the professor asks "what prevents someone from removing the watermark?"

Three things: (1) The PN sequences are secret — seeded from the owner ID, so you can't locate which bins were modified without knowing the owner. (2) The watermark is spread across 650 frequency bins per frame across 61+ frames — you'd have to destroy audio quality to remove it. (3) Even if some bits are corrupted, Reed-Solomon can correct up to 25 byte errors.

### What if the professor asks "what's the minimum audio length and why?"

~5.67 seconds. The math: 109-byte payload + 50 parity bytes = 159 bytes = 1272 bits. Each frame fits 20 bits (650 usable bins / 31 chips per bit). So we need ceil(1272/20) = 64 frames x 4096 samples = 262,144 samples / 44100 Hz = ~5.95 seconds.

---

## Project File Summary

```
audio_warp/
  config.py           -- All parameters (frame size, frequencies, strength)
  audio_io.py         -- Read any audio -> canonical mono 44.1kHz float32
  crypto.py           -- Ed25519 key gen/sign/verify + SHA-256 hashing
  payload.py          -- Build/parse 109-byte payload
  ecc.py              -- Reed-Solomon encode/decode (50 parity symbols)
  framing.py          -- Split audio into 4096-sample FFT frames
  spread_spectrum.py  -- PN sequences, embed/extract bits in frequency domain
  embedder.py         -- Full embed pipeline (hash -> sign -> encode -> embed)
  detector.py         -- Full detect pipeline (extract -> decode -> verify)
  attacks.py          -- 6 attacks: noise, lowpass, resample, scale, timeshift, compress
  cli.py              -- Command-line interface (keygen, embed, detect, attack, demo)

tests/
  test_components.py  -- 19 unit tests (crypto, payload, ECC, spread-spectrum)
  test_pipeline.py    -- 5 end-to-end tests (embed->detect, wrong key, wrong owner)
  test_attacks.py     -- 6 robustness tests (watermark survival under attacks)

run_pipeline.py       -- Full pipeline runner with graphs/logs/reports per stage
```

---

## Pipeline Flow (the 6 objectives)

```
1. STANDARDIZE        audio_io.py     Any audio -> mono, 44.1kHz, float32, normalized
        |
2. HASH + SIGN        crypto.py       SHA-256 hash + Ed25519 signature (identity binding)
        |
3. PAYLOAD            payload.py      109 bytes: MAGIC|VER|OWNER|HASH|SIGNATURE
        |
4. BIT CONVERSION     ecc.py +        RS encode (109->159 bytes) -> 1272 bits MSB-first
                      embedder.py
        |
5. EMBED              spread_         FFT -> multiply magnitudes by (1+a*b*pn) -> IFFT
                      spectrum.py
        |
6. DETECT + VERIFY    detector.py     FFT -> correlate -> RS decode -> verify signature
                                      = IDENTITY PROOF
```

---

## Pipeline Results (already generated in output/)

| Audio File | Duration | SNR | Clean Detect | Noise | Lowpass | Resample | Scale | Compress | Score |
|---|---|---|---|---|---|---|---|---|---|
| collectathon | 85s | 26.2 dB | PASS | PASS | PASS | PASS | PASS | PASS | 100% |
| harvard | 18s | 24.5 dB | PASS | FAIL | PASS | PASS | PASS | FAIL | 67% |
| noise | 30s | 21.3 dB | PASS | PASS | PASS | PASS | PASS | PASS | 100% |
| sine | 1s | -- | SKIP | -- | -- | -- | -- | -- | too short |

---

## Requirements

- Python >= 3.10
- numpy >= 1.24, scipy >= 1.10, soundfile >= 0.12, cryptography >= 41.0, reedsolo >= 1.7, matplotlib >= 3.5
- pytest >= 7.0 (for tests)
- Any WAV file >= 6 seconds long
