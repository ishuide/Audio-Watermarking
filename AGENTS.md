# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

audio_warp is an end-to-end audio watermarking and authentication framework. It embeds a cryptographic ownership certificate (owner ID + SHA-256 hash + Ed25519 signature) into audio using frequency-domain spread-spectrum techniques, protected by Reed–Solomon error correction. Detection is blind: it extracts embedded bits via PN-sequence correlation, applies RS decoding, and verifies the Ed25519 signature.

## Build & Run Commands

```
# Install dependencies
pip install -e ".[dev]"

# Run all tests (expect 30 passed)
python -m pytest tests -v

# Run a single test file
python -m pytest tests/test_pipeline.py -v

# Run a single test
python -m pytest tests/test_components.py::TestCrypto::test_sign_verify -v

# Run the demo (no audio file needed — generates synthetic audio)
python -m audio_warp demo

# Run the full pipeline with graphs/logs/reports (requires WAV files in input dir)
python run_pipeline.py [--input-dir DIR] [--output-dir DIR]

# CLI subcommands
python -m audio_warp keygen --output keys/
python -m audio_warp embed -i input.wav -o watermarked.wav -k keys/private.pem --owner 0102030405060708
python -m audio_warp detect -i watermarked.wav -k keys/public.pem --owner 0102030405060708
python -m audio_warp attack -i watermarked.wav -o attacked.wav --type noise --param 30
```

Input audio must be WAV/FLAC/OGG, at least ~6 seconds long (minimum is `config.min_audio_samples` / 44100 Hz ≈ 5.95 s). Shorter files are rejected with `ValueError`.

## Architecture

The pipeline has six stages, each in its own module:

1. **audio_io.py** — Reads any audio format via `soundfile`, converts to canonical form (mono, 44.1 kHz, float32, peak-normalised). `load_and_canonicalize()` is the convenience entry point used by the CLI. All downstream processing assumes canonical audio.

2. **crypto.py** — Ed25519 key generation/serialisation (PEM), signing, verification, and SHA-256 audio hashing. Uses `cryptography` library.

3. **payload.py** — Constructs and parses the 109-byte watermark payload: `MAGIC(4B) | VERSION(1B) | OWNER_ID(8B) | SHA256_HASH(32B) | ED25519_SIG(64B)`. The `signable_data()` function returns the byte string `MAGIC‖VERSION‖OWNER_ID‖HASH` that is signed/verified.

4. **ecc.py** — Reed–Solomon encoding/decoding via `reedsolo`. Adds 50 parity symbols (corrects up to 25 byte errors), producing 159 encoded bytes (1272 bits).

5. **embedder.py** / **spread_spectrum.py** — Embedding pipeline:
   - `framing.py` splits audio into non-overlapping 4096-sample frames (no windowing — this is intentional to avoid double-window artefacts during detection).
   - `spread_spectrum.py` generates per-bit PN sequences seeded from `SHA256(owner_id ‖ bit_index)`, then embeds each bit by multiplicatively modifying FFT magnitudes in the 1–8 kHz band: `|X'[k]| = |X[k]| * (1 + α * b * pn[k])`.
   - `embedder.embed()` orchestrates: hash → sign → build payload → RS encode → bytes-to-bits → FFT → embed → IFFT → synthesize.

6. **detector.py** — Detection pipeline:
   - Extracts bits via log-magnitude correlation with the same PN sequences.
   - Tries multiple sample offsets (`sync_search_steps`) to handle minor time shifts.
   - Watermark is confirmed only if RS decode succeeds AND Ed25519 signature verifies.
   - Returns a `DetectionResult` dataclass with `found`, `owner_id`, `audio_hash`, `mean_confidence`, and `reason`.

**attacks.py** provides six signal-processing attacks (noise, lowpass, resample, amplitude scale, time shift, lossy compression sim). New attacks should be added as functions and registered in the `ATTACK_REGISTRY` dict (keys are CLI names, values contain `fn`, `param` name, and `default`). `apply_attack()` dispatches by name.

**cli.py** exposes five subcommands: `keygen`, `embed`, `detect`, `attack`, `demo`. Each subcommand is implemented in a `_cmd_*` function and dispatched via a dict.

**run_pipeline.py** — Standalone script that processes all WAV files from an input directory through all 7 stages, generating per-stage logs, matplotlib graphs, and JSON reports under `output/stage{1-7}_*/`. The final comparison stage produces a cross-file summary in `output/final_comparison/`.

## Key Design Decisions

- Non-overlapping frames (hop_size == frame_size) are used deliberately. Overlapping STFT+OLA causes double-windowing that smears the watermark spectrum during detection.
- The embed strength `alpha` is a global constant (0.35). The correlation detector uses mean-removed log-magnitude to suppress host-signal bias.
- Owner ID is required for both embedding and detection because it seeds the PN sequences. Without the correct owner ID, the spread-spectrum correlation fails.
- `_bytes_to_bits` (embedder) and `_bits_to_bytes` (detector) use MSB-first bit ordering.
- `framing.synthesize()` copies modified frames back into the original audio buffer (not overlap-add), preserving unmodified tail samples.

## Testing Patterns

Tests use `pytest` with class-based organisation. All three test files construct synthetic broadband audio (tones at 440/1000/2500/5000 Hz + Gaussian noise, 7 seconds, seeded `rng(42)`) rather than loading files from disk.

- **test_components.py** — Unit tests for config, crypto, payload, ECC, and single-frame spread-spectrum embed/extract.
- **test_pipeline.py** — End-to-end: embed→detect on clean audio, wrong-key rejection, wrong-owner rejection, clean-audio rejection, too-short audio ValueError.
- **test_attacks.py** — Robustness: embed then apply each attack type, verify detection still passes.

## Configuration

All tunable parameters live in `config.py::WatermarkConfig` (dataclass with derived properties). Key parameters: `frame_size=4096`, `freq_low=1000 Hz`, `freq_high=8000 Hz`, `chips_per_bit=31`, `embed_strength=0.35`, `rs_nsym=50`. Derived properties compute `bin_low`, `bin_high`, `bits_per_frame`, `frames_needed`, `min_audio_samples`, etc.

## Dependencies

numpy, scipy, soundfile, cryptography, reedsolo, matplotlib. Dev: pytest. Python >= 3.10 required (uses `X | Y` union type syntax).
