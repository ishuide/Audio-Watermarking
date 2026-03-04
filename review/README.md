# Review Demo — Quick Reference

## Before the review
```powershell
# Make sure everything is installed and tests pass
pip install -e ".[dev]"
python -m pytest tests -v
```

## During the review

### Step 1 — Run with your own audio + sir's clean audio
```powershell
python review/review_demo.py --own "C:\Desktop\audio_watermark_project - Copy\data\raw\collectathon.wav" --clean "PATH_TO_SIR_CLEAN.wav"
```

### Step 2 — Sir gives a 2nd file (watermarked or attacked by you)
```powershell
python review/review_demo.py --test "PATH_TO_SIR_SECOND.wav" --test-key review/output/sir_clean/public.pem
```

### Or run everything at once
```powershell
python review/review_demo.py --own "your_audio.wav" --clean "sir_clean.wav" --test "sir_second.wav"
```

## What the script shows (what sir will ask about)
- **Reed-Solomon**: encode/decode round-trip, error correction demo (inject 25 byte errors → recover)
- **Ed25519 signature**: signing, verification, wrong-key rejection
- **SHA-256 hash**: audio fingerprinting
- **Spread-spectrum SNR**: watermark is inaudible (21-26 dB)
- **Attack robustness**: noise, lowpass, resample, scale, compression
- **Clean audio detection**: proves no false positives
- **Identity proof**: owner ID + crypto verification

## Output structure
```
review/output/
  own_sample/     keys + watermarked.wav
  sir_clean/      keys + watermarked.wav (generated during review)
  sir_test/       detection results only
```

## If sir asks to watermark his file on the spot
The `--clean` flag does exactly this: detects nothing → embeds → detects verified.

## If sir gives a very short file (< 6 seconds)
The script will print an error. Audio must be >= 5.95 seconds.
Explanation: 159 bytes × 8 = 1272 bits, 20 bits/frame, ceil(1272/20) = 64 frames × 4096 samples ÷ 44100 Hz ≈ 5.95 s.
