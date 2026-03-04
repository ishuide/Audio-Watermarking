#!/usr/bin/env python
"""
============================================================
  AUDIO WARP — REVIEW DEMONSTRATION SCRIPT
============================================================

This script is designed for the MFC project review with
Sunil sir. It handles 3 audio samples:

  SAMPLE 1  — Your own audio (full pipeline demo)
  SAMPLE 2  — Sir's clean audio (detect=no watermark, then
              embed + detect to establish identity)
  SAMPLE 3  — Sir's 2nd audio (pre-watermarked or attacked
              by you; detect whether watermark survives)

Usage:
  python review/review_demo.py ^
      --own      "path/to/your_audio.wav" ^
      --clean    "path/to/sir_clean_audio.wav" ^
      --test     "path/to/sir_second_audio.wav" ^
      --test-key "review/output/sir_clean/public.pem"

  The --test-key is the public key from the SAMPLE 2 embed
  step (auto-generated in review/output/sir_clean/). Use
  it only if SAMPLE 3 was watermarked using SAMPLE 2's keys.

  If SAMPLE 3 was watermarked independently (different keys),
  point --test-key to the matching public.pem.

All outputs are saved to review/output/ — nothing in the
main codebase is modified.
============================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ── Add project root to path so audio_warp is importable ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audio_warp.config import WatermarkConfig
from audio_warp import audio_io, crypto, payload, ecc, embedder, detector, attacks

CONFIG = WatermarkConfig()
OWNER_ID = b"\x01\x02\x03\x04\x05\x06\x07\x08"

DIVIDER = "=" * 64
SUB_DIV = "-" * 64


# ── Pretty printing helpers ──────────────────────────────────────────

def header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def sub_header(title: str) -> None:
    print(f"\n{SUB_DIV}")
    print(f"  {title}")
    print(SUB_DIV)


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def info(msg: str) -> None:
    print(f"  {msg}")


# ── Core demonstration functions ─────────────────────────────────────

def show_config() -> None:
    """Print key configuration parameters."""
    sub_header("Watermark Configuration")
    info(f"Sample rate        : {CONFIG.sample_rate} Hz")
    info(f"Frame size         : {CONFIG.frame_size} samples")
    info(f"Embedding band     : {CONFIG.freq_low}–{CONFIG.freq_high} Hz  (bins {CONFIG.bin_low}–{CONFIG.bin_high})")
    info(f"Chips per bit      : {CONFIG.chips_per_bit}")
    info(f"Embed strength (α) : {CONFIG.embed_strength}")
    info(f"Payload size       : {CONFIG.payload_size} bytes")
    info(f"RS parity symbols  : {CONFIG.rs_nsym}  (corrects up to {CONFIG.rs_nsym // 2} byte errors)")
    info(f"RS-encoded size    : {CONFIG.encoded_size} bytes  →  {CONFIG.total_bits} bits")
    info(f"Bits per frame     : {CONFIG.bits_per_frame}")
    info(f"Frames needed      : {CONFIG.frames_needed}")
    info(f"Min audio length   : {CONFIG.min_audio_seconds:.2f} s  ({CONFIG.min_audio_samples} samples)")
    info(f"Owner ID           : {OWNER_ID.hex()}")


def load_audio(path: str, label: str) -> np.ndarray:
    """Load and canonicalize an audio file, printing stats."""
    sub_header(f"Loading: {label}")
    info(f"File: {path}")

    raw, sr = audio_io.read_audio(path)
    info(f"Raw: {raw.shape}, sr={sr}, dtype={raw.dtype}, "
         f"{'stereo' if raw.ndim > 1 else 'mono'}")

    canonical = audio_io.to_canonical(raw, sr, CONFIG)
    duration = len(canonical) / CONFIG.sample_rate
    info(f"Canonical: {len(canonical)} samples, {duration:.2f} s, "
         f"peak={np.max(np.abs(canonical)):.4f}, "
         f"RMS={float(np.sqrt(np.mean(canonical**2))):.4f}")

    if len(canonical) < CONFIG.min_audio_samples:
        fail(f"Audio too short! Need >= {CONFIG.min_audio_seconds:.2f} s, "
             f"got {duration:.2f} s")
        sys.exit(1)

    return canonical


def show_embed_internals(audio: np.ndarray, priv, label: str) -> tuple:
    """
    Walk through embed internals step-by-step, printing everything
    the professor might ask about.
    Returns (audio_hash, signature, raw_payload, encoded, watermarked).
    """
    sub_header(f"Embedding Pipeline — {label}")

    # 1. SHA-256 hash
    t0 = time.perf_counter()
    audio_h = crypto.audio_hash(audio)
    hash_ms = (time.perf_counter() - t0) * 1000
    info(f"1. SHA-256 audio hash : {audio_h.hex()}")
    info(f"   Hash computed in   : {hash_ms:.2f} ms  ({len(audio)} samples → 32 bytes)")

    # 2. Signable data
    signable = payload.signable_data(CONFIG.magic, CONFIG.version, OWNER_ID, audio_h)
    info(f"2. Signable data      : {len(signable)} bytes  "
         f"(MAGIC 4B + VER 1B + OWNER 8B + HASH 32B)")
    info(f"   Hex preview        : {signable.hex()[:60]}...")

    # 3. Ed25519 signature
    t0 = time.perf_counter()
    signature = crypto.sign(priv, signable)
    sign_ms = (time.perf_counter() - t0) * 1000
    info(f"3. Ed25519 signature  : {signature.hex()[:48]}...")
    info(f"   Signature length   : {len(signature)} bytes, computed in {sign_ms:.2f} ms")

    # 4. Payload construction
    raw_payload = payload.build_payload(OWNER_ID, audio_h, signature, CONFIG)
    info(f"4. Payload assembled  : {len(raw_payload)} bytes")
    info(f"   Layout: MAGIC(4B) | VER(1B) | OWNER(8B) | SHA256(32B) | SIG(64B)")

    # 5. Reed-Solomon encoding
    t0 = time.perf_counter()
    encoded = ecc.encode(raw_payload, CONFIG)
    rs_ms = (time.perf_counter() - t0) * 1000
    info(f"5. Reed-Solomon encode: {len(raw_payload)}B → {len(encoded)}B  "
         f"(+{CONFIG.rs_nsym} parity symbols)")
    info(f"   Max correctable    : {CONFIG.rs_nsym // 2} byte errors")
    info(f"   Encoding time      : {rs_ms:.2f} ms")

    # RS round-trip verification
    decoded_check = ecc.decode(encoded, CONFIG)
    if decoded_check == raw_payload:
        ok("RS encode→decode round-trip: PASS")
    else:
        fail("RS encode→decode round-trip: FAIL")

    # RS error correction demo
    corrupted = bytearray(encoded)
    n_errors = CONFIG.rs_nsym // 2
    rng = np.random.default_rng(42)
    err_positions = rng.choice(len(corrupted), size=n_errors, replace=False)
    for pos in err_positions:
        corrupted[pos] ^= 0xFF
    try:
        fixed = ecc.decode(bytes(corrupted), CONFIG)
        if fixed == raw_payload:
            ok(f"RS correction test: injected {n_errors} byte errors → RECOVERED")
        else:
            fail(f"RS correction test: recovered data does not match")
    except Exception:
        fail(f"RS correction test: could not recover from {n_errors} errors")

    # 6. Bits
    total_bits = CONFIG.total_bits
    info(f"6. Total bits         : {total_bits}  ({CONFIG.encoded_size} bytes × 8)")

    # 7. Actual embedding
    t0 = time.perf_counter()
    watermarked = embedder.embed(audio, OWNER_ID, priv, CONFIG)
    embed_ms = (time.perf_counter() - t0) * 1000
    info(f"7. Spread-spectrum embedding completed in {embed_ms:.1f} ms")

    # SNR
    noise_power = float(np.mean((audio - watermarked) ** 2))
    if noise_power > 1e-30:
        snr = 10 * np.log10(float(np.mean(audio ** 2)) / noise_power)
    else:
        snr = float("inf")
    info(f"   SNR (original vs watermarked): {snr:.2f} dB")
    info(f"   Max sample difference: {float(np.max(np.abs(audio - watermarked))):.6f}")

    return audio_h, signature, raw_payload, encoded, watermarked


def show_detect(audio: np.ndarray, pub, label: str,
                expect_found: bool | None = None) -> detector.DetectionResult:
    """Run detection with verbose output."""
    sub_header(f"Detection — {label}")

    t0 = time.perf_counter()
    result = detector.detect(audio, OWNER_ID, pub, CONFIG)
    detect_ms = (time.perf_counter() - t0) * 1000

    if result.found:
        ok(f"WATERMARK VERIFIED")
        info(f"   Owner ID        : {result.owner_id.hex()}")
        info(f"   Audio hash      : {result.audio_hash.hex()}")
        info(f"   Mean confidence : {result.mean_confidence:.4f}")
        info(f"   Detection time  : {detect_ms:.1f} ms")

        # Verify Ed25519 signature explicitly for display
        info(f"   Ed25519 sig verified against public key: YES")
        info(f"   Reed-Solomon decoded successfully: YES")
    else:
        fail(f"WATERMARK NOT VERIFIED")
        info(f"   Reason          : {result.reason}")
        if result.mean_confidence > 0:
            info(f"   Best confidence : {result.mean_confidence:.4f}")
        info(f"   Detection time  : {detect_ms:.1f} ms")

    # Check against expectation if provided
    if expect_found is not None:
        if result.found == expect_found:
            ok(f"Result matches expectation (expected found={expect_found})")
        else:
            fail(f"Expected found={expect_found}, got found={result.found}")

    return result


def show_attacks(watermarked: np.ndarray, pub, label: str) -> None:
    """Run all attacks and show detection results."""
    sub_header(f"Attack Robustness — {label}")

    attack_suite = {
        "noise":    ("Add Gaussian noise (30 dB SNR)", 30.0),
        "lowpass":  ("Low-pass filter (8 kHz cutoff)", 8000.0),
        "resample": ("Resample 44.1k → 22k → 44.1k",  22050),
        "scale":    ("Amplitude scaling (50%)",         0.5),
        "compress": ("Lossy compression sim (12-bit)",  12),
    }

    n_pass = 0
    n_total = len(attack_suite)

    for name, (desc, param) in attack_suite.items():
        attacked = attacks.apply_attack(name, watermarked, param, CONFIG.sample_rate)

        t0 = time.perf_counter()
        result = detector.detect(attacked, OWNER_ID, pub, CONFIG)
        det_ms = (time.perf_counter() - t0) * 1000

        status = "PASS" if result.found else "FAIL"
        if result.found:
            n_pass += 1
        marker = ok if result.found else fail
        marker(f"{desc:42s}  {status}  "
               f"(conf={result.mean_confidence:.4f}, {det_ms:.0f}ms)")

    info(f"\n  Attack survival score: {n_pass}/{n_total} "
         f"({n_pass/n_total*100:.0f}%)")


# ══════════════════════════════════════════════════════════════════════
# SAMPLE 1 — Your own audio (full pipeline)
# ══════════════════════════════════════════════════════════════════════

def run_sample1(wav_path: str, out_dir: str) -> None:
    header("SAMPLE 1 — YOUR OWN AUDIO (Full Pipeline Demo)")
    os.makedirs(out_dir, exist_ok=True)

    show_config()
    audio = load_audio(wav_path, "Your audio")

    # Generate keys
    sub_header("Key Generation")
    priv, pub = crypto.generate_keypair()
    priv_path = os.path.join(out_dir, "private.pem")
    pub_path = os.path.join(out_dir, "public.pem")
    crypto.save_private_key(priv, priv_path)
    crypto.save_public_key(pub, pub_path)
    info(f"Ed25519 key pair generated")
    info(f"  Private key → {priv_path}")
    info(f"  Public key  → {pub_path}")

    # Embed with full internals
    audio_h, sig, raw_payload, encoded, watermarked = \
        show_embed_internals(audio, priv, "Your audio")

    # Save watermarked audio
    wm_path = os.path.join(out_dir, "watermarked.wav")
    audio_io.write_audio(wm_path, watermarked, CONFIG.sample_rate)
    info(f"\n  Watermarked audio saved → {wm_path}")

    # Detect on clean watermarked audio
    show_detect(watermarked, pub, "Your watermarked audio (clean)", expect_found=True)

    # Verify hash matches original
    detected = detector.detect(watermarked, OWNER_ID, pub, CONFIG)
    if detected.found and detected.audio_hash == audio_h:
        ok("Audio hash from detection matches original → identity proven")
    elif detected.found:
        fail("Audio hash mismatch — audio may have been modified after watermarking")

    # Test wrong key rejection
    sub_header("Security Tests")
    _, wrong_pub = crypto.generate_keypair()
    info("Testing with WRONG public key:")
    show_detect(watermarked, wrong_pub, "Wrong key test", expect_found=False)

    info("\nTesting with WRONG owner ID:")
    wrong_owner = b"\xff" * 8
    result_wrong = detector.detect(watermarked, wrong_owner, pub, CONFIG)
    if not result_wrong.found:
        ok(f"Wrong owner (0x{wrong_owner.hex()}) correctly rejected")
    else:
        fail(f"Wrong owner was incorrectly accepted!")

    # Attack robustness
    show_attacks(watermarked, pub, "Your audio")


# ══════════════════════════════════════════════════════════════════════
# SAMPLE 2 — Sir's clean audio
# ══════════════════════════════════════════════════════════════════════

def run_sample2(wav_path: str, out_dir: str) -> None:
    header("SAMPLE 2 — SIR'S CLEAN AUDIO")
    info("Step A: Prove there is NO watermark on this clean audio")
    info("Step B: Embed our watermark and establish identity")
    os.makedirs(out_dir, exist_ok=True)

    audio = load_audio(wav_path, "Sir's clean audio")

    # Generate keys for this sample
    priv, pub = crypto.generate_keypair()
    priv_path = os.path.join(out_dir, "private.pem")
    pub_path = os.path.join(out_dir, "public.pem")
    crypto.save_private_key(priv, priv_path)
    crypto.save_public_key(pub, pub_path)
    info(f"  Keys saved → {out_dir}/")

    # ── Step A: Detect on clean audio (should find nothing) ──
    sub_header("Step A: Detect on CLEAN audio (expect: no watermark)")
    show_detect(audio, pub, "Sir's clean audio — before embedding",
                expect_found=False)

    # ── Step B: Embed watermark ──
    audio_h, sig, raw_payload, encoded, watermarked = \
        show_embed_internals(audio, priv, "Sir's clean audio")

    wm_path = os.path.join(out_dir, "watermarked.wav")
    audio_io.write_audio(wm_path, watermarked, CONFIG.sample_rate)
    info(f"\n  Watermarked audio saved → {wm_path}")

    # ── Step C: Detect on watermarked audio (should succeed) ──
    sub_header("Step B: Detect on WATERMARKED audio (expect: verified)")
    result = show_detect(watermarked, pub,
                         "Sir's audio — after embedding", expect_found=True)

    if result.found:
        info(f"\n  IDENTITY ESTABLISHED:")
        info(f"  Owner ID  : {result.owner_id.hex()}")
        info(f"  Audio hash: {result.audio_hash.hex()}")
        info(f"  This proves: the owner with ID {OWNER_ID.hex()} embedded")
        info(f"  a cryptographic watermark into this audio, verified by")
        info(f"  Ed25519 signature and Reed-Solomon error correction.")

    # Quick attack test on sir's sample
    show_attacks(watermarked, pub, "Sir's audio (watermarked)")


# ══════════════════════════════════════════════════════════════════════
# SAMPLE 3 — Sir's 2nd audio (pre-watermarked or attacked)
# ══════════════════════════════════════════════════════════════════════

def run_sample3(wav_path: str, pub_key_path: str, out_dir: str) -> None:
    header("SAMPLE 3 — SIR'S 2ND AUDIO (Watermarked/Attacked)")
    info("Goal: Detect whether our watermark is still present")
    os.makedirs(out_dir, exist_ok=True)

    audio = load_audio(wav_path, "Sir's 2nd audio")

    # Load the public key
    sub_header("Loading verification key")
    info(f"Public key: {pub_key_path}")
    pub = crypto.load_public_key(pub_key_path)
    ok("Public key loaded successfully")

    # ── Detect ──
    result = show_detect(audio, pub, "Sir's 2nd audio — watermark check")

    if result.found:
        info(f"\n  WATERMARK SURVIVED!")
        info(f"  Owner ID        : {result.owner_id.hex()}")
        info(f"  Audio hash      : {result.audio_hash.hex()}")
        info(f"  Mean confidence : {result.mean_confidence:.4f}")
        info(f"  Ed25519 sig     : VERIFIED")
        info(f"  Reed-Solomon    : DECODED SUCCESSFULLY")
        info(f"\n  Conclusion: Despite any modifications, the watermark")
        info(f"  is intact and ownership is cryptographically proven.")
    else:
        info(f"\n  WATERMARK NOT DETECTED")
        info(f"  Reason: {result.reason}")
        if result.mean_confidence > 0:
            info(f"  Best confidence: {result.mean_confidence:.4f}")
        info(f"\n  This could mean:")
        info(f"  - The audio was heavily attacked (beyond RS correction capacity)")
        info(f"  - The audio was not watermarked with this owner ID / key pair")
        info(f"  - The wrong public key was provided")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio Warp — Review Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run all 3 samples:
  python review/review_demo.py --own audio1.wav --clean sir_clean.wav --test sir_attacked.wav

  # Run only your own sample:
  python review/review_demo.py --own audio1.wav

  # Run only sir's clean sample:
  python review/review_demo.py --clean sir_clean.wav

  # Run only sir's 2nd sample (needs a public key from a previous embed):
  python review/review_demo.py --test sir_attacked.wav --test-key review/output/sir_clean/public.pem
""",
    )
    parser.add_argument("--own", metavar="WAV",
                        help="Your own audio file (full pipeline demo)")
    parser.add_argument("--clean", metavar="WAV",
                        help="Sir's clean audio (detect=nothing, then embed+detect)")
    parser.add_argument("--test", metavar="WAV",
                        help="Sir's 2nd audio (watermarked/attacked; detect only)")
    parser.add_argument("--test-key", metavar="PEM",
                        help="Public key PEM for --test sample "
                             "(default: review/output/sir_clean/public.pem)")
    parser.add_argument("--output-dir", default=os.path.join(
                            os.path.dirname(__file__), "output"),
                        help="Output directory (default: review/output/)")

    args = parser.parse_args()

    if not args.own and not args.clean and not args.test:
        parser.print_help()
        print("\nERROR: Provide at least one of --own, --clean, or --test")
        sys.exit(1)

    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)

    print(DIVIDER)
    print("  AUDIO WARP — REVIEW DEMONSTRATION")
    print(f"  Output directory: {out_root}")
    print(DIVIDER)

    # ── SAMPLE 1: Your own audio ──
    if args.own:
        run_sample1(args.own, os.path.join(out_root, "own_sample"))

    # ── SAMPLE 2: Sir's clean audio ──
    if args.clean:
        run_sample2(args.clean, os.path.join(out_root, "sir_clean"))

    # ── SAMPLE 3: Sir's 2nd audio ──
    if args.test:
        # Default to the key generated during SAMPLE 2 if available
        test_key = args.test_key
        if test_key is None:
            test_key = os.path.join(out_root, "sir_clean", "public.pem")
            if not os.path.exists(test_key):
                # Try own_sample keys as fallback
                test_key = os.path.join(out_root, "own_sample", "public.pem")
        if not os.path.exists(test_key):
            print(f"\nERROR: Public key not found at {test_key}")
            print("Run --clean first to generate keys, or specify --test-key")
            sys.exit(1)
        run_sample3(args.test, test_key, os.path.join(out_root, "sir_test"))

    # ── Summary ──
    header("REVIEW COMPLETE")
    info(f"All outputs saved to: {out_root}")
    if args.own:
        info(f"  own_sample/   — Your audio: keys, watermarked.wav")
    if args.clean:
        info(f"  sir_clean/    — Sir's clean: keys, watermarked.wav")
    if args.test:
        info(f"  sir_test/     — Sir's 2nd: detection results")


if __name__ == "__main__":
    main()
