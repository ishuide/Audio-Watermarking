#!/usr/bin/env python
"""
Full audio watermarking pipeline demonstration.

Processes every WAV file in the input directory through all six stages,
saving logs, graphs and reports into per-stage folders under ``output/``.

Usage:
    python run_pipeline.py [--input-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless / CI
import matplotlib.pyplot as plt
import numpy as np

# ── project imports ──────────────────────────────────────────────────────
from audio_warp.config import WatermarkConfig
from audio_warp import audio_io, crypto, payload, ecc, embedder, detector, attacks
from audio_warp.framing import analyze_frames
from audio_warp.spread_spectrum import generate_pn_sequence

# ── constants ────────────────────────────────────────────────────────────
OWNER_ID = b"\x01\x02\x03\x04\x05\x06\x07\x08"
ATTACK_SUITE: dict[str, float] = {
    "noise":    30.0,
    "lowpass":  8000.0,
    "resample": 22050,
    "scale":    0.5,
    "compress": 12,
}

# ── helpers ──────────────────────────────────────────────────────────────

def _setup_logger(log_path: str, name: str) -> logging.Logger:
    """Return a logger that writes to *log_path* and to stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _save_figure(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _snr(original: np.ndarray, modified: np.ndarray) -> float:
    noise_power = float(np.mean((original - modified) ** 2))
    if noise_power < 1e-30:
        return float("inf")
    return float(10 * np.log10(np.mean(original ** 2) / noise_power))


# ═════════════════════════════════════════════════════════════════════════
# STAGE 1 — Canonicalization
# ═════════════════════════════════════════════════════════════════════════

def stage1_canonicalize(
    wav_path: str, out_dir: str, config: WatermarkConfig
) -> np.ndarray | None:
    """Read, canonicalize, log info, plot waveform + spectrum."""
    stage_dir = os.path.join(out_dir, "stage1_canonicalization")
    os.makedirs(stage_dir, exist_ok=True)
    stem = Path(wav_path).stem
    log = _setup_logger(os.path.join(stage_dir, f"{stem}_log.txt"),
                        f"s1-{stem}")

    log.info("=" * 60)
    log.info("STAGE 1 — CANONICALIZATION")
    log.info("=" * 60)
    log.info(f"Input file : {wav_path}")

    raw, sr = audio_io.read_audio(wav_path)
    log.info(f"Raw samples: {raw.shape}, dtype={raw.dtype}, sr={sr}")
    log.info(f"Raw duration: {len(raw)/sr:.3f} s, channels={'stereo' if raw.ndim > 1 else 'mono'}")

    canonical = audio_io.to_canonical(raw, sr, config)
    log.info(f"Canonical  : {canonical.shape}, dtype={canonical.dtype}, sr={config.sample_rate}")
    log.info(f"Canonical duration: {len(canonical)/config.sample_rate:.3f} s")
    log.info(f"Peak value : {np.max(np.abs(canonical)):.6f}")
    log.info(f"RMS        : {float(np.sqrt(np.mean(canonical**2))):.6f}")

    min_secs = config.min_audio_seconds
    if len(canonical) < config.min_audio_samples:
        log.warning(f"Audio too short ({len(canonical)/config.sample_rate:.2f}s < {min_secs:.2f}s). "
                     "Will SKIP watermark stages for this file.")

    # ── Plot: raw vs canonical waveform ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Stage 1 — Canonicalization: {stem}", fontsize=13, fontweight="bold")

    # Top-left: raw waveform
    ax = axes[0, 0]
    raw_mono = np.mean(raw, axis=1) if raw.ndim > 1 else raw
    t_raw = np.arange(len(raw_mono)) / sr
    ax.plot(t_raw, raw_mono, linewidth=0.3, color="steelblue")
    ax.set_title(f"Raw waveform (sr={sr}, {'stereo→mono' if raw.ndim > 1 else 'mono'})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Top-right: canonical waveform
    ax = axes[0, 1]
    t_can = np.arange(len(canonical)) / config.sample_rate
    ax.plot(t_can, canonical, linewidth=0.3, color="darkorange")
    ax.set_title(f"Canonical waveform (sr={config.sample_rate}, mono, normalised)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Bottom-left: raw spectrum
    ax = axes[1, 0]
    N = min(len(raw_mono), 2**16)
    freqs_raw = np.fft.rfftfreq(N, 1 / sr)
    spec_raw = np.abs(np.fft.rfft(raw_mono[:N]))
    ax.semilogy(freqs_raw, spec_raw + 1e-10, linewidth=0.4, color="steelblue")
    ax.set_title("Raw magnitude spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)

    # Bottom-right: canonical spectrum
    ax = axes[1, 1]
    N2 = min(len(canonical), 2**16)
    freqs_can = np.fft.rfftfreq(N2, 1 / config.sample_rate)
    spec_can = np.abs(np.fft.rfft(canonical[:N2]))
    ax.semilogy(freqs_can, spec_can + 1e-10, linewidth=0.4, color="darkorange")
    ax.axvline(config.freq_low, color="red", ls="--", alpha=0.6, label=f"Embed band {config.freq_low}-{config.freq_high} Hz")
    ax.axvline(config.freq_high, color="red", ls="--", alpha=0.6)
    ax.set_title("Canonical magnitude spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, os.path.join(stage_dir, f"{stem}_waveform_spectrum.png"))
    log.info(f"Graph saved → {stem}_waveform_spectrum.png")

    # ── Report JSON ──
    report = {
        "file": stem,
        "raw_sr": sr,
        "raw_channels": raw.shape[1] if raw.ndim > 1 else 1,
        "raw_samples": len(raw_mono),
        "raw_duration_s": round(len(raw_mono) / sr, 4),
        "canonical_sr": config.sample_rate,
        "canonical_samples": len(canonical),
        "canonical_duration_s": round(len(canonical) / config.sample_rate, 4),
        "peak": round(float(np.max(np.abs(canonical))), 6),
        "rms": round(float(np.sqrt(np.mean(canonical**2))), 6),
        "meets_min_length": len(canonical) >= config.min_audio_samples,
    }
    with open(os.path.join(stage_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → {stem}_report.json")

    return canonical


# ═════════════════════════════════════════════════════════════════════════
# STAGE 2 — Cryptography (key gen + hashing + signing)
# ═════════════════════════════════════════════════════════════════════════

def stage2_crypto(
    canonical: np.ndarray, stem: str, out_dir: str, config: WatermarkConfig
) -> tuple:
    """Generate keys, hash audio, sign payload, log and graph."""
    stage_dir = os.path.join(out_dir, "stage2_crypto")
    os.makedirs(stage_dir, exist_ok=True)
    log = _setup_logger(os.path.join(stage_dir, f"{stem}_log.txt"),
                        f"s2-{stem}")

    log.info("=" * 60)
    log.info("STAGE 2 — CRYPTOGRAPHY (Key Generation, Hashing, Signing)")
    log.info("=" * 60)

    # Key generation
    t0 = time.perf_counter()
    priv, pub = crypto.generate_keypair()
    keygen_ms = (time.perf_counter() - t0) * 1000
    log.info(f"Ed25519 key pair generated in {keygen_ms:.2f} ms")
    log.info(f"Owner ID: {OWNER_ID.hex()}")

    # Save keys for later use
    keys_dir = os.path.join(out_dir, "keys")
    os.makedirs(keys_dir, exist_ok=True)
    priv_path = os.path.join(keys_dir, f"{stem}_private.pem")
    pub_path = os.path.join(keys_dir, f"{stem}_public.pem")
    crypto.save_private_key(priv, priv_path)
    crypto.save_public_key(pub, pub_path)
    log.info(f"Keys saved → {priv_path}, {pub_path}")

    # SHA-256 hash
    t0 = time.perf_counter()
    audio_h = crypto.audio_hash(canonical)
    hash_ms = (time.perf_counter() - t0) * 1000
    log.info(f"SHA-256 hash: {audio_h.hex()}")
    log.info(f"Hashing took {hash_ms:.2f} ms ({len(canonical)} samples → 32 bytes)")

    # Signable data
    signable = payload.signable_data(config.magic, config.version, OWNER_ID, audio_h)
    log.info(f"Signable data length: {len(signable)} bytes")
    log.info(f"Signable data (hex): {signable.hex()[:80]}...")

    # Ed25519 signature
    t0 = time.perf_counter()
    signature = crypto.sign(priv, signable)
    sign_ms = (time.perf_counter() - t0) * 1000
    log.info(f"Ed25519 signature: {signature.hex()}")
    log.info(f"Signature length: {len(signature)} bytes, computed in {sign_ms:.2f} ms")

    # Verify immediately
    valid = crypto.verify(pub, signable, signature)
    log.info(f"Immediate signature verification: {'PASS' if valid else 'FAIL'}")

    # ── Plot: hash byte distribution + signature byte distribution ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Stage 2 — Cryptography: {stem}", fontsize=13, fontweight="bold")

    # Hash byte distribution
    ax = axes[0]
    hash_bytes = list(audio_h)
    ax.bar(range(32), hash_bytes, color="teal", alpha=0.8, width=0.8)
    ax.set_title("SHA-256 hash (32 bytes)")
    ax.set_xlabel("Byte index")
    ax.set_ylabel("Byte value (0–255)")
    ax.set_xlim(-0.5, 31.5)
    ax.grid(True, alpha=0.3)

    # Signature byte distribution
    ax = axes[1]
    sig_bytes = list(signature)
    ax.bar(range(64), sig_bytes, color="coral", alpha=0.8, width=0.8)
    ax.set_title("Ed25519 signature (64 bytes)")
    ax.set_xlabel("Byte index")
    ax.set_ylabel("Byte value (0–255)")
    ax.set_xlim(-0.5, 63.5)
    ax.grid(True, alpha=0.3)

    # Signable data layout
    ax = axes[2]
    sections = [("MAGIC\n4B", 4), ("VER\n1B", 1), ("OWNER_ID\n8B", 8),
                ("SHA-256\n32B", 32)]
    colors = ["#4CAF50", "#FFC107", "#2196F3", "#9C27B0"]
    left = 0
    for (label, size), c in zip(sections, colors):
        ax.barh(0, size, left=left, height=0.5, color=c, edgecolor="black", linewidth=0.5)
        ax.text(left + size / 2, 0, label, ha="center", va="center", fontsize=7, fontweight="bold")
        left += size
    ax.set_xlim(0, 45)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"Signable data layout ({len(signable)} bytes)")
    ax.set_xlabel("Byte offset")
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_figure(fig, os.path.join(stage_dir, f"{stem}_crypto.png"))
    log.info(f"Graph saved → {stem}_crypto.png")

    # Report
    report = {
        "file": stem,
        "owner_id": OWNER_ID.hex(),
        "sha256_hash": audio_h.hex(),
        "signature": signature.hex(),
        "signable_length": len(signable),
        "keygen_ms": round(keygen_ms, 3),
        "hash_ms": round(hash_ms, 3),
        "sign_ms": round(sign_ms, 3),
        "verification": valid,
    }
    with open(os.path.join(stage_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return priv, pub, audio_h, signature


# ═════════════════════════════════════════════════════════════════════════
# STAGE 3 — Payload construction
# ═════════════════════════════════════════════════════════════════════════

def stage3_payload(
    audio_h: bytes, signature: bytes, stem: str, out_dir: str,
    config: WatermarkConfig
) -> bytes:
    stage_dir = os.path.join(out_dir, "stage3_payload")
    os.makedirs(stage_dir, exist_ok=True)
    log = _setup_logger(os.path.join(stage_dir, f"{stem}_log.txt"),
                        f"s3-{stem}")

    log.info("=" * 60)
    log.info("STAGE 3 — PAYLOAD CONSTRUCTION")
    log.info("=" * 60)

    raw_payload = payload.build_payload(OWNER_ID, audio_h, signature, config)
    log.info(f"Payload size: {len(raw_payload)} bytes (expected {config.payload_size})")
    log.info(f"Payload hex : {raw_payload.hex()[:120]}...")

    # Parse back to verify
    parsed = payload.parse_payload(raw_payload, config)
    log.info(f"Parsed magic  : {parsed['magic']}")
    log.info(f"Parsed version: {parsed['version']}")
    log.info(f"Parsed owner  : {parsed['owner_id'].hex()}")
    log.info(f"Parsed hash   : {parsed['audio_hash'].hex()[:32]}...")
    log.info(f"Parsed sig    : {parsed['signature'].hex()[:32]}...")

    # ── Plot: payload structure visualization ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle(f"Stage 3 — Payload Construction: {stem}", fontsize=13, fontweight="bold")

    # Byte value heatmap of entire payload
    ax = axes[0]
    payload_arr = np.array(list(raw_payload), dtype=float).reshape(1, -1)
    im = ax.imshow(payload_arr, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title(f"Payload byte values ({len(raw_payload)} bytes)")
    ax.set_xlabel("Byte index")
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label="Value (0–255)", shrink=0.6)
    # Mark field boundaries
    boundaries = [0, 4, 5, 13, 45, 109]
    labels_b = ["MAGIC", "VER", "OWNER_ID", "SHA-256", "SIGNATURE"]
    for i, b in enumerate(boundaries[:-1]):
        ax.axvline(b - 0.5, color="red", linewidth=1, alpha=0.7)
        mid = (b + boundaries[i + 1]) / 2
        ax.text(mid, -0.8, labels_b[i], ha="center", va="top", fontsize=7,
                color="red", fontweight="bold")

    # Byte histogram
    ax = axes[1]
    ax.hist(list(raw_payload), bins=32, color="mediumpurple", edgecolor="black", alpha=0.8)
    ax.set_title("Payload byte value distribution")
    ax.set_xlabel("Byte value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_figure(fig, os.path.join(stage_dir, f"{stem}_payload.png"))
    log.info(f"Graph saved → {stem}_payload.png")

    report = {
        "file": stem,
        "payload_size": len(raw_payload),
        "magic": parsed["magic"].decode(),
        "version": parsed["version"],
        "owner_id": parsed["owner_id"].hex(),
        "hash_preview": parsed["audio_hash"].hex()[:16],
        "sig_preview": parsed["signature"].hex()[:16],
    }
    with open(os.path.join(stage_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return raw_payload


# ═════════════════════════════════════════════════════════════════════════
# STAGE 4 — Reed-Solomon Error Correction
# ═════════════════════════════════════════════════════════════════════════

def stage4_ecc(
    raw_payload: bytes, stem: str, out_dir: str, config: WatermarkConfig
) -> tuple[bytes, list[int]]:
    stage_dir = os.path.join(out_dir, "stage4_ecc")
    os.makedirs(stage_dir, exist_ok=True)
    log = _setup_logger(os.path.join(stage_dir, f"{stem}_log.txt"),
                        f"s4-{stem}")

    log.info("=" * 60)
    log.info("STAGE 4 — REED-SOLOMON ERROR CORRECTION ENCODING")
    log.info("=" * 60)

    t0 = time.perf_counter()
    encoded = ecc.encode(raw_payload, config)
    enc_ms = (time.perf_counter() - t0) * 1000
    log.info(f"Input  : {len(raw_payload)} bytes (payload)")
    log.info(f"Output : {len(encoded)} bytes (payload + {config.rs_nsym} parity symbols)")
    log.info(f"Encoding time: {enc_ms:.2f} ms")
    log.info(f"Max correctable byte errors: {config.rs_nsym // 2}")

    # Convert to bits
    bits: list[int] = []
    for byte in encoded:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    log.info(f"Total bits: {len(bits)} ({config.total_bits} expected)")
    log.info(f"Bit-0 count: {bits.count(0)}, Bit-1 count: {bits.count(1)}")

    # Verify round-trip
    decoded = ecc.decode(encoded, config)
    assert decoded == raw_payload, "RS round-trip FAILED"
    log.info("RS encode→decode round-trip: PASS")

    # Error correction demo
    corrupted = bytearray(encoded)
    n_errors = config.rs_nsym // 2
    rng = np.random.default_rng(42)
    err_positions = rng.choice(len(corrupted), size=n_errors, replace=False)
    for pos in err_positions:
        corrupted[pos] ^= 0xFF
    try:
        fixed = ecc.decode(bytes(corrupted), config)
        correction_ok = (fixed == raw_payload)
    except Exception:
        correction_ok = False
    log.info(f"Injected {n_errors} byte errors → correction: {'PASS' if correction_ok else 'FAIL'}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Stage 4 — Reed-Solomon ECC: {stem}", fontsize=13, fontweight="bold")

    # Encoded bytes
    ax = axes[0, 0]
    ax.bar(range(len(encoded)), list(encoded), color="teal", alpha=0.7, width=1.0)
    ax.axvline(len(raw_payload) - 0.5, color="red", ls="--", linewidth=2,
               label=f"Payload|Parity boundary ({len(raw_payload)})")
    ax.set_title(f"RS-encoded data ({len(encoded)} bytes)")
    ax.set_xlabel("Byte index")
    ax.set_ylabel("Byte value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bit stream
    ax = axes[0, 1]
    bit_arr = np.array(bits[:200])  # show first 200 bits
    ax.step(range(len(bit_arr)), bit_arr, where="mid", color="navy", linewidth=0.5)
    ax.set_title("Bit stream (first 200 bits, MSB-first)")
    ax.set_xlabel("Bit index")
    ax.set_ylabel("Bit value")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)

    # Error correction capacity
    ax = axes[1, 0]
    test_errors = list(range(0, config.rs_nsym + 5, 1))
    success = []
    for ne in test_errors:
        test_enc = bytearray(encoded)
        if ne > 0:
            positions = rng.choice(len(test_enc), size=min(ne, len(test_enc)), replace=False)
            for p in positions:
                test_enc[p] ^= 0xFF
        try:
            r = ecc.decode(bytes(test_enc), config)
            success.append(1 if r == raw_payload else 0)
        except Exception:
            success.append(0)
    colors = ["green" if s else "red" for s in success]
    ax.bar(test_errors, success, color=colors, width=0.8, alpha=0.8)
    ax.axvline(config.rs_nsym // 2, color="blue", ls="--", linewidth=1.5,
               label=f"Max correctable = {config.rs_nsym // 2}")
    ax.set_title("RS error correction capacity")
    ax.set_xlabel("Number of corrupted bytes")
    ax.set_ylabel("Decoding success")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["FAIL", "PASS"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Parity overhead pie
    ax = axes[1, 1]
    ax.pie([len(raw_payload), config.rs_nsym],
           labels=[f"Payload\n{len(raw_payload)}B", f"Parity\n{config.rs_nsym}B"],
           colors=["#2196F3", "#FF9800"], autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 10})
    ax.set_title("Payload vs Parity overhead")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_figure(fig, os.path.join(stage_dir, f"{stem}_ecc.png"))
    log.info(f"Graph saved → {stem}_ecc.png")

    report = {
        "file": stem,
        "payload_bytes": len(raw_payload),
        "encoded_bytes": len(encoded),
        "parity_symbols": config.rs_nsym,
        "total_bits": len(bits),
        "bit0_count": bits.count(0),
        "bit1_count": bits.count(1),
        "max_correctable_errors": config.rs_nsym // 2,
        "encode_ms": round(enc_ms, 3),
        "roundtrip_pass": True,
        "error_correction_test_pass": correction_ok,
    }
    with open(os.path.join(stage_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return encoded, bits


# ═════════════════════════════════════════════════════════════════════════
# STAGE 5 — Embedding (FFT spread-spectrum)
# ═════════════════════════════════════════════════════════════════════════

def stage5_embedding(
    canonical: np.ndarray, priv, stem: str, out_dir: str,
    audio_out_dir: str, config: WatermarkConfig
) -> np.ndarray:
    stage_dir = os.path.join(out_dir, "stage5_embedding")
    os.makedirs(stage_dir, exist_ok=True)
    log = _setup_logger(os.path.join(stage_dir, f"{stem}_log.txt"),
                        f"s5-{stem}")

    log.info("=" * 60)
    log.info("STAGE 5 — SPREAD-SPECTRUM EMBEDDING")
    log.info("=" * 60)

    log.info(f"Audio length: {len(canonical)} samples ({len(canonical)/config.sample_rate:.2f} s)")
    log.info(f"Frame size: {config.frame_size}, Hop size: {config.hop_size}")
    log.info(f"Embedding band: {config.freq_low}–{config.freq_high} Hz (bins {config.bin_low}–{config.bin_high})")
    log.info(f"Chips/bit: {config.chips_per_bit}, Bits/frame: {config.bits_per_frame}")
    log.info(f"Embed strength (α): {config.embed_strength}")
    log.info(f"Total bits to embed: {config.total_bits}")
    log.info(f"Frames needed: {config.frames_needed}")

    t0 = time.perf_counter()
    watermarked = embedder.embed(canonical, OWNER_ID, priv, config)
    embed_ms = (time.perf_counter() - t0) * 1000
    log.info(f"Embedding completed in {embed_ms:.1f} ms")

    snr_val = _snr(canonical, watermarked)
    log.info(f"SNR (original vs watermarked): {snr_val:.2f} dB")

    max_diff = float(np.max(np.abs(canonical - watermarked)))
    log.info(f"Max sample difference: {max_diff:.6f}")

    # Save watermarked audio
    wm_path = os.path.join(audio_out_dir, f"{stem}_watermarked.wav")
    audio_io.write_audio(wm_path, watermarked, config.sample_rate)
    log.info(f"Watermarked audio saved → {wm_path}")

    # ── Plots ──
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"Stage 5 — Spread-Spectrum Embedding: {stem}", fontsize=13, fontweight="bold")

    # Waveform comparison
    t_axis = np.arange(len(canonical)) / config.sample_rate
    ax = axes[0, 0]
    ax.plot(t_axis, canonical, linewidth=0.3, color="steelblue", label="Original", alpha=0.7)
    ax.plot(t_axis, watermarked, linewidth=0.3, color="darkorange", label="Watermarked", alpha=0.7)
    ax.set_title("Waveform comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Difference signal
    ax = axes[0, 1]
    diff = watermarked - canonical
    ax.plot(t_axis, diff, linewidth=0.3, color="red")
    ax.set_title(f"Watermark signal (difference), max={max_diff:.4f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Spectrum comparison on first frame
    frames_orig, _ = analyze_frames(canonical, config)
    frames_wm, _ = analyze_frames(watermarked, config)
    fft_orig = np.abs(np.fft.rfft(frames_orig[0]))
    fft_wm = np.abs(np.fft.rfft(frames_wm[0]))
    freqs = np.fft.rfftfreq(config.frame_size, 1 / config.sample_rate)

    ax = axes[1, 0]
    ax.semilogy(freqs, fft_orig + 1e-10, linewidth=0.5, color="steelblue", label="Original", alpha=0.7)
    ax.semilogy(freqs, fft_wm + 1e-10, linewidth=0.5, color="darkorange", label="Watermarked", alpha=0.7)
    ax.axvspan(config.freq_low, config.freq_high, alpha=0.1, color="red", label="Embed band")
    ax.set_title("Frame 0 — FFT magnitude spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Spectrum difference in embed band
    ax = axes[1, 1]
    embed_freqs = freqs[config.bin_low:config.bin_high]
    spec_diff = fft_wm[config.bin_low:config.bin_high] - fft_orig[config.bin_low:config.bin_high]
    ax.plot(embed_freqs, spec_diff, linewidth=0.5, color="crimson")
    ax.set_title("Frame 0 — Spectral modification in embed band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude change")
    ax.grid(True, alpha=0.3)

    # PN sequence example
    ax = axes[2, 0]
    pn = generate_pn_sequence(OWNER_ID, 0, config.chips_per_bit)
    ax.stem(range(len(pn)), pn, linefmt="b-", markerfmt="bo", basefmt="k-")
    ax.set_title(f"PN sequence for bit 0 ({config.chips_per_bit} chips)")
    ax.set_xlabel("Chip index")
    ax.set_ylabel("Chip value (±1)")
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)

    # SNR bar
    ax = axes[2, 1]
    ax.barh(["SNR"], [snr_val], color="seagreen", height=0.4)
    ax.set_xlabel("dB")
    ax.set_title(f"Embedding SNR: {snr_val:.2f} dB")
    ax.set_xlim(0, max(60, snr_val + 5))
    ax.axvline(20, color="orange", ls="--", label="20 dB (audible threshold)")
    ax.axvline(40, color="green", ls="--", label="40 dB (transparent)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, os.path.join(stage_dir, f"{stem}_embedding.png"))
    log.info(f"Graph saved → {stem}_embedding.png")

    report = {
        "file": stem,
        "embed_ms": round(embed_ms, 1),
        "snr_db": round(snr_val, 2),
        "max_sample_diff": round(max_diff, 6),
        "total_bits": config.total_bits,
        "frames_used": config.frames_needed,
        "watermarked_path": wm_path,
    }
    with open(os.path.join(stage_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return watermarked


# ═════════════════════════════════════════════════════════════════════════
# STAGE 6 — Detection & Verification
# ═════════════════════════════════════════════════════════════════════════

def stage6_detection(
    watermarked: np.ndarray, canonical: np.ndarray, pub, stem: str,
    out_dir: str, config: WatermarkConfig
) -> None:
    stage_dir = os.path.join(out_dir, "stage6_detection")
    os.makedirs(stage_dir, exist_ok=True)
    log = _setup_logger(os.path.join(stage_dir, f"{stem}_log.txt"),
                        f"s6-{stem}")

    log.info("=" * 60)
    log.info("STAGE 6 — DETECTION & IDENTITY VERIFICATION")
    log.info("=" * 60)

    # Detect on watermarked audio
    t0 = time.perf_counter()
    result = detector.detect(watermarked, OWNER_ID, pub, config)
    detect_ms = (time.perf_counter() - t0) * 1000
    log.info(f"Detection time: {detect_ms:.1f} ms")
    log.info(f"Watermark found: {result.found}")
    log.info(f"Owner ID match : {result.owner_id.hex() if result.owner_id else 'N/A'}")
    log.info(f"Audio hash     : {result.audio_hash.hex() if result.audio_hash else 'N/A'}")
    log.info(f"Mean confidence: {result.mean_confidence:.4f}")
    if result.reason:
        log.info(f"Reason         : {result.reason}")

    # Verify hash matches original audio
    expected_hash = crypto.audio_hash(canonical)
    hash_match = result.audio_hash == expected_hash
    log.info(f"Audio hash matches original: {hash_match}")
    log.info(f"IDENTITY PROOF: {'VERIFIED — Owner {OWNER_ID.hex()} is authenticated' if result.found and hash_match else 'FAILED'}")

    # Test with wrong owner
    wrong_owner = b"\xff" * 8
    result_wrong = detector.detect(watermarked, wrong_owner, pub, config)
    log.info(f"Wrong owner detection: found={result_wrong.found} (expected False)")

    # Test on unwatermarked audio
    _, pub2 = crypto.generate_keypair()
    result_clean = detector.detect(canonical, OWNER_ID, pub2, config)
    log.info(f"Clean audio detection: found={result_clean.found} (expected False)")

    # ── Per-frame confidence extraction for plotting ──
    from audio_warp.framing import analyze_frames as af
    from audio_warp.spread_spectrum import extract_bits_from_frame
    frames, _ = af(watermarked, config)
    frame_confs = []
    bit_idx = 0
    for frame in frames:
        remaining = config.total_bits - bit_idx
        if remaining <= 0:
            break
        n_bits = min(config.bits_per_frame, remaining)
        fft_data = np.fft.rfft(frame)
        _, confs = extract_bits_from_frame(fft_data, n_bits, OWNER_ID, bit_idx, config)
        frame_confs.append(float(np.mean(confs)) if confs else 0.0)
        bit_idx += n_bits

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Stage 6 — Detection & Verification: {stem}", fontsize=13, fontweight="bold")

    # Per-frame confidence
    ax = axes[0, 0]
    ax.bar(range(len(frame_confs)), frame_confs, color="seagreen", alpha=0.8)
    ax.axhline(np.mean(frame_confs), color="red", ls="--", linewidth=1,
               label=f"Mean = {np.mean(frame_confs):.4f}")
    ax.set_title("Per-frame mean correlation confidence")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean |correlation|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Detection outcomes
    ax = axes[0, 1]
    labels = ["Correct owner\n+ correct key", "Wrong owner", "Clean audio\n(no watermark)"]
    outcomes = [1 if result.found else 0,
                1 if result_wrong.found else 0,
                1 if result_clean.found else 0]
    bar_colors = ["green" if o == e else "red"
                  for o, e in zip(outcomes, [1, 0, 0])]
    ax.bar(labels, outcomes, color=bar_colors, alpha=0.8, edgecolor="black")
    ax.set_title("Detection outcome matrix")
    ax.set_ylabel("Detected (1=yes, 0=no)")
    ax.set_ylim(-0.1, 1.3)
    expected_labels = ["Expected: YES", "Expected: NO", "Expected: NO"]
    for i, (lbl, o) in enumerate(zip(expected_labels, outcomes)):
        ax.text(i, o + 0.05, lbl, ha="center", fontsize=8, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Confidence histogram across all bits
    all_bits, all_confs = [], []
    bit_idx = 0
    for frame in frames:
        remaining = config.total_bits - bit_idx
        if remaining <= 0:
            break
        n = min(config.bits_per_frame, remaining)
        fft_data = np.fft.rfft(frame)
        b, c = extract_bits_from_frame(fft_data, n, OWNER_ID, bit_idx, config)
        all_confs.extend(c)
        bit_idx += n

    ax = axes[1, 0]
    ax.hist(all_confs, bins=40, color="mediumpurple", edgecolor="black", alpha=0.8)
    ax.set_title(f"Bit confidence distribution (n={len(all_confs)})")
    ax.set_xlabel("Absolute correlation")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Identity proof summary
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"IDENTITY PROOF RESULT",
        f"",
        f"Watermark found  : {'YES' if result.found else 'NO'}",
        f"Owner ID         : {result.owner_id.hex() if result.owner_id else 'N/A'}",
        f"Audio hash match : {'YES' if hash_match else 'NO'}",
        f"Mean confidence  : {result.mean_confidence:.4f}",
        f"Detection time   : {detect_ms:.1f} ms",
        f"",
        f"VERDICT: {'AUTHENTIC — Ownership verified' if result.found and hash_match else 'NOT VERIFIED'}",
    ]
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen" if result.found else "lightyellow",
                      alpha=0.8))
    ax.set_title("Identity Verification Report")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_figure(fig, os.path.join(stage_dir, f"{stem}_detection.png"))
    log.info(f"Graph saved → {stem}_detection.png")

    report = {
        "file": stem,
        "watermark_found": result.found,
        "owner_id": result.owner_id.hex() if result.owner_id else None,
        "audio_hash_match": hash_match,
        "mean_confidence": round(result.mean_confidence, 4),
        "detect_ms": round(detect_ms, 1),
        "wrong_owner_rejected": not result_wrong.found,
        "clean_audio_rejected": not result_clean.found,
    }
    with open(os.path.join(stage_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════
# STAGE 7 — Attack Testing
# ═════════════════════════════════════════════════════════════════════════

def stage7_attacks(
    watermarked: np.ndarray, pub, stem: str, out_dir: str,
    audio_out_dir: str, config: WatermarkConfig
) -> dict:
    stage_dir = os.path.join(out_dir, "stage7_attacks")
    os.makedirs(stage_dir, exist_ok=True)
    log = _setup_logger(os.path.join(stage_dir, f"{stem}_log.txt"),
                        f"s7-{stem}")

    log.info("=" * 60)
    log.info("STAGE 7 — ATTACK ROBUSTNESS TESTING")
    log.info("=" * 60)

    results_table: dict[str, dict] = {}

    for attack_name, param in ATTACK_SUITE.items():
        log.info(f"\n--- Attack: {attack_name} (param={param}) ---")
        attacked = attacks.apply_attack(attack_name, watermarked, param, config.sample_rate)
        snr_val = _snr(watermarked, attacked)
        log.info(f"  SNR (watermarked vs attacked): {snr_val:.2f} dB")

        t0 = time.perf_counter()
        result = detector.detect(attacked, OWNER_ID, pub, config)
        det_ms = (time.perf_counter() - t0) * 1000

        status = "PASS" if result.found else "FAIL"
        log.info(f"  Detection: {status} (confidence={result.mean_confidence:.4f}, time={det_ms:.1f}ms)")
        if result.reason:
            log.info(f"  Reason: {result.reason}")

        # Save attacked audio
        atk_path = os.path.join(audio_out_dir, f"{stem}_attacked_{attack_name}.wav")
        audio_io.write_audio(atk_path, attacked, config.sample_rate)
        log.info(f"  Saved → {atk_path}")

        results_table[attack_name] = {
            "param": param,
            "survived": result.found,
            "confidence": round(result.mean_confidence, 4),
            "snr_db": round(snr_val, 2),
            "detect_ms": round(det_ms, 1),
        }

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Stage 7 — Attack Robustness: {stem}", fontsize=13, fontweight="bold")

    names = list(results_table.keys())
    survived = [results_table[n]["survived"] for n in names]
    confs = [results_table[n]["confidence"] for n in names]
    snrs = [results_table[n]["snr_db"] for n in names]

    # Survival bar chart
    ax = axes[0, 0]
    bar_colors = ["green" if s else "red" for s in survived]
    ax.bar(names, [1 if s else 0 for s in survived], color=bar_colors, alpha=0.8,
           edgecolor="black")
    ax.set_title("Watermark survival after attacks")
    ax.set_ylabel("Survived (1=yes, 0=no)")
    ax.set_ylim(-0.1, 1.3)
    for i, (s, c) in enumerate(zip(survived, confs)):
        ax.text(i, (1 if s else 0) + 0.05, f"conf={c:.3f}", ha="center", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Confidence comparison
    ax = axes[0, 1]
    bars = ax.bar(names, confs, color="mediumpurple", alpha=0.8, edgecolor="black")
    ax.set_title("Detection confidence after each attack")
    ax.set_ylabel("Mean |correlation|")
    ax.grid(True, alpha=0.3)

    # SNR of attacks
    ax = axes[1, 0]
    ax.bar(names, snrs, color="darkorange", alpha=0.8, edgecolor="black")
    ax.set_title("Attack intensity (SNR: watermarked → attacked)")
    ax.set_ylabel("SNR (dB)")
    ax.grid(True, alpha=0.3)

    # Waveform overlay: original vs attacks
    ax = axes[1, 1]
    show_samples = min(10000, len(watermarked))
    t_show = np.arange(show_samples) / config.sample_rate
    ax.plot(t_show, watermarked[:show_samples], linewidth=0.4, label="Watermarked", alpha=0.6)
    for attack_name in list(ATTACK_SUITE.keys())[:3]:
        atk = attacks.apply_attack(attack_name, watermarked, ATTACK_SUITE[attack_name], config.sample_rate)
        ax.plot(t_show, atk[:show_samples], linewidth=0.3, label=attack_name, alpha=0.5)
    ax.set_title("Waveform overlay (first 10k samples)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_figure(fig, os.path.join(stage_dir, f"{stem}_attacks.png"))
    log.info(f"Graph saved → {stem}_attacks.png")

    report = {"file": stem, "attacks": results_table}
    with open(os.path.join(stage_dir, f"{stem}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return results_table


# ═════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ═════════════════════════════════════════════════════════════════════════

def final_comparison(
    all_results: dict[str, dict], out_dir: str
) -> None:
    stage_dir = os.path.join(out_dir, "final_comparison")
    os.makedirs(stage_dir, exist_ok=True)
    log = _setup_logger(os.path.join(stage_dir, "comparison_log.txt"), "final")

    log.info("=" * 60)
    log.info("FINAL COMPARISON ACROSS ALL AUDIO FILES")
    log.info("=" * 60)

    files = list(all_results.keys())
    attack_names = list(ATTACK_SUITE.keys())

    # Build summary matrix
    summary = {}
    for stem, data in all_results.items():
        log.info(f"\n--- {stem} ---")
        log.info(f"  Embedding SNR: {data['snr_db']:.2f} dB")
        log.info(f"  Detection on clean watermarked: {'PASS' if data['detection_clean'] else 'FAIL'}")
        log.info(f"  Clean confidence: {data['confidence_clean']:.4f}")
        n_survived = sum(1 for a in attack_names if data["attacks"].get(a, {}).get("survived", False))
        log.info(f"  Attacks survived: {n_survived}/{len(attack_names)}")
        for a in attack_names:
            atk = data["attacks"].get(a, {})
            tag = "PASS" if atk.get("survived") else "FAIL"
            log.info(f"    {a:10s}: {tag} (conf={atk.get('confidence', 0):.4f})")
        summary[stem] = {
            "snr_db": data["snr_db"],
            "clean_detection": data["detection_clean"],
            "clean_confidence": data["confidence_clean"],
            "attacks_survived": n_survived,
            "total_attacks": len(attack_names),
        }

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Final Comparison — All Audio Files", fontsize=14, fontweight="bold")

    # SNR comparison
    ax = axes[0, 0]
    snrs = [all_results[f]["snr_db"] for f in files]
    ax.bar(files, snrs, color="teal", alpha=0.8, edgecolor="black")
    ax.set_title("Embedding SNR per file")
    ax.set_ylabel("SNR (dB)")
    for i, v in enumerate(snrs):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Attack survival heatmap
    ax = axes[0, 1]
    matrix = []
    for stem in files:
        row = []
        for a in attack_names:
            row.append(1 if all_results[stem]["attacks"].get(a, {}).get("survived", False) else 0)
        matrix.append(row)
    matrix_arr = np.array(matrix)
    im = ax.imshow(matrix_arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(attack_names)))
    ax.set_xticklabels(attack_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(files)))
    ax.set_yticklabels(files, fontsize=9)
    for i in range(len(files)):
        for j in range(len(attack_names)):
            ax.text(j, i, "PASS" if matrix_arr[i, j] else "FAIL",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if matrix_arr[i, j] == 0 else "black")
    ax.set_title("Attack survival matrix")
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Confidence across attacks for each file
    ax = axes[1, 0]
    x = np.arange(len(attack_names))
    width = 0.8 / len(files)
    for i, stem in enumerate(files):
        confs = [all_results[stem]["attacks"].get(a, {}).get("confidence", 0) for a in attack_names]
        ax.bar(x + i * width, confs, width, label=stem, alpha=0.8)
    ax.set_xticks(x + width * (len(files) - 1) / 2)
    ax.set_xticklabels(attack_names, rotation=45, ha="right", fontsize=9)
    ax.set_title("Detection confidence after attacks")
    ax.set_ylabel("Mean |correlation|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Overall score
    ax = axes[1, 1]
    scores = []
    for stem in files:
        n_pass = summary[stem]["attacks_survived"]
        total = summary[stem]["total_attacks"]
        clean = 1 if summary[stem]["clean_detection"] else 0
        score = (clean + n_pass) / (1 + total) * 100
        scores.append(score)
    bars = ax.bar(files, scores, color="seagreen", alpha=0.8, edgecolor="black")
    ax.set_title("Overall watermark success rate (%)")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    for i, v in enumerate(scores):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_figure(fig, os.path.join(stage_dir, "final_comparison.png"))
    log.info(f"\nGraph saved → final_comparison.png")

    # Summary report
    report = {
        "timestamp": datetime.now().isoformat(),
        "files_processed": files,
        "summary": summary,
    }
    with open(os.path.join(stage_dir, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → final_report.json")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full audio watermark pipeline with reports & graphs")
    parser.add_argument("--input-dir", default=r"C:\Desktop\audio_watermark_project - Copy\data\raw",
                        help="Directory containing input WAV files")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "output"),
                        help="Root output directory for reports, graphs, and audio")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    audio_out_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    config = WatermarkConfig()

    # Discover WAV files
    wav_files = sorted(Path(input_dir).glob("*.wav"))
    if not wav_files:
        print(f"ERROR: No WAV files found in {input_dir}")
        sys.exit(1)

    print("=" * 70)
    print("  AUDIO WATERMARKING & AUTHENTICATION — FULL PIPELINE")
    print("=" * 70)
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Audio files found: {len(wav_files)}")
    for f in wav_files:
        print(f"    • {f.name}")
    print(f"  Min audio length : {config.min_audio_seconds:.2f} s")
    print(f"  Owner ID         : {OWNER_ID.hex()}")
    print("=" * 70)
    print()

    all_results: dict[str, dict] = {}

    for wav_path in wav_files:
        stem = wav_path.stem
        print(f"\n{'-' * 70}")
        print(f"  Processing: {wav_path.name}")
        print(f"{'-' * 70}")

        # Stage 1 — Canonicalization
        canonical = stage1_canonicalize(str(wav_path), output_dir, config)
        if canonical is None:
            continue

        # Check minimum length
        if len(canonical) < config.min_audio_samples:
            print(f"  WARNING: Skipping {stem}: too short "
                  f"({len(canonical)/config.sample_rate:.2f}s < {config.min_audio_seconds:.2f}s)")
            continue

        # Stage 2 — Cryptography
        priv, pub, audio_h, signature = stage2_crypto(canonical, stem, output_dir, config)

        # Stage 3 — Payload
        raw_payload = stage3_payload(audio_h, signature, stem, output_dir, config)

        # Stage 4 — ECC
        encoded, bits = stage4_ecc(raw_payload, stem, output_dir, config)

        # Stage 5 — Embedding
        watermarked = stage5_embedding(canonical, priv, stem, output_dir, audio_out_dir, config)

        # Stage 6 — Detection
        stage6_detection(watermarked, canonical, pub, stem, output_dir, config)

        # Stage 7 — Attacks
        attack_results = stage7_attacks(watermarked, pub, stem, output_dir, audio_out_dir, config)

        # Collect for final comparison
        clean_result = detector.detect(watermarked, OWNER_ID, pub, config)
        all_results[stem] = {
            "snr_db": round(_snr(canonical, watermarked), 2),
            "detection_clean": clean_result.found,
            "confidence_clean": round(clean_result.mean_confidence, 4),
            "attacks": attack_results,
        }

    # Final comparison
    if all_results:
        print(f"\n{'-' * 70}")
        print("  FINAL COMPARISON")
        print(f"{'-' * 70}")
        final_comparison(all_results, output_dir)

    print(f"\n{'=' * 70}")
    print("  PIPELINE COMPLETE")
    print(f"  All outputs saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
