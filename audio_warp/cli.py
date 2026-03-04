"""Command-line interface for audio_warp.

Usage:
    python -m audio_warp keygen  --output keys/
    python -m audio_warp embed   --input in.wav --output out.wav --key keys/private.pem --owner 0102030405060708
    python -m audio_warp detect  --input out.wav --key keys/public.pem --owner 0102030405060708
    python -m audio_warp attack  --input out.wav --output attacked.wav --type noise --param 30
    python -m audio_warp demo
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np


def _cmd_keygen(args: argparse.Namespace) -> None:
    from audio_warp import crypto

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    priv, pub = crypto.generate_keypair()
    priv_path = os.path.join(out_dir, "private.pem")
    pub_path = os.path.join(out_dir, "public.pem")
    crypto.save_private_key(priv, priv_path)
    crypto.save_public_key(pub, pub_path)
    print(f"Key pair saved:\n  Private: {priv_path}\n  Public:  {pub_path}")


def _cmd_embed(args: argparse.Namespace) -> None:
    from audio_warp import audio_io, crypto, embedder
    from audio_warp.config import WatermarkConfig

    config = WatermarkConfig()
    audio = audio_io.load_and_canonicalize(args.input, config)
    private_key = crypto.load_private_key(args.key)
    owner_id = bytes.fromhex(args.owner)

    watermarked = embedder.embed(audio, owner_id, private_key, config)
    audio_io.write_audio(args.output, watermarked, config.sample_rate)

    snr = 10 * np.log10(
        np.mean(audio ** 2) / (np.mean((audio - watermarked) ** 2) + 1e-20)
    )
    print(f"Watermark embedded → {args.output}")
    print(f"  Owner ID : {args.owner}")
    print(f"  SNR      : {snr:.1f} dB")
    print(f"  Duration : {len(audio)/config.sample_rate:.2f} s")


def _cmd_detect(args: argparse.Namespace) -> None:
    from audio_warp import audio_io, crypto, detector
    from audio_warp.config import WatermarkConfig

    config = WatermarkConfig()
    audio = audio_io.load_and_canonicalize(args.input, config)
    public_key = crypto.load_public_key(args.key)
    owner_id = bytes.fromhex(args.owner)

    result = detector.detect(audio, owner_id, public_key, config)

    if result.found:
        print("WATERMARK VERIFIED")
        print(f"  Owner ID   : {result.owner_id.hex()}")
        print(f"  Audio hash : {result.audio_hash.hex()}")
        print(f"  Confidence : {result.mean_confidence:.4f}")
    else:
        print("WATERMARK NOT VERIFIED")
        print(f"  Reason: {result.reason}")
        if result.mean_confidence > 0:
            print(f"  Best confidence: {result.mean_confidence:.4f}")


def _cmd_attack(args: argparse.Namespace) -> None:
    from audio_warp import audio_io, attacks
    from audio_warp.config import WatermarkConfig

    config = WatermarkConfig()
    audio = audio_io.load_and_canonicalize(args.input, config)

    attacked = attacks.apply_attack(args.type, audio, args.param, config.sample_rate)
    audio_io.write_audio(args.output, attacked, config.sample_rate)
    print(f"Attack '{args.type}' applied → {args.output}")


def _cmd_demo(args: argparse.Namespace) -> None:
    """Generate synthetic audio, embed, detect, then test under attacks."""
    from audio_warp import crypto, embedder, detector, attacks
    from audio_warp.config import WatermarkConfig

    config = WatermarkConfig()

    # 1. Generate 7 s broadband synthetic audio (tones + noise)
    duration = 7.0
    rng = np.random.default_rng(42)
    n = int(config.sample_rate * duration)
    t = np.linspace(0, duration, n, dtype=np.float64)
    tones = (
        0.25 * np.sin(2 * np.pi * 440 * t)
        + 0.20 * np.sin(2 * np.pi * 1000 * t)
        + 0.15 * np.sin(2 * np.pi * 2500 * t)
        + 0.10 * np.sin(2 * np.pi * 5000 * t)
    )
    noise = 0.30 * rng.normal(0, 1, n)
    audio = (tones + noise).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-10

    # 2. Generate keys and owner ID
    priv, pub = crypto.generate_keypair()
    owner_id = b"\x01\x02\x03\x04\x05\x06\x07\x08"

    print("=" * 60)
    print("AUDIO WATERMARKING DEMO")
    print("=" * 60)
    print(f"Audio : {duration:.1f} s synthetic ({config.sample_rate} Hz)")
    print(f"Owner : {owner_id.hex()}")
    print()

    # 3. Embed
    watermarked = embedder.embed(audio, owner_id, priv, config)
    snr = 10 * np.log10(
        np.mean(audio ** 2) / (np.mean((audio - watermarked) ** 2) + 1e-20)
    )
    print(f"[EMBED] SNR = {snr:.1f} dB")

    # 4. Detect on clean watermarked audio
    result = detector.detect(watermarked, owner_id, pub, config)
    status = "PASS" if result.found else "FAIL"
    print(f"[DETECT clean]        {status}  (confidence={result.mean_confidence:.4f})")

    # 5. Test under attacks
    attack_params = {
        "noise":     30.0,
        "lowpass":   8000.0,
        "resample":  22050,
        "scale":     0.5,
        "compress":  12,
    }
    print()
    for name, param in attack_params.items():
        attacked = attacks.apply_attack(name, watermarked, param, config.sample_rate)
        res = detector.detect(attacked, owner_id, pub, config)
        tag = "PASS" if res.found else "FAIL"
        print(f"[DETECT {name:10s}]  {tag}  (confidence={res.mean_confidence:.4f})")

    print()
    print("Demo complete.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audio_warp",
        description="Audio Watermarking & Authentication Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python -m audio_warp keygen --output keys/
              python -m audio_warp embed -i song.wav -o wm.wav -k keys/private.pem --owner 0102030405060708
              python -m audio_warp detect -i wm.wav -k keys/public.pem --owner 0102030405060708
              python -m audio_warp demo
        """),
    )
    sub = parser.add_subparsers(dest="command")

    # keygen
    kg = sub.add_parser("keygen", help="Generate Ed25519 key pair")
    kg.add_argument("--output", "-o", default="keys", help="Output directory (default: keys/)")

    # embed
    em = sub.add_parser("embed", help="Embed watermark into a WAV file")
    em.add_argument("--input",  "-i", required=True, help="Input WAV path")
    em.add_argument("--output", "-o", required=True, help="Output WAV path")
    em.add_argument("--key",    "-k", required=True, help="Private key PEM file")
    em.add_argument("--owner",         required=True, help="Owner ID as 16-char hex string")

    # detect
    dt = sub.add_parser("detect", help="Detect and verify watermark")
    dt.add_argument("--input",  "-i", required=True, help="Input WAV path")
    dt.add_argument("--key",    "-k", required=True, help="Public key PEM file")
    dt.add_argument("--owner",         required=True, help="Owner ID as 16-char hex string")

    # attack
    at = sub.add_parser("attack", help="Apply a signal-processing attack")
    at.add_argument("--input",  "-i", required=True, help="Input WAV path")
    at.add_argument("--output", "-o", required=True, help="Output WAV path")
    at.add_argument("--type",   "-t", required=True,
                    choices=["noise", "lowpass", "resample", "scale", "timeshift", "compress"],
                    help="Attack type")
    at.add_argument("--param",  "-p", type=float, default=None, help="Attack parameter value")

    # demo
    sub.add_parser("demo", help="Run full end-to-end demonstration")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "keygen":  _cmd_keygen,
        "embed":   _cmd_embed,
        "detect":  _cmd_detect,
        "attack":  _cmd_attack,
        "demo":    _cmd_demo,
    }
    dispatch[args.command](args)
