"""Cryptographic operations: Ed25519 keys, signing, verification, SHA-256 hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a fresh Ed25519 key pair."""
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def save_private_key(key: Ed25519PrivateKey, path: str) -> None:
    """Serialise a private key to PEM on disk."""
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    Path(path).write_bytes(pem)


def save_public_key(key: Ed25519PublicKey, path: str) -> None:
    """Serialise a public key to PEM on disk."""
    pem = key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    Path(path).write_bytes(pem)


def load_private_key(path: str) -> Ed25519PrivateKey:
    """Load a PEM-encoded Ed25519 private key."""
    pem = Path(path).read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Not an Ed25519 private key")
    return key


def load_public_key(path: str) -> Ed25519PublicKey:
    """Load a PEM-encoded Ed25519 public key."""
    pem = Path(path).read_bytes()
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("Not an Ed25519 public key")
    return key


# ---------------------------------------------------------------------------
# Signing / verification
# ---------------------------------------------------------------------------

def sign(private_key: Ed25519PrivateKey, data: bytes) -> bytes:
    """Sign *data* with Ed25519.  Returns a 64-byte signature."""
    return private_key.sign(data)


def verify(public_key: Ed25519PublicKey, data: bytes, signature: bytes) -> bool:
    """Return True if *signature* is valid for *data* under *public_key*."""
    try:
        public_key.verify(signature, data)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def audio_hash(audio: np.ndarray) -> bytes:
    """SHA-256 digest of raw audio sample bytes."""
    return hashlib.sha256(audio.tobytes()).digest()
