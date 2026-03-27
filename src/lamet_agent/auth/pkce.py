"""PKCE helper functions for authorization-code OAuth flows."""

from __future__ import annotations

import base64
import hashlib
import secrets


def generate_code_verifier(length: int = 64) -> str:
    """Return a high-entropy code verifier suitable for PKCE."""
    return secrets.token_urlsafe(length)[:128]


def build_code_challenge(code_verifier: str) -> str:
    """Return the S256 code challenge for a verifier."""
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def generate_state(length: int = 32) -> str:
    """Return a random OAuth state parameter."""
    return secrets.token_urlsafe(length)
