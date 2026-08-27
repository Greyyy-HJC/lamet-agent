"""Resolve execution-only renormalization parameter defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def effective_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return flat authored parameters with the implicit external apply type."""
    effective = dict(params)
    if params.get("strategy") == "external_denominator":
        effective["type"] = "apply"
    return effective


__all__ = ["effective_params"]
