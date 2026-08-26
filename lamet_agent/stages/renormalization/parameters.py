"""Resolve the active real Renormalization provider nodes for execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def effective_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return common parameters overlaid by the selected strategy and scheme."""
    effective = dict(params)
    strategy = params.get("strategy")
    strategy_params = params.get(strategy) if isinstance(strategy, str) else None
    if not isinstance(strategy_params, Mapping):
        return effective
    effective.update(strategy_params)
    scheme = strategy_params.get("scheme")
    scheme_params = strategy_params.get(scheme) if isinstance(scheme, str) else None
    if isinstance(scheme_params, Mapping):
        effective.update(scheme_params)
    return effective


__all__ = ["effective_params"]
