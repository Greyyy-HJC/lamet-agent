"""Resolve execution-only renormalization parameter defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings


def effective_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return flat authored parameters with the implicit external apply type."""
    effective = dict(params)
    if params.get("strategy") == "external_denominator":
        effective["type"] = "apply"
    elif params.get("strategy") == "self_renormalization":
        effective.setdefault("z_coverage_policy", "extrapolate")
        effective.setdefault("kernel_parameters", {})
    return effective


def authored_kernel_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return kernel overrides and warn when stage-context arguments are replaced."""
    values = dict(effective_params(params)["kernel_parameters"])
    if "z_fm" in values:
        raise ValueError("kernel_parameters.z_fm is supplied by input data and cannot be overridden")
    overridden = sorted({"mu"}.intersection(values))
    if overridden:
        warnings.warn(
            f"renormalization kernel_parameters overrides stage context: {overridden}",
            RuntimeWarning,
            stacklevel=2,
        )
    return values


__all__ = ["effective_params"]
