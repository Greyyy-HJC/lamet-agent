"""Load and align explicitly listed matched distributions."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.stages.extrapolation.physics import load_data


def run(context: ToolContext) -> dict[str, object]:
    """Load every distribution and expose compact coverage diagnostics."""
    if context.params["operation"] != "fit":
        raise ValueError("inspect_scaling is only available for operation='fit'")
    values = context.inputs["distributions"]
    if not isinstance(values, list) or not values:
        raise ValueError("distributions must be a nonempty source list")
    data = [load_data(value) for value in values]
    x = data[0].coords.get("x")
    if (
        x is None
        or not x
        or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in x)
        or any(float(right) <= float(left) for left, right in zip(x, x[1:]))
        or any(not np.allclose(item.coords.get("x", []), x, rtol=0.0, atol=1e-12) for item in data[1:])
    ):
        raise ValueError("all distributions must have identical x coordinates")
    required_attrs = (
        "lattice_spacing_fm",
        "L_s",
        "m_pi",
        "momentum_gev",
        "gfix",
        "kernel_operator",
        "parton",
        "target_observable",
        "renormalization_scheme",
        "kernel_id",
    )
    missing = {
        str(item.attrs.get("ensemble_id", index)): [key for key in required_attrs if key not in item.attrs]
        for index, item in enumerate(data)
    }
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"matched distributions are missing extrapolation provenance: {missing}")
    first_attrs = data[0].attrs
    for index, item in enumerate(data[1:], start=1):
        for key in ("gfix", "kernel_operator", "parton", "target_observable", "renormalization_scheme", "kernel_id"):
            if item.attrs.get(key) != first_attrs.get(key):
                raise ValueError(f"extrapolation provenance field {key} differs at distribution {index}")
    for item in data:
        for key in ("lattice_spacing_fm", "m_pi", "momentum_gev"):
            if (
                not isinstance(item.attrs.get(key), Real)
                or not math.isfinite(float(item.attrs[key]))
                or float(item.attrs[key]) <= 0
            ):
                raise ValueError(f"{key} must be finite and positive on every extrapolation input")
        if (
            not isinstance(item.attrs.get("L_s"), Real)
            or not math.isfinite(float(item.attrs["L_s"]))
            or float(item.attrs["L_s"]) <= 0
        ):
            raise ValueError("L_s must be finite and positive on every extrapolation input")
    context.state["scaling_data"] = data
    context.state["scaling_inspection"] = {
        "distribution_count": len(data),
        "x_count": len(x),
        "sample_count": data[0].n_sample,
        "ensembles": [item.attrs.get("ensemble", item.attrs.get("ensemble_id")) for item in data],
    }
    return {
        "summary": f"aligned {len(data)} distributions on one x grid",
        "metrics": context.state["scaling_inspection"],
        "state_keys": ["scaling_data", "scaling_inspection"],
        "artifacts": [],
    }
