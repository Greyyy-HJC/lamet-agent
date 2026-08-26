"""Inspect signed-z coverage and signal/noise before tail fitting."""

from __future__ import annotations

import math

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.stages.fourier_transform.conventions import derive_conventions
from lamet_agent.stages.fourier_transform.physics import complete_signed_z, load_data


def run(context: ToolContext) -> dict[str, object]:
    """Load and complete the coordinate-space input."""
    value = context.inputs["input"]
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Fourier input accepts one source")
        value = value[0]
    data = load_data(value)
    if data.attrs.get("coord_unit") != "fm":
        raise ValueError("Fourier input coordinates must be in fm")
    momentum = data.attrs.get("momentum_gev")
    if not isinstance(momentum, (int, float)) or isinstance(momentum, bool) or not math.isfinite(float(momentum)) or float(momentum) <= 0:
        raise ValueError("Fourier input requires finite positive momentum_gev")
    conventions = derive_conventions(
        data.attrs,
        target_observable=str(context.manifest["metadata"]["target_observable"]),
        sector=str(context.params["scheme_scan"]["sector"]),
    )
    if conventions["parton"] != context.manifest["metadata"]["parton"]:
        raise ValueError("Fourier input parton provenance conflicts with metadata.parton")
    data = complete_signed_z(data, conventions["symmetry"])
    z = np.asarray(data.coords["z"], dtype=float)
    if z.size < 2 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("Fourier z coordinates must be finite and strictly increasing")
    positive = z[z >= 0]
    if positive.size < 2:
        raise ValueError("Fourier input requires at least two nonnegative z coordinates")
    spacing = float(np.diff(positive)[0])
    if spacing <= 0 or not np.allclose(np.diff(positive), spacing, rtol=0.0, atol=1e-12):
        raise ValueError("Fourier input z coordinates must be uniformly spaced")
    parameters = context.params
    if float(parameters["zmax_ext_fm"]) < float(np.max(np.abs(z))) - 1e-12:
        raise ValueError("zmax_ext_fm cannot be smaller than the input z coverage")
    grid_values = [
        *parameters["zmin_fm"],
        *parameters["zmax_fm"],
    ]
    if any(not math.isclose(round(float(value) / spacing) * spacing, float(value), rel_tol=0.0, abs_tol=1e-12) for value in grid_values):
        raise ValueError("Fourier fit boundaries and tail extent must lie on the input z grid")
    input_max = float(np.max(np.abs(z)))
    if any(float(value) > input_max + 1e-12 for value in parameters["zmax_fm"]):
        raise ValueError("every zmax_fm candidate must be covered by the input z grid")
    context.state["fourier_input"] = data
    context.state["fourier_conventions"] = conventions
    context.state["tail_inspection"] = {"z_min_fm": min(data.coords["z"]), "z_max_fm": max(data.coords["z"]), "n_z": len(data.coords["z"]), "n_sample": data.n_sample, "coord_unit": data.attrs["coord_unit"], "spacing_fm": spacing}
    return {"summary": "inspected and completed signed z grid", "metrics": context.state["tail_inspection"], "state_keys": ["fourier_input", "fourier_conventions", "tail_inspection"], "artifacts": []}
