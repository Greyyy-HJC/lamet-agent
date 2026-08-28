"""Inspect signed-z coverage and signal/noise before tail fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import lattice_spacing_fm
from lamet_agent.stages.fourier_transform.physics import complete_signed_z, load_data


def derive_conventions(attrs: Mapping[str, object], *, target_observable: str, sector: str) -> dict[str, object]:
    """Return the conventions selected implicitly by the reference pipeline."""
    target = str(target_observable).lower()
    parton = str(attrs.get("parton", "")).lower()
    gfix = str(attrs.get("gfix", "")).upper()
    polarization = str(attrs.get("polarization", "")).lower()
    if parton != "quark":
        raise ValueError("the migrated Fourier examples require parton='quark' provenance")
    if gfix not in {"GI", "CG"}:
        raise ValueError("Fourier input must carry GI or CG gfix provenance")
    if target == "da":
        if sector != "full":
            raise ValueError("DA Fourier transformation requires sector='full'")
        component, output_scale = "both", 1.0
    elif target == "pdf":
        if polarization not in {"unpolarized", "helicity", "transversity"}:
            raise ValueError("PDF Fourier input must carry supported polarization provenance")
        try:
            component, output_scale = {
                "valence": ("im" if polarization == "helicity" else "re", 2.0),
                "singlet": ("re" if polarization == "helicity" else "im", 2.0),
                "full": ("both", 1.0),
            }[sector]
        except KeyError as exc:
            raise ValueError("PDF Fourier sector must be valence, singlet, or full") from exc
    else:
        raise ValueError("the migrated Fourier conventions support PDF and DA targets")
    return {
        "parton": parton,
        "gfix": gfix,
        "symmetry": {"real": "even", "imag": "odd"},
        "transform": {"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
        "tail_models": ["gi_nla" if gfix == "GI" else "cg_nla"],
        "component": component,
        "output_scale": output_scale,
    }


def prepare(context: ToolContext) -> tuple[Any, float]:
    """Load and complete the coordinate-space input without requiring fit ranges."""
    existing = context.state.get("fourier_input")
    inspection = context.state.get("tail_inspection")
    if existing is not None and isinstance(inspection, dict):
        return existing, float(inspection["spacing_fm"])
    value = context.inputs["input"]
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Fourier input accepts one source")
        value = value[0]
    data = load_data(value)
    if data.attrs.get("coord_unit") != "fm":
        raise ValueError("Fourier input coordinates must be in fm")
    momentum = data.attrs.get("momentum_gev")
    if (
        not isinstance(momentum, (int, float))
        or isinstance(momentum, bool)
        or not math.isfinite(float(momentum))
        or float(momentum) <= 0
    ):
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
    differences = np.diff(positive)
    grid_step = float(differences[0])
    if grid_step <= 0 or not np.allclose(differences, grid_step, rtol=0.0, atol=1e-12):
        raise ValueError("Fourier input z coordinates must be uniformly spaced")
    stored = lattice_spacing_fm(attrs=data.attrs, ensemble=data.ensemble)
    if stored is None:
        spacing = grid_step
    elif not math.isclose(stored, grid_step, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Fourier input lattice_spacing_fm does not match the z-grid step")
    else:
        spacing = stored
    context.state["fourier_input"] = data
    context.state["fourier_conventions"] = conventions
    context.state["tail_inspection"] = {
        "z_min_fm": min(data.coords["z"]),
        "z_max_fm": max(data.coords["z"]),
        "n_z": len(data.coords["z"]),
        "n_sample": data.n_sample,
        "coord_unit": data.attrs["coord_unit"],
        "spacing_fm": spacing,
    }
    return data, spacing


def run(context: ToolContext) -> dict[str, object]:
    """Inspect the prepared input and validate the effective fit ranges."""
    data, spacing = prepare(context)
    z = np.asarray(data.coords["z"], dtype=float)
    parameters = context.params
    if float(parameters["zmax_ext_fm"]) < float(np.max(np.abs(z))) - 1e-12:
        raise ValueError("zmax_ext_fm cannot be smaller than the input z coverage")
    grid_values = [*parameters["zmin_fm"], *parameters["zmax_fm"]]
    if any(
        not math.isclose(round(float(value) / spacing) * spacing, float(value), rel_tol=0.0, abs_tol=1e-12)
        for value in grid_values
    ):
        raise ValueError("Fourier fit boundaries and tail extent must lie on the input z grid")
    input_max = float(np.max(np.abs(z)))
    if any(float(value) > input_max + 1e-12 for value in parameters["zmax_fm"]):
        raise ValueError("every zmax_fm candidate must be covered by the input z grid")
    return {
        "summary": "inspected and completed signed z grid",
        "metrics": context.state["tail_inspection"],
        "state_keys": ["fourier_input", "fourier_conventions", "tail_inspection"],
        "artifacts": [],
    }


__all__ = ["derive_conventions", "prepare", "run"]
