"""Inspect signed-z coverage and signal/noise before tail fitting."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.stages.fourier_transform.physics import complete_signed_z, load_data


def effective_zmin_fm(context: ToolContext, data: Any) -> list[float]:
    """Apply the job's lattice-site offset to its current lower-range candidates."""
    offset = int(context.params["tail_window_step_offset"])
    spacing = float(data.ensemble.a_s)
    return [round(float(value) + offset * spacing, 12) for value in context.params["zmin_fm"]]


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
    elif target == "gpd":
        if polarization not in {"unpolarized", "helicity", "transversity"}:
            raise ValueError("GPD Fourier input must carry supported polarization provenance")
        if sector not in {"sea", "valence", "singlet", "full"}:
            raise ValueError("GPD Fourier sector must be sea, valence, singlet, or full")
        component, output_scale = "both", 1.0
    else:
        raise ValueError("the migrated Fourier conventions support PDF, DA, and GPD targets")
    return {
        "parton": parton,
        "gfix": gfix,
        "symmetry": {"real": "even", "imag": "odd"},
        "transform": {"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
        "tail_models": ["gi_nla" if gfix == "GI" else "cg_nla"],
        "component": component,
        "output_scale": output_scale,
    }


def _momentum_vector(data: Any, key: str) -> tuple[int, int, int]:
    value = data.attrs.get(key)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Fourier GPD input requires {key} provenance")
    return tuple(int(component) for component in value)


def _gpd_signed_pair(
    primary: Any,
    partner: Any,
    *,
    bilocal_anchor: str,
    delta_momentum_gev: float,
) -> Any:
    """Complete a GPD matrix element with its exchanged-flow Hermitian partner."""
    if primary.dims != ["z"] or partner.dims != ["z"]:
        raise ValueError("GPD Fourier inputs must have exactly one z dimension")
    if primary.ensemble != partner.ensemble or primary.resample != partner.resample:
        raise ValueError("GPD input and hermitian_partner must share ensemble and resampling")
    z = np.asarray(primary.coords["z"], dtype=float)
    partner_z = np.asarray(partner.coords["z"], dtype=float)
    if z.shape != partner_z.shape or not np.allclose(z, partner_z, rtol=0.0, atol=1e-12):
        raise ValueError("GPD input and hermitian_partner must share the z grid")
    if bilocal_anchor not in {"mid_at_0", "barpsi_at_0", "psi_at_0"}:
        raise ValueError("bilocal_anchor must be mid_at_0, barpsi_at_0, or psi_at_0")
    positive_indices = np.where(z >= 0)[0]
    positive_z = z[positive_indices]
    if positive_z.size < 2 or np.any(np.diff(positive_z) <= 0):
        raise ValueError("GPD paired completion requires at least two nonnegative z points")
    primary_values = np.asarray(primary.values)[:, positive_indices].astype(complex)
    partner_values = np.asarray(partner.values)[:, positive_indices].astype(complex)
    phase = np.exp(0.5j * float(delta_momentum_gev) * positive_z / HBAR_C_GEV_FM)[None, :]
    if bilocal_anchor == "mid_at_0":
        target = primary_values * phase
        exchanged = partner_values * np.conjugate(phase)
    elif bilocal_anchor == "barpsi_at_0":
        target = primary_values
        exchanged = partner_values
    else:
        target = np.conjugate(partner_values)
        exchanged = np.conjugate(primary_values)
    negative_z = -positive_z[positive_z > 0][::-1]
    output_z = np.concatenate([negative_z, positive_z])
    values = []
    for sample_target, sample_exchanged in zip(target, exchanged):
        negative = np.conjugate(sample_exchanged[positive_z > 0][::-1])
        if np.isclose(positive_z[0], 0.0):
            center = 0.5 * (sample_target[0] + np.conjugate(sample_exchanged[0]))
            values.append(np.concatenate([negative, [center], sample_target[1:]]))
        else:
            values.append(np.concatenate([negative, sample_target]))
    attrs = dict(primary.attrs)
    attrs.update(
        {
            "symmetry": json.dumps({"real": "explicit", "imag": "explicit"}, sort_keys=True),
            "signed_z_completion": "gpd_hermitian_partner",
            "bilocal_anchor": bilocal_anchor,
            "hermitian_partner_id": str(partner.attrs.get("correlator_id", partner.name or "")),
            "delta_momentum_gev": float(delta_momentum_gev),
            "gpd_completion_mode": "paired_flow",
        }
    )
    from lamet_agent.data import EnsembleData

    return EnsembleData(
        primary.ensemble,
        primary.resample,
        values,
        ["z"],
        {"z": output_z.tolist()},
        attrs=attrs,
        name=primary.name,
    )


def prepare(context: ToolContext) -> tuple[Any, float]:
    """Load and complete the coordinate-space input without requiring fit ranges."""
    existing = context.state.get("fourier_input")
    inspection = context.state.get("tail_inspection")
    if existing is not None and isinstance(inspection, dict):
        return existing, float(inspection["z_grid_step_fm"])
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
    target = str(context.manifest["metadata"]["target_observable"]).lower()
    conventions = derive_conventions(
        data.attrs,
        target_observable=str(context.manifest["metadata"]["target_observable"]),
        sector=str(context.params["scheme_scan"]["sector"]),
    )
    if conventions["parton"] != context.manifest["metadata"]["parton"]:
        raise ValueError("Fourier input parton provenance conflicts with metadata.parton")
    if target == "gpd":
        partner_value = context.inputs.get("hermitian_partner")
        source_momentum = _momentum_vector(data, "source_momentum")
        sink_momentum = _momentum_vector(data, "sink_momentum")
        delta_momentum = float(data.ensemble.k_s) * (sink_momentum[2] - source_momentum[2])
        initial_pz = float(source_momentum[2]) * float(data.ensemble.k_s)
        final_pz = float(sink_momentum[2]) * float(data.ensemble.k_s)
        average_momentum = 0.5 * float(data.ensemble.k_s) * (
            np.linalg.norm(source_momentum) + np.linalg.norm(sink_momentum)
        )
        initial_energy = float(
            np.sqrt(
                float(data.ensemble.m_pi) ** 2
                + (float(data.ensemble.k_s) * np.linalg.norm(source_momentum)) ** 2
            )
        )
        final_energy = float(
            np.sqrt(
                float(data.ensemble.m_pi) ** 2
                + (float(data.ensemble.k_s) * np.linalg.norm(sink_momentum)) ** 2
            )
        )
        delta_spatial_sq = (float(data.ensemble.k_s) ** 2) * sum(
            (sink - source) ** 2 for sink, source in zip(sink_momentum, source_momentum)
        )
        attrs = dict(data.attrs)
        attrs.update(
            {
                "initial_momentum": json.dumps(source_momentum),
                "final_momentum": json.dumps(sink_momentum),
                "delta_momentum_gev": delta_momentum,
                "phase_momentum_source": "ensemble_discrete_momentum",
                "momentum_gev": average_momentum,
                "xi": (final_pz - initial_pz) / (final_pz + initial_pz)
                if not np.isclose(final_pz + initial_pz, 0.0)
                else float("nan"),
                "t_gev2": (final_energy - initial_energy) ** 2 - delta_spatial_sq,
            }
        )
        from lamet_agent.data import EnsembleData

        data = EnsembleData(
            data.ensemble,
            data.resample,
            [sample for sample in data.values],
            data.dims,
            data.coords,
            attrs=attrs,
            name=data.name,
        )
        if partner_value is None:
            if source_momentum != sink_momentum:
                raise ValueError("non-forward GPD input requires hermitian_partner")
            data = complete_signed_z(data, conventions["symmetry"])
            attrs = dict(data.attrs)
            attrs.update({"gpd_completion_mode": "single_flow", "hermitian_partner_id": ""})
            data = EnsembleData(
                data.ensemble,
                data.resample,
                [sample for sample in data.values],
                data.dims,
                data.coords,
                attrs=attrs,
                name=data.name,
            )
        else:
            partner = load_data(partner_value)
            if partner.attrs.get("coord_unit") != "fm":
                raise ValueError("GPD hermitian_partner coordinates must be in fm")
            partner_source = _momentum_vector(partner, "source_momentum")
            partner_sink = _momentum_vector(partner, "sink_momentum")
            if partner_source != sink_momentum or partner_sink != source_momentum:
                raise ValueError("GPD hermitian_partner must exchange source and sink momenta")
            data = _gpd_signed_pair(
                data,
                partner,
                bilocal_anchor=str(context.params["bilocal_anchor"]),
                delta_momentum_gev=delta_momentum,
            )
            attrs = dict(data.attrs)
            attrs["hermitian_partner_id"] = (
                str(partner_value) if isinstance(partner_value, str) else attrs.get("hermitian_partner_id", "")
            )
            data = EnsembleData(
                data.ensemble,
                data.resample,
                [sample for sample in data.values],
                data.dims,
                data.coords,
                attrs=attrs,
                name=data.name,
            )
            context.state["fourier_partner_input"] = partner
    else:
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
    context.state["fourier_input"] = data
    context.state["fourier_conventions"] = conventions
    context.state["tail_inspection"] = {
        "z_min_fm": min(data.coords["z"]),
        "z_max_fm": max(data.coords["z"]),
        "n_z": len(data.coords["z"]),
        "n_sample": data.n_sample,
        "coord_unit": data.attrs["coord_unit"],
        "z_grid_step_fm": grid_step,
    }
    return data, grid_step


def run(context: ToolContext) -> dict[str, object]:
    """Inspect the prepared input and validate the effective fit ranges."""
    data, z_grid_step = prepare(context)
    z = np.asarray(data.coords["z"], dtype=float)
    parameters = context.params
    effective_zmin = effective_zmin_fm(context, data)
    if any(value < 0 for value in effective_zmin):
        raise ValueError("tail_window_step_offset shifts zmin_fm below zero")
    if float(parameters["zmax_ext_fm"]) < float(np.max(np.abs(z))) - 1e-12:
        raise ValueError("zmax_ext_fm cannot be smaller than the input z coverage")
    positive_z = z[z >= 0]
    grid_values = [*effective_zmin, *parameters["zmax_fm"]]
    if any(
        not np.any(np.isclose(positive_z, float(value), rtol=0.0, atol=1e-12))
        for value in grid_values
    ):
        raise ValueError("effective Fourier fit boundaries must lie on the input z grid")
    input_max = float(np.max(np.abs(z)))
    if any(float(value) > input_max + 1e-12 for value in parameters["zmax_fm"]):
        raise ValueError("every zmax_fm candidate must be covered by the input z grid")
    state_keys = ["fourier_input", "fourier_conventions", "tail_inspection"]
    if "fourier_partner_input" in context.state:
        state_keys.append("fourier_partner_input")
    return {
        "summary": "inspected and completed signed z grid",
        "metrics": context.state["tail_inspection"],
        "state_keys": state_keys,
        "artifacts": [],
    }


__all__ = ["derive_conventions", "effective_zmin_fm", "prepare", "run"]
