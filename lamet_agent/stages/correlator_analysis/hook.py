"""Resolve omitted correlator parameters from runtime data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypedDict


class Pt2Window(TypedDict):
    """One half-open two-point fit window."""

    tmin: int
    tmax: int


class Pt3Window(TypedDict):
    """One three-point source-sink and insertion-time selection."""

    tsep_ls: list[int]
    tau_cut: int


def ensure_raw_correlators(
    context: Any,
    correlator_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load the selected configuration-level correlators once for a job."""
    requested = set(
        correlator_ids
        if correlator_ids is not None
        else context.params["correlator_ids"]
    )
    existing = context.state.get("raw_correlators")
    if isinstance(existing, dict):
        if requested != set(existing):
            raise ValueError(
                "correlators were already prepared with a different selection"
            )
        return existing

    from lamet_agent.stages.correlator_analysis.input import (
        load_descriptor,
    )

    source = context.inputs["correlators"]
    if isinstance(source, list):
        if len(source) != 1:
            raise ValueError("one correlator descriptor source is required")
        source = source[0]
    if not isinstance(source, Path):
        raise TypeError("correlators input must resolve to a descriptor Path")
    if not requested:
        raise ValueError("at least one correlator must be selected")

    loaded = load_descriptor(source)
    unknown = requested - set(loaded["correlators"])
    if unknown:
        raise ValueError(f"unknown correlator ids: {sorted(unknown)}")
    context.state["correlator_descriptor_path"] = source
    context.state["correlator_records"] = {
        record["id"]: record
        for record in loaded["descriptor"]["correlators"]
        if record["id"] in requested
    }
    selected = {
        key: value
        for key, value in loaded["correlators"].items()
        if key in requested
    }
    context.state["correlator_configuration_ids"] = list(
        loaded.get("configuration_ids", [])
    )
    context.state["raw_correlators"] = selected
    return selected


def ensure_correlators(
    context: Any,
    correlator_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load and resample the selected correlators once for a job."""
    from lamet_agent.stages.correlator_analysis.input import resample_correlators

    requested = set(
        correlator_ids
        if correlator_ids is not None
        else context.params["correlator_ids"]
    )
    existing = context.state.get("correlators")
    if isinstance(existing, dict):
        if requested != set(existing):
            raise ValueError(
                "correlators were already prepared with a different selection"
            )
        return existing

    raw = ensure_raw_correlators(context, correlator_ids)
    resampled = resample_correlators(
        {
            "correlators": raw,
            "configuration_ids": context.state.get(
                "correlator_configuration_ids", []
            ),
        },
        mode=context.manifest["metadata"]["resample_mode"],
        group=context.params["resample_group"],
        bin_size=context.manifest["metadata"]["bin_size"],
        n_boot=context.manifest["metadata"].get("bootstrap_samples"),
        seed=int(context.manifest["metadata"]["random_seed"]),
    )
    context.state["correlators"] = resampled
    return resampled


def _mean_error(
    data: Any,
    component: str,
    sample_error_mode: str,
) -> tuple[list[Any], list[Any]]:
    import gvar as gv
    import numpy as np

    selected = data.real if component == "real" else data.imag
    average = selected.average(sample_error_mode)
    return (
        np.asarray(gv.mean(average), dtype=float).tolist(),
        np.asarray(gv.sdev(average), dtype=float).tolist(),
    )


def recommend_pt2_windows(
    context: Any,
    ask: Callable[..., Any],
) -> list[Pt2Window]:
    """Ask the model for two-point windows using direct means and errors."""
    correlators = ensure_correlators(context)
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    records = []
    for correlator_id, data in correlators.items():
        if data.attrs.get("correlator_type") != "two_point":
            continue
        if data.dims != ["t"]:
            raise ValueError("two-point recommendation data must have only the t dimension")
        mean, error = _mean_error(data, "real", sample_error_mode)
        records.append(
            {
                "correlator_id": correlator_id,
                "t": [int(value) for value in data.coords["t"]],
                "mean": mean,
                "error": error,
            }
        )
    if not records:
        raise ValueError("two-point window recommendation requires two-point data")
    time_range = context.params["lsqfit"]["time_range"]
    evidence = {
        "two_point_correlators": records,
        "allowed_time_range": {
            "min": int(time_range["min"]),
            "max": int(time_range["max"]),
        },
        "minimum_points": 2 * max(int(value) for value in context.params["nstate"]),
        "window_convention": "tmin is inclusive and tmax is exclusive",
    }
    return ask(
        instruction=(
            "Select a nonempty ordered list of two-point fit windows. Use later-time "
            "regions with stable central behavior while retaining useful precision. "
            "Every window must lie inside allowed_time_range, have tmin < tmax, and "
            "contain at least minimum_points available time coordinates. Include a "
            "small set of defensible alternatives when the onset of stability is ambiguous."
        ),
        evidence=evidence,
    )


def recommend_pt3_windows(
    context: Any,
    ask: Callable[..., Any],
) -> list[Pt3Window]:
    """Ask the model for three-point windows using direct means and errors."""
    import numpy as np

    correlators = ensure_correlators(context)
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    three_points = [
        (correlator_id, data)
        for correlator_id, data in correlators.items()
        if data.attrs.get("correlator_type") == "three_point"
    ]
    if len(three_points) != 1:
        raise ValueError("three-point window recommendation requires exactly one three-point correlator")
    correlator_id, data = three_points[0]
    if data.dims != ["tsep", "tau", "z"]:
        raise ValueError("three-point recommendation data must have tsep, tau, and z dimensions")
    z_values = np.asarray(data.coords["z"], dtype=float)
    tune_z = float(context.params["lsqfit"]["tune_z"])
    matches = np.flatnonzero(np.isclose(z_values, tune_z, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise ValueError("lsqfit.tune_z must name exactly one available z coordinate")
    tuned = data.at("z", data.coords["z"][int(matches[0])])
    requested_components = {
        "re": ("real",),
        "im": ("imag",),
        "both": ("real", "imag"),
    }[context.params["component"]]
    components = {}
    for component in requested_components:
        mean, error = _mean_error(tuned, component, sample_error_mode)
        components[component] = {"mean": mean, "error": error}
    evidence = {
        "correlator_id": correlator_id,
        "tune_z": tune_z,
        "tsep": [int(value) for value in tuned.coords["tsep"]],
        "tau": [int(value) for value in tuned.coords["tau"]],
        "components": components,
        "fit_scope": context.params["lsqfit"]["fit_scope"],
        "constraint": "Each tau_cut must satisfy 2*tau_cut <= every selected tsep.",
    }
    return ask(
        instruction=(
            "Select a nonempty ordered list of three-point windows. Each window must "
            "contain a nonempty unique subset of the available tsep values and one "
            "nonnegative tau_cut. Exclude source and sink contamination while retaining "
            "a visible central signal with useful precision. Use distinct tau_cut values "
            "for alternative windows. An FH scope needs at least two selected tsep values."
        ),
        evidence=evidence,
    )


__all__ = [
    "recommend_pt2_windows",
    "recommend_pt3_windows",
    "ensure_correlators",
    "ensure_raw_correlators",
]
