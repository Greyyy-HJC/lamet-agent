"""Resolve omitted correlator parameters from runtime data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_raw_correlators(
    context: Any,
    correlator_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load the selected configuration-level correlators once for a job."""
    requested = set(correlator_ids if correlator_ids is not None else context.params["correlator_ids"])
    existing = context.state.get("raw_correlators")
    if isinstance(existing, dict):
        if requested != set(existing):
            raise ValueError("correlators were already prepared with a different selection")
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
    ensemble = loaded["descriptor"].get("ensemble", {})
    context.state["correlator_resample_group"] = str(ensemble.get("id", context.job_id))
    context.state["correlator_records"] = {
        record["id"]: record for record in loaded["descriptor"]["correlators"] if record["id"] in requested
    }
    selected = {key: value for key, value in loaded["correlators"].items() if key in requested}
    context.state["correlator_configuration_ids"] = list(loaded.get("configuration_ids", []))
    context.state["raw_correlators"] = selected
    return selected


def ensure_correlators(
    context: Any,
    correlator_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load and resample the selected correlators once for a job."""
    from lamet_agent.stages.correlator_analysis.input import resample_correlators

    requested = set(correlator_ids if correlator_ids is not None else context.params["correlator_ids"])
    existing = context.state.get("correlators")
    if isinstance(existing, dict):
        if requested != set(existing):
            raise ValueError("correlators were already prepared with a different selection")
        return existing

    raw = ensure_raw_correlators(context, correlator_ids)
    resampled = resample_correlators(
        {
            "correlators": raw,
            "configuration_ids": context.state.get("correlator_configuration_ids", []),
        },
        mode=context.manifest["metadata"]["resample_mode"],
        group=str(context.state.get("correlator_resample_group", context.job_id)),
        bin_size=context.manifest["metadata"]["bin_size"],
        n_boot=context.manifest["metadata"].get("samples"),
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


__all__ = [
    "ensure_correlators",
    "ensure_raw_correlators",
]
