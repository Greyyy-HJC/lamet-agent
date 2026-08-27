"""Inspect raw correlators and plan the effective Lanczos moment grid."""

from __future__ import annotations

import warnings

from lamet_agent.agent import ToolContext
from lamet_agent.stages.correlator_analysis.hook import ensure_raw_correlators
from lamet_agent.parallel.lanczos import prepare_lanczos_data


def run(context: ToolContext) -> dict[str, object]:
    """Prepare and report the configuration-level Lanczos input contract."""
    if context.params.get("analysis_method") != "lanczos":
        raise ValueError("inspect_lanczos_inputs is only available for Lanczos jobs")
    settings = context.params
    prepared = prepare_lanczos_data(
        ensure_raw_correlators(context),
        scope=str(settings["scope"]),
        t0=settings.get("t0"),
        time_step=settings.get("time_step"),
    )
    inspection = prepared["inspection"]
    warning = inspection.get("point_usage_warning")
    if warning and inspection.get("point_usage", {}).get("discarded_per_z", 0):
        warnings.warn(str(warning), UserWarning, stacklevel=2)
    context.state["lanczos_prepared"] = prepared
    context.state["lanczos_inspection"] = inspection
    return {
        "summary": "validated raw correlators and planned the effective Lanczos grid",
        "metrics": inspection,
        "state_keys": ["raw_correlators", "lanczos_prepared", "lanczos_inspection"],
        "artifacts": [],
    }
