"""Summarize explicitly scoped upstream results."""

from __future__ import annotations

from pathlib import Path

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData


def _summary(value):
    if isinstance(value, Path):
        if value.suffix.lower() == ".nc":
            value = EnsembleData.from_netcdf(value)
        else:
            return {"type": "Path", "value": str(value)}
    if value.__class__.__name__ == "EnsembleData":
        return {
            "type": "EnsembleData",
            "name": value.name,
            "dims": value.dims,
            "coords": {key: list(values) for key, values in value.coords.items()},
            "n_sample": value.n_sample,
            "resample": value.resample,
            "ensemble": None if value.ensemble is None else value.ensemble._asdict(),
            "attrs": value.attrs,
        }
    return {"type": type(value).__name__, "value": str(value)}


def run(context: ToolContext) -> dict[str, object]:
    """Store compact result summaries without implicit job-history access."""
    results = context.inputs["results"]
    if not isinstance(results, list) or not results:
        raise ValueError("review results must be a nonempty list")
    summary = [_summary(value) for value in results]
    for index, item in enumerate(summary):
        item["terminal_summary"] = (
            context.input_summaries["results"][index]
            if isinstance(context.input_summaries.get("results"), list)
            and index < len(context.input_summaries["results"])
            else None
        )
    context.state["result_summary"] = summary
    return {
        "summary": f"inspected {len(summary)} scoped results",
        "metrics": {
            "result_count": len(summary),
            "dims": [item.get("dims") for item in summary],
            "terminal_summaries": [item.get("terminal_summary") for item in summary],
        },
        "state_keys": ["result_summary"],
        "artifacts": [],
    }
