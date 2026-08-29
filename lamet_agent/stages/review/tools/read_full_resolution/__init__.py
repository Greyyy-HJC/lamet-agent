"""Return full-resolution numerical evidence for one reviewed job."""

from __future__ import annotations

from pathlib import Path

from lamet_agent.agent import ToolContext, _review_output_summary
from lamet_agent.data import EnsembleData


def run(context: ToolContext, *, job_id: str) -> dict[str, object]:
    """Load one selected result at full resolution without expanding every job."""
    review_job = next(
        job
        for job in context.manifest["stages"][context.stage_id]["jobs"]
        if job["id"] == context.job_id
    )
    sources = review_job["inputs"]["results"]
    matches = [index for index, source in enumerate(sources) if source == job_id]
    if len(matches) != 1:
        raise ValueError("job_id must identify exactly one job selected by this Review")
    results = context.inputs["results"]
    value = results[matches[0]]
    if isinstance(value, Path) and value.suffix.lower() == ".nc":
        value = EnsembleData.from_netcdf(value)
    if not isinstance(value, EnsembleData):
        raise TypeError("full-resolution review data requires an EnsembleData result")
    bundle = context.state["review_bundle"]
    job = bundle["jobs"][job_id]
    mode = str(context.manifest["metadata"]["sample_error_mode"])
    reads = context.state.setdefault("full_resolution_reads", [])
    reads.append(job_id)
    return {
        "summary": f"loaded full-resolution data for {job_id}",
        "job_id": job_id,
        "stage_id": job["stage_id"],
        "output": _review_output_summary(value, mode, max_points=None),
        "state_keys": ["full_resolution_reads"],
    }


__all__ = ["run"]
