"""Assemble the complete upstream evidence bundle for Review."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData


def _summary(value):
    if isinstance(value, Path):
        if value.suffix.lower() == ".nc":
            value = EnsembleData.from_netcdf(value)
        else:
            return {"type": "Path", "value": str(value)}
    if isinstance(value, EnsembleData):
        return {
            "type": "EnsembleData",
            "name": value.name,
            "dims": list(value.dims),
            "coords": {
                key: [item.item() if hasattr(item, "item") else item for item in values]
                for key, values in value.coords.items()
            },
            "n_sample": value.n_sample,
            "resample": value.resample,
            "ensemble": None if value.ensemble is None else value.ensemble._asdict(),
            "attrs": json.loads(
                json.dumps(
                    value.attrs,
                    default=lambda item: item.item() if hasattr(item, "item") else str(item),
                )
            ),
        }
    return {"type": type(value).__name__, "value": str(value)}


def run(context: ToolContext) -> dict[str, object]:
    """Write one review bundle containing every preceding report and selected result."""
    results = context.inputs["results"]
    stages = context.manifest["stages"]
    review_job = next(job for job in stages[context.stage_id]["jobs"] if job["id"] == context.job_id)
    authored_sources = review_job["inputs"]["results"]
    artifact_base = context.artifact_directory.parent.parent
    jobs: dict[str, dict[str, object]] = {}
    stage_reports = []
    job_reports = []

    for stage_index, (stage_id, block) in enumerate(stages.items(), start=1):
        if stage_id == context.stage_id:
            break
        stage_directory = artifact_base / f"{stage_index:02d}_{stage_id}"
        stage_report = stage_directory / "report.md"
        stage_reports.append(
            {
                "stage_id": stage_id,
                "path": Path(os.path.relpath(stage_report, context.artifact_directory)).as_posix(),
                "available": stage_report.is_file(),
                "sha256": hashlib.sha256(stage_report.read_bytes()).hexdigest() if stage_report.is_file() else None,
                "text": stage_report.read_text(encoding="utf-8") if stage_report.is_file() else "",
            }
        )
        for job in block["jobs"]:
            job_id = job["id"]
            job_directory = stage_directory / job_id
            report_path = job_directory / "report.md"
            summary_path = job_directory / "summary.json"
            terminal = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
            declared = terminal.get("artifacts", []) if terminal else []
            artifacts = [
                {
                    "path": Path(os.path.relpath(job_directory / relative, context.artifact_directory)).as_posix(),
                    "available": (job_directory / relative).is_file(),
                }
                for relative in declared
            ]
            jobs[job_id] = {
                "job_id": job_id,
                "stage_id": stage_id,
                "stage_index": stage_index,
                "inputs": job["inputs"],
                "params": {key: value for key, value in job.items() if key not in {"id", "inputs"}},
                "artifact_directory": Path(os.path.relpath(job_directory, context.artifact_directory)).as_posix(),
            }
            job_reports.append(
                {
                    "stage_id": stage_id,
                    "job_id": job_id,
                    "path": Path(os.path.relpath(report_path, context.artifact_directory)).as_posix(),
                    "available": report_path.is_file(),
                    "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest() if report_path.is_file() else None,
                    "text": report_path.read_text(encoding="utf-8") if report_path.is_file() else "",
                    "terminal_summary": terminal,
                    "artifacts": artifacts,
                }
            )

    summaries = []
    terminal_summaries = context.input_summaries.get("results", [])
    for index, value in enumerate(results):
        item = _summary(value)
        source = authored_sources[index]
        if isinstance(source, str):
            item.update({"job_id": source, "stage_id": jobs[source]["stage_id"]})
        else:
            item.update({"job_id": None, "stage_id": "external", "source": source})
        item["terminal_summary"] = terminal_summaries[index] if index < len(terminal_summaries) else None
        summaries.append(item)

    bundle = {
        "run_id": context.manifest["metadata"]["run_id"],
        "review_job_id": context.job_id,
        "manifest_path": str(context.manifest_path),
        "stage_reports": stage_reports,
        "job_reports": job_reports,
        "results": summaries,
        "jobs": jobs,
    }
    (context.artifact_directory / "review_bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    context.state["result_summary"] = summaries
    context.state["review_bundle"] = bundle
    return {
        "summary": f"inspected {len(summaries)} results and collected {len(stage_reports)} stage reports",
        "metrics": {
            "result_count": len(summaries),
            "stage_report_count": len(stage_reports),
            "job_report_count": len(job_reports),
            "missing_stage_reports": [item["stage_id"] for item in stage_reports if not item["available"]],
        },
        "reports": [
            {"stage_id": item["stage_id"], "path": item["path"], "text": item["text"]} for item in stage_reports
        ],
        "results": summaries,
        "state_keys": ["result_summary", "review_bundle"],
        "artifacts": ["review_bundle.json"],
    }
