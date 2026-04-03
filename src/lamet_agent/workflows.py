"""Workflow planning and execution entry points."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.artifacts import ArtifactRecord, StageResult
from lamet_agent.errors import StageExecutionError
from lamet_agent.kernel import load_kernel
from lamet_agent.loaders import load_all_correlators
from lamet_agent.reporting import write_run_report, write_stage_summary
from lamet_agent.schemas import Manifest, load_manifest
from lamet_agent.stages import __all__ as _registered_stages  # noqa: F401
from lamet_agent.stages.base import StageContext
from lamet_agent.stages.registry import get_stage
from lamet_agent.utils import ensure_directory, timestamp_slug


@dataclass(slots=True)
class WorkflowPlan:
    """Resolved workflow plan for a given manifest."""

    goal: str
    stage_names: list[str]
    final_observable: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan for CLI output."""
        return {
            "goal": self.goal,
            "stage_names": self.stage_names,
            "final_observable": self.final_observable,
        }


@dataclass(slots=True)
class WorkflowRun:
    """Result of executing a workflow plan."""

    manifest: Manifest
    plan: WorkflowPlan
    run_directory: Path
    stage_results: list[StageResult]


def _restore_matrix_element_family_samples(payload_key: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild hidden sample-wise coordinate-space family payloads from saved artifacts."""
    restored: list[dict[str, Any]] = []
    for family in payload.get(payload_key, []):
        sample_artifact = family.get("sample_artifact")
        if not sample_artifact:
            continue
        sample_path = Path(sample_artifact)
        sample_dump = np.load(sample_path)
        restored.append(
            {
                "metadata": dict(family["metadata"]),
                "z_axis": np.asarray(sample_dump["z_axis"], dtype=float),
                "sample_count": int(family.get("sample_count", sample_dump["real_samples"].shape[0])),
                "real_mean": np.asarray(family["real"]["mean"], dtype=float),
                "real_error": np.asarray(family["real"]["error"], dtype=float),
                "imag_mean": np.asarray(family["imag"]["mean"], dtype=float),
                "imag_error": np.asarray(family["imag"]["error"], dtype=float),
                "real_samples": np.asarray(sample_dump["real_samples"], dtype=float),
                "imag_samples": np.asarray(sample_dump["imag_samples"], dtype=float),
                "sample_artifact": str(sample_path),
            }
        )
    return restored


def _restore_transformed_family_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Restore hidden sample-wise x-space arrays for transformed families."""
    restored: list[dict[str, Any]] = []
    for family in payload.get("transformed_families", []):
        sample_artifact = family.get("sample_artifact")
        if not sample_artifact:
            continue
        sample_path = Path(sample_artifact)
        sample_dump = np.load(sample_path)
        restored.append(
            {
                "metadata": dict(family["metadata"]),
                "momentum": dict(family.get("momentum", {})),
                "lambda_axis": np.asarray(family["lambda_axis"], dtype=float),
                "x_axis": np.asarray(sample_dump["x_axis"], dtype=float),
                "sample_count": int(family["sample_count"]),
                "extrapolation": dict(family["extrapolation"]),
                "coordinate_real_mean": np.asarray(family["coordinate_space"]["real"]["mean"], dtype=float),
                "coordinate_real_error": np.asarray(family["coordinate_space"]["real"]["error"], dtype=float),
                "coordinate_imag_mean": np.asarray(family["coordinate_space"]["imag"]["mean"], dtype=float),
                "coordinate_imag_error": np.asarray(family["coordinate_space"]["imag"]["error"], dtype=float),
                "real_mean": np.asarray(family["momentum_space"]["real"]["mean"], dtype=float),
                "real_error": np.asarray(family["momentum_space"]["real"]["error"], dtype=float),
                "imag_mean": np.asarray(family["momentum_space"]["imag"]["mean"], dtype=float),
                "imag_error": np.asarray(family["momentum_space"]["imag"]["error"], dtype=float),
                "real_samples": np.asarray(sample_dump["real_samples"], dtype=float),
                "imag_samples": np.asarray(sample_dump["imag_samples"], dtype=float),
                "sample_artifact": str(sample_path),
            }
        )
    return restored


def _restore_stage_result(stage_data: dict[str, Any], resume_from: Path) -> StageResult:
    """Rebuild a prior stage result from a saved report entry."""
    artifacts = [
        ArtifactRecord(
            name=str(item["name"]),
            kind=str(item["kind"]),
            path=Path(item["path"]),
            description=str(item["description"]),
            format=str(item["format"]),
        )
        for item in stage_data.get("artifacts", [])
    ]
    payload = dict(stage_data.get("payload", {}))
    stage_name = str(stage_data["stage_name"])
    if stage_name == "correlator_analysis" and payload.get("matrix_element_families"):
        payload["_matrix_element_families"] = _restore_matrix_element_family_samples("matrix_element_families", payload)
    if stage_name == "renormalization" and payload.get("renormalized_families"):
        payload["_renormalized_families"] = _restore_matrix_element_family_samples("renormalized_families", payload)
    if stage_name == "fourier_transform":
        restored_families = _restore_transformed_family_samples(payload)
        if restored_families:
            payload["_transformed_families"] = restored_families
    return StageResult(
        stage_name=stage_name,
        summary=f"Reused from {resume_from.name}: {stage_data['summary']}",
        payload=payload,
        artifacts=artifacts,
    )


def _load_prior_stage_results(resume_from: Path) -> dict[str, StageResult]:
    """Load previously materialized stage results from one run directory."""
    report_path = resume_from / "report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        str(item["stage_name"]): _restore_stage_result(item, resume_from)
        for item in report.get("stage_results", [])
    }


def execute_manifest(
    manifest_path: str | Path,
    planner,
    *,
    resume_from: str | Path | None = None,
    start_stage: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> WorkflowRun:
    """Load, resolve, and execute a manifest with the provided planner."""
    if (resume_from is None) != (start_stage is None):
        raise ValueError("execute_manifest requires both resume_from and start_stage, or neither.")
    manifest = load_manifest(manifest_path)
    plan = planner.resolve(manifest)
    run_directory = ensure_directory(manifest.resolved_output_directory / f"run_{timestamp_slug()}")
    datasets = load_all_correlators(manifest.correlators, manifest.manifest_path)
    kernel = load_kernel(manifest.kernel)
    context = StageContext(
        manifest=manifest,
        run_directory=run_directory,
        datasets=datasets,
        kernel=kernel,
        progress_callback=progress_callback,
    )
    stage_results: list[StageResult] = []
    stage_names = list(plan.stage_names)
    start_index = 0
    if resume_from is not None and start_stage is not None:
        if start_stage not in stage_names:
            raise ValueError(f"Requested start_stage {start_stage!r} is not in the resolved workflow.")
        prior_results = _load_prior_stage_results(Path(resume_from).resolve())
        start_index = stage_names.index(start_stage)
        for stage_name in stage_names[:start_index]:
            if stage_name not in prior_results:
                raise ValueError(
                    f"Cannot resume from stage {start_stage!r}: prior run does not contain stage {stage_name!r}."
                )
            restored = prior_results[stage_name]
            context.stage_payloads[stage_name] = restored.payload
            stage_results.append(restored)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "stage_reused",
                        "stage_name": stage_name,
                        "stage_index": len(stage_results) - 1,
                        "stage_total": len(stage_names),
                        "summary": restored.summary,
                    }
                )
            write_stage_summary(run_directory, restored)
    for stage_name in stage_names[start_index:]:
        stage = get_stage(stage_name)
        stage_index = len(stage_results)
        if progress_callback is not None:
                progress_callback(
                    {
                        "event": "stage_started",
                        "stage_name": stage_name,
                        "stage_index": stage_index,
                        "stage_total": len(stage_names),
                        "stage_description": stage.description,
                    }
                )
        try:
            result = stage.run(context)
        except Exception as exc:
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "stage_failed",
                        "stage_name": stage_name,
                        "stage_index": stage_index,
                        "stage_total": len(stage_names),
                        "error": str(exc),
                    }
                )
            raise StageExecutionError(f"Stage {stage_name!r} failed: {exc}") from exc
        context.stage_payloads[stage_name] = result.payload
        stage_results.append(result)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "stage_completed",
                    "stage_name": stage_name,
                    "stage_index": stage_index,
                    "stage_total": len(stage_names),
                    "summary": result.summary,
                }
            )
        write_stage_summary(run_directory, result)
    write_run_report(run_directory, manifest, plan, stage_results)
    return WorkflowRun(manifest=manifest, plan=plan, run_directory=run_directory, stage_results=stage_results)
