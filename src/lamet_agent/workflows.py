"""Workflow planning and execution entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lamet_agent.artifacts import StageResult
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


def execute_manifest(manifest_path: str | Path, planner) -> WorkflowRun:
    """Load, resolve, and execute a manifest with the provided planner."""
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
    )
    stage_results: list[StageResult] = []
    for stage_name in plan.stage_names:
        stage = get_stage(stage_name)
        try:
            result = stage.run(context)
        except Exception as exc:
            raise StageExecutionError(f"Stage {stage_name!r} failed: {exc}") from exc
        context.stage_payloads[stage_name] = result.payload
        stage_results.append(result)
        write_stage_summary(run_directory, result)
    write_run_report(run_directory, manifest, plan, stage_results)
    return WorkflowRun(manifest=manifest, plan=plan, run_directory=run_directory, stage_results=stage_results)
