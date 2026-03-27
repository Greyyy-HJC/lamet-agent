"""Run-level and stage-level reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lamet_agent.artifacts import StageResult
from lamet_agent.schemas import Manifest
from lamet_agent.utils import ensure_directory, write_json, write_markdown

if TYPE_CHECKING:
    from lamet_agent.workflows import WorkflowPlan


def write_stage_summary(run_directory: Path, result: StageResult) -> None:
    """Write a short markdown summary for one stage."""
    summary_dir = ensure_directory(run_directory / "summaries")
    write_markdown(
        summary_dir / f"{result.stage_name}.md",
        [
            f"# {result.stage_name}",
            "",
            result.summary,
            "",
            "## Artifacts",
            *[
                f"- `{artifact.kind}` `{artifact.path.name}`: {artifact.description}"
                for artifact in result.artifacts
            ],
        ],
    )


def write_run_report(
    run_directory: Path,
    manifest: Manifest,
    plan: WorkflowPlan,
    stage_results: list[StageResult],
) -> None:
    """Write machine-readable and human-readable run reports."""
    report_payload = {
        "goal": manifest.goal,
        "plan": plan.to_dict(),
        "metadata": manifest.metadata,
        "stage_results": [result.to_dict() for result in stage_results],
    }
    write_json(run_directory / "report.json", report_payload)
    lines = [
        "# LaMET Analysis Report",
        "",
        f"- Goal: `{manifest.goal}`",
        f"- Final observable: `{plan.final_observable}`",
        f"- Stage count: `{len(stage_results)}`",
        "",
        "## Metadata",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(manifest.metadata.items())],
        "",
        "## Stage Summaries",
        "",
    ]
    for result in stage_results:
        lines.append(f"### {result.stage_name}")
        lines.append("")
        lines.append(result.summary)
        lines.append("")
    write_markdown(run_directory / "report.md", lines)
