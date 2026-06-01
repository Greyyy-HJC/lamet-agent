"""CLI entrypoint for lamet-agent."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .agent import run_agent
from .manifest import validate_manifest_file
from .skills import select_stage_sequence

app = typer.Typer(help="CLI-first scaffold for LaMET analysis workflows.")


@app.command("validate")
def validate_manifest(path: Path) -> None:
    """Validate workflow manifest schema and kernel function references."""
    try:
        manifest = validate_manifest_file(path)
    except Exception as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "goal": manifest.goal,
                "correlator_count": len(manifest.correlators),
                "kernel_count": len(manifest.kernels),
                "status": "valid",
            },
            indent=2,
        )
    )


@app.command("workflow")
def show_workflow(goal: str = "full_lamet_pipeline") -> None:
    """Print the resolved workflow stages for a goal."""
    stages = select_stage_sequence(goal)
    typer.echo(json.dumps({"goal": goal, "stages": stages}, indent=2))


@app.command("run")
def run_workflow(
    manifest: Path,
    resume_from: str | None = None,
    model: str = "mock",
    max_steps: int = 20,
) -> None:
    """Run the staged agent loop."""
    try:
        parsed = validate_manifest_file(manifest)
    except Exception as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc

    result = run_agent(
        parsed,
        resume_from=resume_from,
        model=model,
        max_steps=max_steps,
    )
    result["manifest"] = str(manifest)
    result["correlators"] = [item.dataset_id for item in parsed.correlators]
    result["kernels"] = [item.kernel_id for item in parsed.kernels]
    typer.echo(json.dumps(result, indent=2))


def entrypoint() -> None:
    """Project console script entrypoint."""
    app()

