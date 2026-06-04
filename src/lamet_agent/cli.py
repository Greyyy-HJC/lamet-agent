"""CLI entrypoint for lamet-agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from .agent import run_agent
from .core.stages import select_stage_sequence
from .manifest import validate_manifest_file

app = typer.Typer(help="CLI-first scaffold for LaMET analysis workflows.")

_CLI_SUMMARY_KEYS = (
    "run_id",
    "status",
    "model",
    "stages",
    "completed_stages",
    "input_issues",
    "summary",
    "manifest",
    "correlators",
    "kernels",
)


def _cli_run_summary(result: dict) -> dict:
    """Return the subset of a run result suitable for stdout (no action trace)."""
    return {key: result[key] for key in _CLI_SUMMARY_KEYS if key in result}


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
    stages: str | None = None,
    resume_from: str | None = None,
    model: str = "mock",
    actions_path: Path | None = None,
    api_key_file: Path = Path("api.key"),
    deepseek_model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print each LLM cycle: prompt, model action, and tool observation.",
    ),
) -> None:
    """Run the staged agent loop.

    Use ``--stages`` (comma-separated) to run a specific ordered subset, e.g.
    ``--stages correlator_analysis``. Running a later stage on its own requires
    the manifest to already provide that stage's inputs.

    With ``--model deepseek`` the loop is driven by the DeepSeek API; the key is
    read from ``--api-key-file`` (default ``api.key``) or ``DEEPSEEK_API_KEY``.
    """
    try:
        parsed = validate_manifest_file(manifest)
    except Exception as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc

    stage_list = [s.strip() for s in stages.split(",") if s.strip()] if stages else None

    api_key = None
    if api_key_file.exists():
        api_key = api_key_file.read_text(encoding="utf-8").strip()
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

    try:
        result = run_agent(
            parsed,
            stages=stage_list,
            resume_from=resume_from,
            model=model,
            actions_path=actions_path,
            api_key=api_key,
            deepseek_model=deepseek_model,
            base_url=base_url,
            verbose=verbose,
        )
    except ValueError as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc
    result["manifest"] = str(manifest)
    result["correlators"] = [item.dataset_id for item in parsed.correlators]
    result["kernels"] = [item.kernel_id for item in parsed.kernels]
    typer.echo(json.dumps(_cli_run_summary(result), indent=2))


def entrypoint() -> None:
    """Project console script entrypoint."""
    app()

