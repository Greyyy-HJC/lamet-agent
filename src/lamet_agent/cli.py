"""CLI entrypoint for lamet-agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from .agent import run_agent
from .core.llm import provider_config
from .manifest import validate_manifest_file

app = typer.Typer(help="CLI-first scaffold for LaMET analysis workflows.")

_CLI_SUMMARY_KEYS = (
    "run_id",
    "status",
    "model",
    "stages",
    "completed_stages",
    "pending_user_input",
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
                "stages": manifest.metadata.stages,
                "correlator_count": len(manifest.inputs.correlators),
                "kernel_count": len(manifest.inputs.kernels),
                "status": "valid",
            },
            indent=2,
        )
    )


@app.command("run")
def run_workflow(
    manifest: Path,
    model: str = typer.Option(
        "mock",
        "--model",
        help="LLM backend: mock, external, deepseek, or openai.",
    ),
    actions_path: Path | None = None,
    api_key_file: Path = Path("api.key"),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="Concrete model name; defaults to the provider's cost-effective model "
        "(deepseek-chat / gpt-4o-mini).",
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override the provider API base URL.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print each LLM cycle: prompt, model action, and tool observation.",
    ),
    max_tool_steps: int = typer.Option(
        40,
        "--max-tool-steps",
        help="Maximum LLM/tool cycles per stage (correlator 2pt+3pt often needs >30).",
    ),
) -> None:
    """Run the staged agent loop.

    With ``--model deepseek`` or ``--model openai`` the loop is driven by that
    provider's API. The key is read from ``--api-key-file`` (default ``api.key``)
    or the provider environment variable (``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``).
    """
    try:
        parsed = validate_manifest_file(manifest)
    except Exception as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc

    api_key = None
    if api_key_file.exists():
        api_key = api_key_file.read_text(encoding="utf-8").strip()
    config = provider_config(model)
    if not api_key and config is not None:
        api_key = os.environ.get(config["key_env"])

    try:
        result = run_agent(
            parsed,
            model=model,
            actions_path=actions_path,
            api_key=api_key,
            llm_model=llm_model,
            base_url=base_url,
            verbose=verbose,
            max_tool_steps=max_tool_steps,
        )
    except ValueError as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc
    result["manifest"] = str(manifest)
    result["correlators"] = [item.correlator_id for item in parsed.correlators]
    result["kernels"] = [item.kernel_id for item in parsed.kernels]
    typer.echo(json.dumps(_cli_run_summary(result), indent=2))


def entrypoint() -> None:
    """Project console script entrypoint."""
    app()
