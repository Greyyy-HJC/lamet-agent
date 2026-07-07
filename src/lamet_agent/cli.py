"""CLI entrypoint for lamet-agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from .agent import run_agent
from .core.llm import parse_api_model, provider_config
from .manifest import validate_manifest_file
from .planning import run_interactive_plan

app = typer.Typer(help="CLI-first scaffold for LaMET analysis workflows.")

_VALID_BACKENDS = frozenset({"mock", "external", "api", "codex"})
_VALID_PLAN_BACKENDS = frozenset({"api", "codex", "mock"})

_CLI_SUMMARY_KEYS = (
    "run_id",
    "status",
    "backend",
    "model",
    "stages",
    "completed_stages",
    "stage_reports",
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


def _resolve_api_config(
    *,
    backend: str,
    model: str | None,
    api_key_file: Path,
    base_url: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve optional OpenAI-compatible API configuration for CLI commands."""
    provider: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    resolved_base_url: str | None = base_url
    if api_key_file.exists():
        api_key = api_key_file.read_text(encoding="utf-8").strip()

    if backend == "api":
        try:
            provider, model_name = parse_api_model(model or "")
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        config = provider_config(provider)
        assert config is not None
        if not api_key:
            api_key = os.environ.get(config["key_env"])
    return provider, model_name, api_key, resolved_base_url


@app.command("plan")
def plan_workflow(
    manifest: Path,
    backend: str = typer.Option(
        ...,
        "--backend",
        help="Planning LLM backend: api or codex. mock is available for tests.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="API model as provider/model_id, e.g. deepseek/deepseek-chat (api backend only).",
    ),
    api_key_file: Path = Path("api.key"),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override the provider API base URL (api backend only).",
    ),
) -> None:
    """Interactively review and repair a draft manifest before running it."""
    if backend not in _VALID_PLAN_BACKENDS:
        raise typer.BadParameter(
            f"--backend must be one of {sorted(_VALID_PLAN_BACKENDS)} for plan; external transcripts are not supported."
        )
    if backend == "api" and not model:
        raise typer.BadParameter("backend='api' requires --model provider/model_id.")
    if backend in {"mock", "codex"} and model:
        print(
            f"warning: --model is ignored for backend={backend!r}.",
            file=sys.stderr,
        )

    provider, model_name, api_key, resolved_base_url = _resolve_api_config(
        backend=backend,
        model=model,
        api_key_file=api_key_file,
        base_url=base_url,
    )
    try:
        run_interactive_plan(
            manifest,
            backend=backend,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=resolved_base_url,
            output_func=typer.echo,
        )
    except ValueError as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc


@app.command("run")
def run_workflow(
    manifest: Path,
    backend: str = typer.Option(
        ...,
        "--backend",
        help="LLM backend: mock, external, api, or codex.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="API model as provider/model_id, e.g. deepseek/deepseek-chat (api backend only).",
    ),
    actions_path: Path | None = None,
    api_key_file: Path = Path("api.key"),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override the provider API base URL (api backend only).",
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
    report_language: str = typer.Option(
        "en",
        "--report_language",
        help="Report language: en or ch.",
    ),
) -> None:
    """Run the staged agent loop.

    With ``--backend codex`` the loop is driven by the Codex Python SDK. With
    ``--backend api`` pass ``--model provider/model_id`` (e.g. ``deepseek/deepseek-chat``).
    The API key is read from ``--api-key-file`` (default ``api.key``) or the provider
    environment variable (``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``).
    """
    try:
        parsed = validate_manifest_file(manifest)
    except Exception as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc
    report_language = report_language.lower()
    if report_language not in {"en", "ch"}:
        raise typer.BadParameter("--report_language must be 'en' or 'ch'")

    if backend not in _VALID_BACKENDS:
        raise typer.BadParameter(
            f"--backend must be one of {sorted(_VALID_BACKENDS)}; got {backend!r}."
        )
    if backend == "external" and actions_path is None:
        raise typer.BadParameter("backend='external' requires --actions-path.")
    if backend == "api" and not model:
        raise typer.BadParameter("backend='api' requires --model provider/model_id.")
    if backend in {"mock", "external", "codex"} and model:
        print(
            f"warning: --model is ignored for backend={backend!r}.",
            file=sys.stderr,
        )

    provider, model_name, api_key, resolved_base_url = _resolve_api_config(
        backend=backend,
        model=model,
        api_key_file=api_key_file,
        base_url=base_url,
    )

    if backend == "api":
        # The matching report's formula generation lives in a self-contained module
        # (stages/matching/reporting.py) that reads its LLM config from LAMET_FORMULA_*
        # env vars rather than receiving it as a parameter. Thread this run's resolved
        # config through so each user's --api-key-file (and chosen provider/model) is
        # what the report uses, instead of whatever happens to be set on the machine.
        if api_key:
            os.environ["LAMET_FORMULA_MODEL"] = provider
            os.environ["LAMET_FORMULA_API_KEY"] = api_key
            os.environ["LAMET_FORMULA_LLM_MODEL"] = model_name
            if resolved_base_url:
                os.environ["LAMET_FORMULA_BASE_URL"] = resolved_base_url

    try:
        result = run_agent(
            parsed,
            backend=backend,
            actions_path=actions_path,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=resolved_base_url,
            verbose=verbose,
            max_tool_steps=max_tool_steps,
            report_language=report_language,
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
