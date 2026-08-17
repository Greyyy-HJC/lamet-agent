"""CLI entrypoint for lamet-agent."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import typer
from pydantic import ValidationError

from .agent import run_agent
from .core.llm import parse_api_model, provider_config
from .core.tools import validate_stage_diagnostics
from .manifest import (
    AnalysisManifest,
    ManifestPathError,
    lamet_agent_project_root,
    validate_manifest_file,
    validate_manifest_paths,
)
from .planning import run_interactive_plan
from .stages.matching.validation import matching_grid_warnings

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


def _format_cli_error(error: BaseException) -> str:
    """Return a short CLI error without Pydantic's docs URL or input dump."""
    if isinstance(error, ValidationError):
        messages: list[str] = []
        for item in error.errors():
            ctx_error = (item.get("ctx") or {}).get("error")
            if isinstance(ctx_error, BaseException):
                message = str(ctx_error).strip()
            else:
                message = str(item.get("msg") or "").strip()
                if message.lower().startswith("value error, "):
                    message = message[len("value error, ") :]
            if message:
                messages.append(message)
        if messages:
            return "\n".join(messages)
    return str(error)


def _render_boxed_notice(title: str, body_lines: list[str], *, wrap: int = 88) -> str:
    """Render a framed terminal notice with wrapped body lines."""
    lines = [title]
    for item in body_lines:
        lines.extend(textwrap.wrap(item, width=wrap) or [""])
    width = max(len(line) for line in lines)
    border = f"+{'-' * (width + 2)}+"
    box = [border, *(f"| {line:<{width}} |" for line in lines), border]
    return "\n".join(box)


def _render_plan_fallback_notice(error: Exception) -> str:
    """Render a prominent notice before a failed run enters plan mode."""
    box = _render_boxed_notice(
        "RUN VALIDATION FAILED",
        [
            "Falling back to interactive PLAN mode.",
            "No workflow stages will run during this command.",
            "Accepting the plan only writes quick/full manifests.",
        ],
    )
    return "\n".join([box, "", "Validation error:", _format_cli_error(error)])


def _matching_grid_warnings_for_cli(manifest: object) -> list[str]:
    """Collect matching-grid warnings for a parsed manifest, if it is strict."""
    if not isinstance(manifest, AnalysisManifest):
        return []
    return matching_grid_warnings(manifest)


def _emit_matching_grid_warnings(manifest: object) -> list[str]:
    """Print boxed matching-grid warnings to stderr and return them."""
    warnings = _matching_grid_warnings_for_cli(manifest)
    if warnings:
        typer.echo(_render_boxed_notice("WARNING: MATCHING GRID DENSITY", warnings), err=True)
        typer.echo(err=True)
    return warnings


@app.command("validate")
def validate_manifest(path: Path) -> None:
    """Validate workflow manifest schema, input paths, and kernel references."""
    try:
        manifest = validate_manifest_file(path)
        validate_manifest_paths(manifest)
    except Exception as exc:  # pragma: no cover - CLI surface
        # Keep long manifest paths and structured validation explanations intact;
        # Click's BadParameter formatter wraps them in the middle of tokens.
        typer.echo(_format_cli_error(exc), err=True)
        raise typer.Exit(code=2) from exc

    warnings = _emit_matching_grid_warnings(manifest)
    issues = []
    for stage in manifest.metadata.stages:
        for job in manifest.stages[stage].jobs:
            for diagnostic in validate_stage_diagnostics(stage, manifest, job):
                issues.append(
                    {
                        "stage": stage,
                        "job_id": job.id,
                        "code": diagnostic.code,
                        "path": diagnostic.path,
                        "message": diagnostic.message,
                        "cause": diagnostic.cause,
                        "physics": diagnostic.physics,
                        "suggested_fix": diagnostic.suggested_fix,
                    }
                )
    typer.echo(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "stages": manifest.metadata.stages,
                "correlator_count": len(manifest.inputs.correlators),
                "kernel_count": len(manifest.inputs.kernels),
                "status": "invalid" if warnings or issues else "valid",
                "warnings": warnings,
                "issues": issues,
            },
            indent=2,
        )
    )
    if warnings or issues:
        raise typer.Exit(code=1)


def _resolve_llm_config(
    *,
    backend: str,
    model: str | None,
    api_key_file: Path,
    base_url: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve model and optional OpenAI-compatible API configuration."""
    provider: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    resolved_base_url: str | None = base_url
    if backend == "api" and api_key_file.exists():
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
    elif backend == "codex" and model:
        model_name = model.strip()
    return provider, model_name, api_key, resolved_base_url


def _run_plan_mode(
    manifest: Path,
    *,
    backend: str,
    model: str | None,
    api_key_file: Path,
    base_url: str | None,
    path_repair_project_root: Path | None = None,
) -> None:
    """Validate planning options and run the interactive planning loop."""
    if backend not in _VALID_PLAN_BACKENDS:
        raise typer.BadParameter(
            f"--backend must be one of {sorted(_VALID_PLAN_BACKENDS)} for plan; external transcripts are not supported."
        )
    if backend == "api" and not model:
        raise typer.BadParameter("backend='api' requires --model provider/model_id.")
    if backend == "mock" and model:
        print(
            f"warning: --model is ignored for backend={backend!r}.",
            file=sys.stderr,
        )

    provider, model_name, api_key, resolved_base_url = _resolve_llm_config(
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
            path_repair_project_root=path_repair_project_root,
        )
    except ValueError as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc


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
        help="Codex model ID, or API model as provider/model_id (api backend).",
    ),
    api_key_file: Path = Path("api.key"),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override the provider API base URL (api backend only).",
    ),
) -> None:
    """Interactively review and repair a draft manifest before running it."""
    _run_plan_mode(
        manifest,
        backend=backend,
        model=model,
        api_key_file=api_key_file,
        base_url=base_url,
    )


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
        help="Codex model ID, or API model as provider/model_id (api backend).",
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

    With ``--backend codex`` the loop is driven by the Codex Python SDK and
    ``--model`` optionally selects its model. With
    ``--backend api`` pass ``--model provider/model_id`` (e.g. ``deepseek/deepseek-chat``).
    The API key is read from ``--api-key-file`` (default ``api.key``) or the provider
    environment variable (``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``).
    With a planning-capable backend, manifest validation failures start the
    interactive planning loop instead of running workflow stages.
    """
    try:
        parsed = validate_manifest_file(manifest)
        validate_manifest_paths(parsed)
    except Exception as exc:  # pragma: no cover - CLI surface
        if backend in _VALID_PLAN_BACKENDS:
            typer.echo(_render_plan_fallback_notice(exc), err=True)
            typer.echo(err=True)
            _run_plan_mode(
                manifest,
                backend=backend,
                model=model,
                api_key_file=api_key_file,
                base_url=base_url,
                path_repair_project_root=(
                    lamet_agent_project_root() if isinstance(exc, ManifestPathError) else None
                ),
            )
            return
        raise typer.BadParameter(_format_cli_error(exc)) from exc
    _emit_matching_grid_warnings(parsed)
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
    if backend in {"mock", "external"} and model:
        print(
            f"warning: --model is ignored for backend={backend!r}.",
            file=sys.stderr,
        )

    provider, model_name, api_key, resolved_base_url = _resolve_llm_config(
        backend=backend,
        model=model,
        api_key_file=api_key_file,
        base_url=base_url,
    )

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


def main() -> None:
    """Project console script entrypoint."""
    app()


if __name__ == "__main__":
    main()
