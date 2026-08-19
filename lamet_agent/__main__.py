"""CLI entrypoint for lamet-agent."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import typer
from pydantic import ValidationError

from .core.llm import parse_api_model, provider_config
from .core.tools import validate_stage_diagnostics
from .manifest import (
    AnalysisManifest,
    ManifestPathError,
    lamet_agent_project_root,
    validate_manifest_file,
    validate_manifest_paths,
)
from .manifest_params import (
    STAGE_PARAM_CONTRACTS,
    render_stage_contract,
    resolve_stage_params,
)
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


def run_agent(*args: object, **kwargs: object) -> dict:
    """Load the numerical runner only when the run command needs it."""
    from .agent import run_agent as run_agent_impl

    return run_agent_impl(*args, **kwargs)


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


@app.command("describe-stage")
def describe_stage(stage: str) -> None:
    """Show one stage's manifest parameters, physics, and compatibility rules."""
    if stage not in STAGE_PARAM_CONTRACTS:
        choices = ", ".join(STAGE_PARAM_CONTRACTS)
        raise typer.BadParameter(f"unknown stage {stage!r}; choose one of: {choices}")
    typer.echo(render_stage_contract(stage))


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

    issues = []
    for stage in manifest.metadata.stages:
        for job in manifest.stages[stage].jobs:
            params = resolve_stage_params(
                stage,
                manifest.stages[stage].defaults,
                job.params,
            )
            if (
                stage == "correlator_analysis"
                and params.get("analysis_method") == "lanczos"
                and params.get("lanczos_precision") == 0
            ):
                typer.echo(
                    f"Warning: {stage}/{job.id} uses lanczos_precision=0; "
                    "Lanczos recurrence matrices will use NumPy double precision. "
                    "Set a positive decimal digit count explicitly to enable "
                    "high-precision matrix construction.",
                    err=True,
                )
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
                "status": "invalid" if issues else "valid",
                "issues": issues,
            },
            indent=2,
        )
    )
    if issues:
        raise typer.Exit(code=1)


def _resolve_llm_config(
    *,
    backend: str,
    model: str | None,
    api_key_file: Path | None,
    base_url: str | None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Resolve model and OpenAI-compatible API configuration.

    For ``backend='api'`` the key comes from ``--api-key-file`` (the file must
    exist and be non-empty) or, if that flag is omitted, from the provider
    environment variable. The sources are not mixed: a missing or empty key
    file does not fall back to the environment.

    Returns ``(provider, model_name, api_key, base_url, key_source)``.
    ``key_source`` is ``file:<path>`` or ``env:<VAR>`` for the api backend.
    """
    provider: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    key_source: str | None = None
    resolved_base_url: str | None = base_url

    if backend == "api":
        try:
            provider, model_name = parse_api_model(model or "")
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        config = provider_config(provider)
        assert config is not None
        if api_key_file is not None:
            if not api_key_file.is_file():
                raise typer.BadParameter(
                    f"--api-key-file {str(api_key_file)!r} does not exist. "
                    "When this flag is set, the api backend does not fall back "
                    f"to {config['key_env']}."
                )
            api_key = api_key_file.read_text(encoding="utf-8").strip()
            if not api_key:
                raise typer.BadParameter(
                    f"--api-key-file {str(api_key_file)!r} is empty."
                )
            key_source = f"file:{api_key_file}"
        else:
            api_key = (os.environ.get(config["key_env"]) or "").strip()
            if not api_key:
                raise typer.BadParameter(
                    f"backend='api' provider={provider!r} requires --api-key-file "
                    f"or the {config['key_env']} environment variable."
                )
            key_source = f"env:{config['key_env']}"
    elif backend == "codex" and model:
        model_name = model.strip()
    return provider, model_name, api_key, resolved_base_url, key_source


def _emit_llm_backend_startup(
    *,
    backend: str,
    provider: str | None,
    model_name: str | None,
    base_url: str | None,
    key_source: str | None,
) -> None:
    """Print a boxed LLM backend summary, then a blank line before the banner."""
    if backend == "api":
        config = provider_config(provider or "")
        effective_base_url = base_url or (config["base_url"] if config else "")
        body = [
            f"backend={backend}",
            f"provider={provider}",
            f"model={model_name}",
            f"base_url={effective_base_url}",
            f"api_key={key_source}",
        ]
    elif backend == "codex":
        body = [
            f"backend={backend}",
            f"model={model_name or 'SDK default'}",
            "auth=Codex login",
        ]
    else:
        return
    typer.echo(_render_boxed_notice("LLM BACKEND", body))
    typer.echo()


def _run_plan_mode(
    manifest: Path,
    *,
    backend: str,
    model: str | None,
    api_key_file: Path | None,
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

    provider, model_name, api_key, resolved_base_url, key_source = _resolve_llm_config(
        backend=backend,
        model=model,
        api_key_file=api_key_file,
        base_url=base_url,
    )
    _emit_llm_backend_startup(
        backend=backend,
        provider=provider,
        model_name=model_name,
        base_url=resolved_base_url,
        key_source=key_source,
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
    api_key_file: Path | None = typer.Option(
        None,
        "--api-key-file",
        help="API key file for --backend api. If omitted, use DEEPSEEK_API_KEY or OPENAI_API_KEY.",
    ),
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
    api_key_file: Path | None = typer.Option(
        None,
        "--api-key-file",
        help="API key file for --backend api. If omitted, use DEEPSEEK_API_KEY or OPENAI_API_KEY.",
    ),
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
    The API key is read from ``--api-key-file`` or, if that flag is omitted, the
    provider environment variable (``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``).
    Startup prints a boxed ``api`` / ``codex`` summary (provider, model, base URL,
    and key source or Codex login), never the key itself, then a blank line
    before the LaMET Agent banner.
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

    provider, model_name, api_key, resolved_base_url, key_source = _resolve_llm_config(
        backend=backend,
        model=model,
        api_key_file=api_key_file,
        base_url=base_url,
    )
    _emit_llm_backend_startup(
        backend=backend,
        provider=provider,
        model_name=model_name,
        base_url=resolved_base_url,
        key_source=key_source,
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
