"""CLI entrypoint for lamet-agent."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import typer
from pydantic import ValidationError

from .core.llm import resolve_llm_provider, validate_api_model
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

_CLI_SUMMARY_KEYS = (
    "run_id",
    "status",
    "provider",
    "provider_type",
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
        lines.extend(
            textwrap.wrap(
                item,
                width=wrap,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
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
    provider: str,
    model: str | None,
    api_key_file: Path | None,
) -> tuple[str, str, str | None, str | None, str | None, str | None]:
    """Resolve the provider type, model, and API authentication configuration.

    For API providers the key comes from ``--api-key-file`` (the file must
    exist and be non-empty) or, if that flag is omitted, from the provider
    environment variable. The sources are not mixed: a missing or empty key
    file does not fall back to the environment.

    Returns ``(provider_type, provider, model_name, api_key, base_url, key_source)``.
    """
    try:
        resolved = resolve_llm_provider(provider, model)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    api_key: str | None = None
    key_source: str | None = None
    if resolved.kind == "api":
        if api_key_file is not None:
            if not api_key_file.is_file():
                fallback = (
                    f"does not fall back to {resolved.key_env}"
                    if resolved.key_env is not None
                    else "has no registered API-key environment variable"
                )
                raise typer.BadParameter(
                    f"--api-key-file {str(api_key_file)!r} does not exist. "
                    f"This provider {fallback}."
                )
            api_key = api_key_file.read_text(encoding="utf-8").strip()
            if not api_key:
                raise typer.BadParameter(
                    f"--api-key-file {str(api_key_file)!r} is empty."
                )
            key_source = f"file:{api_key_file}"
        else:
            if resolved.key_env is None:
                raise typer.BadParameter(
                    "A custom OpenAI-compatible API URL requires --api-key-file."
                )
            api_key = (os.environ.get(resolved.key_env) or "").strip()
            if not api_key:
                raise typer.BadParameter(
                    f"API provider {provider!r} requires --api-key-file "
                    f"or the {resolved.key_env} environment variable."
                )
            key_source = f"env:{resolved.key_env}"
        try:
            resolved = validate_api_model(resolved, api_key=api_key)
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    return (
        resolved.kind,
        resolved.provider,
        resolved.model_name,
        api_key,
        resolved.base_url,
        key_source,
    )


def _emit_llm_provider_startup(
    *,
    provider_type: str,
    provider: str,
    model_name: str | None,
    base_url: str | None,
    key_source: str | None,
) -> None:
    """Print a boxed LLM provider summary, then a blank line before the banner."""
    if provider_type == "api":
        body = [
            f"type={provider_type}",
            f"provider={provider}",
            f"model={model_name}",
            f"base_url={base_url}",
            f"api_key={key_source}",
        ]
    elif provider_type == "cli":
        body = [
            f"type={provider_type}",
            f"provider={provider}",
            f"model={model_name or 'SDK default'}",
            "auth=Codex login",
        ]
    else:
        return
    typer.echo(_render_boxed_notice("LLM PROVIDER", body))
    typer.echo()


def _run_plan_mode(
    manifest: Path,
    *,
    provider: str,
    model: str | None,
    api_key_file: Path | None,
    path_repair_project_root: Path | None = None,
) -> None:
    """Validate planning options and run the interactive planning loop."""
    backend, resolved_provider, model_name, api_key, resolved_base_url, key_source = _resolve_llm_config(
        provider=provider,
        model=model,
        api_key_file=api_key_file,
    )
    _emit_llm_provider_startup(
        provider_type=backend,
        provider=resolved_provider,
        model_name=model_name,
        base_url=resolved_base_url,
        key_source=key_source,
    )
    try:
        run_interactive_plan(
            manifest,
            backend=backend,
            provider=resolved_provider,
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
    provider: str = typer.Option(
        ...,
        "--provider",
        help="Registered agent CLI/API provider, or an OpenAI-compatible API URL.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model ID override. Required for non-loopback custom URLs; inferred when loopback /models returns exactly one ID.",
    ),
    api_key_file: Path | None = typer.Option(
        None,
        "--api-key-file",
        help="API key file. Required for a custom URL; registered APIs may use their configured environment variable.",
    ),
) -> None:
    """Interactively review and repair a draft manifest before running it."""
    _run_plan_mode(
        manifest,
        provider=provider,
        model=model,
        api_key_file=api_key_file,
    )


@app.command("run")
def run_workflow(
    manifest: Path,
    provider: str = typer.Option(
        ...,
        "--provider",
        help="Registered agent CLI/API provider, or an OpenAI-compatible API URL.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model ID override. Required for non-loopback custom URLs; inferred when loopback /models returns exactly one ID.",
    ),
    api_key_file: Path | None = typer.Option(
        None,
        "--api-key-file",
        help="API key file. Required for a custom URL; registered APIs may use their configured environment variable.",
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

    With ``--provider codex`` the loop is driven by the Codex Python SDK and
    ``--model`` optionally selects its model. Registered API providers supply a
    default URL and model; ``--model`` overrides only the model ID. A custom
    OpenAI-compatible API URL passed to ``--provider`` normally requires
    ``--model``. A loopback API may omit it when ``/models`` returns one ID.
    Every API model is checked against ``BASE_URL/models`` before execution.
    The API key is read from ``--api-key-file`` or, if that flag is omitted, the
    registered provider's configured environment variable. Custom URLs require
    ``--api-key-file`` because they have no registered key environment variable.
    Startup prints a boxed ``api`` / ``cli`` summary (provider, model, base URL,
    and key source or Codex login), never the key itself, then a blank line
    before the LaMET Agent banner.
    Manifest validation failures start the
    interactive planning loop instead of running workflow stages.
    """
    try:
        parsed = validate_manifest_file(manifest)
        validate_manifest_paths(parsed)
    except Exception as exc:  # pragma: no cover - CLI surface
        typer.echo(_render_plan_fallback_notice(exc), err=True)
        typer.echo(err=True)
        _run_plan_mode(
            manifest,
            provider=provider,
            model=model,
            api_key_file=api_key_file,
            path_repair_project_root=(
                lamet_agent_project_root() if isinstance(exc, ManifestPathError) else None
            ),
        )
        return
    report_language = report_language.lower()
    if report_language not in {"en", "ch"}:
        raise typer.BadParameter("--report_language must be 'en' or 'ch'")

    backend, resolved_provider, model_name, api_key, resolved_base_url, key_source = _resolve_llm_config(
        provider=provider,
        model=model,
        api_key_file=api_key_file,
    )
    _emit_llm_provider_startup(
        provider_type=backend,
        provider=resolved_provider,
        model_name=model_name,
        base_url=resolved_base_url,
        key_source=key_source,
    )

    try:
        result = run_agent(
            parsed,
            backend=backend,
            provider=resolved_provider,
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


@app.command("precompute-formulas")
def precompute_formulas(
    provider: str = typer.Option(
        "openai",
        "--provider",
        help="Registered agent CLI/API provider, or an OpenAI-compatible API URL.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model ID override. Required for non-loopback custom URLs; inferred when loopback /models returns exactly one ID.",
    ),
    api_key_file: Path | None = typer.Option(
        None,
        "--api-key-file",
        help="API key file. Required for a custom URL; registered APIs may use their configured environment variable.",
    ),
    kernel: list[str] = typer.Option(
        [],
        "--kernel",
        help="Restrict to these kernel_ids (repeatable). Default: every registered kernel.",
    ),
    prune: bool = typer.Option(
        False,
        "--prune",
        help="Delete cached formulas for kernels that are no longer registered.",
    ),
) -> None:
    """Generate the matching formulas that ship inside the package.

    The matching report's formula section is written by an LLM reading the kernel source
    next to its arXiv paper -- about 27k prompt tokens and one paper download per kernel.
    Running this once, and committing the result, means an installed lamet-agent renders
    that section straight from disk: no network, no tokens, on a user's very first run.

    Re-run it after adding or editing a kernel. Entries are keyed by a digest of the
    kernel's own source, so only what actually changed is regenerated and an edit
    overwrites its own file.
    """
    from .stages.matching.functions import KERNEL_REGISTRY
    from .stages.matching.reporting import FormulaLlm, precompute_kernel_formulas

    unknown = sorted(set(kernel) - set(KERNEL_REGISTRY))
    if unknown:
        raise typer.BadParameter(
            f"Unknown kernel_id(s): {unknown}. Available: {sorted(KERNEL_REGISTRY)}"
        )
    kernel_ids = sorted(kernel) if kernel else sorted(KERNEL_REGISTRY)

    backend, resolved_provider, model_name, api_key, resolved_base_url, _key_source = _resolve_llm_config(
        provider=provider,
        model=model,
        api_key_file=api_key_file,
    )
    try:
        result = precompute_kernel_formulas(
            kernel_ids,
            llm=FormulaLlm(
                backend=backend,
                provider=resolved_provider,
                model_name=model_name,
                api_key=api_key,
                base_url=resolved_base_url,
            ),
            prune=prune,
        )
    except RuntimeError as exc:  # pragma: no cover - CLI surface
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


def main() -> None:
    """Project console script entrypoint."""
    app()


if __name__ == "__main__":
    main()
