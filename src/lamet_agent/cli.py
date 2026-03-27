"""CLI entry points for validating, inspecting, and running LaMET workflows.

Example usage:
    lamet-agent validate examples/demo_manifest.json
    lamet-agent workflow examples/demo_manifest.json
    lamet-agent run examples/demo_manifest.json
    lamet-agent auth providers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lamet_agent.auth import OAuthLoginManager
from lamet_agent.errors import LametAgentError
from lamet_agent.logging_utils import configure_logging
from lamet_agent.planners import RuleBasedPlanner
from lamet_agent.schemas import load_manifest
from lamet_agent.workflows import execute_manifest

try:
    import typer
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    typer = None


def _validate_impl(manifest_path: str) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    planner = RuleBasedPlanner()
    plan = planner.resolve(manifest)
    return {
        "manifest": str(Path(manifest_path).resolve()),
        "goal": manifest.goal,
        "stage_names": plan.stage_names,
    }


def _workflow_impl(manifest_path: str) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    planner = RuleBasedPlanner()
    return planner.resolve(manifest).to_dict()


def _run_impl(manifest_path: str) -> dict[str, Any]:
    planner = RuleBasedPlanner()
    run = execute_manifest(manifest_path, planner=planner)
    return {
        "run_directory": str(run.run_directory),
        "stage_names": [result.stage_name for result in run.stage_results],
        "report": str(run.run_directory / "report.md"),
    }


def _dump_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _auth_providers_impl() -> dict[str, Any]:
    manager = OAuthLoginManager()
    return {"providers": manager.list_provider_statuses()}


def _auth_status_impl(provider: str) -> dict[str, Any]:
    manager = OAuthLoginManager()
    return manager.get_status(provider)


def _auth_logout_impl(provider: str) -> dict[str, Any]:
    manager = OAuthLoginManager()
    return manager.logout(provider)


def _auth_login_impl(provider: str, redirect_port: int | None = None) -> dict[str, Any]:
    manager = OAuthLoginManager()
    prepared = manager.begin_login(provider, redirect_port=redirect_port)
    print(
        "Open the following URL in your browser to complete OAuth login:\n"
        f"{prepared['authorization_url']}\n"
        f"Waiting for callback on {prepared['redirect_uri']} ..."
    )
    return manager.complete_login(
        provider=provider,
        state=prepared["state"],
        code_verifier=prepared["code_verifier"],
        authorization_url=prepared["authorization_url"],
        redirect_port=redirect_port,
    )


def _handle_command(
    command: str,
    manifest_path: str | None = None,
    auth_command: str | None = None,
    provider: str | None = None,
    redirect_port: int | None = None,
) -> int:
    configure_logging()
    try:
        if command == "validate":
            assert manifest_path is not None
            _dump_result(_validate_impl(manifest_path))
        elif command == "workflow":
            assert manifest_path is not None
            _dump_result(_workflow_impl(manifest_path))
        elif command == "run":
            assert manifest_path is not None
            _dump_result(_run_impl(manifest_path))
        elif command == "auth":
            if auth_command == "providers":
                _dump_result(_auth_providers_impl())
            elif auth_command == "status":
                assert provider is not None
                _dump_result(_auth_status_impl(provider))
            elif auth_command == "login":
                assert provider is not None
                _dump_result(_auth_login_impl(provider, redirect_port=redirect_port))
            elif auth_command == "logout":
                assert provider is not None
                _dump_result(_auth_logout_impl(provider))
            else:
                raise LametAgentError(f"Unknown auth command: {auth_command}")
        else:
            raise LametAgentError(f"Unknown CLI command: {command}")
    except LametAgentError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


def _build_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lamet-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "workflow", "run"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("manifest_path")
    auth_parser = subparsers.add_parser("auth")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", required=True)
    auth_subparsers.add_parser("providers")
    auth_status_parser = auth_subparsers.add_parser("status")
    auth_status_parser.add_argument("provider", choices=("codex", "claude_code"))
    auth_login_parser = auth_subparsers.add_parser("login")
    auth_login_parser.add_argument("provider", choices=("codex", "claude_code"))
    auth_login_parser.add_argument("--redirect-port", type=int, default=None)
    auth_logout_parser = auth_subparsers.add_parser("logout")
    auth_logout_parser.add_argument("provider", choices=("codex", "claude_code"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI with either Typer or argparse."""
    parser = _build_argparse()
    args = parser.parse_args(argv)
    return _handle_command(
        args.command,
        manifest_path=getattr(args, "manifest_path", None),
        auth_command=getattr(args, "auth_command", None),
        provider=getattr(args, "provider", None),
        redirect_port=getattr(args, "redirect_port", None),
    )


if typer is not None:
    app = typer.Typer(add_completion=False, help="Rule-based LaMET workflow runner.")

    @app.command("validate")
    def validate_command(manifest_path: str) -> None:
        """Validate a manifest and print the resolved stage list."""
        _dump_result(_validate_impl(manifest_path))

    @app.command("workflow")
    def workflow_command(manifest_path: str) -> None:
        """Print the workflow plan that would be executed."""
        _dump_result(_workflow_impl(manifest_path))

    @app.command("run")
    def run_command(manifest_path: str) -> None:
        """Execute the resolved workflow and materialize output artifacts."""
        _dump_result(_run_impl(manifest_path))

    auth_app = typer.Typer(help="OAuth login helpers for provider-backed agent integrations.")
    app.add_typer(auth_app, name="auth")

    @auth_app.command("providers")
    def auth_providers_command() -> None:
        """List built-in OAuth-capable providers and their local status."""
        _dump_result(_auth_providers_impl())

    @auth_app.command("status")
    def auth_status_command(provider: str) -> None:
        """Show OAuth configuration and token status for one provider."""
        _dump_result(_auth_status_impl(provider))

    @auth_app.command("login")
    def auth_login_command(provider: str, redirect_port: int | None = None) -> None:
        """Run a local-browser OAuth login flow for a provider."""
        _dump_result(_auth_login_impl(provider, redirect_port=redirect_port))

    @auth_app.command("logout")
    def auth_logout_command(provider: str) -> None:
        """Delete the stored OAuth token for one provider."""
        _dump_result(_auth_logout_impl(provider))

    def entrypoint() -> None:
        """Invoke the Typer application."""
        configure_logging()
        app()

else:

    def entrypoint() -> None:
        """Fallback CLI entrypoint when typer is unavailable."""
        raise SystemExit(main())
