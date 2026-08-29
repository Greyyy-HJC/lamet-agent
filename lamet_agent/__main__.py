"""Run the LaMET Agent CLI.

Purpose: validate, plan, or execute one JSON manifest.
Inputs: a manifest path plus command-specific output/provider options.
Outputs: deterministic issue text, a completed manifest, or a JSON run summary.
Example: ``python -m lamet_agent validate examples/pion_pdf_gi_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .agent import create_session
from .contract import Issue
from .llm import create_backend
from .manifest import Manifest, load_manifest
from .ui import PlainUi


def _render_issues(issues: Sequence[Issue]) -> str:
    """Render validation issues for CLI output."""
    return "\n".join(f"{issue.path}: {issue.message} ({issue.physics})" for issue in issues)


def _validate(path: Path) -> tuple[Manifest, list[Issue]]:
    manifest = load_manifest(path)
    return manifest, manifest.validate()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lamet-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate one JSON manifest")
    validate.add_argument("manifest", type=Path)
    plan = subparsers.add_parser("plan", help="complete an incomplete manifest through an interactive LLM TUI")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--provider", required=True, help="registered provider or OpenAI-compatible API URL")
    plan.add_argument("--model", help="model ID override")
    plan.add_argument("--api-key-file", type=Path, help="API key file for API providers")
    plan.add_argument("--output", type=Path, help="output path; defaults to <manifest>.planned.json")
    plan.add_argument("--in-place", action="store_true", help="overwrite the input manifest after explicit acceptance")
    run = subparsers.add_parser("run", help="execute one validated manifest")
    run.add_argument("manifest", type=Path)
    run.add_argument(
        "--provider", required=True, help="registered agent CLI/API provider, or an OpenAI-compatible API URL"
    )
    run.add_argument("--model", help="model ID override; optional when a local API exposes exactly one model")
    run.add_argument(
        "--api-key-file", type=Path, help="API key file; required for a custom URL, optional for registered APIs"
    )
    run.add_argument(
        "--progress",
        choices=("auto", "stage", "job", "none"),
        default="auto",
        help="progress granularity; auto uses stage progress when systematics are declared, otherwise job progress",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one CLI command and return its process status."""
    args = _build_parser().parse_args(argv)
    cli_ui = PlainUi()
    if args.command == "validate":
        try:
            _, issues = _validate(args.manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            cli_ui.log(str(exc), level="error")
            return 2
        if issues:
            cli_ui.log(_render_issues(issues))
            return 1
        cli_ui.log("manifest is valid")
        return 0
    if args.command == "plan":
        backend = None
        session = None
        try:
            backend = create_backend(args.provider, args.model, args.api_key_file)
            session = create_session(backend)
            planned_path = session.plan_manifest(
                args.manifest,
                output_path=args.output,
                in_place=args.in_place,
            )
            if planned_path is None:
                return 1
            manifest = load_manifest(planned_path)
            issues = manifest.validate()
            if issues:
                raise ValueError("accepted plan is not valid:\n" + _render_issues(issues))
            session.ui.log(f"manifest written: {planned_path}")
            return 0
        except (EOFError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            (session.ui if session is not None else cli_ui).log(str(exc), level="error")
            return 2
        finally:
            if session is not None:
                session.close()
            else:
                close_backend = getattr(backend, "close", None)
                if callable(close_backend):
                    close_backend()
    backend = None
    session = None
    try:
        manifest = load_manifest(args.manifest)
        backend = create_backend(args.provider, args.model, args.api_key_file)
        session = create_session(backend, progress_mode=args.progress)
        issues = session.validate_manifest(manifest, show_banner=True)
        if issues:
            planned_path = session.plan_manifest(args.manifest)
            if planned_path is None:
                return 1
            manifest = load_manifest(planned_path)
            planned_issues = manifest.validate()
            if planned_issues:
                raise ValueError("accepted plan is not valid:\n" + _render_issues(planned_issues))
        result = session.run_manifest(manifest)
        session.ui.log(json.dumps({"status": "completed", "jobs": sorted(result["summaries"])}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        (session.ui if session is not None else cli_ui).log(str(exc), level="error")
        return 1
    finally:
        if session is not None:
            session.close()
        else:
            close_backend = getattr(backend, "close", None)
            if callable(close_backend):
                close_backend()


if __name__ == "__main__":
    raise SystemExit(main())
