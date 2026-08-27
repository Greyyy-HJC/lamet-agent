"""Run the independent LaMET Agent Neo CLI.

Purpose: validate, plan, or execute one JSON manifest.
Inputs: a manifest path plus command-specific output/provider options.
Outputs: deterministic issue text, a completed manifest, or a JSON run summary.
Example: ``python -m lamet_agent_neo validate examples/neo.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from .agent import create_session
from .contract import Issue
from .llm import create_backend
from .manifest import Manifest, load_manifest


def _render_issues(issues: Sequence[Issue]) -> str:
    """Render validation issues for CLI output."""
    return "\n".join(f"{issue.path}: {issue.message} ({issue.physics})" for issue in issues)


def _render_guidance(issues: Sequence[Issue]) -> str:
    """Render unresolved issues as CLI planning questions."""
    lines: list[str] = []
    for issue in issues:
        prompt = issue.question or issue.message
        lines.append(f"{issue.path}: {prompt} Physical reason: {issue.physics}")
    return "\n".join(lines)


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    parts = [match.group(1) or int(match.group(2)) for match in re.finditer(r"([^\.\[\]]+)|\[(\d+)\]", path)]
    if not parts or ".".join(str(part) for part in parts if isinstance(part, str)) == "":
        raise ValueError(f"invalid answer path: {path}")
    current: Any = document
    for part in parts[:-1]:
        if isinstance(part, int) and isinstance(current, list):
            if part >= len(current):
                raise ValueError(f"answer path index does not exist: {path}")
            current = current[part]
        else:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"answer path does not exist: {path}")
            current = current[part]
    last = parts[-1]
    if isinstance(current, list) and isinstance(last, int):
        if last >= len(current):
            raise ValueError(f"answer path index does not exist: {path}")
        current[last] = value
    elif isinstance(current, dict):
        current[last] = value
    else:
        raise ValueError(f"answer path is not assignable: {path}")


def _parse_assignment(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError("--set expects path=json_value")
    path, text = raw.split("=", 1)
    return path, json.loads(text)


def _ask_for_plan_answers(manifest: Manifest) -> list[Any]:
    """Answer each currently unresolved issue once, then re-evaluate new issues."""
    document = manifest.document
    answered_paths: set[str] = set()
    issues = manifest.validate()
    while True:
        pending = [issue for issue in issues if issue.path not in answered_paths]
        if not pending:
            return issues
        for issue in pending:
            prompt = issue.question or issue.message
            raw = input(f"{issue.path}: {prompt} JSON value: ")
            value = json.loads(raw)
            _set_path(document, issue.path, value)
            answered_paths.add(issue.path)
        issues = manifest.validate()
        if not issues:
            return issues


def _validate(path: Path) -> tuple[Manifest, list[Any]]:
    manifest = load_manifest(path)
    return manifest, manifest.validate()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lamet-agent-neo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate one JSON manifest")
    validate.add_argument("manifest", type=Path)
    plan = subparsers.add_parser("plan", help="write a validated manifest to a new path")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--set", dest="assignments", action="append", default=[], help="set one JSON value at a dot path")
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
    if args.command == "validate":
        try:
            _, issues = _validate(args.manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if issues:
            print(_render_issues(issues))
            return 1
        print("manifest is valid")
        return 0
    if args.command == "plan":
        try:
            manifest = load_manifest(args.manifest)
            document = manifest.document
            if args.assignments:
                for assignment in args.assignments:
                    path, value = _parse_assignment(assignment)
                    _set_path(document, path, value)
                issues = manifest.validate()
            else:
                issues = _ask_for_plan_answers(manifest)
            if issues:
                print(_render_guidance(issues), file=sys.stderr)
                return 1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(args.output)
            return 0
        except (EOFError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    try:
        manifest = load_manifest(args.manifest)
        issues = manifest.validate()
        if issues:
            print(_render_issues(issues), file=sys.stderr)
            return 1
        backend = create_backend(args.provider, args.model, args.api_key_file)
        result = create_session(backend, progress_mode=args.progress).run_manifest(manifest)
        print(json.dumps({"status": "completed", "jobs": sorted(result["summaries"])}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
