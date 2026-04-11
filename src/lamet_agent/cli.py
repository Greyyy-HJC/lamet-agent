"""CLI entry points for validating, inspecting, and running LaMET workflows.

Example usage:
    lamet-agent validate examples/workflow_smoke_manifest.json
    lamet-agent workflow examples/workflow_smoke_manifest.json
    lamet-agent run examples/workflow_smoke_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from lamet_agent.errors import LametAgentError
from lamet_agent.logging_utils import configure_logging
from lamet_agent.planners import RuleBasedPlanner
from lamet_agent.schemas import load_manifest
from lamet_agent.workflows import execute_manifest

try:
    import typer
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    typer = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    tqdm = None


class WorkflowProgressReporter:
    """Render per-stage progress updates for manifest execution."""

    def __init__(self, stage_names: list[str]) -> None:
        self.stage_names = stage_names
        self.total = len(stage_names)
        self.stream = sys.stderr
        self.completed = 0
        self.current_stage: str | None = None
        self.current_bar = None
        self.current_total = 0
        self.current_completed = 0

    def __call__(self, event: dict[str, Any]) -> None:
        kind = str(event["event"])
        stage_name = str(event["stage_name"])
        if kind == "stage_message":
            self._emit_stage_message(str(event.get("message", "")))
            return
        if kind == "stage_progress_start":
            self._start_stage_progress(
                stage_name=stage_name,
                description=str(event.get("description", stage_name)),
                total=int(event.get("total", 0)),
                unit=str(event.get("unit", "step")),
            )
            return
        if kind == "stage_progress_update":
            self._advance_stage_progress(int(event.get("advance", 1)))
            return
        if kind == "stage_progress_end":
            self._finish_stage_progress()
            return

        stage_index = int(event["stage_index"])
        stage_total = int(event["stage_total"])
        if kind == "stage_started":
            self._close_current_bar()
            description = str(event.get("stage_description", "")).strip()
            if description:
                print(
                    f"[{stage_index + 1}/{stage_total}] {stage_name}: {description}",
                    file=self.stream,
                )
            else:
                print(f"[{stage_index + 1}/{stage_total}] Running stage: {stage_name}", file=self.stream)
            self.current_stage = stage_name
            return
        if kind == "stage_reused":
            self._close_current_bar()
            print(f"[{stage_index + 1}/{stage_total}] Reusing stage: {stage_name}", file=self.stream)
            self.completed = max(self.completed, stage_index + 1)
            return
        if kind == "stage_completed":
            self._finish_stage_progress()
            self.completed = max(self.completed, stage_index + 1)
            print(f"[{stage_index + 1}/{stage_total}] Completed stage: {stage_name}", file=self.stream)
            return
        if kind == "stage_failed":
            message = str(event.get("error", "unknown error"))
            self._close_current_bar()
            print(f"[{stage_index + 1}/{stage_total}] Stage failed: {stage_name}: {message}", file=self.stream)

    def _start_stage_progress(self, *, stage_name: str, description: str, total: int, unit: str) -> None:
        self._close_current_bar()
        self.current_stage = stage_name
        self.current_total = max(total, 0)
        self.current_completed = 0
        if tqdm is not None:
            self.current_bar = tqdm(
                total=self.current_total,
                desc=description,
                unit=unit,
                dynamic_ncols=True,
                file=self.stream,
            )
            return
        if self.current_total > 0:
            print(f"{description}: 0/{self.current_total} {unit}", file=self.stream)

    def _advance_stage_progress(self, advance: int) -> None:
        if self.current_total <= 0:
            return
        self.current_completed = min(self.current_total, self.current_completed + max(advance, 0))
        if self.current_bar is not None:
            self.current_bar.update(max(advance, 0))
            self.current_bar.refresh()
            return
        bar_width = 24
        filled = int(round((self.current_completed / self.current_total) * bar_width))
        filled = min(bar_width, max(0, filled))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"[{bar}] {self.current_completed}/{self.current_total}", file=self.stream)

    def _finish_stage_progress(self) -> None:
        if self.current_total > 0 and self.current_completed < self.current_total:
            self._advance_stage_progress(self.current_total - self.current_completed)
        self._close_current_bar()

    def _emit_stage_message(self, message: str) -> None:
        if not message:
            return
        if self.current_bar is not None:
            self.current_bar.write(message)
            self.current_bar.refresh()
            return
        print(message, file=self.stream)

    def close(self) -> None:
        self._close_current_bar()

    def _close_current_bar(self) -> None:
        if self.current_bar is not None:
            self.current_bar.close()
            self.current_bar = None
        self.current_total = 0
        self.current_completed = 0


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


def _run_impl(
    manifest_path: str,
    *,
    resume_from: str | None = None,
    start_stage: str | None = None,
) -> dict[str, Any]:
    planner = RuleBasedPlanner()
    skip_file_check = resume_from is not None and start_stage is not None and start_stage != "correlator_analysis"
    manifest = load_manifest(manifest_path, skip_file_check=skip_file_check)
    plan = planner.resolve(manifest)
    reporter = WorkflowProgressReporter(plan.stage_names)
    try:
        run = execute_manifest(
            manifest_path,
            planner=planner,
            resume_from=resume_from,
            start_stage=start_stage,
            progress_callback=reporter,
        )
    finally:
        reporter.close()
    return {
        "run_directory": str(run.run_directory),
        "stage_names": [result.stage_name for result in run.stage_results],
        "report": str(run.run_directory / "report.md"),
    }


def _dump_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _handle_command(
    command: str,
    manifest_path: str | None = None,
    resume_from: str | None = None,
    start_stage: str | None = None,
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
            _dump_result(_run_impl(manifest_path, resume_from=resume_from, start_stage=start_stage))
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
        if command == "run":
            command_parser.add_argument("--resume-from", dest="resume_from", default=None)
            command_parser.add_argument("--start-stage", dest="start_stage", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI with either Typer or argparse."""
    parser = _build_argparse()
    args = parser.parse_args(argv)
    return _handle_command(
        args.command,
        manifest_path=getattr(args, "manifest_path", None),
        resume_from=getattr(args, "resume_from", None),
        start_stage=getattr(args, "start_stage", None),
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
    def run_command(
        manifest_path: str,
        resume_from: str | None = typer.Option(None, "--resume-from"),
        start_stage: str | None = typer.Option(None, "--start-stage"),
    ) -> None:
        """Execute the resolved workflow and materialize output artifacts."""
        _dump_result(_run_impl(manifest_path, resume_from=resume_from, start_stage=start_stage))

    def entrypoint() -> None:
        """Invoke the Typer application."""
        configure_logging()
        app()

else:

    def entrypoint() -> None:
        """Fallback CLI entrypoint when typer is unavailable."""
        raise SystemExit(main())
