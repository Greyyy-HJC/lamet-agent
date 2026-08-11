"""Human-readable trace formatting for the agent tool loop.

Purpose:
- print ReAct-style cycle logs (prompt, model action, tool observation)
- used when ``run_agent(..., verbose=True)`` or CLI ``--verbose``
- print a startup banner and stage/job headers in quiet (non-verbose) mode

Example usage:
- from lamet_agent.core.trace import AgentTrace
- trace = AgentTrace(quiet_ui=True)
- trace.run_banner(run_id="demo", backend="mock", stages=["correlator_analysis"])
- trace.job_begin("correlator_analysis", "ca")
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

from .banner import BANNER, format_job_header

Emit = Callable[[str], None]


def _default_emit(text: str, *, stream: TextIO | None = None) -> None:
    (stream or sys.stdout).write(text + "\n")
    (stream or sys.stdout).flush()


class AgentTrace:
    """Format and emit one agent step at a time."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        quiet_ui: bool = False,
        emit: Emit | None = None,
        prompt_max_chars: int = 12_000,
    ) -> None:
        self.enabled = enabled
        self.quiet_ui = quiet_ui
        self._emit = emit or _default_emit
        self.prompt_max_chars = prompt_max_chars

    def _write(self, text: str) -> None:
        if self.enabled:
            self._emit(text)

    def _write_quiet(self, text: str) -> None:
        if self.quiet_ui:
            self._emit(text)

    def run_banner(
        self,
        *,
        run_id: str,
        backend: str,
        stages: list[str],
        model_spec: str | None = None,
    ) -> None:
        """Print the LaMET Agent banner and a compact run summary."""
        if not self.quiet_ui:
            return
        self._write_quiet(BANNER)
        self._write_quiet("")
        if model_spec:
            self._write_quiet(f"Run: {run_id}  backend={backend}  model={model_spec}")
        else:
            self._write_quiet(f"Run: {run_id}  backend={backend}")
        self._write_quiet(f"Stages: {', '.join(stages)}")
        self._write_quiet("")

    def job_begin(
        self,
        stage: str,
        job_id: str,
        *,
        input_issues: list[str] | None = None,
    ) -> None:
        """Print a one-line stage/job header before tool execution."""
        if not self.quiet_ui:
            return
        self._write_quiet(format_job_header(stage, job_id))
        if input_issues:
            self._write_quiet(f"Input issues: {input_issues}")

    def run_begin(
        self,
        *,
        run_id: str,
        backend: str,
        stages: list[str],
        model_spec: str | None = None,
    ) -> None:
        self._write("")
        self._write("=" * 60)
        if model_spec:
            self._write(f"Agent run: {run_id}  (backend={backend} model={model_spec})")
        else:
            self._write(f"Agent run: {run_id}  (backend={backend})")
        self._write(f"Stages: {', '.join(stages)}")
        self._write("=" * 60)

    def stage_begin(self, stage: str, *, input_issues: list[str] | None = None) -> None:
        self._write("")
        self._write("#" * 60)
        self._write(f"Stage: {stage}")
        if input_issues:
            self._write(f"Input issues: {input_issues}")
        self._write("#" * 60)

    def stage_context(self, text: str) -> None:
        """Print static stage context once per stage."""
        self._write("")
        self._write("[Stage context]")
        if len(text) <= self.prompt_max_chars:
            self._write(text)
            return
        half = self.prompt_max_chars // 2
        self._write(text[:half])
        self._write(
            f"\n... [{len(text) - self.prompt_max_chars} characters omitted] ...\n"
        )
        self._write(text[-half:])

    def cycle_begin(self, cycle: int) -> None:
        self._write("")
        self._write("-" * 40)
        self._write(f"Cycle {cycle}")
        self._write("-" * 40)

    def llm_call_begin(self, *, backend: str, model_spec: str | None = None) -> None:
        if backend == "external":
            self._write("Loading next action from transcript...")
        elif backend == "mock":
            self._write("Resolving mock action...")
        elif model_spec:
            self._write(f"Calling LLM ({model_spec})...")
        else:
            self._write(f"Calling LLM ({backend})...")

    def llm_call_end(self) -> None:
        self._write("LLM response received.")

    def prompt_delta(self, observation: dict[str, Any]) -> None:
        """Print a compact note for the incremental user turn."""
        tool_name = observation.get("tool_name", "unknown tool")
        self._write("")
        self._write(f"[Observation forwarded to LLM: {tool_name}]")

    def model_output(self, action: dict[str, Any]) -> None:
        self._write("")
        self._write("[Model output]")
        reason = action.get("reason")
        if reason:
            self._write(f"Reason: {reason}")
        act = action.get("action")
        if act == "call_tool":
            tool_name = action.get("tool_name", "")
            args = action.get("args") or {}
            args_text = json.dumps(args, ensure_ascii=False, indent=2)
            self._write("Action: call_tool")
            self._write(f"  tool_name: {tool_name}")
            self._write(f"  args: {args_text}")
        elif act == "request_user_input":
            self._write("Action: request_user_input")
            questions = action.get("questions") or []
            for idx, question in enumerate(questions, start=1):
                self._write(f"  {idx}. {question}")
        elif act == "finish":
            self._write("Action: finish")
        else:
            self._write(json.dumps(action, ensure_ascii=False, indent=2))

    def observation(self, observation: dict[str, Any]) -> None:
        self._write("")
        self._write("[Observation]")
        self._write(json.dumps(observation, ensure_ascii=False, indent=2))

    def stage_end(self, stage: str, *, n_steps: int) -> None:
        self._write("")
        self._write(f"Stage {stage} finished after {n_steps} cycle(s).")

    def run_end(self, *, action_count: int) -> None:
        self._write("")
        self._write("=" * 60)
        self._write(f"Agent run complete ({action_count} action(s) recorded).")
        self._write("=" * 60)
