"""Human-readable trace formatting for the agent tool loop.

Purpose:
- print ReAct-style cycle logs (prompt, model action, tool observation)
- used when ``run_agent(..., verbose=True)`` or CLI ``--verbose``

Example usage:
- from lamet_agent.core.trace import AgentTrace
- trace = AgentTrace()
- trace.stage_begin("correlator_analysis")
- trace.cycle_begin(1)
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

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
        emit: Emit | None = None,
        prompt_max_chars: int = 12_000,
    ) -> None:
        self.enabled = enabled
        self._emit = emit or _default_emit
        self.prompt_max_chars = prompt_max_chars

    def _write(self, text: str) -> None:
        if self.enabled:
            self._emit(text)

    def run_begin(self, *, run_id: str, model: str, stages: list[str]) -> None:
        self._write("")
        self._write("=" * 60)
        self._write(f"Agent run: {run_id}  (model={model})")
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

    def llm_call_begin(self, *, model: str) -> None:
        if model == "external":
            self._write("Loading next action from transcript...")
        elif model == "mock":
            self._write("Resolving mock action...")
        else:
            self._write(f"Calling LLM ({model})...")

    def llm_call_end(self) -> None:
        self._write("LLM response received.")

    def prompt_delta(self, observation: dict[str, Any]) -> None:
        """Print the incremental user turn for multi-turn stages."""
        self._write("")
        self._write("[Observation for LLM]")
        self._write(json.dumps(observation, ensure_ascii=False, indent=2))

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
