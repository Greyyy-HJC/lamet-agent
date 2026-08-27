"""Deterministic renormalization job workflow."""

from __future__ import annotations

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.renormalization._apply import run as apply
from lamet_agent.stages.renormalization._fit import run as fit
from lamet_agent.stages.renormalization._inspection import run as inspect


def run(context: ToolContext, _session: LlmSession) -> None:
    """Inspect and execute one explicit fit or apply job without LLM orchestration."""
    inspect(context)
    (fit if context.params["type"] == "fit" else apply)(context)


__all__ = ["run"]
