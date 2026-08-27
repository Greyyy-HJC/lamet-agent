"""Deterministic perturbative-matching job workflow."""

from __future__ import annotations

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.perturbative_matching._apply import run as apply
from lamet_agent.stages.perturbative_matching._inspection import run as inspect


def run(context: ToolContext, _session: LlmSession) -> None:
    """Inspect the selected kernel and publish its deterministic application."""
    inspect(context)
    apply(context)


__all__ = ["run"]
