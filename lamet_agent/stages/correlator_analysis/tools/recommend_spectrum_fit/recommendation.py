"""Prepare evidence and request one direct-spectrum fit suggestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis.tools._correlator_evidence import prepare
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class SpectrumSuggestion(TypedDict):
    """One complete direct-spectrum fit suggestion."""

    t_min: int
    t_max: int
    n_states: int
    prior_means: dict[str, float]
    prior_widths: dict[str, float]


def recommend(
    context: ToolContext,
    session: LlmSession,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> SpectrumSuggestion:
    """Return typed spectrum parameters from prepared gvar evidence."""
    evidence = prepare(context)
    if diagnostics is not None:
        evidence["previous_fit_diagnostics"] = diagnostics
    instruction = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    schema, _nullable = annotation_schema(SpectrumSuggestion)
    response = session.complete(
        label="spectrum fit recommendation",
        user_message=json.dumps(
            {"instruction": instruction, "evidence": json_compatible(evidence)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "spectrum_fit_recommendation", "schema": schema},
    )
    if response.structured is None:
        raise RuntimeError("spectrum recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(SpectrumSuggestion, result, "spectrum_fit_recommendation")
    return result


__all__ = ["SpectrumSuggestion", "recommend"]
