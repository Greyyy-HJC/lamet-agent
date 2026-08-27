"""Prepare evidence and request one direct-spectrum fit suggestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis.tools._correlator_evidence import ensure_context
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
    fixed_parameters: dict[str, Any] | None = None,
    previous_attempts: dict[str, dict[str, Any]] | None = None,
) -> SpectrumSuggestion:
    """Return typed spectrum parameters from prepared gvar evidence."""
    ensure_context(context, session)
    evidence = {"fixed_parameters": fixed_parameters or {}}
    if previous_attempts is not None:
        evidence["previous_attempts"] = previous_attempts
    instruction = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    if previous_attempts is not None:
        instruction += (
            "\n\nThe previous spectrum parameters were fitted and did not satisfy the quality policy. "
            "Make a conservative adjustment using the parameter-to-quality mapping in previous_attempts."
        )
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
