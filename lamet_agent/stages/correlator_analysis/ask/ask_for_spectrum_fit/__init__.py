"""Request one direct-spectrum fit suggestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class SpectrumSuggestion(TypedDict):
    """One complete direct-spectrum fit suggestion."""

    tmin: int
    tmax: int
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
    evidence = {"fixed_parameters": fixed_parameters or {}}
    if previous_attempts is not None:
        evidence["previous_attempts"] = previous_attempts
    prompt = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    request = {"evidence": json_compatible(evidence)}
    schema, _nullable = annotation_schema(SpectrumSuggestion)
    schema["properties"]["tmin"]["minimum"] = 0
    schema["properties"]["tmax"]["minimum"] = 1
    schema["properties"]["n_states"].update({"minimum": 1, "enum": sorted(set(context.params["nstate"]))})
    schema["properties"]["prior_means"].update(
        {
            "minProperties": 1,
            "additionalProperties": {"type": "number", "exclusiveMinimum": 0.0},
        }
    )
    schema["properties"]["prior_widths"].update(
        {
            "minProperties": 1,
            "additionalProperties": {"type": "number", "exclusiveMinimum": 0.0},
        }
    )
    response = session.complete(
        label="spectrum fit recommendation",
        user_message=json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "spectrum_fit_recommendation", "schema": schema},
        ask_prompt_key="correlator_spectrum_fit_ask",
        ask_prompt=prompt,
    )
    if response.structured is None:
        raise RuntimeError("spectrum recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(SpectrumSuggestion, result, "spectrum_fit_recommendation")
    return result


__all__ = ["SpectrumSuggestion", "recommend"]
