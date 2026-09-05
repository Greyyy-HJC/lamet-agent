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


class _SpectrumResponse(TypedDict):
    """Provider-compatible wire representation of one spectrum suggestion."""

    tmin: int
    tmax: int
    n_states: int
    prior_means: list[float]
    prior_widths: list[float]


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
    schema, _nullable = annotation_schema(_SpectrumResponse)
    schema["properties"]["tmin"]["minimum"] = 0
    schema["properties"]["tmax"]["minimum"] = 1
    allowed_n_states = sorted(set(context.params["nstate"]))
    schema["properties"]["n_states"].update({"minimum": 1, "enum": allowed_n_states})
    for name in ("prior_means", "prior_widths"):
        schema["properties"][name].update({"minItems": 2, "maxItems": 2 * max(allowed_n_states)})
        schema["properties"][name]["items"]["exclusiveMinimum"] = 0.0
    request = {
        "task": "direct_spectrum_fit",
        "phase": "retry" if previous_attempts is not None else "initial",
        "requested_fields": sorted(schema["required"]),
        "evidence": json_compatible(evidence),
    }
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
    wire_result = dict(response.structured)
    validate_value(_SpectrumResponse, wire_result, "spectrum_fit_recommendation")
    n_states = wire_result["n_states"]
    prior_names = [*[f"E{index}" for index in range(n_states)], *[f"A{index}" for index in range(n_states)]]
    if len(wire_result["prior_means"]) != len(prior_names) or len(wire_result["prior_widths"]) != len(prior_names):
        raise ValueError("spectrum recommendation must provide one mean and width for every energy and amplitude")
    return {
        "tmin": wire_result["tmin"],
        "tmax": wire_result["tmax"],
        "n_states": n_states,
        "prior_means": dict(zip(prior_names, wire_result["prior_means"], strict=True)),
        "prior_widths": dict(zip(prior_names, wire_result["prior_widths"], strict=True)),
    }


__all__ = ["SpectrumSuggestion", "recommend"]
