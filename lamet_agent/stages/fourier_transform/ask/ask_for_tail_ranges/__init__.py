"""Request missing or revised Fourier tail-range candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.structured import annotation_schema, json_compatible, validate_unique_items, validate_value


class TailRangeSuggestion(TypedDict, total=False):
    """Candidate physical-distance boundaries for the deterministic scan."""

    zmin_fm: list[float]
    zmax_fm: list[float]


def recommend(
    context: ToolContext,
    session: LlmSession,
    *,
    requested_fields: set[str],
    fixed_parameters: dict[str, Any] | None = None,
    previous_attempts: dict[str, dict[str, Any]] | None = None,
) -> TailRangeSuggestion:
    """Return exactly the requested range fields under a strict dynamic schema."""
    prompt = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    evidence = {
        "fixed_parameters": fixed_parameters or {},
        "scheme_scan": context.params["scheme_scan"],
        "zmax_ext_fm": context.params["zmax_ext_fm"],
    }
    if previous_attempts is not None:
        evidence["previous_attempts"] = previous_attempts
    schema, _nullable = annotation_schema(TailRangeSuggestion)
    schema["properties"]["zmin_fm"]["minItems"] = 1
    schema["properties"]["zmin_fm"]["items"]["minimum"] = 0.5
    schema["properties"]["zmax_fm"]["minItems"] = 1
    schema["properties"]["zmax_fm"]["items"].update(
        {"exclusiveMinimum": 0.0, "maximum": float(context.params["zmax_ext_fm"])}
    )
    schema["properties"] = {name: value for name, value in schema["properties"].items() if name in requested_fields}
    schema["required"] = sorted(requested_fields)
    request = {
        "task": "fourier_tail_range_tuning",
        "phase": "retry" if previous_attempts is not None else "initial",
        "requested_fields": sorted(requested_fields),
        "evidence": json_compatible(evidence),
    }
    response = session.complete(
        label="Fourier tail-range recommendation",
        user_message=json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "fourier_tail_range_recommendation", "schema": schema},
        ask_prompt_key="fourier_tail_ranges_ask",
        ask_prompt=prompt,
    )
    if response.structured is None:
        raise RuntimeError("Fourier tail-range recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(TailRangeSuggestion, result, "fourier_tail_range_recommendation")
    if set(result) != requested_fields:
        raise ValueError(f"Fourier recommendation must return exactly {sorted(requested_fields)}")
    for name in ("zmin_fm", "zmax_fm"):
        if name in result:
            validate_unique_items(result[name], f"fourier_tail_range_recommendation.{name}")
    return result


__all__ = ["TailRangeSuggestion", "recommend"]
