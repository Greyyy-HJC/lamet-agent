"""Request qDA tuning coordinates and two-point windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class Pt2Window(TypedDict):
    tmin: int
    tmax: int


class QdaFitSuggestion(TypedDict, total=False):
    """Joint qDA fit-parameter suggestion."""

    tune_z_values: list[float]
    pt2_windows: list[Pt2Window]


def recommend(
    context: ToolContext,
    session: LlmSession,
    *,
    requested_fields: set[str] | None = None,
    fixed_parameters: dict[str, Any] | None = None,
    previous_attempts: dict[str, dict[str, Any]] | None = None,
) -> QdaFitSuggestion:
    """Return joint qDA parameters from prepared gvar evidence."""
    evidence = {"fixed_parameters": fixed_parameters or {}}
    if previous_attempts is not None:
        evidence["previous_attempts"] = previous_attempts
    prompt = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    request = {"evidence": json_compatible(evidence)}
    if previous_attempts is not None:
        request["instruction"] = (
            "The previous tuning coordinates were tried across every authored fit combination. "
            "Make a conservative adjustment using the parameter-to-quality mapping in previous_attempts."
        )
    schema, _nullable = annotation_schema(QdaFitSuggestion)
    requested = requested_fields or {"tune_z_values"}
    schema["properties"]["tune_z_values"].update({"minItems": 1, "uniqueItems": True})
    schema["properties"]["tune_z_values"]["items"]["not"] = {"const": 0}
    schema["properties"]["pt2_windows"].update({"minItems": 1, "uniqueItems": True})
    schema["properties"]["pt2_windows"]["items"]["properties"]["tmin"]["minimum"] = 0
    schema["properties"]["pt2_windows"]["items"]["properties"]["tmax"]["minimum"] = 1
    schema["properties"] = {name: value for name, value in schema["properties"].items() if name in requested}
    schema["required"] = sorted(requested)
    response = session.complete(
        label="qDA tuning-coordinate recommendation",
        user_message=json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "qda_tune_z_recommendation", "schema": schema},
        ask_prompt_key="correlator_qda_tune_z_ask",
        ask_prompt=prompt,
    )
    if response.structured is None:
        raise RuntimeError("qDA tuning recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(QdaFitSuggestion, result, "qda_tune_z_recommendation")
    if set(result) != requested:
        raise ValueError(f"qDA recommendation must return exactly {sorted(requested)}")
    return result


__all__ = ["Pt2Window", "QdaFitSuggestion", "recommend"]
