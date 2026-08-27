"""Prepare ordinary matrix-fit evidence and request tuning coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis.tools._correlator_evidence import ensure_context
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class MatrixFitSuggestion(TypedDict, total=False):
    """Joint ordinary matrix-element fit-parameter suggestion."""

    tune_z_values: list[float]
    pt2_windows: list[dict[str, int]]
    pt3_windows: list[dict[str, Any]]


def recommend(
    context: ToolContext,
    session: LlmSession,
    *,
    requested_fields: set[str] | None = None,
    fixed_parameters: dict[str, Any] | None = None,
    previous_attempts: dict[str, dict[str, Any]] | None = None,
) -> MatrixFitSuggestion:
    """Return joint ordinary matrix-element parameters from prepared gvar evidence."""
    ensure_context(context, session)
    evidence = {"fixed_parameters": fixed_parameters or {}}
    if previous_attempts is not None:
        evidence["previous_attempts"] = previous_attempts
    instruction = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    if previous_attempts is not None:
        instruction += (
            "\n\nThe previous tuning coordinates were tried across every authored fit combination. "
            "Make a conservative adjustment using the parameter-to-quality mapping in previous_attempts."
        )
    schema, _nullable = annotation_schema(MatrixFitSuggestion)
    requested = requested_fields or {"tune_z_values"}
    schema["properties"] = {name: value for name, value in schema["properties"].items() if name in requested}
    schema["required"] = sorted(requested)
    response = session.complete(
        label="matrix-element tuning-coordinate recommendation",
        user_message=json.dumps(
            {"instruction": instruction, "evidence": json_compatible(evidence)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "matrix_tune_z_recommendation", "schema": schema},
    )
    if response.structured is None:
        raise RuntimeError("matrix tuning recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(MatrixFitSuggestion, result, "matrix_tune_z_recommendation")
    if set(result) != requested:
        raise ValueError(f"matrix recommendation must return exactly {sorted(requested)}")
    return result


__all__ = ["MatrixFitSuggestion", "recommend"]
