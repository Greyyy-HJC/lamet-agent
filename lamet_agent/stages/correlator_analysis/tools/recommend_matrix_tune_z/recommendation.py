"""Prepare ordinary matrix-fit evidence and request tuning coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis.tools._correlator_evidence import prepare
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class MatrixTuneZSuggestion(TypedDict):
    """Representative ordinary matrix-element tuning coordinates."""

    tune_z_values: list[float]


def recommend(
    context: ToolContext,
    session: LlmSession,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[float]:
    """Return representative z coordinates from prepared gvar evidence."""
    evidence = prepare(context)
    if diagnostics is not None:
        evidence["previous_fit_diagnostics"] = diagnostics
    instruction = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    schema, _nullable = annotation_schema(MatrixTuneZSuggestion)
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
    validate_value(MatrixTuneZSuggestion, result, "matrix_tune_z_recommendation")
    return result["tune_z_values"]


__all__ = ["MatrixTuneZSuggestion", "recommend"]
