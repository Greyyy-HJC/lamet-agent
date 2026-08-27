"""Prepare qDA evidence and request representative tuning coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis.tools._correlator_evidence import prepare
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class QdaTuneZSuggestion(TypedDict):
    """Representative nonzero qDA tuning coordinates."""

    tune_z_values: list[float]


def recommend(
    context: ToolContext,
    session: LlmSession,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[float]:
    """Return representative qDA z coordinates from prepared gvar evidence."""
    evidence = prepare(context)
    if diagnostics is not None:
        evidence["previous_fit_diagnostics"] = diagnostics
    instruction = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    schema, _nullable = annotation_schema(QdaTuneZSuggestion)
    response = session.complete(
        label="qDA tuning-coordinate recommendation",
        user_message=json.dumps(
            {"instruction": instruction, "evidence": json_compatible(evidence)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "qda_tune_z_recommendation", "schema": schema},
    )
    if response.structured is None:
        raise RuntimeError("qDA tuning recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(QdaTuneZSuggestion, result, "qda_tune_z_recommendation")
    return result["tune_z_values"]


__all__ = ["QdaTuneZSuggestion", "recommend"]
