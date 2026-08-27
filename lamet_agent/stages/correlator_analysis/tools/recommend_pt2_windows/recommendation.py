"""Prepare two-point gvar evidence and recommend fit windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis.hook import _mean_error, ensure_correlators
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class Pt2Window(TypedDict):
    """One half-open two-point fit window."""

    tmin: int
    tmax: int


class Pt2WindowSuggestion(TypedDict):
    """Ordered two-point fit-window recommendation."""

    windows: list[Pt2Window]


def recommend(context: ToolContext, session: LlmSession) -> list[Pt2Window]:
    """Return two-point windows from direct central values and uncertainties."""
    correlators = ensure_correlators(context)
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    records = []
    for correlator_id, data in correlators.items():
        if data.attrs.get("correlator_type") != "two_point":
            continue
        if data.dims != ["t"]:
            raise ValueError("two-point recommendation data must have only the t dimension")
        mean, error = _mean_error(data, "real", sample_error_mode)
        records.append(
            {
                "correlator_id": correlator_id,
                "t": [int(value) for value in data.coords["t"]],
                "mean": mean,
                "error": error,
            }
        )
    if not records:
        raise ValueError("two-point window recommendation requires two-point data")
    lower = max(min(record["t"]) for record in records)
    upper = min(max(record["t"]) for record in records) + 1
    if lower >= upper:
        raise ValueError("two-point correlators have no shared time-coordinate coverage")
    evidence = {
        "two_point_correlators": records,
        "allowed_time_range": {"min": int(lower), "max": int(upper)},
        "minimum_points": 2 * max(int(value) for value in context.params["nstate"]),
        "window_convention": "tmin is inclusive and tmax is exclusive",
    }
    instruction = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    schema, _nullable = annotation_schema(Pt2WindowSuggestion)
    response = session.complete(
        label="two-point window recommendation",
        user_message=json.dumps(
            {"instruction": instruction, "evidence": json_compatible(evidence)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "pt2_window_recommendation", "schema": schema},
    )
    if response.structured is None:
        raise RuntimeError("two-point window recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(Pt2WindowSuggestion, result, "pt2_window_recommendation")
    return result["windows"]


__all__ = ["Pt2Window", "Pt2WindowSuggestion", "recommend"]
