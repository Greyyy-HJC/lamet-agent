"""Prepare three-point gvar evidence and recommend fit windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.correlator_analysis.hook import _mean_error, ensure_correlators
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class Pt3Window(TypedDict):
    """One three-point source-sink and insertion-time selection."""

    tsep_ls: list[int]
    tau_cut: int


class Pt3WindowSuggestion(TypedDict):
    """Ordered three-point fit-window recommendation."""

    windows: list[Pt3Window]


def recommend(context: ToolContext, session: LlmSession) -> list[Pt3Window]:
    """Return three-point windows from direct central values and uncertainties."""
    correlators = ensure_correlators(context)
    sample_error_mode = str(context.manifest["metadata"]["sample_error_mode"])
    three_points = [
        (correlator_id, data)
        for correlator_id, data in correlators.items()
        if data.attrs.get("correlator_type") == "three_point"
    ]
    if len(three_points) != 1:
        raise ValueError("three-point window recommendation requires exactly one three-point correlator")
    correlator_id, data = three_points[0]
    if data.dims != ["tsep", "tau", "z"]:
        raise ValueError("three-point recommendation data must have tsep, tau, and z dimensions")
    requested_components = {
        "re": ("real",),
        "im": ("imag",),
        "both": ("real", "imag"),
    }[context.params["component"]]
    components = {}
    for component in requested_components:
        mean, error = _mean_error(data, component, sample_error_mode)
        components[component] = {"mean": mean, "error": error}
    evidence = {
        "correlator_id": correlator_id,
        "z": [float(value) for value in data.coords["z"]],
        "tsep": [int(value) for value in data.coords["tsep"]],
        "tau": [int(value) for value in data.coords["tau"]],
        "components": components,
        "fit_scope": context.params["fit_scope"],
        "constraint": "Each tau_cut must satisfy 2*tau_cut <= every selected tsep.",
    }
    instruction = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    schema, _nullable = annotation_schema(Pt3WindowSuggestion)
    response = session.complete(
        label="three-point window recommendation",
        user_message=json.dumps(
            {"instruction": instruction, "evidence": json_compatible(evidence)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "pt3_window_recommendation", "schema": schema},
    )
    if response.structured is None:
        raise RuntimeError("three-point window recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(Pt3WindowSuggestion, result, "pt3_window_recommendation")
    return result["windows"]


__all__ = ["Pt3Window", "Pt3WindowSuggestion", "recommend"]
