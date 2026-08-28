"""Recommend missing or revised Fourier tail-range candidate lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.fourier_transform._inspection import prepare
from lamet_agent.structured import annotation_schema, json_compatible, validate_value


class TailRangeSuggestion(TypedDict, total=False):
    """Candidate physical-distance boundaries for the deterministic scan."""

    zmin_fm: list[float]
    zmax_fm: list[float]


def _ensure_context(context: ToolContext, session: LlmSession) -> None:
    if session.has_context("fourier_tail_fit_data"):
        return
    data, spacing = prepare(context)
    mode = str(context.manifest["metadata"]["sample_error_mode"])
    components = {}
    for name, selected in (("real", data.real), ("imag", data.imag)):
        components[name] = str(selected.average(mode))
    session.add_context(
        "fourier_tail_fit_data",
        {
            "z_fm": [float(value) for value in data.coords["z"]],
            "spacing_fm": spacing,
            "momentum_gev": data.attrs.get("momentum_gev"),
            "components": components,
        },
    )


def recommend(
    context: ToolContext,
    session: LlmSession,
    *,
    requested_fields: set[str],
    fixed_parameters: dict[str, Any] | None = None,
    previous_attempts: dict[str, dict[str, Any]] | None = None,
) -> TailRangeSuggestion:
    """Return exactly the requested range fields under a strict dynamic schema."""
    _ensure_context(context, session)
    instruction = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8").strip()
    evidence = {
        "fixed_parameters": fixed_parameters or {},
        "scheme_scan": context.params["scheme_scan"],
        "zmax_ext_fm": context.params["zmax_ext_fm"],
    }
    if previous_attempts is not None:
        evidence["previous_attempts"] = previous_attempts
        instruction += "\n\nThe complete previous z-range × scheme scan was unacceptable; revise both ranges."
    schema, _nullable = annotation_schema(TailRangeSuggestion)
    schema["properties"]["zmin_fm"].update({"minItems": 1, "uniqueItems": True})
    schema["properties"]["zmin_fm"]["items"]["minimum"] = 0.0
    schema["properties"]["zmax_fm"].update({"minItems": 1, "uniqueItems": True})
    schema["properties"]["zmax_fm"]["items"].update(
        {"exclusiveMinimum": 0.0, "maximum": float(context.params["zmax_ext_fm"])}
    )
    schema["properties"] = {name: value for name, value in schema["properties"].items() if name in requested_fields}
    schema["required"] = sorted(requested_fields)
    response = session.complete(
        label="Fourier tail-range recommendation",
        user_message=json.dumps(
            {"instruction": instruction, "evidence": json_compatible(evidence)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        response_schema={"name": "fourier_tail_range_recommendation", "schema": schema},
    )
    if response.structured is None:
        raise RuntimeError("Fourier tail-range recommendation returned no structured response")
    result = dict(response.structured)
    validate_value(TailRangeSuggestion, result, "fourier_tail_range_recommendation")
    if set(result) != requested_fields:
        raise ValueError(f"Fourier recommendation must return exactly {sorted(requested_fields)}")
    return result


__all__ = ["TailRangeSuggestion", "recommend"]
