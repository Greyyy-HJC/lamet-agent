"""List compact literature records without loading paper bodies."""

from __future__ import annotations

from lamet_agent.agent import ToolContext
from lamet_agent.stages.review.tools._catalog import load_catalog


def run(context: ToolContext, *, stages: list[str] | None = None, hadrons: list[str] | None = None, currents: list[str] | None = None, observables: list[str] | None = None, methods: list[str] | None = None) -> dict[str, object]:
    """Filter compact catalog records by explicit tags."""
    records = load_catalog(context)
    filters = {"stages": stages, "hadrons": hadrons, "currents": currents, "observables": observables, "methods": methods}
    selected = []
    for record in records:
        tags = record.get("tags", {})
        if all(values is None or any(value in tags.get(key, []) for value in values) for key, values in filters.items()):
            selected.append({key: record.get(key) for key in ("id", "title", "authors", "year", "abstract", "source", "tags")})
    context.state["literature_candidates"] = selected
    return {"summary": f"listed {len(selected)} literature records", "metrics": {"paper_ids": [record["id"] for record in selected]}, "state_keys": ["literature_candidates"], "artifacts": []}
