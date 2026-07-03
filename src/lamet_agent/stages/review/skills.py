"""Stage-local skill guidance for review generation."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


STAGE_SKILL = """
The review stage is a report-level synthesis pass. It reads existing stage
reports, preserves their provenance, includes the formulas used by the pipeline,
and marks missing stages explicitly instead of inferring unreported results.
""".strip()

TOOL_CATALOG = {
    "write_review": "Read stage reports from the manifest artifact directory and write review.md or review_CN.md.",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    del manifest, job
    return []
