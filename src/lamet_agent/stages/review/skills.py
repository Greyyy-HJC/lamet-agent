"""Stage-local skill guidance for review generation."""

from __future__ import annotations

from lamet_agent.manifest import AnalysisManifest, StageJob


STAGE_SKILL = """
The review stage is an LLM-written synthesis pass. It asks the configured
backend/model to write the full review from stage reports, NetCDF summaries, and
SVG artifact paths. When `stages.review.defaults.literature` is true, the stage
also injects background-only LaMET literature context from the local SQLite
library, limited by `literature_max_papers` (default 4). The requested report
language is generated directly. SVG paths are provenance only, and figure
statements must be grounded in report text and NetCDF summaries.
""".strip()

TOOL_CATALOG = {
    "write_review": "Collect stage reports, NetCDF summaries, and SVG paths, then ask the configured LLM to write review.md or review_CN.md.",
}


def tool_catalog() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TOOL_CATALOG.items())


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    del manifest, job
    return []
