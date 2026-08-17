"""Executable manifest contract for review generation."""

from __future__ import annotations

from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import ParameterSpec, StageParamContract, StageValidationContext, merge_stage_params


def _paper_limit(value: Any) -> str | None:
    return None if type(value) is int and value >= 1 else "review literature_max_papers must be a positive integer."


def _normalize_draft(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stages = payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {}
    stage = stages.get("review", {}) if isinstance(stages, dict) else {}
    defaults = stage.get("defaults", {}) if isinstance(stage, dict) else {}
    if not isinstance(defaults, dict) or defaults.get("literature") is not True or "literature_max_papers" in defaults:
        return []
    defaults["literature_max_papers"] = 4
    return [{"path": "stages.review.defaults.literature_max_papers", "old": None, "new": 4, "note": "Applied the default literature paper limit."}]


STAGE_PARAM_CONTRACT = StageParamContract(
    code_prefix="review",
    summary="Summarize workflow outputs and optionally place them in literature context.",
    physics="Review settings affect evidence gathering and presentation, not the numerical LQCD analysis.",
    planning_notes=("literature_max_papers defaults to 4 when literature is enabled.",),
    normalize_draft=_normalize_draft,
    schema={
        "literature": ParameterSpec(
            summary="Enable literature comparison.",
            physics="External references provide context but do not alter computed observables.",
            expected=bool,
            default="false",
        ),
        "literature_max_papers": ParameterSpec(
            summary="Maximum number of literature papers included.",
            physics="The limit bounds review breadth and runtime only.",
            expected=int,
            default="4",
            validator=_paper_limit,
        ),
    },
    removed={},
)


def build_validation_context(manifest: AnalysisManifest, job: StageJob) -> StageValidationContext:
    """Build the resolved review context consumed by the shared evaluator."""
    params = merge_stage_params(manifest.stages["review"].defaults, job.params)
    return StageValidationContext(
        stage="review",
        job_id=job.id,
        job_path=f"stages.review.jobs.{job.id}",
        params=params,
        inputs=dict(job.inputs),
        metadata=manifest.metadata.model_dump(),
        authored_params=params,
    )


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return backward-compatible concise diagnostics."""
    return [issue.message for issue in STAGE_PARAM_CONTRACT.evaluate(build_validation_context(manifest, job))]
