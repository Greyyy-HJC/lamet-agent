"""Executable manifest contract for review generation."""

from __future__ import annotations

from typing import Any

from lamet_agent.manifest import AnalysisManifest, StageJob
from lamet_agent.manifest_params import ParameterSpec, StageParamContract, StageValidationContext, merge_stage_params, resolve_stage_params


def _paper_limit(value: Any) -> str | None:
    return None if type(value) is int and value >= 1 else "review literature_max_papers must be a positive integer."



STAGE_PARAM_CONTRACT = StageParamContract(
    code_prefix="review",
    summary="Summarize workflow outputs and optionally place them in literature context.",
    physics="Review settings affect evidence gathering and presentation, not the numerical LQCD analysis.",
    planning_notes=("literature_max_papers defaults to 4 when literature is enabled.",),
    schema={
        "literature": ParameterSpec(
            summary="Enable literature comparison.",
            physics="True adds background-only entries from the local LaMET paper library to the evidence package. References provide context and provenance but never alter computed observables or replace workflow evidence.",
            expected=bool,
            choices=(False, True),
            choice_descriptions={
                False: "Synthesize only the completed workflow reports and artifact summaries.",
                True: "Also retrieve a bounded set of local literature records for background context.",
            },
            default=False,
        ),
        "literature_max_papers": ParameterSpec(
            summary="Maximum number of literature papers included.",
            physics="The limit bounds the breadth, prompt size, and retrieval runtime of the optional background context; it has no effect when literature is false and never changes numerical results.",
            expected=int,
            default=4,
            validator=_paper_limit,
        ),
    },
    removed={},
)


def build_validation_context(manifest: AnalysisManifest, job: StageJob) -> StageValidationContext:
    """Build the resolved review context consumed by the shared evaluator."""
    stage = manifest.stages["review"]
    authored = merge_stage_params(stage.defaults, job.params)
    params = resolve_stage_params("review", stage.defaults, job.params)
    return StageValidationContext(
        stage="review",
        job_id=job.id,
        job_path=f"stages.review.jobs.{job.id}",
        params=params,
        inputs=dict(job.inputs),
        metadata=manifest.metadata.model_dump(),
        authored_params=authored,
    )


def validate_stage_inputs(manifest: AnalysisManifest, job: StageJob) -> list[str]:
    """Return backward-compatible concise diagnostics."""
    return [issue.message for issue in STAGE_PARAM_CONTRACT.evaluate(build_validation_context(manifest, job))]
