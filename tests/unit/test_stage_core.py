import pytest

from lamet_agent.core.prompting import (
    build_stage_static_prompt,
    format_tool_observation,
    get_stage_instruction,
)
from lamet_agent.manifest import AnalysisManifest


def _manifest() -> AnalysisManifest:
    return AnalysisManifest.model_validate({
        "metadata": {
            "run_id": "demo", "root_directory": ".", "target_observable": "pdf",
            "parton": "quark", "resample_mode": "jk", "sample_error_mode": "covariance", "random_seed": 1984, "stages": ["correlator_analysis"],
        },
        "inputs": {"correlators": [], "artifacts": [], "kernels": []},
        "stages": {"correlator_analysis": {"defaults": {"nstate": [2]}, "jobs": [{"id": "ca"}]}},
    })


def test_build_job_prompt_includes_job_and_effective_parameters() -> None:
    manifest = _manifest()
    job = manifest.stages["correlator_analysis"].jobs[0]
    static = build_stage_static_prompt(
        "correlator_analysis", manifest, job=job, effective_params={"nstate": [2]}, completed_stages=[]
    )
    assert "Current job: ca" in static
    assert '"nstate": [2]' in static
    assert "inspect_correlator_scale" in static
    assert "Manifest parameter contract:" in static
    assert "Additional states parameterize excited-state contamination" in static
    assert "Any repository change that adds, removes, renames, or changes a manifest parameter" in static


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("correlator_analysis", "inspect_correlator_scale"),
        ("renormalization", "apply_self_renormalization"),
        ("fourier_transform", "run_fourier_transform"),
        ("perturbative_matching", "build_matching_kernel"),
        ("extrapolation", "run_systematics_budget"),
        ("review", "Generate one LLM-written scientific review"),
    ],
)
def test_stage_instructions_load_from_markdown(stage: str, expected: str) -> None:
    instruction = get_stage_instruction(stage)
    assert expected in instruction
    assert instruction.startswith("# ")
    assert "## Basic Procedure" in instruction
    assert "## Stage Skill" in instruction
    assert "## Available Tools" in instruction


def test_format_tool_observation_omits_ignored_args_for_llm() -> None:
    observation = {"tool_name": "tool", "result": {"ok": True}, "ignored_args": {"large": "payload"}}
    text = format_tool_observation(observation)
    assert "ignored_args" not in text
    assert "payload" not in text
