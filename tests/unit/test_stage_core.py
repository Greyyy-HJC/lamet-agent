from lamet_agent.core.prompting import build_stage_prompt
from lamet_agent.core.stages import select_stage_sequence
from lamet_agent.manifest import AnalysisManifest


def test_select_stage_sequence_keeps_five_stage_pipeline() -> None:
    stages = select_stage_sequence("full_lamet_pipeline")
    assert stages == [
        "correlator_analysis",
        "renormalization",
        "fourier_transform",
        "perturbative_matching",
        "extrapolation",
    ]


def test_build_stage_prompt_uses_stage_package_instruction() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "goal": "full_lamet_pipeline",
            "correlators": [{"dataset_id": "c2", "kind": "2pt", "path": "fake/c2.txt"}],
            "kernels": [
                {
                    "kernel_id": "k1",
                    "function": "lamet_agent.kernels:identity_kernel",
                }
            ],
        }
    )
    prompt = build_stage_prompt(
        "renormalization",
        manifest,
        completed_stages=["correlator_analysis"],
    )
    assert "Apply user-selected renormalization setup deterministically." in prompt
