from lamet_agent.core.prompting import build_stage_static_prompt, format_tool_observation
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


def test_build_stage_static_prompt_excludes_observations() -> None:
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
    static = build_stage_static_prompt(
        "correlator_analysis",
        manifest,
        completed_stages=[],
    )
    assert "Tool results so far" not in static
    assert "inspect_correlator_scale" in static


def test_build_stage_static_prompt_includes_metadata() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "metadata": {"fourier_input": "matrix_element.npz"},
        }
    )
    static = build_stage_static_prompt(
        "fourier_transform",
        manifest,
        completed_stages=[],
    )
    assert "matrix_element.npz" in static
    assert "load_renormalized_matrix_element_samples" in static


def test_build_stage_static_prompt_filters_non_stage_metadata() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "correlators": [
                {"dataset_id": "c2", "kind": "2pt", "path": "fake/c2.txt"}
            ],
            "metadata": {
                "correlator_grid": {"pt2_path": "fake/c2.txt"},
                "renormalization": {"denominator_netcdf_path": "p0.nc"},
                "fourier_input": "matrix_element.npz",
                "matching": {"kernel_id": "unpolarized_gT"},
                "note": "workflow note",
            },
        }
    )
    static = build_stage_static_prompt(
        "renormalization",
        manifest,
        completed_stages=["correlator_analysis"],
    )
    assert "p0.nc" in static
    assert "matrix_element.npz" not in static
    assert "unpolarized_gT" not in static
    assert "fake/c2.txt" not in static
    assert '"dataset_id": "c2"' in static


def test_format_tool_observation_omits_ignored_args_for_llm() -> None:
    observation = {
        "tool_name": "plot_matched_pdf",
        "result": {"path": "artifacts/matched_pdf.pdf"},
        "ignored_args": {"correlator_grid": {"pt2_path": "large/path.h5"}},
    }
    text = format_tool_observation(observation)
    assert "plot_matched_pdf" in text
    assert "matched_pdf.pdf" in text
    assert "ignored_args" not in text
    assert "large/path.h5" not in text
    assert "ignored_args" in observation
