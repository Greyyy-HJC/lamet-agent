"""Unit tests for core tool helpers."""

from __future__ import annotations

from pathlib import Path

from lamet_agent.core.tools import prepare_tool_args, resolve_plot_save_path
from lamet_agent.manifest import AnalysisManifest


def test_resolve_plot_save_path_strips_suffix_and_uses_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    resolved = resolve_plot_save_path(
        "/elsewhere/plots/fit_on_data.png",
        artifacts_dir=artifacts,
    )
    assert resolved == str(artifacts / "fit_on_data")


def test_resolve_plot_save_path_default_stem(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    resolved = resolve_plot_save_path(None, artifacts_dir=artifacts)
    assert resolved == str(artifacts / "fit_on_data")


def test_prepare_tool_args_merges_fourier_manifest_options(tmp_path: Path) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "fourier",
            "metadata": {
                "fourier_input": "matrix_element.h5",
                "fourier": {
                    "input_format": "h5",
                    "h5_group": "Pz=6",
                    "coord_unit": "fm",
                    "pz_gev": 2.43,
                    "method": "GI",
                    "order": "Empirical",
                    "observable": "nucleon_quark_transversity_quasi_pdf",
                    "k_grid": {"start": -2.0, "stop": 2.0, "num": 401},
                    "plot_fourier": {"save_path": "ft.pdf", "title": "FT"},
                    "plot_extension": {"scheme_index": 2, "save_path": "ext_re.pdf"},
                },
            },
        }
    )

    load_args = prepare_tool_args(
        "load_renormalized_matrix_element_samples",
        {},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert load_args["path"] == "matrix_element.h5"
    assert load_args["input_format"] == "h5"
    assert load_args["h5_group"] == "Pz=6"

    run_args = prepare_tool_args(
        "run_fourier_transform",
        {"method": "CG"},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert run_args["method"] == "GI"
    assert run_args["order"] == "Empirical"
    assert run_args["k_grid"]["num"] == 401

    plot_args = prepare_tool_args(
        "plot_fourier_result",
        {},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert plot_args == {"save_path": "ft.pdf", "title": "FT"}

    extension_args = prepare_tool_args(
        "plot_fourier_extension_quality_result",
        {},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert extension_args == {"scheme_index": 2, "save_path": "ext_re.pdf"}
