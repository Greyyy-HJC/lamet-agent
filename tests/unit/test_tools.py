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
                    "resample_mode": "jk",
                    "coord_unit": "fm",
                    "pz_gev": 2.43,
                    "method": "GI",
                    "order": "NLA",
                    "observable": "nucleon_quark_transversity_quasi_pdf",
                    "Lambda0": 0.2,
                    "save_path": "ft_result.npz",
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
    assert load_args["resample_mode"] == "jk"

    run_args = prepare_tool_args(
        "run_fourier_transform",
        {"method": "CG"},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert run_args["method"] == "GI"
    assert run_args["order"] == "NLA"
    assert run_args["Lambda0"] == 0.2
    assert run_args["save_path"] == "ft_result.npz"
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


def test_prepare_tool_args_fills_correlator_grid_defaults(tmp_path: Path) -> None:
    manifest = AnalysisManifest(
        run_id="workflow_cg_qpdf_p5",
        correlators=[],
        kernels=[],
        metadata={
            "correlator_grid": {
                "pt2_path": "data/pt2.h5",
                "pt3_paths": {"8": "data/ts8.h5", "10": "/abs/ts10.h5"},
                "tsep_ls": [8, 10],
                "z_values": [0, 1],
                "ensemble": "HISQa060_X",
                "tag": "CG52bxp30_CG52bxp30",
                "source_sink": "SS",
                "pt2_gamma": "5",
                "pt3_gamma": "T",
                "momentum": "PX5PY0PZ0",
                "pt2_windows": [{"tmin": 2, "tmax": 12}],
                "pt3_tau_cuts": [2, 3],
                "fit_strategy": "joint",
                "resample_mode": "jk",
            }
        },
        manifest_dir=tmp_path / "examples",
        project_root=tmp_path,
    )

    ground_args = prepare_tool_args(
        "tune_ground_state",
        {"correlator_rescale": 1e20},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert ground_args["pt2_path"] == str(tmp_path / "data" / "pt2.h5")
    assert ground_args["source_sink"] == "SS"
    assert ground_args["gamma"] == "5"
    assert ground_args["momentum"] == "PX5PY0PZ0"
    assert ground_args["pt2_windows"] == [{"tmin": 2, "tmax": 12}]

    bare_args = prepare_tool_args(
        "tune_bare_matrix",
        {"correlator_rescale": 1e20},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert bare_args["pt2_path"] == str(tmp_path / "data" / "pt2.h5")
    assert bare_args["pt3_paths"]["8"] == str(tmp_path / "data" / "ts8.h5")
    assert bare_args["pt3_paths"]["10"] == "/abs/ts10.h5"
    assert bare_args["momentum"] == "PX5PY0PZ0"
    assert bare_args["pt2_gamma"] == "5"
    assert bare_args["pt3_gamma"] == "T"
    assert bare_args["tsep_ls"] == [8, 10]
    assert bare_args["pt3_tau_cuts"] == [2, 3]


def test_prepare_tool_args_merges_renormalization_manifest_options(tmp_path: Path) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "renorm",
            "metadata": {
                "renormalization": {
                    "denominator_report_json": "p0_report.json",
                    "zs": 4,
                    "delta_m": 0.0,
                    "m0": 0.0,
                    "save_path": "renorm_npz",
                    "plot": {"save_path": "renorm_plot.pdf", "title": "Renorm"},
                }
            },
        }
    )

    load_target = prepare_tool_args(
        "load_bare_matrix_element_grid",
        {},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={"bare_matrix_grid_report": {"outputs": []}},
    )
    assert load_target["out"] == "target_bare_matrix_element"
    assert "report_json" not in load_target

    load_denom = prepare_tool_args(
        "load_bare_matrix_element_grid",
        {"out": "denominator_bare_matrix_element"},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={"target_bare_matrix_element": object()},
    )
    assert load_denom["report_json"] == "p0_report.json"

    apply_args = prepare_tool_args(
        "apply_ratio_scheme_renormalization",
        {},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert apply_args["zs"] == 4
    assert apply_args["save_path"] == str(tmp_path / "artifacts" / "renorm_npz")
    assert apply_args["artifacts_dir"] == str(tmp_path / "artifacts")

    plot_args = prepare_tool_args(
        "plot_renormalized_matrix_element",
        {},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert plot_args["save_path"] == str(tmp_path / "artifacts" / "renorm_plot")
    assert plot_args["title"] == "Renorm"


def test_prepare_tool_args_merges_matching_plot_options(tmp_path: Path) -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "matching",
            "metadata": {
                "matching": {
                    "plot": {
                        "save_path": "matched_pdf.pdf",
                        "xlim": [-1.5, 1.5],
                        "ylim": [-0.2, 2.0],
                    }
                }
            },
        }
    )

    plot_args = prepare_tool_args(
        "plot_matched_pdf",
        {},
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )

    assert plot_args["save_path"] == str(tmp_path / "artifacts" / "matched_pdf")
    assert plot_args["artifacts_dir"] == str(tmp_path / "artifacts")
    assert plot_args["xlim"] == [-1.5, 1.5]
    assert plot_args["ylim"] == [-0.2, 2.0]
