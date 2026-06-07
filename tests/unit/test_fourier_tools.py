from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lamet_agent.core.tools import resolve_stage_tools
from lamet_agent.core.plotting import plot_fourier_npz
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.stages.fourier.functions import (
    load_matrix_element_samples,
    load_renormalized_matrix_element_samples,
    plot_fourier_extension_quality_result,
    plot_fourier_result,
    run_fourier_transform,
    summarize_fourier_result,
)
from lamet_agent.stages.fourier.skills import validate_stage_inputs


def _write_npz(path: Path) -> None:
    coord = np.arange(0.0, 5.0)
    base_re = np.exp(-0.45 * coord)
    base_im = 0.1 * np.exp(-0.45 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re])
    im_samples = np.vstack([base_im, 0.98 * base_im, 1.02 * base_im])
    np.savez(path, coord=coord, re_samples=re_samples, im_samples=im_samples)


def _write_h5(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    coord = np.arange(0.0, 5.0)
    base_re = np.exp(-0.45 * coord)
    base_im = 0.1 * np.exp(-0.45 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re]).T
    im_samples = np.vstack([base_im, 0.98 * base_im, 1.02 * base_im]).T
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("Pz=4")
        group.create_dataset("z_ary", data=coord)
        group.create_dataset("Re", data=re_samples)
        group.create_dataset("Im", data=im_samples)


def test_fourier_stage_tools_are_registered() -> None:
    tools = resolve_stage_tools("fourier_transform")
    assert "load_renormalized_matrix_element_samples" in tools
    assert "load_matrix_element_samples" in tools
    assert "run_fourier_transform" in tools
    assert "summarize_fourier_result" in tools
    assert "plot_fourier_result" in tools
    assert "plot_fourier_extension_quality_result" in tools


def test_fourier_tool_chain_writes_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}

    loaded = load_renormalized_matrix_element_samples(store, path=str(data_path))
    assert loaded["out"] == "matrix_element"

    run = run_fourier_transform(
        store,
        k_grid=[-0.5, 0.0, 0.5],
        schemes=[{"label": "scan_a", "zmin": 1.0, "zmax": 4.0, "z_ext_max": 5.0}],
        method="GI",
        order="LA",
    )
    assert run["n_schemes"] == 1
    assert run["n_samples"] == 3
    assert Path(run["artifact"]).is_file()
    assert Path(run["fit_info_artifact"]).is_file()
    fit_info = np.load(run["fit_info_artifact"])
    assert fit_info["fit_param_labels"].tolist() == ["A2", "phi2", "Lambda"]
    assert fit_info["fit_params"].shape == (1, 3, 3)
    assert fit_info["fit_param_center"].shape == (1, 3)
    assert fit_info["fit_param_sdev"].shape == (1, 3)
    assert fit_info["fit_chi2"].shape == (1, 3)
    assert fit_info["fit_q"].shape == (1, 3)

    summary = summarize_fourier_result(store)
    assert summary["out"] == "fourier_summary"
    assert len(summary["ft_re_mean"]) == 3
    assert summary["best_scheme_label"] == "scan_a"
    assert summary["scheme_weights"] == [1.0]
    assert summary["fit_info_artifact"] == run["fit_info_artifact"]

    plot = plot_fourier_result(store)
    assert Path(plot["plot"]).is_file()

    extension_plot = plot_fourier_extension_quality_result(store)
    assert Path(extension_plot["plot_re"]).is_file()
    assert Path(extension_plot["plot_im"]).is_file()


def test_fourier_tool_chain_accepts_h5_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "rnmlzd_C0_h102_bootstrap_zs0.30_pz4.h5"
    _write_h5(data_path)
    store = {}

    loaded = load_renormalized_matrix_element_samples(store, path=str(data_path), input_format="h5")
    assert loaded["input_format"] == "h5"
    assert loaded["h5_group"] == "Pz=4"
    assert loaded["re_shape"] == [5, 3]

    run = run_fourier_transform(
        store,
        k_grid=[-0.5, 0.0, 0.5],
        schemes=[{"label": "scan_a", "zmin": 1.0, "zmax": 4.0, "z_ext_max": 5.0}],
        method="GI",
        order="LA",
        sample_axis=1,
    )

    assert run["n_schemes"] == 1
    assert run["n_samples"] == 3
    assert Path(run["artifact"]).is_file()
    assert Path(run["fit_info_artifact"]).is_file()


def test_fourier_tool_chain_passes_observable_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 13.0)
    base_re = np.exp(-0.25 * coord)
    base_im = 0.1 * np.exp(-0.25 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re])
    im_samples = np.vstack([base_im, 0.98 * base_im, 1.02 * base_im])
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid=[0.0],
        schemes=[{"label": "pion", "zmin": 1.0, "zmax": 11.0, "z_ext_max": 12.0}],
        method="GI",
        order="NLA",
        observable="pion_quark_quasi_pdf",
    )

    fit_info = np.load(run["fit_info_artifact"])
    assert fit_info["fit_param_labels"].tolist() == [
        "A2",
        "phi2",
        "A1",
        "phi1",
        "A3",
        "phi3",
        "A2p",
        "phi2p",
        "A1p",
        "phi1p",
        "A3p",
        "phi3p",
        "Lambda",
    ]
    assert fit_info["fit_params"].shape == (1, 3, 13)


def test_fourier_tool_chain_accepts_empirical_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 9.0)
    base_re = np.exp(-0.25 * coord)
    base_im = 0.1 * np.exp(-0.25 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re])
    im_samples = np.vstack([base_im, 0.98 * base_im, 1.02 * base_im])
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid=[0.0],
        schemes=[{"label": "empirical", "zmin": 1.0, "zmax": 7.0, "z_ext_max": 8.0}],
        method="GI",
        order="Empirical",
    )

    fit_info = np.load(run["fit_info_artifact"])
    assert fit_info["fit_param_labels"].tolist() == ["c1", "c2", "a", "b", "lambda0"]
    assert fit_info["fit_params"].shape == (1, 3, 5)


def test_fourier_scheme_scan_scores_and_model_averages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid=[-0.6, -0.3, 0.0, 0.3, 0.6],
        scheme_scan={
            "zmin_values": [1.0, 2.0],
            "zmax_values": [3.0, 4.0],
            "min_width": 2.0,
            "z_ext_max": 5.0,
            "smooth": "linear",
            "y_range": [-0.6, 0.6],
            "roughness_weight": 2.0,
        },
        method="GI",
        order="LA",
    )
    summary = summarize_fourier_result(store)

    assert run["n_schemes"] == 3
    assert summary["best_scheme_index"] in {0, 1, 2}
    assert len(summary["scheme_weights"]) == 3
    assert np.isclose(sum(summary["scheme_weights"]), 1.0)
    assert len(summary["scheme_fit_chi2_dof"]) == 3
    assert len(summary["scheme_roughness"]) == 3


def test_fourier_accepts_compact_k_grid_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid={"start": -1.0, "stop": 1.0, "num": 21},
        schemes=[{"label": "scan_a", "zmin": 1.0, "zmax": 4.0, "z_ext_max": 5.0}],
        method="GI",
        order="LA",
    )
    summary = summarize_fourier_result(store)

    assert run["n_k"] == 21
    assert len(summary["k_grid"]) == 21


def test_fourier_stage_validation_accepts_declared_matrix_element() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "metadata": {"fourier_input": "matrix_element.npz"},
        }
    )
    assert validate_stage_inputs(manifest) == []


def test_fourier_stage_validation_flags_missing_matrix_element() -> None:
    manifest = AnalysisManifest.model_validate({"run_id": "demo"})
    assert validate_stage_inputs(manifest)


def test_plot_fourier_npz_writes_figure(tmp_path: Path) -> None:
    path = tmp_path / "fourier_result.npz"
    save_path = tmp_path / "fourier.pdf"
    np.savez(
        path,
        k_grid=np.array([-0.5, 0.0, 0.5]),
        ft_re_mean=np.array([0.2, 0.3, 0.2]),
        ft_im_mean=np.array([-0.1, 0.0, 0.1]),
        ft_re_stat_sdev=np.array([0.01, 0.02, 0.01]),
        ft_im_stat_sdev=np.array([0.02, 0.01, 0.02]),
        ft_re_sys_sdev=np.array([0.005, 0.005, 0.005]),
        ft_im_sys_sdev=np.array([0.005, 0.005, 0.005]),
        observable=np.asarray("nucleon_quark_transversity_quasi_pdf"),
    )

    fig, (ax_re, _ax_im) = plot_fourier_npz(path, save_path=save_path)

    assert save_path.is_file()
    assert ax_re.get_title() == "FT nucleon quark transversity quasi pdf"
    fig.clf()
