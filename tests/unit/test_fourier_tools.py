from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.tools import resolve_stage_tools
from lamet_agent.core.plotting import _band_segment, plot_fourier_extension_quality, plot_fourier_npz
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.stages.fourier.functions import (
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


def test_fourier_band_segment_inserts_exact_range_edges() -> None:
    x, mean, sdev = _band_segment(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 10.0, 20.0, 30.0]),
        np.ones(4),
        start=1.25,
        stop=2.75,
    )

    assert np.isclose(x[0], 1.25)
    assert np.isclose(x[-1], 2.75)
    assert np.isclose(mean[0], 12.5)
    assert np.isclose(mean[-1], 27.5)


def test_fourier_stage_tools_are_registered() -> None:
    tools = resolve_stage_tools("fourier_transform")
    assert "load_renormalized_matrix_element_samples" in tools
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
    assert loaded["data"] == "matrix_element_data"
    assert loaded["resample_mode"] == "bootstrap"
    assert "matrix_element_data" in store

    run = run_fourier_transform(
        store,
        k_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
        Lambda0=0.3,
    )
    assert run["n_schemes"] == 1
    assert run["n_samples"] == 3
    assert Path(run["artifact"]).is_file()
    ft_data, ft_extra = EnsembleData.load_npz(run["artifact"])
    assert ft_data.dims == ["x"]
    assert ft_data.resample == "bootstrap"
    assert ft_data.values.shape == (3, 3)
    assert "ft_re_mean" in ft_extra
    assert Path(run["fit_info_artifact"]).is_file()
    fit_data, fit_extra = EnsembleData.load_npz(run["fit_info_artifact"])
    assert fit_data.dims == ["scheme", "parameter"]
    assert fit_data.resample == "bootstrap"
    assert fit_data.values.shape == (3, 1, 3)
    assert "fit_chi2" in fit_extra
    fit_info = np.load(run["fit_info_artifact"])
    assert fit_info["fit_param_labels"].tolist() == ["A2", "phi2", "Lambda"]
    assert fit_info["fit_params"].shape == (1, 3, 3)
    assert np.all(fit_info["fit_params"][:, :, 2] >= 0.3)
    assert fit_info["fit_param_center"].shape == (1, 3)
    assert fit_info["fit_param_sdev"].shape == (1, 3)
    assert fit_info["fit_chi2"].shape == (1, 3)
    assert fit_info["fit_q"].shape == (1, 3)

    summary = summarize_fourier_result(store)
    assert summary["out"] == "fourier_summary"
    assert len(summary["ft_re_mean"]) == 3
    assert summary["best_scheme_label"] == "zmin_1_zmax_4"
    assert summary["scheme_weights"] == [1.0]
    assert summary["fit_info_artifact"] == run["fit_info_artifact"]

    plot = plot_fourier_result(store)
    assert Path(plot["plot"]).is_file()

    extension_plot = plot_fourier_extension_quality_result(store)
    assert Path(extension_plot["plot_re"]).is_file()
    assert Path(extension_plot["plot_im"]).is_file()

    data = store["fourier_result"]
    fig, ax = plot_fourier_extension_quality(
        store["matrix_element"]["coord"],
        store["matrix_element"]["re_samples"],
        data,
        component="re",
    )
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Extension Endpoint" not in labels
    assert r"\mathrm{Re}\,\tilde{h}^R" in ax.get_ylabel()
    fig.clf()


def test_fourier_tool_chain_accepts_h5_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "rnmlzd_C0_h102_bootstrap_zs0.30_pz4.h5"
    _write_h5(data_path)
    store = {}

    loaded = load_renormalized_matrix_element_samples(store, path=str(data_path), input_format="h5")
    assert loaded["input_format"] == "h5"
    assert loaded["h5_group"] == "Pz=4"
    assert loaded["re_shape"] == [3, 5]

    run = run_fourier_transform(
        store,
        k_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
    )

    assert run["n_schemes"] == 1
    assert run["n_samples"] == 3
    assert Path(run["artifact"]).is_file()
    assert Path(run["fit_info_artifact"]).is_file()


def test_fourier_tool_chain_preserves_jackknife_resampling(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}

    loaded = load_renormalized_matrix_element_samples(store, path=str(data_path), resample_mode="jk")
    assert loaded["resample_mode"] == "jackknife"
    assert store["matrix_element_data"].resample == "jackknife"

    run = run_fourier_transform(
        store,
        k_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
    )

    ft_data, _ft_extra = EnsembleData.load_npz(run["artifact"])
    fit_data, _fit_extra = EnsembleData.load_npz(run["fit_info_artifact"])
    assert store["fourier_result"]["resample_mode"] == "jackknife"
    assert store["fourier_result_data"].resample == "jackknife"
    assert ft_data.resample == "jackknife"
    assert fit_data.resample == "jackknife"


def test_fourier_loader_accepts_ensemble_data_npz(tmp_path: Path) -> None:
    coord = np.arange(0.0, 5.0)
    base_re = np.exp(-0.45 * coord)
    base_im = 0.1 * np.exp(-0.45 * coord)
    data = EnsembleData(
        ensemble=None,
        resample="jackknife",
        values=[
            base_re + 1j * base_im,
            1.01 * base_re + 0.98j * base_im,
            0.99 * base_re + 1.02j * base_im,
        ],
        dims=("z",),
        coords={"z": coord.tolist()},
        name="renormalized_matrix_element",
    )
    path = tmp_path / "matrix_element_ensemble.npz"
    data.save_npz(path)
    store = {}

    loaded = load_renormalized_matrix_element_samples(store, path=str(path), input_format="npz", resample_mode="bs")

    assert loaded["resample_mode"] == "jackknife"
    assert store["matrix_element_data"].resample == "jackknife"
    assert store["matrix_element"]["re_samples"].shape == (3, 5)
    assert store["matrix_element"]["im_samples"].shape == (3, 5)


def test_fourier_transform_accepts_upstream_ensemble_data(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    coord = np.arange(0.0, 5.0)
    base_re = np.exp(-0.45 * coord)
    base_im = 0.1 * np.exp(-0.45 * coord)
    store = {
        "matrix_element_data": EnsembleData(
            ensemble=None,
            resample="bootstrap",
            values=[
                base_re + 1j * base_im,
                1.01 * base_re + 0.98j * base_im,
                0.99 * base_re + 1.02j * base_im,
            ],
            dims=("z",),
            coords={"z": coord.tolist()},
            name="renormalized_matrix_element",
        )
    }

    run = run_fourier_transform(
        store,
        k_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
    )

    assert run["n_samples"] == 3
    assert "fourier_result_data" in store
    assert store["fourier_result_data"].dims == ["x"]


def test_fourier_tool_chain_passes_observable_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 16.0)
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
        scheme_scan={"zmin_values": [1.0], "zmax_values": [13.0], "z_ext_max": 15.0},
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
        scheme_scan={"zmin_values": [1.0], "zmax_values": [7.0], "z_ext_max": 8.0},
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
    load_renormalized_matrix_element_samples(store, path=str(data_path))

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


def test_fourier_auto_generates_scheme_scan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.linspace(0.0, 1.2, 13)
    base_re = np.exp(-1.5 * coord)
    base_im = 0.15 * np.exp(-1.2 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re, 1.02 * base_re])
    im_samples = np.vstack([base_im, 0.98 * base_im, 1.02 * base_im, 0.99 * base_im])
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid={"start": -0.5, "stop": 0.5, "num": 5},
        method="GI",
        order="LA",
        observable="nucleon_quark_transversity_quasi_pdf",
        coord_unit="fm",
        pz_gev=2.0,
    )

    auto = run["auto_scheme_scan"]
    assert auto["auto_generated"] is True
    assert len(auto["zmin_values"]) == 4
    assert len(auto["zmax_values"]) == 4
    assert auto["zmin_values"][0] > 0.0
    assert auto["zmax_values"] == pytest.approx([0.9, 1.0, 1.1, 1.2])
    assert auto["min_width"] > 0
    assert auto["z_ext_max"] == pytest.approx(1.2 + 8.0 / (5.067731237 * 2.0))
    assert auto["smooth"] == "linear"
    assert auto["y_range"] == [-2.0, 2.0]
    assert auto["roughness_weight"] == 1.0
    assert auto["model_average"] is True
    assert run["n_schemes"] >= 4


def test_fourier_auto_completes_partial_scheme_scan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.linspace(0.0, 1.2, 13)
    base_re = np.exp(-1.5 * coord)
    base_im = 0.15 * np.exp(-1.2 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re, 1.02 * base_re])
    im_samples = np.vstack([base_im, 0.98 * base_im, 1.02 * base_im, 0.99 * base_im])
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid={"start": -0.5, "stop": 0.5, "num": 5},
        scheme_scan={"y_range": [-0.5, 0.5], "roughness_weight": 2.0},
        method="GI",
        order="LA",
        observable="nucleon_quark_transversity_quasi_pdf",
        coord_unit="fm",
        pz_gev=2.0,
    )

    auto = run["auto_scheme_scan"]
    assert auto["y_range"] == [-0.5, 0.5]
    assert auto["roughness_weight"] == 2.0
    assert len(auto["zmin_values"]) == 4
    assert len(auto["zmax_values"]) == 4
    assert "min_width" in auto
    assert "z_ext_max" in auto
    assert auto["smooth"] == "linear"


def test_fourier_auto_zmin_uses_tail_fit_stability(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.linspace(0.0, 1.6, 17)
    z_fit = coord * 5.067731237
    tail_re = 0.8 * np.exp(-0.65 * z_fit)
    tail_im = 0.18 * np.exp(-0.65 * z_fit)
    contaminated_re = tail_re.copy()
    contaminated_im = tail_im.copy()
    short = coord < 0.6
    contaminated_re[short] += 2.0 * (1.0 - coord[short] / 0.6) ** 2
    contaminated_im[short] -= 1.0 * (1.0 - coord[short] / 0.6)
    scales = np.array([0.98, 1.0, 1.02, 1.01])
    re_samples = scales[:, None] * contaminated_re[None, :]
    im_samples = scales[:, None] * contaminated_im[None, :]
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid={"start": -0.5, "stop": 0.5, "num": 5},
        method="GI",
        order="LA",
        observable="nucleon_quark_transversity_quasi_pdf",
        coord_unit="fm",
        pz_gev=2.0,
    )

    auto = run["auto_scheme_scan"]
    assert len(auto["zmin_values"]) == 4
    assert min(auto["zmin_values"]) >= 0.6 - 1e-12


def test_fourier_auto_zmax_stops_before_noisy_tail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.linspace(0.0, 1.2, 13)
    base_re = np.exp(-1.5 * coord)
    base_im = 0.15 * np.exp(-1.2 * coord)
    scales = np.array([0.98, 1.0, 1.02, 1.01])
    re_samples = scales[:, None] * base_re[None, :]
    im_samples = scales[:, None] * base_im[None, :]
    re_samples[:, -2:] += np.array([[-0.8, 0.9], [0.8, -0.9], [-0.7, 0.85], [0.7, -0.85]])
    im_samples[:, -2:] += np.array([[0.6, -0.7], [-0.6, 0.7], [0.5, -0.65], [-0.5, 0.65]])
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid={"start": -0.5, "stop": 0.5, "num": 5},
        method="GI",
        order="LA",
        observable="nucleon_quark_transversity_quasi_pdf",
        coord_unit="fm",
        pz_gev=2.0,
    )

    assert max(run["auto_scheme_scan"]["zmax_values"]) < coord[-1]


def test_fourier_defaults_scheme_scoring_options_for_complete_scan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run_fourier_transform(
        store,
        k_grid={"start": -0.5, "stop": 0.5, "num": 5},
        scheme_scan={
            "zmin_values": [1.0],
            "zmax_values": [4.0],
            "min_width": 1.0,
            "z_ext_max": 5.0,
            "smooth": "linear",
        },
        method="GI",
        order="LA",
    )

    assert store["fourier_result"]["scheme_weights"] == [1.0]
    assert len(store["fourier_result"]["scheme_roughness"]) == 1


def test_fourier_accepts_compact_k_grid_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid={"start": -1.0, "stop": 1.0, "num": 21},
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
    )
    summary = summarize_fourier_result(store)

    assert run["n_k"] == 21
    assert len(summary["k_grid"]) == 21


def test_fourier_accepts_covariance_fit_error_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        k_grid={"start": -0.5, "stop": 0.5, "num": 5},
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
        fit_error_mode="covariance",
    )

    assert run["n_schemes"] == 1
    assert store["fourier_result"]["fit_error_mode"] == "covariance"


def test_fourier_stage_validation_accepts_declared_matrix_element() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "metadata": {
                "fourier_input": "matrix_element.npz",
                "fourier": {
                    "method": "GI",
                    "order": "NLA",
                    "observable": "nucleon_quark_transversity_quasi_pdf",
                    "coord_unit": "lambda",
                    "k_grid": {"start": -2.0, "stop": 2.0, "num": 401},
                    "scheme_scan": {
                        "zmin_values": [1.0],
                        "zmax_values": [4.0],
                        "z_ext_max": 5.0,
                    },
                },
            },
        }
    )
    assert validate_stage_inputs(manifest) == []


def test_fourier_stage_validation_allows_auto_scheme_scan() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "metadata": {
                "fourier_input": "matrix_element.npz",
                "fourier": {
                    "method": "GI",
                    "order": "Empirical",
                    "observable": "nucleon_quark_transversity_quasi_pdf",
                    "coord_unit": "fm",
                    "pz_gev": 2.0,
                    "k_grid": {"start": -2.0, "stop": 2.0, "num": 401},
                },
            },
        }
    )
    assert validate_stage_inputs(manifest) == []


def test_fourier_stage_validation_explains_missing_options() -> None:
    manifest = AnalysisManifest.model_validate(
        {
            "run_id": "demo",
            "metadata": {"fourier_input": "matrix_element.npz"},
        }
    )

    issues = validate_stage_inputs(manifest)
    text = "\n".join(issues)

    assert "Fourier stage needs more metadata" in text
    assert "Missing metadata.fourier.observable/order" in text
    assert "pion_quark_quasi_pdf" in text
    assert "2601.12189 2.1/2.2" in text
    assert "Empirical uses arXiv:2208.08008 Eq. (6)" in text
    assert "Missing metadata.fourier.coord_unit" in text
    assert "Missing metadata.fourier.k_grid" in text
    assert "metadata.fourier.scheme_scan" not in text
    assert "sample_axis" not in text


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
