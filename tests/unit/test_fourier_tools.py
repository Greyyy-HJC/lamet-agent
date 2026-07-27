from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.tools import resolve_stage_tools
from lamet_agent.core.plotting import _band_segment, plot_fourier_artifact, plot_fourier_extension_quality
from lamet_agent.stages.fourier import functions as fourier_functions
from lamet_agent.stages.fourier.functions import (
    _asymptotic_values,
    _param_labels,
    _param_template,
    complete_z_negative,
    load_renormalized_matrix_element_samples,
    plot_fourier_extension_quality_result,
    plot_fourier_result,
    report_fourier_result,
    run_fourier_workflow,
    run_fourier_transform,
    summarize_fourier_result,
    sum_ft_re_im,
)


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


def test_complete_z_negative_preserves_shortest_negative_point_without_zero() -> None:
    lam, _, _ = complete_z_negative([1.0, 2.0], [10.0, 20.0], [1.0, 2.0])

    assert lam.tolist() == [-2.0, -1.0, 1.0, 2.0]


def test_sum_ft_re_im_uses_unshifted_fourier_phase() -> None:
    lam = np.array([0.0, 1.0, 2.0])
    re = np.array([1.0, 2.0, 3.0])
    im = np.array([0.2, 0.3, 0.4])
    x_grid = np.array([0.0, 0.5, 1.0])
    pref = (lam[1] - lam[0]) / (2 * np.pi)
    phase = np.multiply.outer(lam, x_grid)

    got_re, got_im = sum_ft_re_im(lam, re, im, x_grid)

    expected_re = pref * np.sum(np.cos(phase) * re[:, None], axis=0) - pref * np.sum(
        np.sin(phase) * im[:, None], axis=0
    )
    expected_im = pref * np.sum(np.sin(phase) * re[:, None], axis=0) + pref * np.sum(
        np.cos(phase) * im[:, None], axis=0
    )
    assert np.allclose(got_re, expected_re)
    assert np.allclose(got_im, expected_im)


@pytest.mark.parametrize(
    ("coord_unit", "momentum_gev", "lattice_spacing_fm", "scale"),
    [
        ("lambda", 2.0, None, 1.0),
        ("gev_inv", 2.0, None, 2.0),
        ("fm", 2.0, None, 2.0 * fourier_functions.FM_TO_GEV_INV),
        ("lattice", 2.0, 0.1, 0.2 * fourier_functions.FM_TO_GEV_INV),
    ],
)
def test_da_symmetry_projection_runs_before_range_selection(
    monkeypatch, coord_unit: str, momentum_gev: float, lattice_spacing_fm: float | None, scale: float
) -> None:
    coord = np.array([0.0, 1.0, 2.0])
    values = np.array([[1.0 + 0.2j, 0.8 + 0.1j, 0.6 - 0.1j], [1.1 + 0.1j, 0.7 + 0.2j, 0.5 - 0.2j]])
    store = {
        "input": EnsembleData(
            ensemble=None,
            resample="bootstrap",
            values=list(values),
            dims=("z",),
            coords={"z": coord.tolist()},
        )
    }
    captured: dict[str, np.ndarray] = {}

    def capture_scan(**kwargs):
        captured["values"] = kwargs["re_samples"] + 1j * kwargs["im_samples"]
        raise RuntimeError("captured projected matrix element")

    monkeypatch.setattr(fourier_functions, "_auto_scheme_scan", capture_scan)
    with pytest.raises(RuntimeError, match="captured projected matrix element"):
        run_fourier_transform(
            store,
            y_grid=[0.0],
            target_observable="da",
            sector="full",
            coord_unit=coord_unit,
            momentum_gev=momentum_gev,
            lattice_spacing_fm=lattice_spacing_fm,
        )

    phase = np.exp(0.5j * coord * scale)[None, :]
    expected = np.real(values * phase) * np.conjugate(phase)
    assert np.allclose(captured["values"], expected)
    assert np.allclose(np.imag(captured["values"] * phase), 0.0)


@pytest.mark.parametrize(("target", "symmetry_guarantee"), [("da", False), ("pdf", True)])
def test_da_symmetry_projection_is_optional_and_da_only(
    monkeypatch, target: str, symmetry_guarantee: bool
) -> None:
    coord = np.array([0.0, 1.0, 2.0])
    values = np.array([[1.0 + 0.2j, 0.8 + 0.1j, 0.6 - 0.1j]])
    store = {
        "input": EnsembleData(
            ensemble=None,
            resample="bootstrap",
            values=list(values),
            dims=("z",),
            coords={"z": coord.tolist()},
        )
    }
    captured: dict[str, np.ndarray] = {}

    def capture_scan(**kwargs):
        captured["values"] = kwargs["re_samples"] + 1j * kwargs["im_samples"]
        raise RuntimeError("captured unchanged matrix element")

    monkeypatch.setattr(fourier_functions, "_auto_scheme_scan", capture_scan)
    with pytest.raises(RuntimeError, match="captured unchanged matrix element"):
        run_fourier_transform(
            store,
            y_grid=[0.0],
            target_observable=target,
            sector="full",
            coord_unit="lambda",
            momentum_gev=2.0,
            symmetry_guarantee=symmetry_guarantee,
        )

    assert np.allclose(captured["values"], values)


def test_sum_ft_re_im_uses_quadrature_weights_for_nonuniform_grid() -> None:
    lam = np.array([0.0, 0.8, 1.5, 3.0])
    re = np.array([1.0, 2.0, 1.5, 0.5])
    im = np.array([0.2, 0.1, -0.1, -0.2])
    x_grid = np.array([0.0, 0.4])
    weights = np.array([0.4, 0.75, 1.1, 0.75])
    phase = np.multiply.outer(lam, x_grid)

    got_re, got_im = sum_ft_re_im(lam, re, im, x_grid)

    pref = weights[:, None] / (2 * np.pi)
    expected_re = np.sum(pref * np.cos(phase) * re[:, None], axis=0) - np.sum(
        pref * np.sin(phase) * im[:, None], axis=0
    )
    expected_im = np.sum(pref * np.sin(phase) * re[:, None], axis=0) + np.sum(
        pref * np.cos(phase) * im[:, None], axis=0
    )
    assert np.allclose(got_re, expected_re)
    assert np.allclose(got_im, expected_im)


def test_fourier_workflow_omits_missing_short_distance_grid() -> None:
    coord = np.arange(2.0, 8.0)
    base = np.exp(-0.2 * coord) * np.cos(0.3 * coord)
    re_samples = np.vstack([base * (1.0 + 0.001 * idx) for idx in range(8)])
    im_samples = np.zeros_like(re_samples)

    result = run_fourier_workflow(
        coord,
        re_samples,
        im_samples,
        [-0.5, 0.0, 0.5],
        schemes=[{"label": "manual", "zmin": 2.0, "zmax": 4.0, "z_ext_max": 7.0, "smooth": "linear"}],
        method="GI",
        order="LA",
        observable="meson_quasi_da",
        coord_unit="lattice",
        momentum_gev=2.4,
        lattice_spacing_fm=0.04,
        resample_mode="jackknife",
        sample_error_mode="mean",
        part="re",
        sector="valence",
        hadron="pion",
        psi1_flavor_class="light",
        psi2_flavor_class="light",
    )

    assert result["short_distance_policy"] == "truncate_missing"
    assert result["missing_short_distance_coord"] == [0.0, 1.0]
    assert result["fourier_positive_coord_start"] == 2.0
    assert np.isclose(result["scheme_results"][0]["z_ext"][0], 2.0 * 0.04 / 0.197327)


def test_fourier_workflow_accepts_nonuniform_input_grid() -> None:
    coord = np.array([0.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    base = np.exp(-0.2 * coord) * np.cos(0.3 * coord)
    re_samples = np.vstack([base * (1.0 + 0.001 * idx) for idx in range(8)])
    im_samples = np.zeros_like(re_samples)

    result = run_fourier_workflow(
        coord,
        re_samples,
        im_samples,
        [-0.5, 0.0, 0.5],
        schemes=[{"label": "manual", "zmin": 2.0, "zmax": 5.0, "z_ext_max": 6.0, "smooth": "linear"}],
        method="GI",
        order="LA",
        observable="meson_quasi_da",
        coord_unit="lattice",
        momentum_gev=2.4,
        lattice_spacing_fm=0.04,
        resample_mode="jackknife",
        sample_error_mode="mean",
        part="re",
        sector="valence",
        hadron="pion",
        psi1_flavor_class="light",
        psi2_flavor_class="light",
    )

    assert result["input_coord_step"] == 1.0
    assert np.allclose(result["scheme_results"][0]["z_ext"], coord * 0.04 / 0.197327)
    assert result["ft_re_samples"].shape == (1, 8, 3)


def test_fourier_parallel_sample_fits_match_serial() -> None:
    coord = np.arange(2.0, 8.0)
    base = np.exp(-0.2 * coord) * np.cos(0.3 * coord)
    re_samples = np.vstack([base * (1.0 + 0.001 * idx) for idx in range(4)])
    im_samples = np.zeros_like(re_samples)
    kwargs = {
        "schemes": [{"label": "manual", "zmin": 2.0, "zmax": 4.0, "z_ext_max": 7.0, "smooth": "linear"}],
        "method": "GI",
        "order": "LA",
        "observable": "meson_quasi_da",
        "coord_unit": "lattice",
        "momentum_gev": 2.4,
        "lattice_spacing_fm": 0.04,
        "resample_mode": "jackknife",
        "sample_error_mode": "mean",
        "part": "re",
        "sector": "valence",
        "hadron": "pion",
        "psi1_flavor_class": "light",
        "psi2_flavor_class": "light",
    }

    serial = run_fourier_workflow(coord, re_samples, im_samples, [-0.5, 0.0, 0.5], workers=1, **kwargs)
    parallel = run_fourier_workflow(coord, re_samples, im_samples, [-0.5, 0.0, 0.5], workers=2, **kwargs)

    assert serial["workers"] == 1
    assert parallel["workers"] == 2
    for key in ("fit_params", "fit_chi2", "fit_q", "fit_log_gbf", "ft_re_samples", "ft_im_samples"):
        assert np.allclose(serial["scheme_results"][0][key], parallel["scheme_results"][0][key])


def test_fourier_stage_tools_are_registered() -> None:
    tools = resolve_stage_tools("fourier_transform")
    assert "load_renormalized_matrix_element_samples" in tools
    assert "run_fourier_transform" in tools
    assert "summarize_fourier_result" in tools
    assert "plot_fourier_result" in tools
    assert "plot_fourier_extension_quality_result" in tools
    assert "report_fourier_result" in tools


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
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
        Lambda0_gev=0.3,
        bz_direction="X",
        artifacts_dir=str(tmp_path / "artifacts"),
        workers=2,
    )
    assert run["n_schemes"] == 1
    assert run["n_samples"] == 3
    assert run["workers"] == 2
    assert store["fourier_result"]["symmetry_guarantee"] is False
    assert store["fourier_result"]["Lambda0_gev"] == pytest.approx(0.3)
    assert run["Lambda0_gev"] == pytest.approx(0.3)
    assert Path(run["artifact"]).is_file()
    assert Path(run["artifact"]).parent == tmp_path / "artifacts"
    assert Path(run["artifact"]).suffix == ".nc"
    assert Path(run["fit_info_artifact"]).suffix == ".nc"
    ft_data = EnsembleData.from_netcdf(run["artifact"])
    assert ft_data.dims == ["x"]
    assert ft_data.resample == "bootstrap"
    assert ft_data.attrs["workers"] == "2"
    assert ft_data.attrs["symmetry_guarantee"] == "False"
    assert "phase_shift" not in ft_data.attrs
    assert ft_data.attrs["Lambda0_gev"] == "0.3"
    assert ft_data.attrs["bz_direction"] == "X"
    assert ft_data.values.shape == (3, 3)
    assert "ft_re_mean" in ft_data.attrs
    assert Path(run["fit_info_artifact"]).is_file()
    assert Path(run["plot"]).is_file()
    assert Path(run["plot"]).with_suffix(".svg").is_file()
    assert Path(run["plot_re"]).is_file()
    assert Path(run["plot_re"]).with_suffix(".svg").is_file()
    assert Path(run["plot_im"]).is_file()
    assert Path(run["plot_im"]).with_suffix(".svg").is_file()
    assert run["report"] is None
    fit_data = EnsembleData.from_netcdf(run["fit_info_artifact"])
    assert fit_data.dims == ["scheme", "parameter"]
    assert fit_data.resample == "bootstrap"
    assert fit_data.attrs["Lambda0_gev"] == "0.3"
    assert fit_data.values.shape == (3, 1, 3)
    assert "fit_chi2" in fit_data.attrs
    assert fit_data.coords["parameter"] == ["A2", "phi2", "m"]

    summary = summarize_fourier_result(store)
    assert summary["out"] == "fourier_summary"
    assert len(summary["ft_re_mean"]) == 3
    assert summary["selected_range_label"] == "zmin_1_zmax_4"
    assert summary["fit_model_labels"] == ["LA_prior_3"]
    assert summary["fit_model_mean_weights"] == [1.0]
    assert summary["fit_info_artifact"] == run["fit_info_artifact"]
    assert summary["Lambda0_gev"] == pytest.approx(0.3)

    plot = plot_fourier_result(store)
    assert Path(plot["plot"]).is_file()

    extension_plot = plot_fourier_extension_quality_result(store)
    assert Path(extension_plot["plot_re"]).is_file()
    assert Path(extension_plot["plot_im"]).is_file()

    report = report_fourier_result(store)
    report_path = Path(report["report"])
    assert report_path.is_file()
    assert "report_cn" not in report
    assert not report_path.with_name("report_fourier_CN.md").exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "# Fourier Transform Analysis Report" in report_text
    assert "nucleon_quark_transversity_quasi_pdf" in report_text
    assert "GI" in report_text
    assert "LA" in report_text
    assert "Lambda0_gev" in report_text
    assert "Active fitted component" in report_text
    assert "fits $\\mathrm{Re}\\,\\tilde h^R$ and $\\mathrm{Im}\\,\\tilde h^R$ together" in report_text
    assert "Model Diagnostics" in report_text
    assert "q(x)=\\frac{\\Delta\\lambda}{2\\pi}" in report_text
    assert "![Fourier result]" in report_text
    assert "fourier_xdep.svg" in report_text
    assert "Reading the NetCDF Outputs" in report_text
    assert "fourier_result.nc" in report_text
    assert "fourier_fit_info.nc" in report_text
    assert Path(run["artifact"]).name in report_text
    assert Path(run["fit_info_artifact"]).name in report_text
    monkeypatch.setattr("lamet_agent.stages.fourier.reporting.translate_markdown_report", lambda markdown, **kwargs: markdown)
    report_cn = report_fourier_result(
        store,
        save_path=str(tmp_path / "report_fourier_ch.md"),
        report_language="ch",
    )
    report_cn_path = Path(report_cn["report"])
    assert report_cn_path.name == "report_fourier_ch_CN.md"
    assert report_cn_path.is_file()
    assert (tmp_path / "report_fourier_ch.md").exists()
    report_cn_text = report_cn_path.read_text(encoding="utf-8")
    assert "# Fourier Transform Analysis Report" in report_cn_text
    assert "Active fitted component" in report_cn_text
    assert "fits $\\mathrm{Re}\\,\\tilde h^R$ and $\\mathrm{Im}\\,\\tilde h^R$ together" in report_cn_text
    assert "Figures and Visual Assessment" in report_cn_text
    assert "Lambda0_gev" in report_cn_text
    assert "Reading the NetCDF Outputs" in report_cn_text
    assert "fourier_result.nc" in report_cn_text
    assert "fourier_fit_info.nc" in report_cn_text
    assert "fourier_xdep.svg" in report_cn_text
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


def test_fourier_shift_relabels_range_and_inherits_zs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))
    store["matrix_element_data"].array.attrs["zs_fm"] = "0.2"

    run = run_fourier_transform(
        store,
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        zmin_shift=1,
        method="GI",
        order="LA",
        coord_unit="lambda",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    data = EnsembleData.from_netcdf(run["artifact"])
    assert data.attrs["selected_range_label"] == "zmin_2_zmax_4"
    assert data.attrs["zs_fm"] == "0.2"


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
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
    )

    assert run["n_schemes"] == 1
    assert run["n_samples"] == 3
    assert store["fourier_result"]["Lambda0_gev"] == 0.0
    assert Path(run["artifact"]).is_file()
    assert Path(run["fit_info_artifact"]).is_file()
    assert EnsembleData.from_netcdf(run["artifact"]).attrs["Lambda0_gev"] == "0.0"
    assert EnsembleData.from_netcdf(run["fit_info_artifact"]).attrs["Lambda0_gev"] == "0.0"


def test_fourier_part_selects_active_fit_channel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    coord = np.arange(0.0, 7.0)
    base_re = np.exp(-0.35 * coord)
    base_im = 0.7 * np.exp(-0.30 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re])
    im_samples = np.vstack([base_im, 1.02 * base_im, 0.98 * base_im])
    data_path = tmp_path / "matrix_element.npz"
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)

    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))
    run_re = run_fourier_transform(
        store,
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [6.0], "z_ext_max": 7.0},
        method="GI",
        order="LA",
        observable="nucleon_quark_unpolarized_quasi_pdf",
        part="re",
    )
    result_re = store["fourier_result"]
    assert result_re["part"] == "re"
    assert np.allclose(result_re["ft_im_samples"], 0.0)
    assert np.allclose(result_re["scheme_results"][0]["extended_im_samples"], 0.0)
    artifact_re = EnsembleData.from_netcdf(run_re["artifact"])
    assert artifact_re.attrs["part"] == "re"
    fig_re, ax_re = plot_fourier_extension_quality(coord, re_samples, result_re, component="re")
    assert len(ax_re.collections) == 2
    assert len(ax_re.lines) == 4
    fig_re.clf()
    fig_im_inactive, ax_im_inactive = plot_fourier_extension_quality(coord, im_samples, result_re, component="im")
    assert len(ax_im_inactive.collections) == 1
    assert len(ax_im_inactive.lines) == 3
    fig_im_inactive.clf()

    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))
    run_im = run_fourier_transform(
        store,
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [6.0], "z_ext_max": 7.0},
        method="GI",
        order="LA",
        observable="nucleon_quark_unpolarized_quasi_pdf",
        part="im",
    )
    result_im = store["fourier_result"]
    assert result_im["part"] == "im"
    assert np.allclose(result_im["scheme_results"][0]["extended_re_samples"], 0.0)
    assert np.all(np.isfinite(result_im["ft_re_samples"]))
    assert np.all(np.isfinite(result_im["ft_im_samples"]))
    artifact_im = EnsembleData.from_netcdf(run_im["artifact"])
    assert artifact_im.attrs["part"] == "im"
    fig_re_inactive, ax_re_inactive = plot_fourier_extension_quality(coord, re_samples, result_im, component="re")
    assert len(ax_re_inactive.collections) == 1
    assert len(ax_re_inactive.lines) == 3
    fig_re_inactive.clf()
    fig_im, ax_im = plot_fourier_extension_quality(coord, im_samples, result_im, component="im")
    assert len(ax_im.collections) == 2
    assert len(ax_im.lines) == 4
    fig_im.clf()


def test_fourier_sector_valence_resolves_projection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
        sector="valence",
        target_observable="pdf",
    )

    result = store["fourier_result"]
    artifact = EnsembleData.from_netcdf(run["artifact"])
    assert result["sector"] == "valence"
    assert result["part"] == "re"
    assert result["output_scale"] == 2.0
    assert result["im_flip_for_ft"] is False
    assert artifact.attrs["sector"] == "valence"
    assert artifact.attrs["part"] == "re"


def test_fourier_gpd_sector_valence_resolves_projection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 12.0)
    base_re = np.exp(-0.35 * coord)
    base_im = 0.1 * np.exp(-0.35 * coord)
    np.savez(
        data_path,
        coord=coord,
        re_samples=np.vstack([base_re, 1.01 * base_re, 0.99 * base_re]),
        im_samples=np.vstack([base_im, 0.98 * base_im, 1.02 * base_im]),
    )
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [7.0], "z_ext_max": 8.0},
        method="GI",
        order="LA",
        sector="valence",
        target_observable="gpd",
        observable="nucleon_quark_quasi_gpd",
    )

    result = store["fourier_result"]
    artifact = EnsembleData.from_netcdf(run["artifact"])
    assert result["sector"] == "valence"
    assert result["part"] == "re"
    assert result["output_scale"] == 2.0
    assert result["im_flip_for_ft"] is False
    assert artifact.attrs["sector"] == "valence"
    assert artifact.attrs["part"] == "re"


def test_fourier_sector_sea_combines_total_and_valence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    common = dict(
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
        target_observable="pdf",
    )

    total_store = {}
    load_renormalized_matrix_element_samples(total_store, path=str(data_path))
    run_fourier_transform(total_store, sector="total", **common)

    valence_store = {}
    load_renormalized_matrix_element_samples(valence_store, path=str(data_path))
    run_fourier_transform(valence_store, sector="valence", **common)

    sea_store = {}
    load_renormalized_matrix_element_samples(sea_store, path=str(data_path))
    run_fourier_transform(sea_store, sector="sea", **common)

    total = total_store["fourier_result"]
    valence = valence_store["fourier_result"]
    sea = sea_store["fourier_result"]
    assert sea["sector"] == "sea"
    assert sea["part"] == "sea"
    assert np.allclose(sea["final_ft_re_samples"], 0.5 * (total["final_ft_re_samples"] - valence["final_ft_re_samples"]))
    assert np.allclose(sea["final_ft_im_samples"], 0.5 * (total["final_ft_im_samples"] - valence["final_ft_im_samples"]))


def test_fourier_gpd_sector_sea_combines_total_and_valence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 12.0)
    base_re = np.exp(-0.35 * coord)
    base_im = 0.1 * np.exp(-0.35 * coord)
    np.savez(
        data_path,
        coord=coord,
        re_samples=np.vstack([base_re, 1.01 * base_re, 0.99 * base_re]),
        im_samples=np.vstack([base_im, 0.98 * base_im, 1.02 * base_im]),
    )
    common = dict(
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [7.0], "z_ext_max": 8.0},
        method="GI",
        order="LA",
        target_observable="gpd",
        observable="nucleon_quark_quasi_gpd",
    )

    total_store = {}
    load_renormalized_matrix_element_samples(total_store, path=str(data_path))
    run_fourier_transform(total_store, sector="total", **common)

    valence_store = {}
    load_renormalized_matrix_element_samples(valence_store, path=str(data_path))
    run_fourier_transform(valence_store, sector="valence", **common)

    sea_store = {}
    load_renormalized_matrix_element_samples(sea_store, path=str(data_path))
    run_fourier_transform(sea_store, sector="sea", **common)

    total = total_store["fourier_result"]
    valence = valence_store["fourier_result"]
    sea = sea_store["fourier_result"]
    assert sea["sector"] == "sea"
    assert sea["part"] == "sea"
    assert sea["output_scale"] == 1.0
    assert np.allclose(sea["final_ft_re_samples"], 0.5 * (total["final_ft_re_samples"] - valence["final_ft_re_samples"]))
    assert np.allclose(sea["final_ft_im_samples"], 0.5 * (total["final_ft_im_samples"] - valence["final_ft_im_samples"]))


def test_fourier_output_scale_multiplies_fourier_space_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)

    base_store = {}
    load_renormalized_matrix_element_samples(base_store, path=str(data_path))
    run_fourier_transform(
        base_store,
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
        part="re",
        output_scale=1.0,
    )
    base = base_store["fourier_result"]
    base_artifact_values = np.asarray(base_store["fourier_result_data"].values)

    scaled_store = {}
    load_renormalized_matrix_element_samples(scaled_store, path=str(data_path))
    scaled_run = run_fourier_transform(
        scaled_store,
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
        part="re",
        output_scale=2.0,
    )
    scaled = scaled_store["fourier_result"]

    assert scaled["output_scale"] == 2.0
    assert scaled_run["output_scale"] == 2.0
    assert np.allclose(scaled["ft_re_samples"], 2.0 * base["ft_re_samples"])
    assert np.allclose(scaled["final_ft_re_samples"], 2.0 * base["final_ft_re_samples"])
    assert np.allclose(scaled["ft_re_mean"], 2.0 * base["ft_re_mean"])
    assert np.allclose(scaled["ft_re_stat_sdev"], 2.0 * base["ft_re_stat_sdev"])
    assert np.allclose(scaled["ft_re_sys_sdev"], 2.0 * base["ft_re_sys_sdev"])
    artifact = EnsembleData.from_netcdf(scaled_run["artifact"])
    assert np.allclose(np.real(artifact.values), 2.0 * np.real(base_artifact_values))
    assert float(json.loads(artifact.attrs["output_scale"])) == 2.0


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
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
    )

    ft_data = EnsembleData.from_netcdf(run["artifact"])
    fit_data = EnsembleData.from_netcdf(run["fit_info_artifact"])
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


def test_fourier_loader_accepts_ensemble_data_netcdf(tmp_path: Path) -> None:
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
    path = tmp_path / "matrix_element.nc"
    data.to_netcdf(path)
    store = {}

    loaded = load_renormalized_matrix_element_samples(store, path=str(path), input_format="nc")

    assert loaded["input_format"] == "nc"
    assert loaded["resample_mode"] == "jackknife"
    assert store["matrix_element_data"].resample == "jackknife"
    assert store["matrix_element"]["re_samples"].shape == (3, 5)


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
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
    )

    assert run["n_samples"] == 3
    assert "fourier_result_data" in store
    assert store["fourier_result_data"].dims == ["x"]
    assert store["output"] is store["fourier_result_data"]


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
        y_grid=[0.0],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [13.0], "z_ext_max": 15.0},
        method="GI",
        order="NLA",
        observable="pion_quark_quasi_pdf",
    )

    fit_info = EnsembleData.from_netcdf(run["fit_info_artifact"])
    assert json.loads(fit_info.attrs["fit_param_labels"]) == [
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
        "m",
    ]
    assert np.asarray(json.loads(fit_info.attrs["fit_params"])).shape == (1, 3, 13)


def test_fourier_pion_pdf_valence_tail_constraints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 16.0)
    base_re = np.exp(-0.25 * coord)
    base_im = 0.1 * np.exp(-0.25 * coord)
    np.savez(
        data_path,
        coord=coord,
        re_samples=np.vstack([base_re, 1.01 * base_re, 0.99 * base_re]),
        im_samples=np.vstack([base_im, 0.98 * base_im, 1.02 * base_im]),
    )
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid=[0.0],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [13.0], "z_ext_max": 15.0},
        method="GI",
        order="NLA",
        observable="pion_quark_quasi_pdf",
        sector="valence",
        target_observable="pdf",
    )

    fit_info = EnsembleData.from_netcdf(run["fit_info_artifact"])
    labels = json.loads(fit_info.attrs["fit_param_labels"])
    params = np.asarray(json.loads(fit_info.attrs["fit_params"]))[0]
    idx = {label: labels.index(label) for label in labels}
    assert np.allclose(params[:, idx["phi2"]], 0.0)
    assert np.allclose(params[:, idx["phi2p"]], 0.0)
    assert np.allclose(params[:, idx["A3"]], params[:, idx["A1"]])
    assert np.allclose(params[:, idx["A3p"]], params[:, idx["A1p"]])
    assert np.allclose(params[:, idx["phi3"]], -params[:, idx["phi1"]])
    assert np.allclose(params[:, idx["phi3p"]], -params[:, idx["phi1p"]])


def test_fourier_pion_gpd_valence_has_no_pdf_valence_tail_constraints() -> None:
    labels = _param_labels("GI", "NLA", "pion_quark_quasi_gpd", sector="valence")
    fit_labels = _param_labels("GI", "NLA", "pion_quark_quasi_gpd", sector="valence", fit=True)

    assert fit_labels == labels
    for label in ("A1", "A3", "A1p", "A3p", "phi2", "phi2p"):
        assert label in fit_labels


def test_fourier_meson_da_pion_tail_constraints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 12.0)
    base_re = np.exp(-0.3 * coord)
    base_im = 0.05 * np.exp(-0.3 * coord)
    np.savez(
        data_path,
        coord=coord,
        re_samples=np.vstack([base_re, 1.01 * base_re, 0.99 * base_re]),
        im_samples=np.vstack([base_im, 0.98 * base_im, 1.02 * base_im]),
    )
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))
    original_values = np.asarray(store["matrix_element_data"].values).copy()
    plotted_samples: dict[str, np.ndarray] = {}
    original_extension_plot = fourier_functions.plot_fourier_extension_quality

    def capture_extension_plot(coord_values, sample_values, result, **kwargs):
        plotted_samples[str(kwargs["component"])] = np.asarray(sample_values).copy()
        return original_extension_plot(coord_values, sample_values, result, **kwargs)

    monkeypatch.setattr(
        fourier_functions,
        "plot_fourier_extension_quality",
        capture_extension_plot,
    )

    run = run_fourier_transform(
        store,
        y_grid=[0.0],
        scheme_scan={"zmin_values": [1.0], "zmax_values": [10.0], "z_ext_max": 13.0},
        method="GI",
        order="NLA",
        observable="meson_quasi_da",
        target_observable="da",
        coord_unit="lambda",
        psi1_flavor_class="light",
        psi2_flavor_class="light",
    )

    projected_input = np.asarray(store["matrix_element_data"].values)
    phase = np.exp(0.5j * coord)[None, :]
    expected = np.real(original_values * phase) * np.conjugate(phase)
    assert np.allclose(projected_input, expected)
    assert np.allclose(np.imag(projected_input * phase), 0.0)
    assert np.allclose(plotted_samples["re"], np.real(expected))
    assert np.allclose(plotted_samples["im"], np.imag(expected))
    assert Path(run["plot_re"]).is_file()
    assert Path(run["plot_im"]).is_file()
    scheme_result = store["fourier_result"]["scheme_results"][0]
    assert float(np.max(scheme_result["lambda_ext"])) == pytest.approx(13.0)
    assert scheme_result["extended_re_samples"].shape[1] > original_values.shape[1]

    fit_info = EnsembleData.from_netcdf(run["fit_info_artifact"])
    labels = json.loads(fit_info.attrs["fit_param_labels"])
    params = np.asarray(json.loads(fit_info.attrs["fit_params"]))[0]
    idx = {label: labels.index(label) for label in labels}
    assert np.allclose(params[:, idx["A2"]], params[:, idx["A1"]])
    assert np.allclose(params[:, idx["A2p"]], params[:, idx["A1p"]])
    assert np.allclose(params[:, idx["phi2"]], -params[:, idx["phi1"]])
    assert np.allclose(params[:, idx["phi2p"]], -params[:, idx["phi1p"]])
    result_data = EnsembleData.from_netcdf(run["artifact"])
    assert result_data.attrs["psi1_flavor_class"] == "light"
    assert result_data.attrs["psi2_flavor_class"] == "light"
    assert result_data.attrs["symmetry_guarantee"] == "True"
    assert fit_info.attrs["psi1_flavor_class"] == "light"
    assert fit_info.attrs["psi2_flavor_class"] == "light"
    assert fit_info.attrs["symmetry_guarantee"] == "True"
    assert run["symmetry_guarantee"] is True
    report = report_fourier_result(store, save_path=str(tmp_path / "da_report.md"))
    report_text = Path(report["report"]).read_text(encoding="utf-8")
    assert "### Scope and Equivalence" in report_text
    assert "`symmetry_guarantee=true`" in report_text
    assert "discards $\\operatorname{Im}h_{+}$" in report_text
    assert "e^{-izP_z/2}" in report_text
    assert "phase_shift" not in report_text


def test_fourier_meson_da_flavor_classes_control_fit_labels() -> None:
    full = _param_labels("GI", "NLA", "meson_quasi_da")
    default_fit = _param_labels("GI", "NLA", "meson_quasi_da", fit=True)
    light_light_fit = _param_labels(
        "GI",
        "NLA",
        "meson_quasi_da",
        psi1_flavor_class="light",
        psi2_flavor_class="light",
        fit=True,
    )
    light_heavy_fit = _param_labels(
        "GI",
        "NLA",
        "meson_quasi_da",
        psi1_flavor_class="light",
        psi2_flavor_class="heavy",
        fit=True,
    )
    heavy_light_fit = _param_labels(
        "GI",
        "NLA",
        "meson_quasi_da",
        psi1_flavor_class="heavy",
        psi2_flavor_class="light",
        fit=True,
    )

    assert full == ["A1", "phi1", "A2", "phi2", "A1p", "phi1p", "A2p", "phi2p", "m"]
    assert default_fit == full
    assert light_light_fit == ["A1", "phi1", "A1p", "phi1p", "m"]
    assert light_heavy_fit == ["A2", "phi2", "A2p", "phi2p", "m"]
    assert heavy_light_fit == ["A1", "phi1", "A1p", "phi1p", "m"]


def test_fourier_tool_chain_accepts_gluon_observables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    coord = np.arange(0.0, 14.0)
    base_re = (coord + 0.2) * np.exp(-0.25 * coord)
    base_re[0] = base_re[1]
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re])
    im_samples = np.zeros_like(re_samples)

    cases = [
        ("nucleon_gluon_quasi_pdf", "LA", ["A", "m"]),
        ("nucleon_gluon_quasi_pdf", "NLA", ["A", "Ap", "m"]),
        ("pion_gluon_quasi_pdf", "LA", ["A2", "m"]),
        ("pion_gluon_quasi_pdf", "NLA", ["A2", "A2p", "A1", "phi", "m"]),
    ]
    for observable, order, expected_labels in cases:
        data_path = tmp_path / f"{observable}_{order}.npz"
        np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
        store = {}
        load_renormalized_matrix_element_samples(store, path=str(data_path))

        run = run_fourier_transform(
            store,
            y_grid=[0.0],
            scheme_scan={"zmin_values": [1.0], "zmax_values": [10.0], "z_ext_max": 12.0},
            method="GI",
            order=order,
            observable=observable,
        )

        fit_info = EnsembleData.from_netcdf(run["fit_info_artifact"])
        assert json.loads(fit_info.attrs["fit_param_labels"]) == expected_labels
        assert np.asarray(json.loads(fit_info.attrs["fit_params"])).shape == (1, 3, len(expected_labels))


def test_fourier_gluon_observables_use_appendix_f_forms() -> None:
    z = np.array([2.0, 3.0])

    shifted_re, _shifted_im = _asymptotic_values(
        z,
        np.array([1.5, 0.4]),
        method="GI",
        order="LA",
        observable="nucleon_gluon_quasi_pdf",
        phase_scale=2.0,
        Lambda0_gev=0.3,
    )
    assert np.asarray(shifted_re, dtype=float).tolist() == pytest.approx(
        (1.5 * z * np.exp(-0.7 * z)).tolist()
    )

    re, im = _asymptotic_values(
        z,
        np.array([1.5, 0.4]),
        method="GI",
        order="LA",
        observable="nucleon_gluon_quasi_pdf",
        phase_scale=2.0,
        Lambda0_gev=0.0,
    )
    assert np.asarray(re, dtype=float).tolist() == pytest.approx((1.5 * z * np.exp(-0.4 * z)).tolist())
    assert np.asarray(im, dtype=float).tolist() == pytest.approx([0.0, 0.0])

    re, _im = _asymptotic_values(
        z,
        np.array([1.5, 0.2, 0.4]),
        method="GI",
        order="NLA",
        observable="nucleon_gluon_quasi_pdf",
        phase_scale=2.0,
        Lambda0_gev=0.0,
    )
    assert np.asarray(re, dtype=float).tolist() == pytest.approx(((1.5 * z + 0.2) * np.exp(-0.4 * z)).tolist())

    re, _im = _asymptotic_values(
        z,
        np.array([1.5, 0.2, 0.3, 0.1, 0.4]),
        method="GI",
        order="NLA",
        observable="pion_gluon_quasi_pdf",
        phase_scale=2.0,
        Lambda0_gev=0.0,
    )
    expected = (1.5 * z + 0.2 + 0.6 * np.cos(0.1 - 2.0 * z)) * np.exp(-0.4 * z)
    assert np.asarray(re, dtype=float).tolist() == pytest.approx(expected.tolist())


def test_fourier_cg_parameter_order_keeps_lambda_before_power() -> None:
    labels = _param_labels("CG", "NLA", "pion_gluon_quasi_pdf")
    p0, bounds = _param_template("CG", "NLA", "pion_gluon_quasi_pdf", Lambda0_gev=0.3)

    assert labels == ["A2", "A2p", "A1", "phi", "m", "n"]
    assert p0.shape == (6,)
    assert bounds[0].shape == (6,)
    assert bounds[1].shape == (6,)
    assert bounds[0][4] == pytest.approx(0.0)
    assert np.isinf(bounds[1][4])
    assert bounds[0][5] == pytest.approx(-2.0)
    assert bounds[1][5] == pytest.approx(4.0)


def test_fourier_scheme_scan_scores_and_model_averages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid=[-0.6, -0.3, 0.0, 0.3, 0.6],
        scheme_scan={
            "zmin_values": [1.0, 2.0],
            "zmax_values": [3.0, 4.0],
            "z_ext_max": 5.0,
            "smooth": "linear",
        },
        method="GI",
        order="LA",
    )
    summary = summarize_fourier_result(store)

    assert run["n_schemes"] == 1
    assert summary["selected_range_label"] in store["fourier_result"]["candidate_scheme_labels"]
    assert len(store["fourier_result"]["candidate_scheme_labels"]) == 4
    assert len(summary["fit_model_chi2_dof"]) == 1
    assert len(summary["fit_model_logGBF"]) == 1
    assert np.asarray(store["fourier_result"]["fit_model_weights"]).shape == (1, 3)


def test_fourier_model_average_false_selects_one_scheme_from_mean_scan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid=[-0.6, -0.3, 0.0, 0.3, 0.6],
        scheme_scan={
            "zmin_values": [1.0, 2.0],
            "zmax_values": [3.0, 4.0],
            "z_ext_max": 5.0,
            "smooth": "linear",
            "model_average": False,
        },
        method="GI",
        order="LA",
    )
    result = store["fourier_result"]

    assert run["n_schemes"] == 1
    assert result["selection_mode"] == "sample_range_then_sample_best_fit_model"
    assert len(result["candidate_scheme_labels"]) == 4
    assert len(result["candidate_scheme_fit_chi2_dof"]) == 4
    assert result["selected_candidate_label"] in result["candidate_scheme_labels"]
    assert store["fourier_result_data"].values.shape == (3, 5)


def test_fourier_model_average_scans_order_and_prior_width_per_sample(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    coord = np.arange(0.0, 14.0)
    base_re = np.exp(-0.22 * coord)
    base_im = 0.08 * np.exp(-0.22 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re])
    im_samples = np.vstack([base_im, 0.98 * base_im, 1.02 * base_im])
    data_path = tmp_path / "matrix_element.npz"
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid=[-0.5, 0.0, 0.5],
        scheme_scan={"zmin_values": [1.0, 2.0], "zmax_values": [10.0, 11.0], "z_ext_max": 13.0},
        method="GI",
        order=["LA", "NLA"],
        posterior_prior_error_scale=[2.0, 3.0],
        observable="pion_quark_quasi_pdf",
    )

    result = store["fourier_result"]
    weights = np.asarray(result["fit_model_weights"], dtype=float)
    assert run["n_schemes"] >= 2
    assert weights.shape == (len(result["fit_model_labels"]), 3)
    assert np.allclose(np.sum(weights, axis=0), 1.0)
    assert result["selected_range_label"] in result["candidate_scheme_labels"]
    fit_info = EnsembleData.from_netcdf(run["fit_info_artifact"])
    labels = json.loads(fit_info.attrs["fit_param_labels"])
    assert "A2" in labels
    assert "m" in labels


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
        y_grid={"start": -0.5, "stop": 0.5, "num": 5},
        method="GI",
        order="LA",
        observable="nucleon_quark_transversity_quasi_pdf",
        coord_unit="fm",
        momentum_gev=2.0,
    )

    auto = run["auto_scheme_scan"]
    assert auto["auto_generated"] is True
    assert len(auto["zmin_values"]) == 4
    assert len(auto["zmax_values"]) == 5
    assert auto["zmin_values"][0] > 0.0
    assert auto["zmax_values"] == pytest.approx([0.8, 0.9, 1.0, 1.1, 1.2])
    assert auto["z_ext_max"] == pytest.approx(1.2 + 8.0 / (5.067731237 * 2.0))
    assert auto["smooth"] == "linear"
    assert "y_range" not in auto
    assert auto["model_average"] is True
    assert run["n_schemes"] == 1
    assert len(store["fourier_result"]["candidate_scheme_labels"]) >= 4


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
        y_grid={"start": -0.5, "stop": 0.5, "num": 5},
        scheme_scan={"model_average": False},
        method="GI",
        order="LA",
        observable="nucleon_quark_transversity_quasi_pdf",
        coord_unit="fm",
        momentum_gev=2.0,
    )

    auto = run["auto_scheme_scan"]
    assert "y_range" not in auto
    assert auto["model_average"] is False
    assert len(auto["zmin_values"]) == 4
    assert len(auto["zmax_values"]) == 5
    assert "z_ext_max" in auto
    assert auto["smooth"] == "linear"


def test_fourier_gpd_auto_scheme_uses_nonzero_second_momentum_for_scale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.linspace(0.0, 10.0, 11)
    base_re = np.exp(-0.2 * coord)
    base_im = 0.05 * np.exp(-0.2 * coord)
    re_samples = np.vstack([base_re, 1.01 * base_re, 0.99 * base_re, 1.02 * base_re])
    im_samples = np.vstack([base_im, 0.98 * base_im, 1.02 * base_im, 0.99 * base_im])
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid={"start": -0.5, "stop": 0.5, "num": 5},
        method="GI",
        order="LA",
        observable="pion_quark_quasi_gpd",
        coord_unit="lattice",
        momentum_gev=0.0,
        final_momentum_gev=0.49,
        lattice_spacing_fm=0.105,
    )

    auto = run["auto_scheme_scan"]
    expected_ft_scale = 0.105 * 5.067731237 * 0.49
    assert auto["z_ext_max"] == pytest.approx(10.0 + 8.0 / expected_ft_scale)


def test_fourier_auto_scan_counts_real_and_imaginary_fit_channels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 25.0)
    base_re = np.exp(-0.08 * coord) * np.cos(0.15 * coord)
    base_im = 0.2 * np.exp(-0.08 * coord) * np.sin(0.12 * coord)
    scales = np.array([0.98, 1.0, 1.02, 1.01])
    re_samples = scales[:, None] * base_re[None, :]
    im_samples = scales[:, None] * base_im[None, :]
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path), resample_mode="jk")

    run = run_fourier_transform(
        store,
        y_grid={"start": -0.5, "stop": 0.5, "num": 6},
        method="CG",
        order="NLA",
        observable="pion_quark_quasi_pdf",
        part="both",
    )

    assert run["n_schemes"] > 0
    assert min(run["auto_scheme_scan"]["zmin_values"]) > 0.0


def test_fourier_auto_scan_prefers_tail_region_for_lattice_units(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.arange(0.0, 25.0)
    base_re = np.exp(-0.08 * coord) * np.cos(0.15 * coord)
    base_im = 0.2 * np.exp(-0.08 * coord) * np.sin(0.12 * coord)
    scales = np.array([0.98, 1.0, 1.02, 1.01, 0.99])
    re_samples = scales[:, None] * base_re[None, :]
    im_samples = scales[:, None] * base_im[None, :]
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path), resample_mode="jk")

    run = run_fourier_transform(
        store,
        y_grid={"start": -0.5, "stop": 0.5, "num": 6},
        method="CG",
        order="NLA",
        observable="pion_quark_quasi_pdf",
        coord_unit="lattice",
        momentum_gev=2.15,
        lattice_spacing_fm=0.0574,
        part="both",
    )

    auto = run["auto_scheme_scan"]
    assert auto["zmax_values"] == [20.0, 21.0, 22.0, 23.0, 24.0]
    assert auto["zmin_values"] == [9.0, 10.0, 11.0, 12.0]
    assert min(auto["zmin_values"]) > 8.0


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
        y_grid={"start": -0.5, "stop": 0.5, "num": 5},
        method="GI",
        order="LA",
        observable="nucleon_quark_transversity_quasi_pdf",
        coord_unit="fm",
        momentum_gev=2.0,
    )

    auto = run["auto_scheme_scan"]
    assert len(auto["zmin_values"]) == 4
    assert min(auto["zmin_values"]) >= 0.5 - 1e-12


def test_fourier_auto_zmax_keeps_nearby_zero_compatible_tail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    coord = np.linspace(0.0, 1.2, 13)
    base_re = np.exp(-1.5 * coord)
    base_im = 0.15 * np.exp(-1.2 * coord)
    base_re[7:] = 0.0
    base_im[7:] = 0.0
    scales = np.array([0.98, 1.0, 1.02, 1.01])
    re_samples = scales[:, None] * base_re[None, :]
    im_samples = scales[:, None] * base_im[None, :]
    re_samples[:, 7:] += np.array([[-0.02], [0.02], [-0.015], [0.015]])
    im_samples[:, 7:] += np.array([[0.015], [-0.015], [0.012], [-0.012]])
    np.savez(data_path, coord=coord, re_samples=re_samples, im_samples=im_samples)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid={"start": -0.5, "stop": 0.5, "num": 5},
        method="GI",
        order="LA",
        observable="nucleon_quark_transversity_quasi_pdf",
        coord_unit="fm",
        momentum_gev=2.0,
    )

    assert run["auto_scheme_scan"]["zmax_values"] == pytest.approx([0.7, 0.8, 0.9, 1.0, 1.1])


def test_fourier_defaults_scheme_scoring_options_for_complete_scan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run_fourier_transform(
        store,
        y_grid={"start": -0.5, "stop": 0.5, "num": 5},
        scheme_scan={
            "zmin_values": [1.0],
            "zmax_values": [4.0],
            "z_ext_max": 5.0,
            "smooth": "linear",
        },
        method="GI",
        order="LA",
    )

    assert len(store["fourier_result"]["fit_model_logGBF"]) == 1


def test_fourier_accepts_compact_y_grid_spec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid={"start": -1.0, "stop": 1.0, "num": 21},
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
    )
    summary = summarize_fourier_result(store)

    assert run["n_y"] == 21
    assert len(summary["y_grid"]) == 21


def test_fourier_accepts_covariance_sample_error_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "matrix_element.npz"
    _write_npz(data_path)
    store = {}
    load_renormalized_matrix_element_samples(store, path=str(data_path))

    run = run_fourier_transform(
        store,
        y_grid={"start": -0.5, "stop": 0.5, "num": 5},
        scheme_scan={"zmin_values": [1.0], "zmax_values": [4.0], "z_ext_max": 5.0},
        method="GI",
        order="LA",
        sample_error_mode="covariance",
    )

    assert run["n_schemes"] == 1
    assert store["fourier_result"]["sample_error_mode"] == "covariance"


def test_plot_fourier_artifact_writes_figure(tmp_path: Path) -> None:
    path = tmp_path / "fourier_result.npz"
    save_path = tmp_path / "fourier.pdf"
    np.savez(
        path,
        y_grid=np.array([-0.5, 0.0, 0.5]),
        ft_re_mean=np.array([0.2, 0.3, 0.2]),
        ft_im_mean=np.array([-0.1, 0.0, 0.1]),
        ft_re_stat_sdev=np.array([0.01, 0.02, 0.01]),
        ft_im_stat_sdev=np.array([0.02, 0.01, 0.02]),
        ft_re_sys_sdev=np.array([0.005, 0.005, 0.005]),
        ft_im_sys_sdev=np.array([0.005, 0.005, 0.005]),
        observable=np.asarray("nucleon_quark_transversity_quasi_pdf"),
    )

    fig, (ax_re, _ax_im) = plot_fourier_artifact(path, save_path=save_path)

    assert save_path.is_file()
    assert ax_re.get_title() == "FT nucleon quark transversity quasi pdf"
    fig.clf()


def test_plot_fourier_artifact_uses_stored_means_for_median_complex_nc(tmp_path: Path) -> None:
    """NetCDF attrs carry real/im means; avoid median gvar on complex bootstrap samples."""
    k = np.array([-0.5, 0.0, 0.5])
    re_samples = np.array([[0.2, 0.3, 0.2], [0.21, 0.31, 0.21]])
    im_samples = np.array([[-0.1, 0.0, 0.1], [-0.11, 0.01, 0.11]])
    values = [re_samples[idx] + 1j * im_samples[idx] for idx in range(re_samples.shape[0])]
    data = EnsembleData(
        ensemble=None,
        resample="bootstrap",
        values=values,
        dims=("x",),
        coords={"x": k.tolist()},
        attrs={
            "sample_error_mode": "median",
            "observable": "pion_quark_quasi_pdf",
            "ft_re_mean": json.dumps([0.205, 0.305, 0.205]),
            "ft_im_mean": json.dumps([-0.105, 0.005, 0.105]),
            "ft_re_stat_sdev": json.dumps([0.01, 0.02, 0.01]),
            "ft_im_stat_sdev": json.dumps([0.02, 0.01, 0.02]),
            "ft_re_sys_sdev": json.dumps([0.005, 0.005, 0.005]),
            "ft_im_sys_sdev": json.dumps([0.005, 0.005, 0.005]),
        },
        name="fourier_transform",
    )
    path = tmp_path / "fourier_result.nc"
    save_path = tmp_path / "fourier.pdf"
    data.to_netcdf(path)

    fig, _ax_re = plot_fourier_artifact(path, save_path=save_path)

    assert save_path.is_file()
    fig.clf()
