"""Unit tests for the refactored correlator stage tools and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gvar as gv
import h5py
import numpy as np
import pytest
from matplotlib.legend import Legend

pytest.importorskip("lsqfit")

import lamet_agent.stages.correlator.functions as correlator_functions
from lamet_agent.core.plotting import (
    _pt3_ratio_data_tau_slice,
    _ratio_denominator_correction,
    _ylim_middle_third,
    _ylim_mean_middle_third,
    plot_pt2_fit_on_data,
    plot_pt2_meff_on_data,
)
from lamet_agent.core.resampling import jackknife
from lamet_agent.core.resampling import sample_mean_err as core_sample_mean_err
from lamet_agent.core.tools import log_nonlinear_fit_quality, prepare_tool_args, setup_logger
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.stages.correlator.functions import (
    PT2_PRIOR_ERROR_SCALE,
    STAGE_TOOLS,
    _anchor_pt2_prior,
    _bare_samples,
    _candidate_specs,
    _check_mode,
    _check_rescale,
    _fit_summary,
    _fit_usable,
    _loggbf_weights,
    _normalise_pt2_windows,
    _normalise_pt3_windows,
    _normalise_strategy,
    _overlaps,
    _ratio_samples,
    _resample_pt2,
    _scaled_prior,
    _write_outputs,
    asymptotic_ratio,
    bayesian_average,
    fit_bare_matrix_grid,
    fit_joint,
    fit_ratio,
    fit_two_point,
    inspect_correlator_scale,
    pt2_prior,
    pt2_re_fcn,
    pt3_ratio_fcn,
    pt3_ratio_prior,
    select_best,
    tune_bare_matrix,
    tune_ground_state,
)


# --- toy data builders -------------------------------------------------------


def _toy_pt2_gv(*, Lt: int = 24, E0: float = 0.45) -> np.ndarray:
    t = np.arange(Lt, dtype=int)
    p = gv.BufferDict()
    p["E0"] = gv.gvar(E0, 1e-4)
    p["log(dE1)"] = gv.gvar(np.log(0.5), 1e-3)
    p["z0"] = gv.gvar(1.0, 1e-4)
    p["z1"] = gv.gvar(0.05, 1e-4)
    return pt2_re_fcn(t, p, Lt, nstate=2)


def _toy_ratio_gv(*, tsep_ls: list[int], Lt: int = 32):
    p = gv.BufferDict()
    p["E0"] = gv.gvar(0.45, 1e-4)
    p["log(dE1)"] = gv.gvar(np.log(0.5), 1e-3)
    p["z0"] = gv.gvar(1.0, 1e-4)
    p["z1"] = gv.gvar(0.05, 1e-4)
    p["O00_re"] = gv.gvar(0.30, 1e-4)
    p["O00_im"] = gv.gvar(0.0, 1e-4)
    p["O01_re"] = gv.gvar(0.02, 1e-4)
    p["O01_im"] = gv.gvar(0.0, 1e-4)
    p["O11_re"] = gv.gvar(0.01, 1e-4)
    p["O11_im"] = gv.gvar(0.0, 1e-4)
    ratio_re: dict[int, np.ndarray] = {}
    ratio_im: dict[int, np.ndarray] = {}
    for tsep in tsep_ls:
        tau = np.arange(tsep + 1, dtype=float)
        t_arr = np.full_like(tau, float(tsep))
        ratio_re[tsep] = pt3_ratio_fcn(t_arr, tau, p, Lt, nstate=2, part="re")
        ratio_im[tsep] = pt3_ratio_fcn(t_arr, tau, p, Lt, nstate=2, part="im")
    return ratio_re, ratio_im, p


def _write_fake_correlators(
    tmp_path: Path,
    *,
    Lt: int = 24,
    n_cfg: int = 24,
    tsep_ls: tuple[int, ...] = (6, 8),
    z_values: tuple[int, ...] = (0, 1),
    momentum: str = "PX0PY0PZ0",
    seed: int = 0,
) -> tuple[str, dict[str, str]]:
    """Write recoverable fake 2pt/3pt HDF5 files (ratio = model up to per-cfg noise)."""
    rng = np.random.default_rng(seed)
    p2 = {"E0": 0.45, "dE1": 0.5, "z0": 1.0, "z1": 0.1}
    t = np.arange(Lt)
    base = pt2_re_fcn(t, p2, Lt, nstate=2)
    pt2_cfg = np.empty((n_cfg, Lt), dtype=complex)
    for c in range(n_cfg):
        pt2_cfg[c] = base * (1 + rng.normal(0, 2e-3, Lt))
    pt2_path = tmp_path / "pt2.h5"
    with h5py.File(pt2_path, "w") as h5f:
        h5f.create_dataset(f"SS/5/{momentum}", data=pt2_cfg.T)

    pt3_paths: dict[str, str] = {}
    for tsep in tsep_ls:
        path = tmp_path / f"pt3_ts{tsep}.h5"
        with h5py.File(path, "w") as h5f:
            for z in z_values:
                p3 = {
                    "E0": 0.45, "dE1": 0.5, "z0": 1.0, "z1": 0.1,
                    "O00_re": 0.30 - 0.03 * z, "O00_im": 0.05,
                    "O01_re": 0.02, "O01_im": 0.01, "O11_re": 0.01, "O11_im": 0.0,
                }
                tau = np.arange(tsep + 1, dtype=float)
                t_arr = np.full_like(tau, float(tsep))
                ratio = pt3_ratio_fcn(t_arr, tau, p3, Lt, nstate=2, part="re") + 1j * pt3_ratio_fcn(
                    t_arr, tau, p3, Lt, nstate=2, part="im"
                )
                pt3_cfg = np.empty((n_cfg, tsep + 1), dtype=complex)
                for c in range(n_cfg):
                    eps = rng.normal(0, 2e-3, tsep + 1)
                    pt3_cfg[c] = ratio * (1 + eps) * pt2_cfg[c, tsep]
                h5f.create_dataset(f"SS/T/{momentum}/b_X/eta0/bT0/bz{z}", data=pt3_cfg.T)
        pt3_paths[str(tsep)] = str(path)
    return str(pt2_path), pt3_paths


# --- registry ----------------------------------------------------------------


def test_stage_tools_expose_the_four_agentic_tools() -> None:
    assert set(STAGE_TOOLS) == {
        "inspect_correlator_scale",
        "tune_ground_state",
        "tune_bare_matrix",
        "fit_bare_matrix_grid",
    }


# --- physics models and fits -------------------------------------------------


def test_pt3_ratio_fcn_part_selects_matrix_element() -> None:
    p = {"E0": 0.45, "dE1": 0.5, "z0": 1.0, "z1": 0.1,
         "O00_re": 0.3, "O00_im": 0.1, "O01_re": 0.0, "O01_im": 0.0, "O11_re": 0.0, "O11_im": 0.0}
    re = pt3_ratio_fcn(np.array([8.0]), np.array([4.0]), p, 32, nstate=2, part="re")
    im = pt3_ratio_fcn(np.array([8.0]), np.array([4.0]), p, 32, nstate=2, part="im")
    assert re[0] != im[0]


def test_pt3_ratio_prior_extends_pt2_prior_with_matrix_elements() -> None:
    prior = pt3_ratio_prior(nstate=2)
    base = pt2_prior(nstate=2)
    assert set(base).issubset(set(prior))
    assert {"O00_re", "O00_im", "O01_re", "O11_re"}.issubset(set(prior))


def test_fit_two_point_recovers_e0() -> None:
    fit = fit_two_point(_toy_pt2_gv(), 2, 10, 24, svdcut=1e-8)
    assert abs(gv.mean(fit.p["E0"]) - 0.45) < 0.05


def test_fit_two_point_rescale_recovers_tiny_pt2() -> None:
    scale = 1.0e18
    fit = fit_two_point(_toy_pt2_gv() / scale, 2, 10, 24, svdcut=1e-8, rescale=scale)
    assert abs(gv.mean(fit.p["E0"]) - 0.45) < 0.05
    diag = _overlaps(fit.p, 2, scale)
    assert gv.mean(diag["z0_physical"]) == pytest.approx(1.0e-9, rel=0.1)


def test_fit_ratio_recovers_parameters() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, p_true = _toy_ratio_gv(tsep_ls=tsep_ls)
    fit = fit_ratio(ratio_re, ratio_im, tsep_ls, 1, 32, prior=pt3_ratio_prior(2))
    assert abs(gv.mean(fit.p["E0"]) - gv.mean(p_true["E0"])) < 0.05
    assert abs(gv.mean(fit.p["O00_re"]) - gv.mean(p_true["O00_re"])) < 0.05


def test_fit_joint_recovers_parameters_and_is_rescale_invariant() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, p_true = _toy_ratio_gv(tsep_ls=tsep_ls, Lt=32)
    scale = 1.0e18
    scaled = fit_joint(
        _toy_pt2_gv(Lt=32) / scale, 2, 12, ratio_re, ratio_im, tsep_ls, 1, 32,
        prior=pt3_ratio_prior(2), svdcut=1e-8, rescale=scale,
    )
    unscaled = fit_joint(
        _toy_pt2_gv(Lt=32), 2, 12, ratio_re, ratio_im, tsep_ls, 1, 32,
        prior=pt3_ratio_prior(2), svdcut=1e-8,
    )
    scaled_plateau = gv.mean(scaled.p["O00_re"] / (2 * scaled.p["E0"]))
    unscaled_plateau = gv.mean(unscaled.p["O00_re"] / (2 * unscaled.p["E0"]))
    true_plateau = gv.mean(p_true["O00_re"] / (2 * p_true["E0"]))
    assert scaled_plateau == pytest.approx(unscaled_plateau, rel=0.05)
    assert scaled_plateau == pytest.approx(true_plateau, rel=0.05)


def test_fit_ratio_rejects_empty_tau_window() -> None:
    ratio_re, ratio_im, _ = _toy_ratio_gv(tsep_ls=[4])
    with pytest.raises(ValueError, match="empty tau"):
        fit_ratio(ratio_re, ratio_im, [4], 3, 32, prior=pt3_ratio_prior(2))


# --- selection and averaging helpers ----------------------------------------


def test_select_best_prefers_loggbf_after_q_cut() -> None:
    records = [{"Q": 0.20, "logGBF": 1.0}, {"Q": 0.06, "logGBF": 3.0}, {"Q": 0.90, "logGBF": 2.0}]
    index, fallback = select_best(records, q_min=0.05)
    assert index == 1
    assert fallback is False


def test_select_best_falls_back_to_max_q() -> None:
    records = [{"Q": 0.01, "logGBF": 10.0}, {"Q": 0.04, "logGBF": 1.0}]
    index, fallback = select_best(records, q_min=0.05)
    assert index == 1
    assert fallback is True


def test_loggbf_weights_normalise_and_favour_high_loggbf() -> None:
    weights = _loggbf_weights([{"logGBF": 0.0}, {"logGBF": 2.0}])
    assert weights.sum() == pytest.approx(1.0)
    assert weights[1] > weights[0]


def test_bayesian_average_adds_systematic_spread() -> None:
    values = np.array([gv.gvar(1.0, 0.1), gv.gvar(2.0, 0.1)], dtype=object)
    weights = np.array([0.5, 0.5])
    avg = bayesian_average(values, weights)
    assert gv.mean(avg) == pytest.approx(1.5)
    assert gv.sdev(avg) > 0.1


def test_scaled_prior_inflates_all_errors() -> None:
    template = pt3_ratio_prior(nstate=2)
    posterior = gv.BufferDict()
    for key in template:
        posterior[key] = gv.gvar(0.25, 0.03)

    class Fit:
        pass

    fit = Fit()
    fit.p = posterior
    prior = _scaled_prior(fit, template, error_scale=3.0)
    for key in template:
        assert gv.mean(prior[key]) == pytest.approx(0.25)
        assert gv.sdev(prior[key]) == pytest.approx(0.09)


def test_anchor_pt2_prior_widens_only_e0_z0() -> None:
    class Fit:
        pass

    fit = Fit()
    fit.p = {"E0": gv.gvar(0.45, 0.01), "z0": gv.gvar(1.0, 0.02)}
    prior = pt3_ratio_prior(nstate=2)
    _anchor_pt2_prior(prior, fit)
    assert gv.sdev(prior["E0"]) == pytest.approx(PT2_PRIOR_ERROR_SCALE * 0.01)
    assert gv.sdev(prior["z0"]) == pytest.approx(PT2_PRIOR_ERROR_SCALE * 0.02)
    assert gv.sdev(prior["z1"]) == gv.sdev(pt3_ratio_prior(2)["z1"])


def test_fit_usable_rejects_non_physical_e0() -> None:
    template = pt3_ratio_prior(nstate=2)
    posterior = gv.BufferDict()
    for key in template:
        posterior[key] = gv.gvar(0.25, 0.03)
    posterior["E0"] = gv.gvar(1e-7, 0.03)

    class Fit:
        pass

    fit = Fit()
    fit.p = posterior
    usable, reason = _fit_usable(fit, template)
    assert usable is False
    assert "E0" in str(reason)


def test_bare_samples_use_o00_over_two_e0() -> None:
    class Fit:
        pass

    fit = Fit()
    fit.p = {"E0": gv.gvar(0.5, 0.01), "O00_re": gv.gvar(2.0, 0.1), "O00_im": gv.gvar(-1.0, 0.1)}
    real, imag = _bare_samples([{"fit": fit}, {"fit": None}])
    assert real[0] == pytest.approx(2.0)
    assert imag[0] == pytest.approx(-1.0)
    assert np.isnan(real[1])


def test_fit_summary_reports_physical_overlaps() -> None:
    class Fit:
        pass

    fit = Fit()
    fit.p = {"E0": gv.gvar(0.5, 0.01), "z0": gv.gvar(1.0, 0.01), "z1": gv.gvar(2.0, 0.02)}
    summary = _fit_summary(
        {"fit": fit, "nstate": 2, "chi2_dof": 1.0, "Q": 0.5, "logGBF": 3.0, "correlator_rescale": 1.0e18},
        fallback=False,
        index=0,
    )
    assert summary["correlator_rescale"] == pytest.approx(1.0e18)
    assert "z0_physical" in summary
    assert "z1_over_z0_physical" in summary


def test_overlaps_apply_sqrt_rescale() -> None:
    p = {"z0": gv.gvar(1.0, 0.01), "z1": gv.gvar(2.0, 0.02)}
    diag = _overlaps(p, 2, 1.0e18)
    assert gv.mean(diag["z0_physical"]) == pytest.approx(1.0e-9, rel=1e-6)
    assert gv.mean(diag["z1_over_z0_physical"]) == pytest.approx(2.0, rel=1e-6)


# --- window grids and strategy ----------------------------------------------


def test_default_pt2_windows_use_l_over_four() -> None:
    windows = _normalise_pt2_windows(None, Lt=32)
    assert windows
    assert {window["tmax"] for window in windows} == {8}
    assert all(window["tmax"] - window["tmin"] >= 4 for window in windows)


def test_normalise_pt3_windows_expands_tau_cuts() -> None:
    windows = _normalise_pt3_windows(None, tsep_ls=[6, 8], tau_cuts=[1, 2])
    assert [w["tau_cut"] for w in windows] == [1, 2]
    assert windows[0]["tsep_ls"] == [6, 8]


def test_candidate_specs_joint_is_cartesian() -> None:
    pt2 = [{"tmin": 2, "tmax": 10}, {"tmin": 3, "tmax": 10}]
    pt3 = [{"tsep_ls": [6, 8], "tau_cut": 1}, {"tsep_ls": [6, 8], "tau_cut": 2}]
    specs = _candidate_specs(strategy="joint", pt2_window_specs=pt2, pt3_window_specs=pt3, pt2_best=None)
    assert len(specs) == 4
    assert {(s["tmin"], s["tau_cut"]) for s in specs} == {(2, 1), (2, 2), (3, 1), (3, 2)}


def test_candidate_specs_chained_uses_pt2_best_window() -> None:
    pt3 = [{"tsep_ls": [6, 8], "tau_cut": 1}]
    best = {"tmin": 3, "tmax": 11}
    specs = _candidate_specs(strategy="chained", pt2_window_specs=[], pt3_window_specs=pt3, pt2_best=best)
    assert specs == [{"tmin": 3, "tmax": 11, "tsep_ls": [6, 8], "tau_cut": 1}]


def test_normalise_strategy_aliases() -> None:
    assert _normalise_strategy("joint") == ("joint", "joint_2pt_ratio")
    assert _normalise_strategy("chain") == ("chained", "chained_2pt_ratio")
    with pytest.raises(ValueError, match="fit_strategy"):
        _normalise_strategy("nonsense")


def test_check_helpers_validate_arguments() -> None:
    assert _check_rescale(10.0) == 10.0
    with pytest.raises(ValueError, match="correlator_rescale"):
        _check_rescale(-1.0)
    assert _check_mode("jk") == "jk"
    with pytest.raises(ValueError, match="resample_mode"):
        _check_mode("nope")


# --- resampling and ratio ----------------------------------------------------


def test_ratio_after_resample_differs_from_resample_after_ratio() -> None:
    rng = np.random.default_rng(0)
    n_cfg, lt, tsep = 40, 24, 8
    pt2 = rng.normal(size=(n_cfg, lt)) + 2.0
    pt3 = rng.normal(size=(n_cfg, tsep + 1)) + 1.0
    from lamet_agent.core.resampling import resample_config_samples

    _, pt2_complex_samples, indices = _resample_pt2(pt2.astype(complex), mode="bs", n_boot=200, seed=1984)
    pt3_samples, _ = resample_config_samples(pt3, mode="bs", n_boot=200, seed=1984, indices=indices)
    ratio_first, _ = _ratio_samples(pt2_complex_samples, pt3_samples, tsep)
    wrong_ratio, _ = resample_config_samples(pt3 / pt2[:, tsep][:, None], mode="bs", n_boot=200, seed=1984, indices=indices)
    assert not np.allclose(ratio_first, wrong_ratio)


def test_resample_config_samples_jackknife_matches_helper() -> None:
    data = np.arange(12, dtype=float).reshape(4, 3)
    _, complex_samples, indices = _resample_pt2(data.astype(complex), mode="jk", n_boot=99, seed=7)
    assert indices is None
    assert np.allclose(np.real(complex_samples), jackknife(data))


def test_sample_mean_err_matches_core_helper() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert core_sample_mean_err(values, mode="jk")[0] == pytest.approx(2.0)


# --- output writer -----------------------------------------------------------


def test_write_outputs_writes_txt_plot_and_report(tmp_path) -> None:
    records = [
        {
            "z": 0,
            "real_samples": np.array([1.0, 1.1, 0.9]),
            "imag_samples": np.array([0.0, 0.1, -0.1]),
            "window": {"tau_cut": 1, "tsep_ls": [8, 10]},
            "sample0_plot_paths": {"ratio_re_pdf": "re.pdf"},
        },
        {
            "z": 1,
            "real_samples": np.array([0.8, 0.9, 0.7]),
            "imag_samples": np.array([0.2, 0.3, 0.1]),
            "window": {"tau_cut": 2, "tsep_ls": [8, 10]},
        },
    ]
    result = _write_outputs(
        records,
        artifacts_dir=tmp_path,
        save_path=str(tmp_path / "bare"),
        ensemble="HISQa060_X",
        tag="CG52bxp00_CG52bxp00",
        variant="free",
        direction="X",
        momentum="PX0PY0PZ0",
        b_label="b0",
        resample_mode="jk",
        output_subdir="bare_qpdf",
    )
    txt0 = tmp_path / "bare_qpdf" / "HISQa060_X_CG52bxp00_CG52bxp00_free_X_PX0PY0PZ0_b0_z0.txt"
    assert txt0.is_file()
    loaded = np.loadtxt(txt0)
    assert loaded.shape == (3, 2)
    assert np.allclose(loaded[:, 0], records[0]["real_samples"])
    assert (tmp_path / "bare.pdf").is_file()
    report = json.loads((tmp_path / "bare_report.json").read_text())
    assert report["resample_mode"] == "jk"
    assert report["outputs"][0]["sample0_plot_paths"] == {"ratio_re_pdf": "re.pdf"}
    assert result["n_z"] == 2


# --- inspect tool ------------------------------------------------------------


def test_inspect_correlator_scale_accepts_selector_momentum(tmp_path) -> None:
    path = tmp_path / "pt2_px5.h5"
    px5_data = np.full((12, 4), 3.0e-18, dtype=np.complex128)
    px0_data = np.full((12, 4), 9.0e-18, dtype=np.complex128)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=px0_data)
        h5f.create_dataset("SS/5/PX5PY0PZ0", data=px5_data)

    result = inspect_correlator_scale(
        {},
        pt2_path=str(path),
        pt2_windows=[{"tmin": 2, "tmax": 5}],
        selectors={"source_sink": "SS", "gamma": "5", "momentum": "PX5PY0PZ0"},
    )
    assert result["momentum"] == "PX5PY0PZ0"
    assert result["windows"][0]["median_abs"] == pytest.approx(3.0e-18)


# --- tune tools --------------------------------------------------------------


def test_tune_ground_state_returns_windows_and_stores_avg(tmp_path) -> None:
    pt2_path, _ = _write_fake_correlators(tmp_path, tsep_ls=(6,), z_values=(0,))
    store: dict = {}
    result = tune_ground_state(
        store,
        pt2_path=pt2_path,
        pt2_windows=[{"tmin": 2, "tmax": 10}, {"tmin": 3, "tmax": 10}],
        resample_mode="jk",
        svdcut=1e-6,
        artifacts_dir=tmp_path / "artifacts",
        window_indices=[0, 1],
    )
    assert len(result["windows"]) == 2
    assert result["E0_avg"] is not None
    assert "E0_avg" in store and "z0_avg" in store
    assert abs(gv.mean(store["E0_avg"]) - 0.45) < 0.05
    assert Path(result["meff_pdf"]).is_file()


def test_tune_bare_matrix_returns_ranked_candidates(tmp_path) -> None:
    pt2_path, pt3_paths = _write_fake_correlators(tmp_path, tsep_ls=(6, 8), z_values=(0,))
    result = tune_bare_matrix(
        {},
        pt2_path=pt2_path,
        pt3_paths=pt3_paths,
        tsep_ls=[6, 8],
        momentum="PX0PY0PZ0",
        tune_z=0,
        pt2_windows=[{"tmin": 2, "tmax": 10}],
        pt3_tau_cuts=[1, 2],
        fit_strategy="joint",
        resample_mode="jk",
        svdcut=1e-6,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["candidates"]
    assert "O00_re_over_2E0" in result["candidates"][0]
    assert "recommended_index" in result
    assert Path(result["tuning_ratio_pdf"]["ratio_re_pdf"]).is_file()


# --- apply tool (end to end) -------------------------------------------------


def test_fit_bare_matrix_grid_single_shared_window(tmp_path) -> None:
    pt2_path, pt3_paths = _write_fake_correlators(tmp_path, tsep_ls=(6, 8), z_values=(0, 1))
    result = fit_bare_matrix_grid(
        {},
        pt2_path=pt2_path,
        pt3_paths=pt3_paths,
        tsep_ls=[6, 8],
        z_values=[0, 1],
        ensemble="E",
        tag="T",
        momentum="PX0PY0PZ0",
        pt2_windows=[{"tmin": 2, "tmax": 10}, {"tmin": 3, "tmax": 10}],
        pt3_tau_cuts=[1, 2],
        fit_strategy="joint",
        resample_mode="jk",
        svdcut=1e-6,
        artifacts_dir=tmp_path / "artifacts",
    )
    # one shared window applied to every z
    assert len(result["shared_window_specs"]) == 1
    report = json.loads(Path(result["report_json"]).read_text())
    tau_cuts = {z["window"]["tau_cut"] for z in report["z_fits"]}
    assert len(tau_cuts) == 1
    for z in (0, 1):
        txt = tmp_path / "artifacts" / "bare_qpdf" / f"E_T_free_X_PX0PY0PZ0_b0_z{z}.txt"
        assert txt.is_file()
        assert np.loadtxt(txt).shape[1] == 2


def test_fit_bare_matrix_grid_explicit_window_and_chained(tmp_path) -> None:
    pt2_path, pt3_paths = _write_fake_correlators(tmp_path, tsep_ls=(6, 8), z_values=(0,))
    result = fit_bare_matrix_grid(
        {},
        pt2_path=pt2_path,
        pt3_paths=pt3_paths,
        tsep_ls=[6, 8],
        z_values=[0],
        ensemble="E",
        tag="T",
        momentum="PX0PY0PZ0",
        pt2_window={"tmin": 2, "tmax": 10},
        pt3_window={"tsep_ls": [6, 8], "tau_cut": 1},
        fit_strategy="chained",
        resample_mode="jk",
        svdcut=1e-6,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["fit_strategy"] == "chained"
    assert result["shared_window_specs"][0]["tau_cut"] == 1
    assert result["sample0_pt2_plot_paths"].get("meff_pdf")


def test_fit_bare_matrix_grid_model_average_uses_window_set(tmp_path) -> None:
    pt2_path, pt3_paths = _write_fake_correlators(tmp_path, tsep_ls=(6, 8), z_values=(0,))
    result = fit_bare_matrix_grid(
        {},
        pt2_path=pt2_path,
        pt3_paths=pt3_paths,
        tsep_ls=[6, 8],
        z_values=[0],
        ensemble="E",
        tag="T",
        momentum="PX0PY0PZ0",
        pt2_windows=[{"tmin": 2, "tmax": 10}],
        pt3_tau_cuts=[1, 2],
        model_average=True,
        fit_strategy="joint",
        resample_mode="jk",
        svdcut=1e-6,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["model_average"] is True
    assert len(result["shared_window_specs"]) == 2
    txt = tmp_path / "artifacts" / "bare_qpdf" / "E_T_free_X_PX0PY0PZ0_b0_z0.txt"
    assert txt.is_file()


# --- agent plumbing ----------------------------------------------------------


def test_fit_bare_matrix_grid_path_args_resolve_under_artifacts(tmp_path) -> None:
    manifest = AnalysisManifest(
        run_id="demo",
        correlators=[],
        kernels=[],
        manifest_dir=tmp_path / "examples",
        project_root=tmp_path,
    )
    args = {"pt2_path": "data/pt2.h5", "pt3_paths": {"8": "data/ts8.h5", "10": "/abs/ts10.h5"}, "save_path": None}
    resolved = prepare_tool_args(
        "fit_bare_matrix_grid", args, manifest=manifest, artifacts_dir=tmp_path / "artifacts", _store={}
    )
    assert resolved["pt2_path"] == str(tmp_path / "data" / "pt2.h5")
    assert resolved["pt3_paths"]["8"] == str(tmp_path / "data" / "ts8.h5")
    assert resolved["pt3_paths"]["10"] == "/abs/ts10.h5"
    assert resolved["save_path"] == str(tmp_path / "artifacts" / "bare_matrix_elements")
    assert resolved["artifacts_dir"] == str(tmp_path / "artifacts")


def test_tune_tools_get_artifacts_dir_injected(tmp_path) -> None:
    manifest = AnalysisManifest(run_id="demo", correlators=[], kernels=[], manifest_dir=tmp_path, project_root=tmp_path)
    resolved = prepare_tool_args(
        "tune_bare_matrix", {"save_path": None}, manifest=manifest, artifacts_dir=tmp_path / "artifacts", _store={}
    )
    assert resolved["artifacts_dir"] == str(tmp_path / "artifacts")


# --- plotting helpers retained ----------------------------------------------


def test_asymptotic_ratio_differs_from_o00_by_two_e0() -> None:
    plat = asymptotic_ratio(gv.gvar(0.30, 0.01), gv.gvar(0.45, 0.01), tsep=10, Lt=32)
    assert abs(gv.mean(plat) - 0.30 / (2 * 0.45)) < 0.002


def test_ratio_denominator_correction_converts_periodic_to_forward() -> None:
    E0 = gv.gvar(0.09, 0.001)
    correction = _ratio_denominator_correction(12, energy=E0, Lt=64)
    assert gv.mean(correction) == pytest.approx(gv.mean(1.0 + gv.exp(-E0 * (64 - 2 * 12))))


def test_pt3_ratio_data_tau_slice_includes_tsep_minus_one() -> None:
    row = np.arange(12)
    assert list(row[_pt3_ratio_data_tau_slice(10)]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_ylim_mean_middle_third_places_means_in_center_band() -> None:
    y_lo, y_hi = _ylim_mean_middle_third(np.array([1.0, 2.0]))
    height = y_hi - y_lo
    assert np.isclose(y_lo + height / 3, 1.0, rtol=1e-9)
    assert np.isclose(y_lo + 2 * height / 3, 2.0, rtol=1e-9)


def test_ylim_middle_third_places_data_in_center_band() -> None:
    y_lo, y_hi = _ylim_middle_third([np.array([1.0, 2.0])], [np.array([0.1, 0.1])])
    height = y_hi - y_lo
    assert np.isclose(y_lo + height / 3, 0.9, rtol=1e-9)
    assert np.isclose(y_lo + 2 * height / 3, 2.1, rtol=1e-9)


def test_plot_pt2_meff_on_data_respects_t_max(tmp_path) -> None:
    pt2_gv = _toy_pt2_gv()
    t_band = np.arange(2, 10, dtype=int)
    save = tmp_path / "meff_only"
    _, ax = plot_pt2_meff_on_data(pt2_gv, fit_bands=[{"fit_t": t_band, "fit_gv": pt2_gv[t_band], "label": "w1"}], t_max=6, save_path=save)
    assert ax.get_xlim()[1] == pytest.approx(6.0)
    assert (tmp_path / "meff_only_meff.pdf").is_file()


def test_plot_pt2_fit_on_data_respects_t_max() -> None:
    pt2_gv = _toy_pt2_gv()
    t_band = np.arange(2, 10, dtype=int)
    (_, ax_c2), (_, ax_meff) = plot_pt2_fit_on_data(
        pt2_gv,
        fit_bands=[{"fit_t": t_band, "fit_gv": pt2_gv[t_band], "label": "w1"}],
        t_max=6,
    )
    assert ax_meff.get_xlim()[1] == pytest.approx(6.0)
    assert ax_c2.get_xlim()[1] > 6.0


def test_plot_pt2_legend_upper_right() -> None:
    pt2_gv = _toy_pt2_gv()
    t_band = np.arange(2, 10, dtype=int)
    (_, ax_c2), (_, ax_meff) = plot_pt2_fit_on_data(pt2_gv, fit_bands=[{"fit_t": t_band, "fit_gv": pt2_gv[t_band], "label": "w1"}])
    upper_right = Legend.codes["upper right"]
    assert ax_c2.get_legend()._loc == upper_right
    assert ax_meff.get_legend()._loc == upper_right


# --- logging -----------------------------------------------------------------


def test_setup_logger_writes_log_file(tmp_path) -> None:
    log_path = tmp_path / "fit.log"
    logger = setup_logger(log_path)
    logger.info("joint fit message")
    for handler in logger.handlers:
        handler.flush()
    assert "joint fit message" in log_path.read_text()


def test_log_nonlinear_fit_quality_writes_good_and_bad(tmp_path) -> None:
    class Fit:
        def __init__(self, q: float) -> None:
            self.Q = q
            self.chi2 = 2.0
            self.dof = 4
            self.logGBF = -1.5

    log_path = tmp_path / "quality.log"
    logger = setup_logger(log_path, logger_name="quality_test_logger")
    assert log_nonlinear_fit_quality(Fit(0.2), kind="toy", label="good", logger=logger) == "Good"
    assert log_nonlinear_fit_quality(Fit(0.01), kind="toy", label="bad", logger=logger) == "Bad"
    for handler in logger.handlers:
        handler.flush()
    log_text = log_path.read_text()
    assert "Good toy good" in log_text
    assert "WARNING - Bad toy bad" in log_text
