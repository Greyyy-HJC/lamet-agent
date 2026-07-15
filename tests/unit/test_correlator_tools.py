"""Unit tests for the refactored correlator stage tools and helpers."""

from __future__ import annotations

import inspect
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
from lamet_agent.stages.correlator.skills import TOOL_CATALOG
from lamet_agent.core.plotting import (
    FIT_LOG_YLIM_BOTTOM_FACTOR,
    FIT_LOG_YLIM_DATA_HIGH_NUM,
    FIT_LOG_YLIM_DATA_LOW_NUM,
    FIT_LOG_YLIM_TOP_FACTOR,
    _pt3_ratio_data_tau_slice,
    _ratio_denominator_correction,
    _ylim_middle_third,
    _ylim_mean_middle_third,
    plot_pt2_fit_on_data,
    plot_pt2_meff_on_data,
)
from lamet_agent.core.resampling import jackknife
from lamet_agent.core.resampling import sample_mean_and_sdev
from lamet_agent.core.tools import log_nonlinear_fit_quality, setup_logger
from lamet_agent.stages.correlator.functions import (
    PT2_PRIOR_ERROR_SCALE,
    STAGE_TOOLS,
    _anchor_pt2_prior,
    _bare_matrix_element_mean_for_part,
    _candidate_specs,
    _check_mode,
    _check_rescale,
    _fit_summary,
    _fit_usable,
    _fh_samples_from_ratios,
    _loggbf_weights,
    _normalise_pt2_windows,
    _normalise_pt3_windows,
    _normalise_fit_scope,
    _normalise_strategy,
    _non_forward_ratio_samples,
    _overlaps,
    _ratio_samples,
    _read_2pt,
    _read_3pt,
    _resample_pt2,
    _scaled_prior,
    _vary_prior_width,
    bayesian_average,
    fit_bare_matrix_grid,
    fit_matrix_element,
    fit_two_point,
    fh_prior,
    inspect_correlator_scale,
    pt2_prior,
    pt2_re_fcn,
    pt3_nonbreit_ratio_fcn,
    pt3_nonbreit_ratio_prior,
    pt3_ratio_fcn,
    pt3_ratio_prior,
    select_data_window,
    select_best,
    _summarise_cross_z_feasibility,
    _window_candidate_key,
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
        h5f.create_dataset(f"g5/g5/{momentum}", data=pt2_cfg.T)

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
                h5f.create_dataset(
                    f"g5/g5/gT_nonlocal/{momentum}/tsep{tsep}/bT0/bz{z}", data=pt3_cfg.T
                )
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
    assert set(TOOL_CATALOG) == set(STAGE_TOOLS)


def test_terminal_tool_uses_bz_direction_and_removes_variant() -> None:
    parameters = inspect.signature(fit_bare_matrix_grid).parameters
    assert "bz_direction" in parameters
    assert "variant" not in parameters


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


def test_pt3_nonbreit_ratio_fcn_uses_separate_initial_final_spectra() -> None:
    p = gv.BufferDict()
    p["E0_i"] = gv.gvar(0.45, 1e-4)
    p["log(dE1_i)"] = gv.gvar(np.log(0.5), 1e-4)
    p["z0_i"] = gv.gvar(1.0, 1e-4)
    p["z1_i"] = gv.gvar(0.0, 1e-4)
    p["E0_f"] = gv.gvar(0.60, 1e-4)
    p["log(dE1_f)"] = gv.gvar(np.log(0.5), 1e-4)
    p["z0_f"] = gv.gvar(1.2, 1e-4)
    p["z1_f"] = gv.gvar(0.0, 1e-4)
    for snk in range(2):
        for src in range(2):
            p[f"O{snk}{src}_re"] = gv.gvar(0.0, 1e-4)
            p[f"O{snk}{src}_im"] = gv.gvar(0.0, 1e-4)
    p["O00_re"] = gv.gvar(0.42, 1e-4)
    ratio = pt3_nonbreit_ratio_fcn(np.array([8.0]), np.array([4.0]), p, 64, nstate=2, part="re")
    expected = gv.mean(p["O00_re"] / (2 * gv.sqrt(p["E0_i"] * p["E0_f"])))
    assert gv.mean(ratio[0]) == pytest.approx(expected, rel=1e-4)


def test_non_forward_ratio_samples_omits_kinematic_prefactor() -> None:
    n_sample = 2
    tsep = 4
    tau = np.arange(tsep + 1)
    pt2_i = np.full((n_sample, tsep + 1), 2.0 + 0.0j)
    pt2_f = np.full((n_sample, tsep + 1), 8.0 + 0.0j)
    pt3 = np.full((n_sample, tsep + 1), 16.0 + 0.0j)
    re, im = _non_forward_ratio_samples(pt2_i, pt2_f, pt3, tsep)
    expected = (pt3 / pt2_f[:, tsep][:, None]) * np.sqrt(
        (pt2_i[:, tsep - tau] * pt2_f[:, tau] * pt2_f[:, tsep][:, None])
        / (pt2_f[:, tsep - tau] * pt2_i[:, tau] * pt2_i[:, tsep][:, None])
    )
    assert np.allclose(re, expected.real)
    assert np.allclose(im, 0.0)


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
    fit = fit_matrix_element(
        ratio_re,
        ratio_im,
        tsep_ls,
        1,
        32,
        strategy="chained",
        fit_scope="ratio",
        fitting_form="Breit",
        prior=pt3_ratio_prior(2),
    )
    assert abs(gv.mean(fit.p["E0"]) - gv.mean(p_true["E0"])) < 0.05
    assert abs(gv.mean(fit.p["O00_re"]) - gv.mean(p_true["O00_re"])) < 0.05


def test_fh_samples_from_ratios_finite_differences_summed_ratio() -> None:
    ratio_re = {
        4: np.full((2, 5), 0.3),
        6: np.full((2, 7), 0.3),
        8: np.full((2, 9), 0.3),
    }
    ratio_im = {tsep: np.zeros_like(values) for tsep, values in ratio_re.items()}
    fh_re, fh_im = _fh_samples_from_ratios(ratio_re, ratio_im, [4, 6, 8], tau_cut=1)
    assert fh_re.shape == (2, 2)
    assert np.allclose(fh_re, 0.3)
    assert np.allclose(fh_im, 0.0)


def test_fit_fh_one_state_recovers_plateau() -> None:
    tsep_ls = [4, 6, 8]
    ratio_re = {
        tsep: np.asarray([gv.gvar(0.30, 0.01) for _ in range(tsep + 1)], dtype=object)
        for tsep in tsep_ls
    }
    ratio_im = {
        tsep: np.asarray([gv.gvar(0.05, 0.01) for _ in range(tsep + 1)], dtype=object)
        for tsep in tsep_ls
    }
    fit = fit_matrix_element(
        ratio_re,
        ratio_im,
        tsep_ls,
        1,
        32,
        strategy="chained",
        fit_scope="FH",
        fitting_form="Breit",
        nstate=1,
        prior=fh_prior(1),
        svdcut=1e-8,
    )
    assert gv.mean(fit.p["O00_re"] / (2 * fit.p["E0"])) == pytest.approx(0.30, abs=0.03)
    assert gv.mean(fit.p["O00_im"] / (2 * fit.p["E0"])) == pytest.approx(0.05, abs=0.03)


def test_fit_joint_recovers_parameters_and_is_rescale_invariant() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, p_true = _toy_ratio_gv(tsep_ls=tsep_ls, Lt=32)
    scale = 1.0e18
    scaled = fit_matrix_element(
        ratio_re,
        ratio_im,
        tsep_ls,
        1,
        32,
        strategy="joint",
        fit_scope="ratio",
        fitting_form="Breit",
        pt2_gv=_toy_pt2_gv(Lt=32) / scale,
        tmin=2,
        tmax=12,
        prior=pt3_ratio_prior(2),
        svdcut=1e-8,
        rescale=scale,
    )
    unscaled = fit_matrix_element(
        ratio_re,
        ratio_im,
        tsep_ls,
        1,
        32,
        strategy="joint",
        fit_scope="ratio",
        fitting_form="Breit",
        pt2_gv=_toy_pt2_gv(Lt=32),
        tmin=2,
        tmax=12,
        prior=pt3_ratio_prior(2),
        svdcut=1e-8,
    )
    scaled_plateau = gv.mean(scaled.p["O00_re"] / (2 * scaled.p["E0"]))
    unscaled_plateau = gv.mean(unscaled.p["O00_re"] / (2 * unscaled.p["E0"]))
    true_plateau = gv.mean(p_true["O00_re"] / (2 * p_true["E0"]))
    assert scaled_plateau == pytest.approx(unscaled_plateau, rel=0.05)
    assert scaled_plateau == pytest.approx(true_plateau, rel=0.05)


def test_fit_matrix_element_supports_nonbreit_joint_data() -> None:
    Lt = 32
    tsep_ls = [6, 8, 10]
    parameters = {
        "E0_i": 0.40,
        "z0_i": 1.00,
        "E0_f": 0.50,
        "z0_f": 0.90,
        "O00_re": 0.30,
        "O00_im": 0.05,
    }
    time = np.arange(Lt, dtype=float)
    pt2_i = parameters["z0_i"] ** 2 / (2 * parameters["E0_i"]) * (
        np.exp(-parameters["E0_i"] * time)
        + np.exp(-parameters["E0_i"] * (Lt - time))
    )
    pt2_f = parameters["z0_f"] ** 2 / (2 * parameters["E0_f"]) * (
        np.exp(-parameters["E0_f"] * time)
        + np.exp(-parameters["E0_f"] * (Lt - time))
    )
    pt2_i_gv = np.asarray([gv.gvar(value, max(abs(value) * 1e-3, 1e-8)) for value in pt2_i])
    pt2_f_gv = np.asarray([gv.gvar(value, max(abs(value) * 1e-3, 1e-8)) for value in pt2_f])
    ratio_re: dict[int, np.ndarray] = {}
    ratio_im: dict[int, np.ndarray] = {}
    for tsep in tsep_ls:
        tau = np.arange(tsep + 1, dtype=float)
        tsep_array = np.full_like(tau, float(tsep))
        re_mean = pt3_nonbreit_ratio_fcn(
            tsep_array,
            tau,
            parameters,
            Lt,
            nstate=1,
            part="re",
        )
        im_mean = pt3_nonbreit_ratio_fcn(
            tsep_array,
            tau,
            parameters,
            Lt,
            nstate=1,
            part="im",
        )
        ratio_re[tsep] = np.asarray([gv.gvar(value, 1e-3) for value in re_mean])
        ratio_im[tsep] = np.asarray([gv.gvar(value, 1e-3) for value in im_mean])

    prior = pt3_nonbreit_ratio_prior(1)
    for key in ("E0_i", "z0_i", "E0_f", "z0_f", "O00_re", "O00_im"):
        prior[key] = gv.gvar(parameters[key], 0.2)
    fit = fit_matrix_element(
        ratio_re,
        ratio_im,
        tsep_ls,
        1,
        Lt,
        strategy="joint",
        fit_scope="ratio",
        fitting_form="NonBreit",
        pt2_gv=pt2_i_gv,
        pt2_f_gv=pt2_f_gv,
        tmin=2,
        tmax=12,
        nstate=1,
        prior=prior,
        svdcut=1e-8,
    )
    fitted = gv.mean(fit.p["O00_re"] / (fit.p["E0_i"] + fit.p["E0_f"]))
    expected = parameters["O00_re"] / (parameters["E0_i"] + parameters["E0_f"])
    assert fitted == pytest.approx(expected, rel=0.05)


def test_fit_ratio_rejects_empty_tau_window() -> None:
    ratio_re, ratio_im, _ = _toy_ratio_gv(tsep_ls=[4])
    with pytest.raises(ValueError, match="empty tau"):
        fit_matrix_element(
            ratio_re,
            ratio_im,
            [4],
            3,
            32,
            strategy="chained",
            fit_scope="ratio",
            fitting_form="Breit",
            prior=pt3_ratio_prior(2),
        )


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


def test_select_data_window_rejects_low_q_and_underdetermined() -> None:
    records = [
        {"Q": 0.90, "chi2_dof": 0.8, "n_data": 4, "n_params": 4},
        {"Q": 0.01, "chi2_dof": 0.7, "n_data": 20, "n_params": 4},
        {"Q": 0.20, "chi2_dof": 1.0, "n_data": 12, "n_params": 4},
    ]
    index, fallback = select_data_window(records, q_min=0.05)
    assert index == 2
    assert fallback is False


def test_select_data_window_prefers_more_data_when_chi2_close() -> None:
    records = [
        {"Q": 0.20, "chi2_dof": 0.90, "n_data": 10, "n_params": 4},
        {"Q": 0.30, "chi2_dof": 1.05, "n_data": 18, "n_params": 4},
    ]
    index, fallback = select_data_window(records, q_min=0.05, chi2_dof_tolerance=0.25)
    assert index == 1
    assert fallback is False


def test_select_data_window_prefers_clear_chi2_improvement() -> None:
    records = [
        {"Q": 0.20, "chi2_dof": 0.90, "n_data": 10, "n_params": 4},
        {"Q": 0.30, "chi2_dof": 1.40, "n_data": 18, "n_params": 4},
    ]
    index, fallback = select_data_window(records, q_min=0.05, chi2_dof_tolerance=0.25)
    assert index == 0
    assert fallback is False


def test_select_data_window_falls_back_without_q_passing() -> None:
    records = [
        {"Q": 0.01, "chi2_dof": 1.2, "n_data": 10, "n_params": 4},
        {"Q": 0.02, "chi2_dof": 1.0, "n_data": 12, "n_params": 4},
    ]
    index, fallback = select_data_window(records, q_min=0.05)
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


def test_vary_prior_width_scales_template_errors() -> None:
    template = pt3_ratio_prior(nstate=2)
    varied = _vary_prior_width(template, 0.5)
    for key in template:
        assert gv.mean(varied[key]) == pytest.approx(gv.mean(template[key]))
        assert gv.sdev(varied[key]) == pytest.approx(0.5 * gv.sdev(template[key]))


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


def test_bare_matrix_element_mean_zeros_unfit_component() -> None:
    p = {
        "E0": gv.gvar(0.5, 0.01),
        "O00_re": gv.gvar(2.0, 0.1),
        "O00_im": gv.gvar(3.0, 1.0),
    }
    assert _bare_matrix_element_mean_for_part(
        p, output_part="re", fit_part="re", fitting_form="Breit"
    ) == pytest.approx(2.0)
    assert _bare_matrix_element_mean_for_part(
        p, output_part="im", fit_part="re", fitting_form="Breit"
    ) == pytest.approx(0.0)


def test_nonbreit_bare_matrix_element_uses_overlap_sign_convention() -> None:
    p = {
        "E0_i": gv.gvar(0.6, 0.01),
        "E0_f": gv.gvar(0.4, 0.01),
        "z0_i": gv.gvar(-2.0, 0.1),
        "z0_f": gv.gvar(3.0, 0.1),
        "O00_re": gv.gvar(1.5, 0.1),
        "O00_im": gv.gvar(-0.5, 0.1),
    }
    assert _bare_matrix_element_mean_for_part(
        p, output_part="re", fit_part="both", fitting_form="NonBreit"
    ) == pytest.approx(-1.5)
    p["z0_i"] = gv.gvar(2.0, 0.1)
    assert _bare_matrix_element_mean_for_part(
        p, output_part="re", fit_part="both", fitting_form="NonBreit"
    ) == pytest.approx(1.5)


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


def test_normalise_pt3_windows_preserves_explicit_tsep_subsets() -> None:
    windows = _normalise_pt3_windows(
        [{"tsep_ls": [6, 8], "tau_cut": 1}, {"tau_cut": 2}],
        tsep_ls=[6, 8, 10],
        tau_cuts=[3],
    )
    assert windows == [
        {"tsep_ls": [6, 8], "tau_cut": 1},
        {"tsep_ls": [6, 8, 10], "tau_cut": 2},
    ]


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


def test_normalise_fit_scope_aliases() -> None:
    assert _normalise_fit_scope(None) == ("ratio", "ratio")
    assert _normalise_fit_scope("FH") == ("FH", "fh")
    assert _normalise_fit_scope("ratio+FH") == ("ratio+FH", "ratio_fh")
    with pytest.raises(ValueError, match="fit_scope"):
        _normalise_fit_scope("summed")


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


def test_sample_mean_and_sdev_matches_core_helper() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert sample_mean_and_sdev(values, mode="jk")[0] == pytest.approx(2.0)


# --- inspect tool ------------------------------------------------------------


def test_standard_hdf5_reader_selects_shared_momenta_operators_and_tseps(tmp_path: Path) -> None:
    path = tmp_path / "shared.h5"
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=np.full((8, 3), 1.0))
        h5f.create_dataset("g5/g5/PX2PY0PZ0", data=np.full((8, 3), 2.0))
        h5f.create_dataset("g5/g5_nonlocal/PX0PY0PZ0", data=np.full((8, 3), 3.0))
        h5f.create_dataset("g5/g5/gT_nonlocal/PX2PY0PZ0/tsep4/bT0/bz1", data=np.full((5, 3), 4.0))
        h5f.create_dataset("g5/g5/gT_nonlocal/PX2PY0PZ0/tsep6/bT0/bz1", data=np.full((7, 3), 6.0))

    pt2 = _read_2pt(
        str(path), source_operator="g5", sink_operator="g5", momentum="PX2PY0PZ0", temporal_extent=8
    )
    pt3_t4 = _read_3pt(
        str(path), source_operator="g5", sink_operator="g5", current_operator="gT_nonlocal",
        momentum="PX2PY0PZ0", tsep=4, bT=0, bz=1,
    )
    pt3_t6 = _read_3pt(
        str(path), source_operator="g5", sink_operator="g5", current_operator="gT_nonlocal",
        momentum="PX2PY0PZ0", tsep=6, bT=0, bz=1,
    )

    assert pt2.shape == (3, 8) and np.all(pt2 == 2.0)
    assert pt3_t4.shape == (3, 5) and np.all(pt3_t4 == 4.0)
    assert pt3_t6.shape == (3, 7) and np.all(pt3_t6 == 6.0)
    with pytest.raises(ValueError, match="expected 9"):
        _read_2pt(
            str(path), source_operator="g5", sink_operator="g5", momentum="PX0PY0PZ0", temporal_extent=9
        )
    with pytest.raises(KeyError):
        _read_3pt(
            str(path), source_operator="g5", sink_operator="g5", current_operator="gT_nonlocal",
            momentum="PX2PY0PZ0", tsep=8, bT=0, bz=1,
        )


def test_standard_hdf5_reader_rejects_wrong_three_point_tau_extent(tmp_path: Path) -> None:
    path = tmp_path / "bad_3pt.h5"
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("g5/g5/gT_nonlocal/PX0PY0PZ0/tsep4/bT0/bz0", data=np.ones((4, 3)))
    with pytest.raises(ValueError, match="expected 5"):
        _read_3pt(
            str(path), source_operator="g5", sink_operator="g5", current_operator="gT_nonlocal",
            momentum="PX0PY0PZ0", tsep=4, bT=0, bz=0,
        )


def test_inspect_correlator_scale_accepts_selector_momentum(tmp_path) -> None:
    path = tmp_path / "pt2_px5.h5"
    px5_data = np.full((12, 4), 3.0e-18, dtype=np.complex128)
    px0_data = np.full((12, 4), 9.0e-18, dtype=np.complex128)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("g5/g5/PX0PY0PZ0", data=px0_data)
        h5f.create_dataset("g5/g5/PX5PY0PZ0", data=px5_data)

    result = inspect_correlator_scale(
        {},
        pt2_path=str(path),
        pt2_windows=[{"tmin": 2, "tmax": 5}],
        selectors={"source_operator": "g5", "sink_operator": "g5", "momentum": "PX5PY0PZ0"},
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
        tune_z_values=[0],
        z_values=[0],
        pt2_windows=[{"tmin": 2, "tmax": 10}],
        pt3_tau_cuts=[1, 2],
        fit_strategy="joint",
        prior_width=1.0,
        resample_mode="jk",
        svdcut=1e-6,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["candidates"]
    assert "O00_re_over_2E0" in result["candidates"][0]
    assert result["candidates"][0]["prior_width"] == pytest.approx(1.0)
    assert "recommended_index" in result
    assert "recommended_robust_index" in result
    assert "feasible_at_all_tune_z" in result["candidates"][0]
    assert "tune_z_diagnostics" in result["candidates"][0]
    assert result["tune_z_values"] == [0]
    assert result["tuning_diagnostic_pdfs"] == {}
    assert not list((tmp_path / "artifacts").glob("tune_*_sample0_pt3_ratio_*.pdf"))
    assert result["candidates"][0]["fit_scope"] == "ratio"
    assert result["candidates"][0]["n_data"] > result["candidates"][0]["n_params"]
    assert result["candidates"][0]["dof_is_positive"] is True
    assert result["recommended_window"]["n_data"] > result["recommended_window"]["n_params"]


def test_correlator_parallel_sample_fits_match_serial(tmp_path) -> None:
    pt2_path, pt3_paths = _write_fake_correlators(
        tmp_path,
        n_cfg=8,
        tsep_ls=(6, 8),
        z_values=(0,),
    )
    common = {
        "pt2_path": pt2_path,
        "pt3_paths": pt3_paths,
        "tsep_ls": [6, 8],
        "z_values": [0],
        "ensemble": "toy",
        "momentum": "PX0PY0PZ0",
        "bz_direction": "Z",
        "pt2_window": {"tmin": 2, "tmax": 10},
        "pt3_window": {"tsep_ls": [6, 8], "tau_cut": 1},
        "fit_strategy": "joint",
        "fit_scope": "ratio",
        "nstate": 2,
        "prior_width": 1.0,
        "resample_mode": "jk",
        "sample_error_mode": "mean",
        "svdcut": 1e-6,
    }
    serial_store: dict = {}
    parallel_store: dict = {}
    serial = fit_bare_matrix_grid(
        serial_store,
        tag="serial",
        artifacts_dir=tmp_path / "serial",
        save_path=str(tmp_path / "serial" / "bare"),
        workers=1,
        **common,
    )
    parallel = fit_bare_matrix_grid(
        parallel_store,
        tag="parallel",
        artifacts_dir=tmp_path / "parallel",
        save_path=str(tmp_path / "parallel" / "bare"),
        workers=2,
        **common,
    )

    assert serial["workers"] == 1
    assert parallel["workers"] == 2
    assert parallel_store["bare_matrix_element_data"].attrs["workers"] == "2"
    assert parallel_store["bare_matrix_element_data"].attrs["bz_direction"] == "Z"
    assert "variant" not in parallel_store["bare_matrix_element_data"].attrs
    assert np.allclose(
        serial_store["bare_matrix_element_data"].values,
        parallel_store["bare_matrix_element_data"].values,
        equal_nan=True,
    )
    assert serial["z_fits"][0]["n_failed_samples"] == parallel["z_fits"][0]["n_failed_samples"]
    assert np.allclose(serial["z_fits"][0]["fit_model_weights"], parallel["z_fits"][0]["fit_model_weights"])
    assert parallel["z_fits"][0]["sample0_plot_paths"]
    assert all(Path(path).is_file() for path in parallel["z_fits"][0]["sample0_plot_paths"].values())


def test_tune_bare_matrix_requires_tune_z_values(tmp_path) -> None:
    pt2_path, pt3_paths = _write_fake_correlators(tmp_path, tsep_ls=(6, 8), z_values=(0,))
    with pytest.raises(ValueError, match="tune_z_values is required"):
        tune_bare_matrix(
            {},
            pt2_path=pt2_path,
            pt3_paths=pt3_paths,
            tsep_ls=[6, 8],
            momentum="PX0PY0PZ0",
            z_values=[0],
            pt2_windows=[{"tmin": 2, "tmax": 10}],
            pt3_tau_cuts=[1],
            resample_mode="jk",
        )


def test_tune_bare_matrix_rejects_invalid_tune_z(tmp_path) -> None:
    pt2_path, pt3_paths = _write_fake_correlators(tmp_path, tsep_ls=(6, 8), z_values=(0, 1))
    with pytest.raises(ValueError, match="not in the job z grid"):
        tune_bare_matrix(
            {},
            pt2_path=pt2_path,
            pt3_paths=pt3_paths,
            tsep_ls=[6, 8],
            momentum="PX0PY0PZ0",
            tune_z_values=[99],
            z_values=[0, 1],
            pt2_windows=[{"tmin": 2, "tmax": 10}],
            pt3_tau_cuts=[1],
            resample_mode="jk",
        )


def test_summarise_cross_z_feasibility_tracks_failures() -> None:
    per_z = {
        0: {"usable": True, "Q": 0.99, "chi2_dof": 0.5},
        9: {"usable": False, "reason": "non-physical E0"},
    }
    summary = _summarise_cross_z_feasibility(per_z, [0, 9])
    assert summary["feasible_at_all_tune_z"] is False
    assert summary["failure_reasons"] == {"9": "non-physical E0"}
    assert summary["bottleneck_z"] == 9


def test_window_candidate_key_is_stable() -> None:
    meta = {
        "fit_strategy": "joint",
        "fit_scope": "ratio",
        "nstate": 2,
        "prior_width": 1.0,
        "tmin": 3,
        "tmax": 12,
        "tsep_ls": [8, 10, 12],
        "tau_cut": 3,
    }
    assert _window_candidate_key(meta) == _window_candidate_key(dict(meta))


# --- plotting helpers retained ----------------------------------------------


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


def test_ylim_middle_third_fit_log_factors_place_data_at_three_to_seven_twelfths() -> None:
    y_lo, y_hi = _ylim_middle_third(
        [np.array([1.0, 2.0])],
        [np.array([0.1, 0.1])],
        bottom_margin_factor=FIT_LOG_YLIM_BOTTOM_FACTOR,
        top_margin_factor=FIT_LOG_YLIM_TOP_FACTOR,
    )
    height = y_hi - y_lo
    assert FIT_LOG_YLIM_BOTTOM_FACTOR == pytest.approx(0.75)
    assert FIT_LOG_YLIM_TOP_FACTOR == pytest.approx(1.25)
    assert np.isclose(y_lo + height * FIT_LOG_YLIM_DATA_LOW_NUM / 12, 0.9, rtol=1e-9)
    assert np.isclose(y_lo + height * FIT_LOG_YLIM_DATA_HIGH_NUM / 12, 2.1, rtol=1e-9)


def test_plot_pt2_meff_on_data_respects_t_max(tmp_path) -> None:
    pt2_gv = _toy_pt2_gv()
    t_band = np.arange(2, 10, dtype=int)
    save = tmp_path / "meff_only"
    _, ax = plot_pt2_meff_on_data(pt2_gv, fit_bands=[{"fit_t": t_band, "fit_gv": pt2_gv[t_band], "label": "w1"}], t_max=6, save_path=save)
    assert ax.get_xlim()[0] == pytest.approx(-0.5)
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
    assert ax_meff.get_xlim()[0] == pytest.approx(0.5)
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
