"""Unit tests for correlator stage tools and plotting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gvar as gv
import numpy as np
import pytest
from matplotlib.legend import Legend

pytest.importorskip("lsqfit")

import lamet_agent.stages.correlator.functions as correlator_functions
from lamet_agent.stages.correlator.prompts import STAGE_PROMPT
from lamet_agent.stages.correlator.skills import tool_catalog
from lamet_agent.core.plotting import (
    _pt3_ratio_data_tau_slice,
    _ratio_denominator_correction,
    _ylim_middle_third,
    plot_pt2_fit_on_data,
    plot_pt2_meff_on_data,
)
from lamet_agent.core.resampling import sample_mean_err as core_sample_mean_err
from lamet_agent.core.tools import log_nonlinear_fit_quality, prepare_tool_args, setup_logger
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.stages.correlator.functions import (
    MAX_FIT_WINDOWS,
    MAX_PT3_FIT_WINDOWS,
    PT2_PRIOR_ERROR_SCALE,
    STAGE_TOOLS,
    _bare_matrix_samples_from_records,
    _normalise_pt2_windows,
    _scaled_posterior_as_prior,
    _fit_posterior_is_usable,
    _fit_summary,
    _resample_config_samples,
    _sample_mean_err,
    _select_best_fit_index,
    _pt2_posterior_as_prior,
    _write_bare_matrix_grid_outputs,
    asymptotic_ratio_real_gvar,
    compute_pt3_ratio,
    inspect_correlator_scale,
    read_pt3,
    fit_pt3_window,
    fit_window,
    jackknife,
    model_average,
    plot_fit_on_data,
    plot_pt3_fit_on_data,
    pt2_ratio_joint_fit,
    pt2_re_fcn,
    pt3_ratio_fit,
    pt3_ratio_im_fcn,
    pt3_ratio_prior,
    pt3_ratio_prior_from_pt2_avg,
    pt3_ratio_re_fcn,
)


def _toy_pt2_gv(*, Lt: int = 24, E0: float = 0.45) -> np.ndarray:
    t = np.arange(Lt, dtype=int)
    p = gv.BufferDict()
    p["E0"] = gv.gvar(E0, 1e-4)
    p["log(dE1)"] = gv.gvar(np.log(0.5), 1e-3)
    p["z0"] = gv.gvar(1.0, 1e-4)
    p["z1"] = gv.gvar(0.05, 1e-4)
    return pt2_re_fcn(t, p, Lt, nstate=2)


def test_stage_tools_excludes_scan_tmin() -> None:
    assert "fit_window" in STAGE_TOOLS
    assert "fit_pt3_window" in STAGE_TOOLS
    assert "read_pt3" in STAGE_TOOLS
    assert "scan_tmin" not in STAGE_TOOLS


def _toy_pt3_ratio_gv(*, tsep_ls: list[int], Lt: int = 32) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], gv.BufferDict]:
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
        ratio_re[tsep] = pt3_ratio_re_fcn(t_arr, tau, p, Lt, nstate=2)
        ratio_im[tsep] = pt3_ratio_im_fcn(t_arr, tau, p, Lt, nstate=2)
    return ratio_re, ratio_im, p


def test_fit_pt3_window_and_model_average() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, p_true = _toy_pt3_ratio_gv(tsep_ls=tsep_ls)
    store = {"ratio_real_gv": ratio_re, "ratio_imag_gv": ratio_im}
    result = fit_pt3_window(
        store,
        tsep_ls=tsep_ls,
        tau_cut=1,
        Lt=32,
        append=False,
        use_pt2_avg_prior=False,
    )
    assert result["Q"] > 0
    fit = store["pt3_scan"][0]["fit"]
    assert abs(gv.mean(fit.p["O00_re"]) - gv.mean(p_true["O00_re"])) < 0.05
    avg = model_average(
        store,
        scan="pt3_scan",
        param="O00_re",
        window_indices=[0],
        out="O00_re_avg",
    )
    assert avg["n_windows"] == 1


def test_pt2_posterior_as_prior_inflates_error() -> None:
    posterior = gv.gvar(0.45, 0.01)
    prior = _pt2_posterior_as_prior(posterior)
    assert gv.mean(prior) == gv.mean(posterior)
    assert gv.sdev(prior) == pytest.approx(PT2_PRIOR_ERROR_SCALE * gv.sdev(posterior))


def test_pt3_ratio_prior_from_pt2_avg_widens_only_E0_z0() -> None:
    store = {
        "E0_avg": gv.gvar(0.45, 0.01),
        "log(dE1)_avg": gv.gvar(np.log(0.5), 0.02),
        "z0_avg": gv.gvar(1.0, 0.03),
        "z1_avg": gv.gvar(0.05, 0.04),
    }
    prior = pt3_ratio_prior_from_pt2_avg(store)
    for key in ("E0", "z0"):
        assert gv.mean(prior[key]) == gv.mean(store[f"{key}_avg"])
        assert gv.sdev(prior[key]) == pytest.approx(
            PT2_PRIOR_ERROR_SCALE * gv.sdev(store[f"{key}_avg"])
        )
    broad = pt3_ratio_prior(nstate=2)
    assert gv.sdev(prior["log(dE1)"]) == gv.sdev(broad["log(dE1)"])
    assert gv.sdev(prior["z1"]) == gv.sdev(broad["z1"])


def test_fit_pt3_window_uses_pt2_avg_prior(tmp_path) -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    Lt = 24
    for tmin in (2, 3, 4):
        fit_window(store, tmin=tmin, tmax=12, Lt=Lt, append=True)
    for param in ("E0", "log(dE1)", "z0", "z1"):
        model_average(store, scan="scan", param=param, window_indices=[0, 1, 2])

    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=[6, 8, 10])
    store["ratio_real_gv"] = ratio_re
    store["ratio_imag_gv"] = ratio_im

    result = fit_pt3_window(
        store,
        tsep_ls=[6, 8, 10],
        tau_cut=1,
        Lt=32,
        append=False,
    )
    assert set(result["pt2_prior_from"]) == {"E0_avg", "z0_avg"}
    fit = store["pt3_scan"][0]["fit"]
    assert gv.sdev(fit.p["O00_re"]) < 0.5
    assert abs(gv.mean(fit.p["E0"]) - gv.mean(store["E0_avg"])) < 0.05


def _store_with_pt2_avg_for_pt3(
    ratio_re: dict[int, np.ndarray],
    ratio_im: dict[int, np.ndarray],
) -> dict:
    store: dict = {"ratio_real_gv": ratio_re, "ratio_imag_gv": ratio_im, "pt2_gv": _toy_pt2_gv()}
    for tmin in (2, 3):
        fit_window(store, tmin=tmin, tmax=12, Lt=24, append=True)
    for param in ("E0", "log(dE1)", "z0", "z1"):
        model_average(store, scan="scan", param=param, window_indices=[0, 1])
    return store


def test_fit_pt3_window_requires_pt2_scan_for_priors() -> None:
    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=[6, 8, 10])
    store = {"ratio_real_gv": ratio_re, "ratio_imag_gv": ratio_im, "Lt": 32}
    with pytest.raises(ValueError, match="scan"):
        fit_pt3_window(store, tsep_ls=[6, 8, 10], tau_cut=1, append=False)


def test_fit_pt3_window_rejects_insufficient_points() -> None:
    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=[8])
    store = _store_with_pt2_avg_for_pt3(ratio_re, ratio_im)
    with pytest.raises(ValueError, match="data points"):
        fit_pt3_window(store, tsep_ls=[8], tau_cut=3, Lt=32, append=False)


def test_fit_pt3_window_rejects_third_append() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=tsep_ls)
    store = _store_with_pt2_avg_for_pt3(ratio_re, ratio_im)
    for tau_cut in (1, 2):
        fit_pt3_window(
            store,
            tsep_ls=tsep_ls,
            tau_cut=tau_cut,
            Lt=32,
            append=True,
        )
    assert len(store["pt3_scan"]) == MAX_PT3_FIT_WINDOWS
    with pytest.raises(ValueError, match=str(MAX_PT3_FIT_WINDOWS)):
        fit_pt3_window(store, tsep_ls=[8], tau_cut=1, Lt=32, append=True)


def test_fit_pt3_window_rejects_tau_cut_zero() -> None:
    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=[8])
    store = _store_with_pt2_avg_for_pt3(ratio_re, ratio_im)
    with pytest.raises(ValueError, match="tau_cut must be >= 1"):
        fit_pt3_window(store, tsep_ls=[8], tau_cut=0, Lt=32, append=False)


def test_fit_pt3_window_rejects_empty_tau() -> None:
    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=[4])
    store = _store_with_pt2_avg_for_pt3(ratio_re, ratio_im)
    with pytest.raises(ValueError, match="empty tau"):
        fit_pt3_window(store, tsep_ls=[4], tau_cut=3, Lt=32, append=False)


def test_compute_pt3_ratio_from_samples() -> None:
    n_cfg, lt = 8, 32
    tsep = 8
    pt2_re = np.ones((n_cfg, lt))
    pt2_im = np.zeros((n_cfg, lt))
    pt3_re = {tsep: np.full((n_cfg, tsep + 1), 0.5)}
    pt3_im = {tsep: np.zeros((n_cfg, tsep + 1))}
    store = {
        "pt2_samples": pt2_re,
        "pt2_imag_samples": pt2_im,
        "pt3_samples_re": pt3_re,
        "pt3_samples_im": pt3_im,
    }
    out = compute_pt3_ratio(store)
    assert out["tsep_keys"] == [tsep]
    assert np.allclose(store["ratio_samples_re"][tsep], 0.5)


def test_read_pt2_pt3_share_bootstrap_indices(tmp_path) -> None:
    import h5py

    from lamet_agent.stages.correlator.functions import read_pt2, resample_ratio_to_gvar

    n_cfg, lt = 6, 10
    tsep = 4
    pt2_path = tmp_path / "pt2.h5"
    pt3_path = tmp_path / "pt3.h5"
    pt2_data = np.arange(n_cfg * lt, dtype=float).reshape(lt, n_cfg) + 1.0
    pt3_data = np.arange(n_cfg * (tsep + 1), dtype=float).reshape(tsep + 1, n_cfg) + 2.0
    with h5py.File(pt2_path, "w") as h5f:
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=pt2_data)
    with h5py.File(pt3_path, "w") as h5f:
        h5f.create_dataset("SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0", data=pt3_data)

    store: dict = {}
    read_pt2(store, path=str(pt2_path), resample_mode="bs", n_boot=20, seed=123)
    read_pt3(store, path=str(pt3_path), append=False, resample_mode="bs", n_boot=20, seed=999)
    compute_pt3_ratio(store)
    resample_ratio_to_gvar(store)

    assert store["pt2_samples"].shape[0] == 20
    assert store["pt3_samples_re"][tsep].shape[0] == 20
    assert store["resample_indices"] is not None
    assert store["ratio_real_gv"][tsep].shape == (tsep + 1,)


def test_resample_ratio_to_gvar_rejects_config_level_ratio() -> None:
    from lamet_agent.stages.correlator.functions import resample_ratio_to_gvar

    store = {
        "n_cfg": 4,
        "ratio_samples_re": {8: np.ones((4, 9))},
        "ratio_samples_im": {8: np.zeros((4, 9))},
    }
    with pytest.raises(ValueError, match="configuration-level"):
        resample_ratio_to_gvar(store)


def test_ratio_after_resample_differs_from_resample_after_ratio() -> None:
    rng = np.random.default_rng(0)
    n_cfg, lt, tsep = 40, 24, 8
    pt2 = rng.normal(size=(n_cfg, lt)) + 2.0
    pt3 = rng.normal(size=(n_cfg, tsep + 1)) + 1.0
    pt2_complex = pt2.astype(complex)
    pt2_samples, pt2_complex_samples, indices = correlator_functions._resample_pt2_complex(
        pt2_complex,
        mode="bs",
        n_boot=200,
        seed=1984,
    )
    pt3_samples, _ = _resample_config_samples(
        pt3,
        mode="bs",
        n_boot=200,
        seed=1984,
        indices=indices,
    )
    ratio_first, _ = correlator_functions._ratio_samples_from_resampled(
        pt2_complex_samples, pt3_samples, tsep
    )
    wrong_ratio, _ = _resample_config_samples(
        pt3 / pt2[:, tsep][:, None],
        mode="bs",
        n_boot=200,
        seed=1984,
        indices=indices,
    )
    assert not np.allclose(ratio_first, wrong_ratio)


def test_ylim_middle_third_places_data_in_center_band() -> None:
    y = [np.array([1.0, 2.0])]
    err = [np.array([0.1, 0.1])]
    y_lo, y_hi = _ylim_middle_third(y, err)
    data_min, data_max = 0.9, 2.1
    height = y_hi - y_lo
    assert np.isclose(y_lo + height / 3, data_min, rtol=1e-9)
    assert np.isclose(y_lo + 2 * height / 3, data_max, rtol=1e-9)


def test_asymptotic_ratio_differs_from_O00_by_two_E0() -> None:
    O00 = gv.gvar(0.30, 0.01)
    E0 = gv.gvar(0.45, 0.01)
    plat = asymptotic_ratio_real_gvar(O00, E0, tsep=10, Lt=32)
    assert abs(gv.mean(plat) - gv.mean(O00) / (2 * gv.mean(E0))) < 0.002


def test_ratio_denominator_correction_converts_periodic_to_forward() -> None:
    E0 = gv.gvar(0.09, 0.001)
    correction = _ratio_denominator_correction(12, energy=E0, Lt=64)
    expected = 1.0 + gv.exp(-E0 * (64 - 2 * 12))
    assert gv.mean(correction) == pytest.approx(gv.mean(expected))


def test_pt3_ratio_data_tau_slice_includes_tsep_minus_one() -> None:
    row = np.arange(12)
    sl = _pt3_ratio_data_tau_slice(10)
    assert list(row[sl]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_read_pt3_ignores_legacy_out_kwarg(tmp_path) -> None:
    """Agent sometimes passes out=; tool should not raise."""
    import h5py

    path = tmp_path / "fake.h5"
    data = np.ones((4, 5), dtype=float)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset(
            "SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0",
            data=data.T,
        )
    store: dict = {}
    read_pt3(store, path=str(path), append=False, out="pt3_samples")
    assert "pt3_samples_re" in store


def test_fit_pt3_window_autofills_missing_z0_avg() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=tsep_ls)
    store: dict = {"ratio_real_gv": ratio_re, "ratio_imag_gv": ratio_im, "pt2_gv": _toy_pt2_gv(), "Lt": 32}
    for tmin in (2, 3):
        fit_window(store, tmin=tmin, tmax=12, Lt=32, append=True)
    model_average(store, scan="scan", param="E0", window_indices=[0, 1], out="E0_avg")
    assert "z0_avg" not in store
    result = fit_pt3_window(store, tsep_ls=tsep_ls, tau_cut=1, append=False)
    assert "z0_avg" in store
    assert "z0_avg" in result.get("pt2_prior_autofill", [])
    assert "log(dE1)_avg" not in store


def test_fit_pt3_window_infers_Lt_from_store() -> None:
    tsep_ls = [8]
    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=tsep_ls)
    store = _store_with_pt2_avg_for_pt3(ratio_re, ratio_im)
    store["Lt"] = 32
    fit_pt3_window(store, tsep_ls=tsep_ls, tau_cut=1, append=False)


def test_plot_pt3_fit_on_data_writes_pdfs(tmp_path) -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, _ = _toy_pt3_ratio_gv(tsep_ls=tsep_ls)
    store = _store_with_pt2_avg_for_pt3(ratio_re, ratio_im)
    fit_pt3_window(store, tsep_ls=tsep_ls, tau_cut=1, Lt=32, append=False)
    for param, out_key in (("O00_re", "O00_re_avg"), ("O00_im", "O00_im_avg")):
        model_average(
            store,
            scan="pt3_scan",
            param=param,
            window_indices=[0],
            out=out_key,
        )
    artifacts = tmp_path / "artifacts"
    result = plot_pt3_fit_on_data(
        store,
        Lt=32,
        window_indices=[0],
        artifacts_dir=artifacts,
        save_path=str(tmp_path / "pt3_fit"),
    )
    assert (artifacts / "pt3_fit_pt3_ratio_re.pdf").is_file()
    assert (artifacts / "pt3_fit_pt3_ratio_im.pdf").is_file()
    assert result["n_windows"] == 1


def test_pt3_ratio_fit_recovers_parameters() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, p_true = _toy_pt3_ratio_gv(tsep_ls=tsep_ls)
    fit = pt3_ratio_fit(tsep_ls, tau_cut=1, ratio_real=ratio_re, ratio_imag=ratio_im, Lt=32)
    assert abs(gv.mean(fit.p["E0"]) - gv.mean(p_true["E0"])) < 0.05
    assert abs(gv.mean(fit.p["O00_re"]) - gv.mean(p_true["O00_re"])) < 0.05


def test_inspect_correlator_scale_accepts_selector_momentum(tmp_path) -> None:
    import h5py

    path = tmp_path / "pt2_px5.h5"
    px5_data = np.full((12, 4), 3.0e-18, dtype=np.complex128)
    px0_data = np.full((12, 4), 9.0e-18, dtype=np.complex128)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("SS/5/PX0PY0PZ0", data=px0_data)
        h5f.create_dataset("SS/5/PX5PY0PZ0", data=px5_data)

    store: dict = {}
    result = inspect_correlator_scale(
        store,
        pt2_path=str(path),
        pt2_windows=[{"tmin": 2, "tmax": 5}],
        selectors={"source_sink": "SS", "gamma": "5", "momentum": "PX5PY0PZ0"},
    )

    assert result["momentum"] == "PX5PY0PZ0"
    assert result["windows"][0]["median_abs"] == pytest.approx(3.0e-18)


def test_fit_window_correlator_rescale_recovers_tiny_pt2() -> None:
    scale = 1.0e18
    store: dict = {"pt2_gv": _toy_pt2_gv() / scale}
    result = fit_window(
        store,
        tmin=2,
        tmax=10,
        Lt=24,
        append=False,
        correlator_rescale=scale,
    )
    fit = store["scan"][0]["fit"]
    assert result["correlator_rescale"] == pytest.approx(scale)
    assert abs(gv.mean(fit.p["E0"]) - 0.45) < 0.05
    diag = correlator_functions._physical_overlap_diagnostics(fit.p, 2, scale)
    assert gv.mean(diag["z0_physical"]) == pytest.approx(1.0e-9, rel=0.1)


def test_joint_ratio_fit_rescale_preserves_o00_plateau() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, p_true = _toy_pt3_ratio_gv(tsep_ls=tsep_ls, Lt=32)
    scale = 1.0e18
    fit_scaled = pt2_ratio_joint_fit(
        _toy_pt2_gv(Lt=32, E0=0.45) / scale,
        tmin=2,
        tmax=12,
        ratio_real=ratio_re,
        ratio_imag=ratio_im,
        tsep_ls=tsep_ls,
        tau_cut=1,
        Lt=32,
        svdcut=1e-8,
        correlator_rescale=scale,
    )
    fit_unscaled = pt2_ratio_joint_fit(
        _toy_pt2_gv(Lt=32, E0=0.45),
        tmin=2,
        tmax=12,
        ratio_real=ratio_re,
        ratio_imag=ratio_im,
        tsep_ls=tsep_ls,
        tau_cut=1,
        Lt=32,
        svdcut=1e-8,
    )
    scaled_plateau = gv.mean(fit_scaled.p["O00_re"] / (2 * fit_scaled.p["E0"]))
    unscaled_plateau = gv.mean(fit_unscaled.p["O00_re"] / (2 * fit_unscaled.p["E0"]))
    true_plateau = gv.mean(p_true["O00_re"] / (2 * p_true["E0"]))
    assert scaled_plateau == pytest.approx(unscaled_plateau, rel=0.05)
    assert scaled_plateau == pytest.approx(true_plateau, rel=0.05)


def test_fit_window_append_and_index() -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    Lt = 24
    r1 = fit_window(store, tmin=2, tmax=10, Lt=Lt, append=True)
    r2 = fit_window(store, tmin=3, tmax=11, Lt=Lt, append=True)
    assert r1["index"] == 0
    assert r2["index"] == 1
    assert len(store["scan"]) == 2


def test_fit_window_rejects_past_half() -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    with pytest.raises(ValueError, match="first half"):
        fit_window(store, tmin=2, tmax=20, Lt=24, append=False)


def test_fit_window_rejects_tmin_zero() -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    with pytest.raises(ValueError, match="tmin must be >= 1"):
        fit_window(store, tmin=0, tmax=10, Lt=24, append=False)


def test_fit_window_rejects_insufficient_points() -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv(Lt=24)}
    with pytest.raises(ValueError, match="data points"):
        fit_window(store, tmin=10, tmax=11, Lt=24, nstate=2, append=False)


def test_fit_window_rejects_seventh_append() -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    Lt = 24
    for tmin in range(1, MAX_FIT_WINDOWS + 1):
        fit_window(store, tmin=tmin, tmax=12, Lt=Lt, append=True)
    assert len(store["scan"]) == MAX_FIT_WINDOWS
    with pytest.raises(ValueError, match=str(MAX_FIT_WINDOWS)):
        fit_window(store, tmin=6, tmax=12, Lt=Lt, append=True)


def test_fit_window_six_windows_model_average_and_plot(tmp_path) -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    Lt = 24
    for tmin in range(1, MAX_FIT_WINDOWS + 1):
        fit_window(store, tmin=tmin, tmax=12, Lt=Lt, append=True)
    avg = model_average(store, param="E0", window_indices=list(range(MAX_FIT_WINDOWS)))
    assert avg["n_windows"] == MAX_FIT_WINDOWS
    artifacts = tmp_path / "artifacts"
    plot = plot_fit_on_data(
        store,
        Lt=Lt,
        window_indices=[0, 1],
        E0_avg="E0_avg",
        artifacts_dir=artifacts,
    )
    assert plot["n_bands"] == 2


def test_model_average_subset_indices() -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    Lt = 24
    for tmin in (2, 3, 4):
        fit_window(store, tmin=tmin, tmax=10, Lt=Lt, append=True)
    full = model_average(store, param="E0", window_indices=None)
    subset = model_average(store, param="E0", window_indices=[0, 2], out="E0_subset")
    assert full["n_windows"] == 3
    assert subset["window_indices"] == [0, 2]
    assert subset["n_windows"] == 2
    assert full["mean"] != subset["mean"] or full["sdev"] != subset["sdev"]


def test_plot_fit_on_data_writes_pdfs(tmp_path) -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    Lt = 24
    fit_window(store, tmin=2, tmax=10, Lt=Lt, append=False)
    fit_window(store, tmin=3, tmax=11, Lt=Lt, append=True)
    model_average(store, param="E0", window_indices=[0, 1])
    artifacts = tmp_path / "artifacts"
    result = plot_fit_on_data(
        store,
        Lt=Lt,
        window_indices=[0, 1],
        E0_avg="E0_avg",
        save_path=str(tmp_path / "fit"),
        artifacts_dir=artifacts,
    )
    assert (artifacts / "fit_c2pt.pdf").is_file()
    assert (artifacts / "fit_meff.pdf").is_file()
    assert result["n_bands"] == 2


def test_plot_fit_on_data_rewrites_outside_path(tmp_path) -> None:
    store: dict = {"pt2_gv": _toy_pt2_gv()}
    Lt = 24
    fit_window(store, tmin=2, tmax=10, Lt=Lt, append=False)
    artifacts = tmp_path / "artifacts"
    result = plot_fit_on_data(
        store,
        Lt=Lt,
        window_indices=[0],
        save_path="/tmp/elsewhere/custom.png",
        artifacts_dir=artifacts,
    )
    assert (artifacts / "custom_c2pt.pdf").is_file()
    assert result["c2pt_pdf"].startswith(str(artifacts))


def test_plot_pt2_meff_on_data_respects_t_max(tmp_path) -> None:
    pt2_gv = _toy_pt2_gv()
    t_band = np.arange(2, 10, dtype=int)
    band_gv = pt2_gv[t_band]
    save = tmp_path / "meff_only"
    _, ax = plot_pt2_meff_on_data(
        pt2_gv,
        fit_bands=[{"fit_t": t_band, "fit_gv": band_gv, "label": "w1"}],
        t_max=6,
        save_path=save,
    )
    assert ax.get_xlim()[1] == pytest.approx(6.0)
    assert (tmp_path / "meff_only_meff.pdf").is_file()
    assert not (tmp_path / "meff_only_c2pt.pdf").exists()


def test_plot_pt2_legend_upper_right() -> None:
    pt2_gv = _toy_pt2_gv()
    t_band = np.arange(2, 10, dtype=int)
    band_gv = pt2_gv[t_band]
    (_, ax_c2), (_, ax_meff) = plot_pt2_fit_on_data(
        pt2_gv,
        fit_bands=[{"fit_t": t_band, "fit_gv": band_gv, "label": "w1"}],
    )
    upper_right = Legend.codes["upper right"]
    assert ax_c2.get_legend()._loc == upper_right
    assert ax_meff.get_legend()._loc == upper_right


def test_plot_pt2_multi_band_and_e0_hspan() -> None:
    pt2_gv = _toy_pt2_gv()
    t_band = np.arange(2, 10, dtype=int)
    band_gv = pt2_gv[t_band]
    e0 = gv.gvar(0.45, 0.01)
    (_, ax_c2), (_, ax_meff) = plot_pt2_fit_on_data(
        pt2_gv,
        fit_bands=[
            {"fit_t": t_band, "fit_gv": band_gv, "label": "w1", "color": "#4E79A7"},
            {"fit_t": t_band + 1, "fit_gv": band_gv, "label": "w2", "color": "#E69F00"},
        ],
        E0_band=e0,
    )
    assert len(ax_c2.collections) + len(ax_c2.patches) >= 2
    assert any(
        hasattr(artist, "get_facecolor") or hasattr(artist, "get_xy")
        for artist in list(ax_meff.patches) + list(ax_meff.collections)
    )

def test_select_best_fit_index_prefers_loggbf_after_q_cut() -> None:
    records = [
        {"Q": 0.20, "logGBF": 1.0},
        {"Q": 0.06, "logGBF": 3.0},
        {"Q": 0.90, "logGBF": 2.0},
    ]
    index, fallback = _select_best_fit_index(records, q_min=0.05)
    assert index == 1
    assert fallback is False


def test_select_best_fit_index_falls_back_to_max_q() -> None:
    records = [
        {"Q": 0.01, "logGBF": 10.0},
        {"Q": 0.04, "logGBF": 1.0},
    ]
    index, fallback = _select_best_fit_index(records, q_min=0.05)
    assert index == 1
    assert fallback is True


def test_bare_matrix_samples_use_o00_over_two_e0() -> None:
    p = gv.BufferDict()
    p["E0"] = gv.gvar(0.5, 0.01)
    p["O00_re"] = gv.gvar(2.0, 0.1)
    p["O00_im"] = gv.gvar(-1.0, 0.1)

    class Fit:
        pass

    fit = Fit()
    fit.p = p
    real, imag = _bare_matrix_samples_from_records([{"fit": fit}])
    assert real[0] == pytest.approx(2.0)
    assert imag[0] == pytest.approx(-1.0)


def test_fit_summary_reports_physical_overlap_rescale() -> None:
    p = gv.BufferDict()
    p["E0"] = gv.gvar(0.5, 0.01)
    p["z0"] = gv.gvar(1.0, 0.01)
    p["z1"] = gv.gvar(2.0, 0.02)

    class Fit:
        pass

    fit = Fit()
    fit.p = p
    summary = _fit_summary(
        {
            "fit": fit,
            "nstate": 2,
            "chi2_dof": 1.0,
            "Q": 0.5,
            "logGBF": 3.0,
            "correlator_rescale": 1.0e18,
        },
        fallback=False,
        index=0,
    )
    assert summary["correlator_rescale"] == pytest.approx(1.0e18)
    assert summary["overlap_rescale"] == pytest.approx(1.0e9)
    assert "z0_physical" in summary
    assert "z1_over_z0_physical" in summary


def test_write_sample0_ratio_plot_uses_momentum_and_o00_band(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_plot(ratio_real, ratio_imag, **kwargs):
        captured["ratio_real"] = ratio_real
        captured["ratio_imag"] = ratio_imag
        captured.update(kwargs)
        return []

    monkeypatch.setattr(correlator_functions, "plot_pt3_ratio_fit_on_data", fake_plot)

    p = gv.BufferDict()
    p["E0"] = gv.gvar(0.5, 0.01)
    p["z0"] = gv.gvar(1.0, 0.01)
    p["O00_re"] = gv.gvar(2.0, 0.1)
    p["O00_im"] = gv.gvar(-1.0, 0.1)

    class Fit:
        pass

    fit = Fit()
    fit.p = p
    ratio = {6: np.ones(7)}
    result = correlator_functions._write_sample0_ratio_plot(
        ratio_real_sample=ratio,
        ratio_imag_sample=ratio,
        fit_record={
            "fit": fit,
            "tmin": 2,
            "tmax": 8,
            "tsep_ls": [6],
            "tau_cut": 1,
            "nstate": 1,
        },
        Lt=32,
        log_dir=tmp_path,
        momentum="PX5PY0PZ0",
        z=5,
    )

    assert result["ratio_re_pdf"].endswith("joint_fit_PX5PY0PZ0_z5_sample0_pt3_ratio_re.pdf")
    assert result["ratio_im_pdf"].endswith("joint_fit_PX5PY0PZ0_z5_sample0_pt3_ratio_im.pdf")
    assert Path(captured["save_path"]).name == "joint_fit_PX5PY0PZ0_z5_sample0"
    assert gv.mean(captured["plateau_ref_re"]) == pytest.approx(2.0)
    assert gv.mean(captured["plateau_ref_im"]) == pytest.approx(-1.0)
    assert captured["denominator_correction_energy"] is p["E0"]
    assert captured["denominator_correction_Lt"] == 32
    assert captured["plateau_label"] == r"Sample-0 fit $\mathcal{O}_{00}/(2E_0)$"


def test_write_sample0_pt2_plot_uses_momentum_and_e0_band(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_plot(pt2_gv, **kwargs):
        captured["pt2_gv"] = pt2_gv
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(correlator_functions, "plot_pt2_meff_on_data", fake_plot)
    monkeypatch.setattr(correlator_functions.plt, "close", lambda _fig: None)

    p = gv.BufferDict()
    p["E0"] = gv.gvar(0.5, 0.01)
    p["z0"] = gv.gvar(1.0, 0.01)
    p["log(dE1)"] = gv.gvar(-1.0, 0.1)
    p["z1"] = gv.gvar(0.2, 0.05)

    class Fit:
        pass

    fit = Fit()
    fit.p = p
    pt2 = _toy_pt2_gv()
    result = correlator_functions._write_sample0_pt2_plot(
        pt2_sample=pt2,
        fit_record={
            "fit": fit,
            "tmin": 2,
            "tmax": 10,
            "nstate": 2,
            "correlator_rescale": 1.0,
        },
        Lt=24,
        log_dir=tmp_path,
        momentum="PX5PY0PZ0",
    )

    assert "c2pt_pdf" not in result
    assert result["meff_pdf"].endswith("chained_fit_PX5PY0PZ0_sample0_meff.pdf")
    assert Path(captured["save_path"]).name == "chained_fit_PX5PY0PZ0_sample0"
    assert captured["E0_band"] is p["E0"]
    assert captured["E0_label"] == r"Sample-0 fit $E_0$"
    assert captured["boundary"] == "none"
    assert captured["t_max"] == 6
    assert len(captured["fit_bands"]) == 1


def test_write_bare_matrix_grid_outputs_writes_txt_plot_and_report(tmp_path) -> None:
    records = [
        {
            "z": 0,
            "real_samples": np.array([1.0, 1.1, 0.9]),
            "imag_samples": np.array([0.0, 0.1, -0.1]),
            "pt3_window": {"tau_cut": 1, "tsep_ls": [8, 10, 12]},
            "sample0_plot_paths": {"ratio_re_pdf": "re.pdf", "ratio_im_pdf": "im.pdf"},
        },
        {
            "z": 1,
            "real_samples": np.array([0.8, 0.9, 0.7]),
            "imag_samples": np.array([0.2, 0.3, 0.1]),
            "pt3_window": {"tau_cut": 2, "tsep_ls": [8, 10, 12]},
        },
    ]
    result = _write_bare_matrix_grid_outputs(
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
    )
    txt0 = tmp_path / "bare_qpdf" / "HISQa060_X_CG52bxp00_CG52bxp00_free_X_PX0PY0PZ0_b0_z0.txt"
    assert txt0.is_file()
    loaded = np.loadtxt(txt0)
    assert loaded.shape == (3, 2)
    assert np.allclose(loaded[:, 0], records[0]["real_samples"])
    assert np.allclose(loaded[:, 1], records[0]["imag_samples"])
    assert (tmp_path / "bare.pdf").is_file()
    report_path = tmp_path / "bare_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["resample_mode"] == "jk"
    assert report["plot_ylim"] == [-0.2, 1.2]
    assert report["outputs"][0]["sample0_plot_paths"] == {"ratio_re_pdf": "re.pdf", "ratio_im_pdf": "im.pdf"}
    assert "c2pt_pdf" not in report["outputs"][0]["sample0_plot_paths"]
    assert result["plot_ylim"] == [-0.2, 1.2]
    assert result["n_txt"] == 2


def test_fit_bare_matrix_grid_path_args_resolve_under_artifacts(tmp_path) -> None:
    manifest = AnalysisManifest(
        run_id="demo",
        correlators=[],
        kernels=[],
        manifest_dir=tmp_path / "examples",
        project_root=tmp_path,
    )
    args = {
        "pt2_path": "data/pt2.h5",
        "pt3_paths": {"8": "data/ts8.h5", "10": "/abs/ts10.h5"},
        "save_path": None,
    }
    resolved = prepare_tool_args(
        "fit_bare_matrix_grid",
        args,
        manifest=manifest,
        artifacts_dir=tmp_path / "artifacts",
        _store={},
    )
    assert resolved["pt2_path"] == str(tmp_path / "data" / "pt2.h5")
    assert resolved["pt3_paths"]["8"] == str(tmp_path / "data" / "ts8.h5")
    assert resolved["pt3_paths"]["10"] == "/abs/ts10.h5"
    assert resolved["save_path"] == str(tmp_path / "artifacts" / "bare_matrix_elements")
    assert resolved["artifacts_dir"] == str(tmp_path / "artifacts")

def test_resample_config_samples_jackknife_count_and_values() -> None:
    data = np.arange(12, dtype=float).reshape(4, 3)
    samples, indices = _resample_config_samples(data, mode="jk", n_boot=99, seed=7)
    assert indices is None
    assert samples.shape == (4, 3)
    assert np.allclose(samples, jackknife(data))


def test_sample_mean_err_uses_jackknife_scaling() -> None:
    values = np.array([1.0, 2.0, 3.0])
    mean, err = _sample_mean_err(values, mode="jk")
    assert mean == pytest.approx(2.0)
    assert err == pytest.approx(np.sqrt(4.0 / 3.0))
    _, bs_err = _sample_mean_err(values, mode="bs")
    assert bs_err == pytest.approx(1.0)
    assert core_sample_mean_err(values, mode="jk") == pytest.approx((mean, err))


def test_default_pt2_windows_use_l_over_four() -> None:
    windows = _normalise_pt2_windows(None, Lt=32)
    assert windows
    assert {window["tmax"] for window in windows} == {8}
    assert all(window["tmax"] - window["tmin"] >= 4 for window in windows)


def test_pt2_ratio_joint_fit_recovers_toy_parameters() -> None:
    tsep_ls = [6, 8, 10]
    ratio_re, ratio_im, p_true = _toy_pt3_ratio_gv(tsep_ls=tsep_ls, Lt=32)
    pt2_gv = _toy_pt2_gv(Lt=32, E0=0.45)
    fit = pt2_ratio_joint_fit(
        pt2_gv,
        tmin=2,
        tmax=12,
        ratio_real=ratio_re,
        ratio_imag=ratio_im,
        tsep_ls=tsep_ls,
        tau_cut=1,
        Lt=32,
        svdcut=1e-8,
    )
    assert abs(gv.mean(fit.p["E0"]) - gv.mean(p_true["E0"])) < 0.05
    assert abs(gv.mean(fit.p["O00_re"]) - gv.mean(p_true["O00_re"])) < 0.05


def test_scaled_posterior_as_prior_inflates_all_errors() -> None:
    template = pt3_ratio_prior(nstate=2)
    posterior = gv.BufferDict()
    for key in template:
        posterior[key] = gv.gvar(0.25, 0.03)

    class Fit:
        pass

    fit = Fit()
    fit.p = posterior
    prior = _scaled_posterior_as_prior(fit, template, error_scale=3.0)
    for key in template:
        assert gv.mean(prior[key]) == pytest.approx(0.25)
        assert gv.sdev(prior[key]) == pytest.approx(0.09)


def test_fit_posterior_is_usable_rejects_degenerate_values() -> None:
    template = pt3_ratio_prior(nstate=2)
    posterior = gv.BufferDict()
    for key in template:
        posterior[key] = gv.gvar(0.25, 0.03)
    posterior["E0"] = gv.gvar(1e-7, 0.03)

    class Fit:
        pass

    fit = Fit()
    fit.p = posterior
    usable, reason = _fit_posterior_is_usable(fit, template)
    assert usable is False
    assert "E0" in str(reason)


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
