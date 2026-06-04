"""Unit tests for correlator stage tools and plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import gvar as gv
import numpy as np
import pytest
from matplotlib.legend import Legend

pytest.importorskip("lsqfit")

from lamet_agent.core.plotting import (
    _pt3_ratio_data_tau_slice,
    _ylim_middle_third,
    plot_pt2_fit_on_data,
)
from lamet_agent.stages.correlator.functions import (
    MAX_FIT_WINDOWS,
    MAX_PT3_FIT_WINDOWS,
    PT2_PRIOR_ERROR_SCALE,
    STAGE_TOOLS,
    _pt2_posterior_as_prior,
    asymptotic_ratio_real_gvar,
    compute_pt3_ratio,
    read_pt3,
    fit_pt3_window,
    fit_window,
    model_average,
    plot_fit_on_data,
    plot_pt3_fit_on_data,
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
        tau = np.arange(tsep + 2, dtype=float)
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
    pt3_re = {tsep: np.full((n_cfg, tsep + 2), 0.5)}
    pt3_im = {tsep: np.zeros((n_cfg, tsep + 2))}
    store = {
        "pt2_samples": pt2_re,
        "pt2_imag_samples": pt2_im,
        "pt3_samples_re": pt3_re,
        "pt3_samples_im": pt3_im,
    }
    out = compute_pt3_ratio(store)
    assert out["tsep_keys"] == [tsep]
    assert np.allclose(store["ratio_samples_re"][tsep], 0.5)


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


def test_pt3_ratio_data_tau_slice_includes_tsep_minus_one() -> None:
    row = np.arange(12)
    sl = _pt3_ratio_data_tau_slice(10)
    assert list(row[sl]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_read_pt3_ignores_legacy_out_kwarg(tmp_path) -> None:
    """Agent sometimes passes out=; tool should not raise."""
    import h5py

    path = tmp_path / "fake.h5"
    data = np.ones((4, 6), dtype=float)
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
