"""Unit tests for correlator stage tools and plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import gvar as gv
import numpy as np
import pytest
from matplotlib.legend import Legend

pytest.importorskip("lsqfit")

from lamet_agent.core.plotting import plot_pt2_fit_on_data
from lamet_agent.stages.correlator.functions import (
    MAX_FIT_WINDOWS,
    STAGE_TOOLS,
    fit_window,
    model_average,
    plot_fit_on_data,
    pt2_re_fcn,
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
    assert "scan_tmin" not in STAGE_TOOLS


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
