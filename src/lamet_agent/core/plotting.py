"""Shared plotting conventions for correlator-stage figures.

Purpose:
- provide a single, self-contained plotting module for the agent project
- mirror the LaMETLat publication style for 2pt correlator and effective-mass plots

Expected inputs:
- resampled correlator values as ``gvar`` arrays
- optional per-window fit bands and a model-averaged E0 band on meff

Expected outputs:
- matplotlib figures, optionally saved to PDF

Example usage:
- from lamet_agent.core.plotting import plot_pt2_fit_on_data
- plot_pt2_fit_on_data(pt2_gv, fit_bands=[...], E0_band=e0_gv, save_path="run/c2pt")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gvar as gv
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.resampling import bs_ls_avg, jk_ls_avg

# Publication-oriented palette and styles copied from LaMETLat plot_settings.
BLUE = "#4E79A7"
ORANGE = "#E69F00"
GREEN = "#2CA02C"
RED = "#D62728"
VIOLET = "#7B6FD0"
FUCHSIA = "#CC79A7"

COLOR_CYCLE = [BLUE, ORANGE, GREEN, RED, VIOLET, FUCHSIA]

FONT_CONFIG = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
}

FIG_WIDTH = 6.75
GOLDEN_RATIO = 1.618034333
FIG_SIZE = (FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO)

FONT_SIZE = {"fontsize": 18}
LEGEND_SETS = {"fontsize": 14, "loc": "upper right"}
LABEL_SIZE = {"labelsize": 18}

ERRORBAR_STYLE = {
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1,
}

TSEP_LABEL = r"${t_{\mathrm{sep}}~/~a}$"
MEFF_LABEL = r"${m}_{\mathrm{eff}}$"
TAU_CENTER_LABEL = r"$(\tau - t_{\mathrm{sep}}/2)~/~a$"
RATIO_REAL_LABEL = r"$\Re\left[\mathcal{R}(t_{\mathrm{sep}},\tau)\right]$"
RATIO_IMAG_LABEL = r"$\Im\left[\mathcal{R}(t_{\mathrm{sep}},\tau)\right]$"
TSEP_TAG = r"$t_{\mathrm{sep}}$"


def apply_plot_style() -> None:
    """Apply package default font settings to matplotlib rcParams."""
    rcParams.update(FONT_CONFIG)


def default_plot() -> tuple[Figure, Axes]:
    """Create a default single-panel plot."""
    apply_plot_style()
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.axes()
    ax.tick_params(direction="in", top=True, right=True, **LABEL_SIZE)
    ax.grid(linestyle=":")
    return fig, ax


def pt2_to_meff(pt2_array: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """Convert a 1D 2pt correlator to effective-mass values."""
    data = np.asarray(pt2_array)
    if boundary in ("periodic", "anti-periodic"):
        return np.arccosh((data[2:] + data[:-2]) / (2 * data[1:-1]))
    if boundary == "none":
        return np.log(data[:-1] / data[1:])
    raise ValueError(f"unsupported boundary mode: {boundary!r}")


def _meff_trange(t: np.ndarray, boundary: str) -> np.ndarray:
    if boundary in ("periodic", "anti-periodic"):
        return t[1:-1]
    return t[:-1]


def _draw_fit_band(
    ax: Axes,
    fit_t: np.ndarray,
    fit_gv: np.ndarray,
    *,
    color: str,
    label: str,
    boundary: str | None = None,
    t_max: int | None = None,
) -> None:
    """Draw a fit curve with uncertainty band on C2pt or meff axes."""
    if boundary is None:
        fit_mean = gv.mean(fit_gv)
        fit_sdev = gv.sdev(fit_gv)
        plot_t = np.asarray(fit_t, dtype=int)
        if t_max is not None:
            keep = plot_t <= int(t_max)
            plot_t = plot_t[keep]
            fit_mean = fit_mean[keep]
            fit_sdev = fit_sdev[keep]
        ax.plot(plot_t, fit_mean, color=color, label=label)
        ax.fill_between(
            plot_t,
            fit_mean - fit_sdev,
            fit_mean + fit_sdev,
            color=color,
            alpha=0.35,
        )
        return

    fit_meff = pt2_to_meff(fit_gv, boundary=boundary)
    meff_t = _meff_trange(np.asarray(fit_t, dtype=int), boundary)
    if t_max is not None:
        keep = meff_t <= int(t_max)
        meff_t = meff_t[keep]
        fit_meff = fit_meff[keep]
    fit_mean = gv.mean(fit_meff)
    fit_sdev = gv.sdev(fit_meff)
    ax.plot(meff_t, fit_mean, color=color, label=label)
    ax.fill_between(
        meff_t,
        fit_mean - fit_sdev,
        fit_mean + fit_sdev,
        color=color,
        alpha=0.35,
    )


def plot_pt2_fit_on_data(
    pt2_gv: np.ndarray,
    *,
    boundary: str = "periodic",
    fit_t: np.ndarray | None = None,
    fit_gv: np.ndarray | None = None,
    fit_label: str = "Fit",
    fit_bands: list[dict[str, Any]] | None = None,
    E0_band: gv.GVar | None = None,
    E0_label: str = r"Model-averaged $E_0$",
    t_max: int | None = None,
    save_path: str | Path | None = None,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot C2pt and effective mass with optional per-window fit bands.

    ``fit_bands`` entries may contain ``fit_t``, ``fit_gv``, ``label``, and
    optional ``color``. When ``E0_band`` is given, a horizontal uncertainty band
    is drawn on the meff panel at the model-averaged ground-state energy.

    When ``t_max`` is set, the meff panel shows only points with ``t <= t_max``
    and sets ``xlim`` accordingly; the C2pt panel is unchanged.

    Legacy single-band usage: pass ``fit_t`` and ``fit_gv`` instead of
    ``fit_bands``. ``save_path`` writes ``<save_path>_c2pt.pdf`` and
    ``<save_path>_meff.pdf``.
    """
    t = np.arange(len(pt2_gv), dtype=int)

    if fit_bands is None and fit_t is not None and fit_gv is not None:
        fit_bands = [{"fit_t": fit_t, "fit_gv": fit_gv, "label": fit_label, "color": COLOR_CYCLE[0]}]

    fig_c2, ax_c2 = default_plot()
    ax_c2.errorbar(
        t,
        gv.mean(pt2_gv),
        yerr=gv.sdev(pt2_gv),
        label="Data",
        **ERRORBAR_STYLE,
    )
    ax_c2.set_yscale("log")
    ax_c2.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_c2.set_ylabel(r"$C_{2\mathrm{pt}}(t_{\mathrm{sep}})$", **FONT_SIZE)

    meff_gv = pt2_to_meff(pt2_gv, boundary=boundary)
    fig_meff, ax_meff = default_plot()
    meff_x = _meff_trange(t, boundary)
    if t_max is not None:
        keep = meff_x <= int(t_max)
        meff_x = meff_x[keep]
        meff_gv = meff_gv[keep]
    meff_mean = gv.mean(meff_gv)
    ax_meff.errorbar(
        meff_x,
        meff_mean,
        yerr=gv.sdev(meff_gv),
        label="Data",
        **ERRORBAR_STYLE,
    )
    ax_meff.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_meff.set_ylabel(MEFF_LABEL, **FONT_SIZE)
    if t_max is not None:
        ax_meff.set_xlim(left=float(np.min(meff_x)) - 0.5 if meff_x.size else -0.5, right=float(t_max))
    ax_meff.set_ylim(_ylim_mean_middle_half(meff_mean))

    if fit_bands:
        for i, band in enumerate(fit_bands):
            band_t = np.asarray(band["fit_t"], dtype=int)
            band_gv = band["fit_gv"]
            color = band.get("color", COLOR_CYCLE[i % len(COLOR_CYCLE)])
            label = band.get("label", f"Fit {i}")
            _draw_fit_band(ax_c2, band_t, band_gv, color=color, label=label)
            _draw_fit_band(
                ax_meff,
                band_t,
                band_gv,
                color=color,
                label=label,
                boundary=boundary,
                t_max=t_max,
            )

    if E0_band is not None:
        e0_mean = float(gv.mean(E0_band))
        e0_sdev = float(gv.sdev(E0_band))
        ax_meff.axhspan(
            e0_mean - e0_sdev,
            e0_mean + e0_sdev,
            color=COLOR_CYCLE[0],
            alpha=0.2,
            label=E0_label,
        )
        ax_meff.axhline(e0_mean, color=COLOR_CYCLE[0], linestyle="--", linewidth=1)

    ax_c2.legend(**LEGEND_SETS)
    ax_meff.legend(**LEGEND_SETS)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_c2.savefig(path.with_name(f"{path.name}_c2pt.pdf"), bbox_inches="tight", transparent=True)
        fig_meff.savefig(path.with_name(f"{path.name}_meff.pdf"), bbox_inches="tight", transparent=True)

    return (fig_c2, ax_c2), (fig_meff, ax_meff)


def plot_pt2_meff_on_data(
    pt2_gv: np.ndarray,
    *,
    boundary: str = "none",
    fit_t: np.ndarray | None = None,
    fit_gv: np.ndarray | None = None,
    fit_label: str = "Fit",
    fit_bands: list[dict[str, Any]] | None = None,
    E0_band: gv.GVar | None = None,
    E0_label: str = r"Model-averaged $E_0$",
    t_max: int | None = None,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot effective mass with optional per-window fit bands.

    When ``t_max`` is set, only points with ``t <= t_max`` are shown.
    ``save_path`` writes ``<save_path>_meff.pdf``.
    """
    if fit_bands is None and fit_t is not None and fit_gv is not None:
        fit_bands = [{"fit_t": fit_t, "fit_gv": fit_gv, "label": fit_label, "color": COLOR_CYCLE[0]}]

    if t_max is not None:
        end = min(len(pt2_gv), int(t_max) + 2)
        pt2_plot = pt2_gv[:end]
    else:
        pt2_plot = pt2_gv

    t = np.arange(len(pt2_plot), dtype=int)
    meff_gv = pt2_to_meff(pt2_plot, boundary=boundary)
    meff_x = _meff_trange(t, boundary)
    if t_max is not None:
        keep = meff_x <= int(t_max)
        meff_x = meff_x[keep]
        meff_gv = meff_gv[keep]

    fig_meff, ax_meff = default_plot()
    meff_mean = gv.mean(meff_gv)
    ax_meff.errorbar(
        meff_x,
        meff_mean,
        yerr=gv.sdev(meff_gv),
        label="Data",
        **ERRORBAR_STYLE,
    )
    ax_meff.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_meff.set_ylabel(MEFF_LABEL, **FONT_SIZE)
    if t_max is not None:
        ax_meff.set_xlim(left=float(np.min(meff_x)) - 0.5 if meff_x.size else -0.5, right=float(t_max))
    ax_meff.set_ylim(_ylim_mean_middle_half(meff_mean))

    if fit_bands:
        for i, band in enumerate(fit_bands):
            band_t = np.asarray(band["fit_t"], dtype=int)
            band_gv = band["fit_gv"]
            color = band.get("color", COLOR_CYCLE[i % len(COLOR_CYCLE)])
            label = band.get("label", f"Fit {i}")
            _draw_fit_band(
                ax_meff,
                band_t,
                band_gv,
                color=color,
                label=label,
                boundary=boundary,
                t_max=t_max,
            )

    if E0_band is not None:
        e0_mean = float(gv.mean(E0_band))
        e0_sdev = float(gv.sdev(E0_band))
        ax_meff.axhspan(
            e0_mean - e0_sdev,
            e0_mean + e0_sdev,
            color=COLOR_CYCLE[0],
            alpha=0.2,
            label=E0_label,
        )
        ax_meff.axhline(e0_mean, color=COLOR_CYCLE[0], linestyle="--", linewidth=1)

    ax_meff.legend(**LEGEND_SETS)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_meff.savefig(path.with_name(f"{path.name}_meff.pdf"), bbox_inches="tight", transparent=True)

    return fig_meff, ax_meff


def _pt3_ratio_data_tau_slice(tsep: int) -> slice:
    """Tau indices for ratio data points: ``1`` through ``tsep - 1`` inclusive."""
    if int(tsep) < 2:
        raise ValueError(f"tsep must be >= 2 for ratio plots, got {tsep}")
    return slice(1, int(tsep))


def _tau_center_limits(ratio_dict: dict[int, np.ndarray]) -> tuple[float, float]:
    centers = []
    for tsep, row in ratio_dict.items():
        sl = _pt3_ratio_data_tau_slice(int(tsep))
        tau = np.arange(row.shape[-1], dtype=float)[sl]
        centers.append(tau - float(tsep) / 2)
    stacked = np.concatenate(centers)
    return float(np.min(stacked)), float(np.max(stacked))


def _ylim_middle_third(
    y_data: list[np.ndarray],
    yerr_data: list[np.ndarray],
) -> tuple[float, float]:
    """Y limits so data±error spans the middle third of the axis."""
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for y, err in zip(y_data, yerr_data):
        y_arr = np.asarray(y, dtype=float)
        err_arr = np.asarray(err, dtype=float)
        lows.append(y_arr - err_arr)
        highs.append(y_arr + err_arr)
    data_min = float(np.min(np.concatenate(lows)))
    data_max = float(np.max(np.concatenate(highs)))
    span = data_max - data_min
    if span <= 0.0:
        err_scale = float(np.max([np.max(np.asarray(e, dtype=float)) for e in yerr_data]))
        span = max(err_scale, 1e-6) * 2.0
    margin = span
    return data_min - margin, data_max + margin


def _ylim_mean_middle_third(y: np.ndarray) -> tuple[float, float]:
    """Y limits so the mean-value span occupies the middle third of the axis."""
    finite = np.asarray(y, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    span = data_max - data_min
    if span <= 0.0:
        span = max(abs(data_min), 1e-6) * 0.2
    margin = span
    return data_min - margin, data_max + margin


def _ylim_mean_middle_half(y: np.ndarray) -> tuple[float, float]:
    """Y limits so the mean-value span occupies the middle half of the axis."""
    finite = np.asarray(y, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    span = data_max - data_min

    if span <= 0.0:
        span = max(abs(data_min), 1e-6) * 0.2

    margin = 0.5 * span

    return data_min - margin, data_max + margin


def _draw_O00_band(
    ax: Axes,
    o00: gv.GVar,
    x_min: float,
    x_max: float,
    *,
    label: str,
) -> None:
    band_x = np.linspace(x_min, x_max, 2)
    band_mean = np.full(2, gv.mean(o00), dtype=float)
    band_sdev = np.full(2, gv.sdev(o00), dtype=float)
    ax.fill_between(
        band_x,
        band_mean - band_sdev,
        band_mean + band_sdev,
        color="grey",
        alpha=0.35,
        label=label,
    )


def _ratio_denominator_correction(
    tsep: int,
    *,
    energy: gv.GVar | float,
    Lt: int,
) -> gv.GVar | float:
    """Convert C3/C2_periodic ratios to a forward-denominator convention."""
    return 1.0 + gv.exp(-energy * (float(int(Lt)) - 2.0 * float(int(tsep))))


def plot_pt3_ratio_fit_on_data(
    ratio_real: dict[int, np.ndarray],
    ratio_imag: dict[int, np.ndarray],
    *,
    denominator_correction_energy: gv.GVar | float,
    denominator_correction_Lt: int,
    window_bands: list[dict[str, Any]] | None = None,
    plateau_ref_re: gv.GVar | None = None,
    plateau_ref_im: gv.GVar | None = None,
    plateau_label: str = r"Model-averaged $\mathcal{O}_{00}/(2E_0)$",
    save_path: str | Path | None = None,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot 3pt/2pt ratio real and imag vs centered tau with optional fit bands.

    Data use error bars on tau in ``[1, tsep - 1]``. Fit bands (``fill_between``)
    cover only each window's fit range ``[tau_cut, tsep + 1 - tau_cut)``.
    Plotted ratios are always multiplied by
    C2_periodic_ground(tsep) / C2_forward_ground(tsep) so the reference
    band remains the infinite-time O00/(2E0) value.
    """
    tsep_ls = sorted(ratio_real.keys())
    x_min, x_max = _tau_center_limits(ratio_real)

    fig_re, ax_re = default_plot()
    y_re: list[np.ndarray] = []
    yerr_re: list[np.ndarray] = []
    for tsep in tsep_ls:
        sl = _pt3_ratio_data_tau_slice(int(tsep))
        tau = np.arange(ratio_real[tsep].shape[-1], dtype=float)[sl]
        x = tau - tsep / 2
        correction = _ratio_denominator_correction(
            int(tsep),
            energy=denominator_correction_energy,
            Lt=denominator_correction_Lt,
        )
        corrected = ratio_real[tsep][sl] * correction
        mean = np.asarray(gv.mean(corrected), dtype=float)
        sdev = np.asarray(gv.sdev(corrected), dtype=float)
        y_re.append(mean)
        yerr_re.append(sdev)
        ax_re.errorbar(
            x,
            mean,
            yerr=sdev,
            label=f"{TSEP_TAG}={tsep} $a$",
            **ERRORBAR_STYLE,
        )

    if window_bands:
        for win in window_bands:
            for band in win["bands"]:
                fit_x = band["fit_tau"] - band["tsep"] / 2
                correction = _ratio_denominator_correction(
                    int(band["tsep"]),
                    energy=denominator_correction_energy,
                    Lt=denominator_correction_Lt,
                )
                corrected_fit = band["fit_re"] * correction
                fit_mean = gv.mean(corrected_fit)
                fit_sdev = gv.sdev(corrected_fit)
                color = band.get("color", COLOR_CYCLE[0])
                ax_re.fill_between(
                    fit_x,
                    fit_mean - fit_sdev,
                    fit_mean + fit_sdev,
                    color=color,
                    alpha=0.3,
                )

    if plateau_ref_re is not None:
        _draw_O00_band(ax_re, plateau_ref_re, x_min, x_max, label=plateau_label)

    ax_re.set_xlabel(TAU_CENTER_LABEL, **FONT_SIZE)
    ax_re.set_ylabel(RATIO_REAL_LABEL, **FONT_SIZE)
    ax_re.set_ylim(_ylim_middle_third(y_re, yerr_re))
    ax_re.legend(**LEGEND_SETS)

    fig_im, ax_im = default_plot()
    y_im: list[np.ndarray] = []
    yerr_im: list[np.ndarray] = []
    for tsep in tsep_ls:
        sl = _pt3_ratio_data_tau_slice(int(tsep))
        tau = np.arange(ratio_imag[tsep].shape[-1], dtype=float)[sl]
        x = tau - tsep / 2
        correction = _ratio_denominator_correction(
            int(tsep),
            energy=denominator_correction_energy,
            Lt=denominator_correction_Lt,
        )
        corrected = ratio_imag[tsep][sl] * correction
        mean = np.asarray(gv.mean(corrected), dtype=float)
        sdev = np.asarray(gv.sdev(corrected), dtype=float)
        y_im.append(mean)
        yerr_im.append(sdev)
        ax_im.errorbar(
            x,
            mean,
            yerr=sdev,
            label=f"{TSEP_TAG}={tsep} $a$",
            **ERRORBAR_STYLE,
        )
    if window_bands:
        for win in window_bands:
            for band in win["bands"]:
                fit_x = band["fit_tau"] - band["tsep"] / 2
                correction = _ratio_denominator_correction(
                    int(band["tsep"]),
                    energy=denominator_correction_energy,
                    Lt=denominator_correction_Lt,
                )
                corrected_fit = band["fit_im"] * correction
                fit_mean = gv.mean(corrected_fit)
                fit_sdev = gv.sdev(corrected_fit)
                color = band.get("color", COLOR_CYCLE[0])
                ax_im.fill_between(
                    fit_x,
                    fit_mean - fit_sdev,
                    fit_mean + fit_sdev,
                    color=color,
                    alpha=0.3,
                )
    if plateau_ref_im is not None:
        _draw_O00_band(ax_im, plateau_ref_im, x_min, x_max, label=plateau_label)
    ax_im.set_xlabel(TAU_CENTER_LABEL, **FONT_SIZE)
    ax_im.set_ylabel(RATIO_IMAG_LABEL, **FONT_SIZE)
    ax_im.set_ylim(_ylim_middle_third(y_im, yerr_im))
    ax_im.legend(**LEGEND_SETS)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_re.savefig(
            path.with_name(f"{path.name}_pt3_ratio_re.pdf"),
            bbox_inches="tight",
            transparent=True,
        )
        fig_im.savefig(
            path.with_name(f"{path.name}_pt3_ratio_im.pdf"),
            bbox_inches="tight",
            transparent=True,
        )

    return (fig_re, ax_re), (fig_im, ax_im)


def plot_fourier_artifact(
    path: str | Path,
    *,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot real and imaginary momentum-space distributions from a Fourier artifact."""
    path = Path(path)
    try:
        if path.suffix.lower() == ".nc":
            ft_data = EnsembleData.from_netcdf(path)
            extra = {
                key: np.asarray(json.loads(value))
                for key, value in ft_data.attrs.items()
                if key
                in {
                    "ft_re_mean",
                    "ft_im_mean",
                    "ft_re_stat_sdev",
                    "ft_im_stat_sdev",
                    "ft_re_sys_sdev",
                    "ft_im_sys_sdev",
                    "scheme_labels",
                    "fit_failures",
                    "scheme_weights",
                    "scheme_fit_chi2_dof",
                    "scheme_roughness",
                    "scheme_scores",
                    "best_scheme_index",
                }
            }
        else:
            ft_data, extra = EnsembleData.load_npz(path)
        if ft_data.dims != ["x"]:
            raise ValueError("Fourier EnsembleData artifact must have dimension ['x']")
        k = np.asarray(ft_data.coords["x"], dtype=float)
        mean = np.asarray(ft_data.mean)
        re = np.real(mean)
        im = np.imag(mean)
        re_stat = np.asarray(extra.get("ft_re_stat_sdev", np.std(np.real(ft_data.values), axis=0, ddof=1)), dtype=float)
        im_stat = np.asarray(extra.get("ft_im_stat_sdev", np.std(np.imag(ft_data.values), axis=0, ddof=1)), dtype=float)
        re_sys = np.asarray(extra.get("ft_re_sys_sdev", 0.0), dtype=float)
        im_sys = np.asarray(extra.get("ft_im_sys_sdev", 0.0), dtype=float)
        observable = str(extra["observable"]) if "observable" in extra else ft_data.attrs.get("observable", "")
        pz_raw = extra.get("pz_gev", ft_data.attrs.get("pz_gev"))
        pz_gev = float(pz_raw) if pz_raw is not None and np.isfinite(float(pz_raw)) else None
    except ValueError:
        data = np.load(path)
        k = np.asarray(data["k_grid"], dtype=float)
        re = np.asarray(data["ft_re_mean"], dtype=float)
        im = np.asarray(data["ft_im_mean"], dtype=float)
        re_stat = np.asarray(data["ft_re_stat_sdev"], dtype=float)
        im_stat = np.asarray(data["ft_im_stat_sdev"], dtype=float)
        re_sys = np.asarray(data["ft_re_sys_sdev"], dtype=float) if "ft_re_sys_sdev" in data else 0.0
        im_sys = np.asarray(data["ft_im_sys_sdev"], dtype=float) if "ft_im_sys_sdev" in data else 0.0
        observable = str(data["observable"]) if "observable" in data else ""
        pz_gev = float(data["pz_gev"]) if "pz_gev" in data and np.isfinite(data["pz_gev"]) else None
    re_total = np.sqrt(re_stat**2 + re_sys**2)
    im_total = np.sqrt(im_stat**2 + im_sys**2)
    roundoff_floor = 1e-14
    re = np.where(np.abs(re) < roundoff_floor, 0.0, re)
    im = np.where(np.abs(im) < roundoff_floor, 0.0, im)
    re_total = np.where(re_total < roundoff_floor, 0.0, re_total)
    im_total = np.where(im_total < roundoff_floor, 0.0, im_total)
    default_title = "FT" if not observable else "FT " + observable.replace("_", " ")
    legend_label = rf"$P_z={pz_gev:g}\,\mathrm{{GeV}}$" if pz_gev is not None else r"$P_z$"

    apply_plot_style()
    fig, (ax_re, ax_im) = plt.subplots(
        2,
        1,
        figsize=FIG_SIZE,
        gridspec_kw={"height_ratios": [1, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)
    for ax in (ax_re, ax_im):
        ax.tick_params(direction="in", top=True, right=True, **LABEL_SIZE)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(linestyle=":")

    for ax in (ax_re, ax_im):
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)

    ax_re.fill_between(k, re - re_total, re + re_total, color=COLOR_CYCLE[0], alpha=0.32, linewidth=0, label=legend_label)
    ax_re.plot(k, re, color=COLOR_CYCLE[0], linewidth=0.9, alpha=0.65)
    ax_im.fill_between(k, im - im_total, im + im_total, color=COLOR_CYCLE[1], alpha=0.32, linewidth=0, label=legend_label)
    ax_im.plot(k, im, color=COLOR_CYCLE[1], linewidth=0.9, alpha=0.65)
    ax_re.set_xlim(-2.0, 2.0)
    ax_im.set_xlim(-2.0, 2.0)
    ax_re.set_ylabel(r"$\mathrm{Re}\,\tilde{q}(x)$", **FONT_SIZE)
    ax_im.set_ylabel(r"$\mathrm{Im}\,\tilde{q}(x)$", **FONT_SIZE)
    ax_re.yaxis.set_label_coords(-0.11, 0.5)
    ax_im.yaxis.set_label_coords(-0.11, 0.5)
    ax_im.set_xlabel(r"$x$", **FONT_SIZE)
    ax_re.legend(**LEGEND_SETS)
    ax_im.legend(**LEGEND_SETS)
    ax_re.set_title(default_title if title is None else title, **FONT_SIZE)

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (ax_re, ax_im)


def _band_segment(
    x: np.ndarray,
    mean: np.ndarray,
    sdev: np.ndarray,
    *,
    start: float,
    stop: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a band segment with exact start/stop points inserted for plotting."""
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    mean_sorted = np.asarray(mean, dtype=float)[order]
    sdev_sorted = np.asarray(sdev, dtype=float)[order]
    start = max(float(start), float(x_sorted[0]))
    stop = min(float(stop), float(x_sorted[-1]))
    mask = (x_sorted > start) & (x_sorted < stop)
    x_seg = np.concatenate(([start], x_sorted[mask], [stop]))
    mean_seg = np.interp(x_seg, x_sorted, mean_sorted)
    sdev_seg = np.interp(x_seg, x_sorted, sdev_sorted)
    return x_seg, mean_seg, sdev_seg


def _coord_to_lambda(
    coord: np.ndarray,
    *,
    coord_unit: str,
    pz_gev: float | None,
    a_fm: float | None,
) -> np.ndarray:
    unit = coord_unit.lower()
    fm_to_gev_inv = 5.067731237
    if unit == "lambda":
        return coord
    if unit == "gev_inv":
        if pz_gev is None:
            raise ValueError("pz_gev is required when coord_unit='gev_inv'")
        return coord * float(pz_gev)
    if unit == "fm":
        if pz_gev is None:
            raise ValueError("pz_gev is required when coord_unit='fm'")
        return coord * fm_to_gev_inv * float(pz_gev)
    if unit == "lattice":
        if pz_gev is None or a_fm is None:
            raise ValueError("pz_gev and a_fm are required when coord_unit='lattice'")
        return coord * float(a_fm) * fm_to_gev_inv * float(pz_gev)
    raise ValueError("coord_unit must be 'lambda', 'gev_inv', 'fm', or 'lattice'")


def plot_fourier_extension_quality(
    coord: np.ndarray,
    samples: np.ndarray,
    result: dict[str, Any],
    *,
    scheme_index: int = 0,
    component: str = "re",
    pz_gev: float | None = None,
    a_fm: float | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot coordinate-space data against the fitted long-distance extrapolation."""
    component = component.lower()
    if component not in {"re", "im"}:
        raise ValueError("component must be 're' or 'im'")
    scheme = result["scheme_results"][scheme_index]
    coord_unit = str(result.get("coord_unit", "lambda"))
    if pz_gev is None:
        pz_gev = result.get("pz_gev")
    if a_fm is None:
        a_fm = result.get("a_fm")

    coord_arr = np.asarray(coord, dtype=float)
    lambda_data = _coord_to_lambda(coord_arr, coord_unit=coord_unit, pz_gev=pz_gev, a_fm=a_fm)
    resample_mode = str(result.get("resample_mode", "bootstrap"))

    lambda_ext = np.asarray(scheme["lambda_ext"], dtype=float)
    model_key = "fit_re_samples" if component == "re" else "fit_im_samples"
    mode = resample_mode.strip().lower()
    band_stats = []
    for sample_values in (samples, scheme[model_key]):
        arr = np.asarray(sample_values, dtype=float)
        if arr.shape[0] < 2:
            mean = np.mean(arr, axis=0)
            sdev = np.zeros_like(mean, dtype=float)
        elif mode in {"jk", "jackknife"}:
            values = jk_ls_avg(arr)
            mean = np.asarray(gv.mean(values), dtype=float)
            sdev = np.asarray(gv.sdev(values), dtype=float)
        elif mode in {"bs", "boot", "bootstrap"}:
            values = bs_ls_avg(arr)
            mean = np.asarray(gv.mean(values), dtype=float)
            sdev = np.asarray(gv.sdev(values), dtype=float)
        elif mode == "raw":
            mean = np.mean(arr, axis=0)
            sdev = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        else:
            raise ValueError("resample_mode must be 'bs'/'bootstrap', 'jk'/'jackknife', or 'raw'")
        band_stats.append((mean, sdev))
    (data_mean, data_sdev), (ext_mean, ext_sdev) = band_stats

    zmin, zmax = scheme["fit_range"]
    fit_lambda = _coord_to_lambda(
        np.asarray([zmin, zmax], dtype=float),
        coord_unit=coord_unit,
        pz_gev=pz_gev,
        a_fm=a_fm,
    )
    ext_endpoint_lambda = _coord_to_lambda(
        np.asarray([scheme["z_ext_max"]], dtype=float),
        coord_unit=coord_unit,
        pz_gev=pz_gev,
        a_fm=a_fm,
    )[0]
    lambda_ext_plot, ext_mean_plot, ext_sdev_plot = _band_segment(
        lambda_ext,
        ext_mean,
        ext_sdev,
        start=fit_lambda[0],
        stop=ext_endpoint_lambda,
    )
    z_unit = coord_unit
    if coord_unit.lower() == "lambda":
        z_unit = r"\lambda"

    apply_plot_style()
    fig, ax = default_plot()
    data_color = "#9ecae1"
    ext_color = "#a1d99b"

    method = str(result.get("method", "")).upper()
    order = str(result.get("order", "")).upper()
    model_label = "Extrapolation"
    if method or order:
        model_label = f"Extrapolation ({'+'.join(item for item in (method, order) if item)})"

    ax.fill_between(
        lambda_data,
        data_mean - data_sdev,
        data_mean + data_sdev,
        color=data_color,
        alpha=0.7,
        linewidth=0,
        label="Lattice Data",
    )
    ax.fill_between(
        lambda_ext_plot,
        ext_mean_plot - ext_sdev_plot,
        ext_mean_plot + ext_sdev_plot,
        color=ext_color,
        alpha=0.5,
        linewidth=0,
        label=model_label,
        zorder=1,
    )

    for idx, value in enumerate(fit_lambda):
        ax.axvline(
            value,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label="Fit Range" if idx == 0 else None,
        )

    ax.set_xlabel(r"$\lambda = zP^z$", **FONT_SIZE)
    component_label = r"\mathrm{Re}" if component == "re" else r"\mathrm{Im}"
    if pz_gev is None:
        ax.set_ylabel(rf"${component_label}\,\tilde{{h}}^R(\lambda, P^z)$", **FONT_SIZE)
    else:
        ax.set_ylabel(
            rf"${component_label}\,\tilde{{h}}^R(\lambda, P^z={float(pz_gev):g}\,\mathrm{{GeV}})$",
            **FONT_SIZE,
        )
    if title is None:
        if coord_unit.lower() == "lambda" and pz_gev is None:
            title = rf"$\lambda$-extrapolation: $z_{{\min}}={zmin:g}\,\lambda$, $z_{{\max}}={zmax:g}\,\lambda$"
        else:
            unit = coord_unit.lower()
            if unit == "fm":
                zmin_fm, zmax_fm = zmin, zmax
            elif unit == "lattice":
                zmin_fm, zmax_fm = zmin * float(a_fm), zmax * float(a_fm)
            elif unit == "gev_inv":
                zmin_fm, zmax_fm = zmin / 5.067731237, zmax / 5.067731237
            else:
                scale = 5.067731237 * float(pz_gev)
                zmin_fm, zmax_fm = zmin / scale, zmax / scale
            title = rf"$\lambda$-extrapolation: $z_{{\min}}={zmin_fm:g}\,\mathrm{{fm}}$, $z_{{\max}}={zmax_fm:g}\,\mathrm{{fm}}$"
    ax.set_title(title, **FONT_SIZE)
    chi2_values = result.get("scheme_fit_chi2_dof", [])
    if chi2_values and scheme_index < len(chi2_values):
        ax.text(
            0.03,
            0.95,
            rf"$\chi^2/\mathrm{{dof}}={float(chi2_values[scheme_index]):.3g}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    ax.legend(**LEGEND_SETS)

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
