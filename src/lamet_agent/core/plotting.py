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

from pathlib import Path
from typing import Any

import gvar as gv
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
) -> None:
    """Draw a fit curve with uncertainty band on C2pt or meff axes."""
    if boundary is None:
        fit_mean = gv.mean(fit_gv)
        fit_sdev = gv.sdev(fit_gv)
        ax.plot(fit_t, fit_mean, color=color, label=label)
        ax.fill_between(
            fit_t,
            fit_mean - fit_sdev,
            fit_mean + fit_sdev,
            color=color,
            alpha=0.35,
        )
        return

    fit_meff = pt2_to_meff(fit_gv, boundary=boundary)
    meff_t = _meff_trange(fit_t, boundary)
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
    save_path: str | Path | None = None,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot C2pt and effective mass with optional per-window fit bands.

    ``fit_bands`` entries may contain ``fit_t``, ``fit_gv``, ``label``, and
    optional ``color``. When ``E0_band`` is given, a horizontal uncertainty band
    is drawn on the meff panel at the model-averaged ground-state energy.

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
    ax_meff.errorbar(
        meff_x,
        gv.mean(meff_gv),
        yerr=gv.sdev(meff_gv),
        label="Data",
        **ERRORBAR_STYLE,
    )
    ax_meff.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_meff.set_ylabel(MEFF_LABEL, **FONT_SIZE)

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
