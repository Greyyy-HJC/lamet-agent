"""Small publication-style plotting primitives shared by neo stages."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

import gvar
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.markers import MarkerStyle

_BLUE = "#4E79A7"
_ORANGE = "#E69F00"
_GREEN = "#2CA02C"
_RED = "#D62728"
_VIOLET = "#7B6FD0"
_FUCHSIA = "#CC79A7"
COLOR_CYCLE = [_BLUE, _ORANGE, _GREEN, _RED, _VIOLET, _FUCHSIA]

X_LABEL = r"$x$"
Z_OVER_A_LABEL = r"$z~/~a$"
BARE_MATRIX_ELEMENT_LABEL = "bare matrix element"
QUASI_DISTRIBUTION_LABELS = {
    "real": r"$\mathrm{Re}\,\tilde q(x)$",
    "imag": r"$\mathrm{Im}\,\tilde q(x)$",
    "both": r"$\tilde q(x)$",
}

_FONT_CONFIG = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
}
_FIG_WIDTH = 6.75
_GOLDEN_RATIO = 1.618034333
_FIG_SIZE = (_FIG_WIDTH, _FIG_WIDTH / _GOLDEN_RATIO)
_FONT_SIZE = {"fontsize": 18}
_LEGEND_SETTINGS = {"fontsize": 14, "loc": "upper right"}
_LABEL_SIZE = {"labelsize": 18}
_ERRORBAR_STYLE = {
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1,
}
_LINE_STYLES = {"solid", "dashed", "dotted", "dashdot", "-", "--", ":", "-."}
_LINE_ALPHA = 0.65
_BAND_ALPHA = 0.32
_DRAW_STYLES = {"default", "steps", "steps-pre", "steps-mid", "steps-post"}
_CURRENT_FIGURE: Any | None = None
_CURRENT_AXIS: Any | None = None
_COLOR_INDEX = 0


def _apply_plot_style() -> None:
    """Apply the original package font settings to matplotlib."""
    rcParams.update(_FONT_CONFIG)


def _unpack_gvar(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract central values and standard deviations from gvars."""
    array = np.asarray(values)
    is_gvar = isinstance(values, gvar.GVar) or (
        array.dtype == object and all(isinstance(value, gvar.GVar) for value in array.flat)
    )
    if not is_gvar:
        raise TypeError("plot values must be a gvar or an array of gvars")
    return np.asarray(gvar.mean(values)), np.asarray(gvar.sdev(values), dtype=float)


def _axis() -> Any:
    if _CURRENT_AXIS is None:
        raise RuntimeError("start_plot() must be called before plotting")
    return _CURRENT_AXIS


def _resolve_color(color: str | None) -> str:
    global _COLOR_INDEX
    if color is not None:
        return color
    selected = COLOR_CYCLE[_COLOR_INDEX % len(COLOR_CYCLE)]
    _COLOR_INDEX += 1
    return selected


def series_color(index: int) -> str:
    """Return a stable color for a related group of plot primitives."""
    return COLOR_CYCLE[index % len(COLOR_CYCLE)]


def momentum_label(momentum_gev: object, *, default: str | None = None) -> str:
    """Format a momentum legend label, optionally falling back for missing metadata."""
    if isinstance(momentum_gev, Real) and not isinstance(momentum_gev, bool):
        momentum = float(momentum_gev)
        if math.isfinite(momentum):
            return rf"$P_z={round(momentum, 2):g}\,\mathrm{{GeV}}$"
    if default is not None:
        return default
    raise ValueError("momentum_gev must be a finite real number")


def quasi_distribution_label(component: str) -> str:
    """Return the shared quasi-distribution label for one component."""
    normalized = {"re": "real", "im": "imag"}.get(component, component)
    try:
        return QUASI_DISTRIBUTION_LABELS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported quasi-distribution component {component!r}") from exc


def start_plot() -> None:
    """Create the one module-owned publication-style figure and axis."""
    global _COLOR_INDEX, _CURRENT_AXIS, _CURRENT_FIGURE
    if _CURRENT_FIGURE is not None:
        raise RuntimeError("the current plot must be saved before starting another")
    _apply_plot_style()
    _COLOR_INDEX = 0
    _CURRENT_FIGURE, _CURRENT_AXIS = plt.subplots(figsize=_FIG_SIZE)
    _CURRENT_AXIS.tick_params(direction="in", top=True, right=True, **_LABEL_SIZE)
    _CURRENT_AXIS.grid(linestyle=":")


def errorline(
    x: Any,
    values: Any,
    *,
    color: str | None = None,
    marker: str = "o",
    label: str | None = None,
) -> None:
    """Plot central values with marker-based error lines."""
    try:
        MarkerStyle(marker)
    except ValueError as exc:
        raise ValueError(f"unsupported marker {marker!r}") from exc
    mean, sdev = _unpack_gvar(values)
    _axis().errorbar(
        x,
        mean,
        yerr=sdev,
        fmt=marker,
        color=_resolve_color(color),
        label=label,
        **_ERRORBAR_STYLE,
    )


def line(
    x: Any,
    y: Any,
    *,
    color: str | None = None,
    marker: str | None = None,
    label: str | None = None,
    linestyle: str = "solid",
    linewidth: float = 0.9,
    drawstyle: str = "default",
) -> None:
    """Plot one ordinary line with an optional marker and label."""
    if marker is not None:
        try:
            MarkerStyle(marker)
        except ValueError as exc:
            raise ValueError(f"unsupported marker {marker!r}") from exc
    _validate_line_style(linestyle)
    if drawstyle not in _DRAW_STYLES:
        raise ValueError(f"unsupported draw style {drawstyle!r}")
    _axis().plot(
        x,
        y,
        color=_resolve_color(color),
        marker=marker,
        alpha=_LINE_ALPHA,
        linestyle=linestyle,
        linewidth=linewidth,
        drawstyle=drawstyle,
        label=label,
    )


def band(
    x: Any,
    lower: Any,
    upper: Any,
    *,
    color: str | None = None,
    label: str | None = None,
) -> None:
    """Fill one vertical interval between lower and upper curves."""
    _axis().fill_between(
        x,
        lower,
        upper,
        color=_resolve_color(color),
        alpha=_BAND_ALPHA,
        linewidth=0,
        label=label,
    )


def vband(
    lower: float,
    upper: float,
    *,
    color: str | None = None,
    label: str | None = None,
) -> None:
    """Fill one vertical coordinate interval."""
    _axis().axvspan(lower, upper, color=_resolve_color(color), alpha=_BAND_ALPHA, linewidth=0, label=label)


def hband(
    lower: float,
    upper: float,
    *,
    color: str | None = None,
    label: str | None = None,
) -> None:
    """Fill one horizontal coordinate interval."""
    _axis().axhspan(lower, upper, color=_resolve_color(color), alpha=_BAND_ALPHA, linewidth=0, label=label)


def bar(
    x: Any,
    height: Any,
    *,
    width: float = 0.8,
    color: str | None = None,
    label: str | None = None,
) -> None:
    """Plot one bar series with an explicit width."""
    _axis().bar(x, height, width=width, color=_resolve_color(color), label=label)


def errorband(
    x: Any,
    values: Any,
    *,
    color: str | None = None,
    label: str | None = None,
) -> None:
    """Plot a central line over its one-sigma error band."""
    mean, sdev = _unpack_gvar(values)
    selected_color = _resolve_color(color)
    band(x, mean - sdev, mean + sdev, color=selected_color, label=label)
    line(x, mean, color=selected_color)


def histogram(
    values: Any,
    bins: Any,
    *,
    color: str | None = None,
    label: str | None = None,
    linewidth: float = 1.4,
) -> None:
    """Plot one unfilled step histogram with explicitly shared bins."""
    _axis().hist(
        values,
        bins=bins,
        histtype="step",
        color=_resolve_color(color),
        label=label,
        linewidth=linewidth,
    )


def _validate_line_style(linestyle: str) -> None:
    if linestyle not in _LINE_STYLES:
        raise ValueError(f"unsupported line style {linestyle!r}")


def hline(
    value: float,
    *,
    color: str | None = None,
    linestyle: str = "solid",
) -> None:
    """Add a horizontal line with a supported color and line style."""
    _validate_line_style(linestyle)
    _axis().axhline(
        value,
        color=_resolve_color(color),
        linestyle=linestyle,
        linewidth=0.8,
        alpha=0.45,
    )


def vline(
    value: float,
    *,
    color: str | None = None,
    linestyle: str = "solid",
) -> None:
    """Add a vertical line with a supported color and line style."""
    _validate_line_style(linestyle)
    _axis().axvline(
        value,
        color=_resolve_color(color),
        linestyle=linestyle,
        linewidth=0.8,
        alpha=0.45,
    )


def configure_plot(
    *,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool = False,
    legend_loc: str | None = None,
    title: str | None = None,
) -> None:
    """Apply the supported labels, limits, and optional standard legend."""
    axis = _axis()
    axis.set_xlabel(xlabel, **_FONT_SIZE)
    axis.set_ylabel(ylabel, **_FONT_SIZE)
    if xlim is not None:
        axis.set_xlim(*xlim)
    if ylim is not None:
        axis.set_ylim(*ylim)
    if title is not None:
        axis.set_title(title, **_FONT_SIZE)
    if legend:
        settings = dict(_LEGEND_SETTINGS)
        if legend_loc is not None:
            settings["loc"] = legend_loc
        axis.legend(**settings)


def save_figure(*paths: str | Path) -> None:
    """Save the current figure to every path, then close and clear it."""
    global _COLOR_INDEX, _CURRENT_AXIS, _CURRENT_FIGURE
    if _CURRENT_FIGURE is None:
        raise RuntimeError("there is no current plot to save")
    outputs = [Path(path) for path in paths]
    figure = _CURRENT_FIGURE
    try:
        if not outputs:
            raise ValueError("save_figure requires at least one output path")
        if any(output.suffix.lower() not in {".pdf", ".svg"} for output in outputs):
            raise ValueError("figures must be saved as .pdf or .svg")
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, bbox_inches="tight")
    finally:
        plt.close(figure)
        _COLOR_INDEX = 0
        _CURRENT_FIGURE = None
        _CURRENT_AXIS = None


__all__ = [
    "COLOR_CYCLE",
    "X_LABEL",
    "Z_OVER_A_LABEL",
    "BARE_MATRIX_ELEMENT_LABEL",
    "QUASI_DISTRIBUTION_LABELS",
    "series_color",
    "momentum_label",
    "quasi_distribution_label",
    "start_plot",
    "configure_plot",
    "line",
    "band",
    "vband",
    "hband",
    "bar",
    "errorline",
    "errorband",
    "histogram",
    "hline",
    "vline",
    "save_figure",
]
