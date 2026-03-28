"""Shared plotting presets migrated from incoming analysis drafts.

The helpers are intentionally lightweight wrappers around Matplotlib defaults so
new analysis code can keep visual style consistent.
"""

from __future__ import annotations

import numpy as np

from lamet_agent.errors import OptionalDependencyError

try:  # pragma: no cover - optional dependency checks are environment dependent.
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency checks are environment dependent.
    plt = None

PALETTE = {
    "grey": "#808080",
    "red": "#FF6F6F",
    "peach": "#FF9E6F",
    "orange": "#FFBC6F",
    "sunkist": "#FFDF6F",
    "yellow": "#FFEE6F",
    "lime": "#CBF169",
    "green": "#5CD25C",
    "turquoise": "#4AAB89",
    "blue": "#508EAD",
    "grape": "#635BB1",
    "violet": "#7C5AB8",
    "fuschia": "#C3559F",
    "brown": "#6B3F3F",
}

COLOR_CYCLE = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["violet"],
    PALETTE["fuschia"],
    PALETTE["turquoise"],
    PALETTE["grape"],
    PALETTE["lime"],
    PALETTE["peach"],
    PALETTE["sunkist"],
    PALETTE["yellow"],
    PALETTE["brown"],
]

MARKER_CYCLE = [".", "o", "s", "P", "X", "*", "p", "D", "<", ">", "^", "v", "1", "2", "3", "4", "+", "x"]

FONT_CONFIG = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"],
}

FIGURE_WIDTH = 6.75
GOLDEN_RATIO = 1.618034333
FIGURE_SIZE = (FIGURE_WIDTH, FIGURE_WIDTH / GOLDEN_RATIO)

AXIS_FONT = {"fontsize": 18}
SMALL_AXIS_FONT = {"fontsize": 15}
TICK_STYLE = {"labelsize": 18}

ERRORBAR_STYLE = {
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1,
}

ERRORBAR_CIRCLE_STYLE = {
    "marker": "o",
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1.5,
}

COMMON_LABELS = {
    "tmin": r"$t_{\mathrm{min}}~/~a$",
    "tmax": r"$t_{\mathrm{max}}~/~a$",
    "tau_center": r"$(\tau - t_{\rm{sep}}/2)~/~a$",
    "tsep": r"${t_{\mathrm{sep}}~/~a}$",
    "z": r"${z~/~a}$",
    "lambda": r"$\lambda = z P^z$",
    "meff": r"$m_{\rm eff}$",
}


def _require_matplotlib() -> None:
    if plt is None:
        raise OptionalDependencyError(
            "matplotlib is required for plotting presets. Install with `pip install matplotlib`."
        )


def apply_publication_style() -> None:
    """Apply global rcParams for consistent publication-style plots."""
    _require_matplotlib()
    from matplotlib import rcParams

    rcParams.update(FONT_CONFIG)


def auto_ylim(y_data: list[np.ndarray], yerr_data: list[np.ndarray], y_range_ratio: float = 4.0) -> tuple[float, float]:
    """Compute padded y-axis limits from central values and uncertainties."""
    upper = [np.asarray(y) + np.asarray(yerr) for y, yerr in zip(y_data, yerr_data, strict=True)]
    lower = [np.asarray(y) - np.asarray(yerr) for y, yerr in zip(y_data, yerr_data, strict=True)]
    all_values = np.concatenate(upper + lower)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_span = y_max - y_min
    return y_min - y_span / y_range_ratio, y_max + y_span / y_range_ratio


def default_plot():
    """Create a single-axis plot with the shared styling defaults."""
    _require_matplotlib()
    apply_publication_style()
    figure = plt.figure(figsize=FIGURE_SIZE)
    axis = plt.axes()
    axis.tick_params(direction="in", top=True, right=True, **TICK_STYLE)
    axis.grid(linestyle=":")
    return figure, axis


def default_sub_plot(height_ratio: int = 3):
    """Create a two-row subplot layout with a shared x-axis."""
    _require_matplotlib()
    apply_publication_style()
    figure, (upper_axis, lower_axis) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        gridspec_kw={"height_ratios": [height_ratio, 1]},
        sharex=True,
    )
    figure.subplots_adjust(hspace=0)
    for axis in (upper_axis, lower_axis):
        axis.tick_params(direction="in", top=True, right=True, **TICK_STYLE)
        axis.grid(linestyle=":")
    return figure, (upper_axis, lower_axis)
