"""Plotting helpers and common visual conventions for workflow artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from lamet_agent.errors import OptionalDependencyError
from lamet_agent.extensions.plot_presets import (
    AXIS_FONT,
    COLOR_CYCLE,
    COMMON_LABELS,
    ERRORBAR_CIRCLE_STYLE,
    default_plot,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    matplotlib = None
    plt = None


PLOT_STYLE = {
    "figure_size": (8, 5),
    "line_color": "#0b5fff",
    "line_width": 2.2,
    "grid_alpha": 0.25,
}


def apply_plot_style() -> None:
    """Apply the shared Matplotlib style when available."""
    if plt is None:
        return
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = PLOT_STYLE["figure_size"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = PLOT_STYLE["grid_alpha"]
    plt.rcParams["savefig.bbox"] = "tight"


def save_line_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    *,
    yscale: str = "linear",
) -> None:
    """Save a line plot in one of the supported formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if plt is None and suffix == ".svg":
        path.write_text(
            build_svg_line_plot(x_values, y_values, title=title, x_label=x_label, y_label=y_label),
            encoding="utf-8",
        )
        return
    if plt is None:
        raise OptionalDependencyError(
            f"Plot format {suffix} requires matplotlib, which is not installed in the current environment."
        )
    figure, axis = default_plot()
    axis.plot(x_values, y_values, color=COLOR_CYCLE[0], linewidth=PLOT_STYLE["line_width"])
    axis.set_title(title, **AXIS_FONT)
    axis.set_xlabel(_resolve_label(x_label), **AXIS_FONT)
    axis.set_ylabel(_resolve_label(y_label), **AXIS_FONT)
    axis.set_yscale(yscale)
    _save_figure(figure, path)


def save_band_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_errors: np.ndarray,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Save a mean curve with an uncertainty band."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    lower = np.asarray(y_values, dtype=float) - np.asarray(y_errors, dtype=float)
    upper = np.asarray(y_values, dtype=float) + np.asarray(y_errors, dtype=float)
    if plt is None and suffix == ".svg":
        path.write_text(
            build_svg_band_plot(
                x_values,
                y_values,
                lower,
                upper,
                title=title,
                x_label=x_label,
                y_label=y_label,
            ),
            encoding="utf-8",
        )
        return
    if plt is None:
        raise OptionalDependencyError(
            f"Plot format {suffix} requires matplotlib, which is not installed in the current environment."
        )
    figure, axis = default_plot()
    axis.plot(x_values, y_values, color=COLOR_CYCLE[0], linewidth=PLOT_STYLE["line_width"])
    axis.fill_between(x_values, lower, upper, color=COLOR_CYCLE[0], alpha=0.25)
    axis.set_title(title, **AXIS_FONT)
    axis.set_xlabel(_resolve_label(x_label), **AXIS_FONT)
    axis.set_ylabel(_resolve_label(y_label), **AXIS_FONT)
    _save_figure(figure, path)


def save_uncertainty_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_errors: np.ndarray,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    *,
    fit_x: np.ndarray | None = None,
    fit_y: np.ndarray | None = None,
    fit_error: np.ndarray | None = None,
    yscale: str = "linear",
    data_label: str | None = None,
    fit_label: str | None = None,
) -> None:
    """Save a plot with data as error bars and optional fit bands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None:
        raise OptionalDependencyError(
            f"Plot format {path.suffix.lower()} requires matplotlib, which is not installed in the current environment."
        )

    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    error_array = np.asarray(y_errors, dtype=float)
    finite_mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(error_array)
    x_array = x_array[finite_mask]
    y_array = y_array[finite_mask]
    error_array = error_array[finite_mask]

    figure, axis = default_plot()
    color = COLOR_CYCLE[0]
    has_uncertainty = np.any(error_array > 0)

    if has_uncertainty:
        axis.errorbar(
            x_array,
            y_array,
            yerr=error_array,
            color=color,
            label=data_label,
            **ERRORBAR_CIRCLE_STYLE,
        )
    else:
        axis.plot(
            x_array,
            y_array,
            color=color,
            linewidth=PLOT_STYLE["line_width"],
            label=data_label,
        )

    if fit_x is not None and fit_y is not None and fit_error is not None:
        fit_x_array = np.asarray(fit_x, dtype=float)
        fit_y_array = np.asarray(fit_y, dtype=float)
        fit_error_array = np.asarray(fit_error, dtype=float)
        fit_mask = np.isfinite(fit_x_array) & np.isfinite(fit_y_array) & np.isfinite(fit_error_array)
        axis.fill_between(
            fit_x_array[fit_mask],
            fit_y_array[fit_mask] - fit_error_array[fit_mask],
            fit_y_array[fit_mask] + fit_error_array[fit_mask],
            color=COLOR_CYCLE[1],
            alpha=0.30,
            label=fit_label,
        )

    axis.set_title(title, **AXIS_FONT)
    axis.set_xlabel(_resolve_label(x_label), **AXIS_FONT)
    axis.set_ylabel(_resolve_label(y_label), **AXIS_FONT)
    axis.set_yscale(yscale)
    handles, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend()
    _save_figure(figure, path)


def build_svg_line_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    """Build a lightweight SVG line plot without external plotting dependencies."""
    width = 720
    height = 460
    margin_left = 70
    margin_right = 24
    margin_top = 48
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    x_min, x_max = float(np.min(x_array)), float(np.max(x_array))
    y_min, y_max = float(np.min(y_array)), float(np.max(y_array))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def scale_x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        return height - margin_bottom - (value - y_min) / (y_max - y_min) * plot_height

    points = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(x_array, y_array, strict=False))
    x_ticks = generate_ticks(x_min, x_max, 5)
    y_ticks = generate_ticks(y_min, y_max, 5)
    x_tick_markup = "\n".join(
        f'<line x1="{scale_x(tick):.2f}" y1="{height - margin_bottom}" x2="{scale_x(tick):.2f}" y2="{height - margin_bottom + 6}" stroke="#334155" />'
        f'<text x="{scale_x(tick):.2f}" y="{height - margin_bottom + 22}" font-size="12" text-anchor="middle" fill="#0f172a">{tick:.3g}</text>'
        for tick in x_ticks
    )
    y_tick_markup = "\n".join(
        f'<line x1="{margin_left - 6}" y1="{scale_y(tick):.2f}" x2="{margin_left}" y2="{scale_y(tick):.2f}" stroke="#334155" />'
        f'<text x="{margin_left - 12}" y="{scale_y(tick) + 4:.2f}" font-size="12" text-anchor="end" fill="#0f172a">{tick:.3g}</text>'
        for tick in y_ticks
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc" />
  <text x="{width / 2:.2f}" y="28" font-size="20" text-anchor="middle" fill="#0f172a">{title}</text>
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#334155" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#334155" stroke-width="1.5" />
  {x_tick_markup}
  {y_tick_markup}
  <polyline fill="none" stroke="#0b5fff" stroke-width="2.5" points="{points}" />
  <text x="{width / 2:.2f}" y="{height - 18}" font-size="14" text-anchor="middle" fill="#0f172a">{x_label}</text>
  <text x="20" y="{height / 2:.2f}" font-size="14" text-anchor="middle" transform="rotate(-90 20 {height / 2:.2f})" fill="#0f172a">{y_label}</text>
</svg>
"""


def build_svg_band_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    lower_values: np.ndarray,
    upper_values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    """Build a lightweight SVG plot with a filled uncertainty band."""
    width = 720
    height = 460
    margin_left = 70
    margin_right = 24
    margin_top = 48
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    lower_array = np.asarray(lower_values, dtype=float)
    upper_array = np.asarray(upper_values, dtype=float)

    mask = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(lower_array) & np.isfinite(upper_array)
    x_array = x_array[mask]
    y_array = y_array[mask]
    lower_array = lower_array[mask]
    upper_array = upper_array[mask]
    if len(x_array) == 0:
        return build_svg_line_plot(np.array([0.0, 1.0]), np.array([0.0, 0.0]), title, x_label, y_label)

    x_min, x_max = float(np.min(x_array)), float(np.max(x_array))
    y_min = float(np.min(lower_array))
    y_max = float(np.max(upper_array))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def scale_x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        return height - margin_bottom - (value - y_min) / (y_max - y_min) * plot_height

    mean_points = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(x_array, y_array, strict=False))
    polygon_points = " ".join(
        [f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(x_array, upper_array, strict=False)]
        + [f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(x_array[::-1], lower_array[::-1], strict=False)]
    )
    x_ticks = generate_ticks(x_min, x_max, 5)
    y_ticks = generate_ticks(y_min, y_max, 5)
    x_tick_markup = "\n".join(
        f'<line x1="{scale_x(tick):.2f}" y1="{height - margin_bottom}" x2="{scale_x(tick):.2f}" y2="{height - margin_bottom + 6}" stroke="#334155" />'
        f'<text x="{scale_x(tick):.2f}" y="{height - margin_bottom + 22}" font-size="12" text-anchor="middle" fill="#0f172a">{tick:.3g}</text>'
        for tick in x_ticks
    )
    y_tick_markup = "\n".join(
        f'<line x1="{margin_left - 6}" y1="{scale_y(tick):.2f}" x2="{margin_left}" y2="{scale_y(tick):.2f}" stroke="#334155" />'
        f'<text x="{margin_left - 12}" y="{scale_y(tick) + 4:.2f}" font-size="12" text-anchor="end" fill="#0f172a">{tick:.3g}</text>'
        for tick in y_ticks
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc" />
  <text x="{width / 2:.2f}" y="28" font-size="20" text-anchor="middle" fill="#0f172a">{title}</text>
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#334155" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#334155" stroke-width="1.5" />
  {x_tick_markup}
  {y_tick_markup}
  <polygon fill="#0b5fff" fill-opacity="0.18" stroke="none" points="{polygon_points}" />
  <polyline fill="none" stroke="#0b5fff" stroke-width="2.5" points="{mean_points}" />
  <text x="{width / 2:.2f}" y="{height - 18}" font-size="14" text-anchor="middle" fill="#0f172a">{x_label}</text>
  <text x="20" y="{height / 2:.2f}" font-size="14" text-anchor="middle" transform="rotate(-90 20 {height / 2:.2f})" fill="#0f172a">{y_label}</text>
</svg>
"""


def generate_ticks(start: float, stop: float, count: int) -> Iterable[float]:
    """Return evenly spaced tick locations."""
    return np.linspace(start, stop, count)


def _resolve_label(label: str) -> str:
    return COMMON_LABELS.get(label, label)


def _save_figure(figure, path: Path) -> None:
    """Save a Matplotlib figure with the shared export defaults."""
    figure.tight_layout()
    figure.savefig(path, transparent=True)
    plt.close(figure)
