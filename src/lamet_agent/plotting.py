"""Plotting helpers and common visual conventions for workflow artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from lamet_agent.errors import OptionalDependencyError

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
) -> None:
    """Save a line plot in one of the supported formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".svg":
        path.write_text(
            build_svg_line_plot(x_values, y_values, title=title, x_label=x_label, y_label=y_label),
            encoding="utf-8",
        )
        return
    if plt is None:
        raise OptionalDependencyError(
            f"Plot format {suffix} requires matplotlib, which is not installed in the current environment."
        )
    apply_plot_style()
    figure, axis = plt.subplots()
    axis.plot(x_values, y_values, color=PLOT_STYLE["line_color"], linewidth=PLOT_STYLE["line_width"])
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    figure.savefig(path)
    plt.close(figure)


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


def generate_ticks(start: float, stop: float, count: int) -> Iterable[float]:
    """Return evenly spaced tick locations."""
    return np.linspace(start, stop, count)
