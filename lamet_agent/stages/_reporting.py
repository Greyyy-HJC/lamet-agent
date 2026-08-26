"""Shared, deterministic helpers for stage-level Markdown reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StageReportRecord:
    """One completed job presented to its owning stage reporter."""

    job_id: str
    params: Mapping[str, Any]
    inputs: Mapping[str, Any]
    output: Any
    summary: Mapping[str, Any]
    artifact_directory: Path


def format_value(value: Any) -> str:
    """Format compact scalar or structured report values deterministically."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    if isinstance(value, (dict, list, tuple)):
        return f"`{json.dumps(value, sort_keys=True, ensure_ascii=False)}`"
    return str(value)


def output_attrs(record: StageReportRecord) -> Mapping[str, Any]:
    attrs = getattr(record.output, "attrs", {})
    return attrs if isinstance(attrs, Mapping) else {}


def artifact_rows(record: StageReportRecord, stage_directory: Path) -> list[str]:
    """Return verified Markdown rows for every artifact declared by a job."""
    raw = record.summary.get("artifacts", [])
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise TypeError(f"job '{record.job_id}' summary.artifacts must be a string list")
    rows: list[str] = []
    for relative in raw:
        path = (record.artifact_directory / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"job '{record.job_id}' declared missing artifact: {path}")
        link = path.relative_to(stage_directory.resolve()).as_posix()
        rows.append(f"| `{record.job_id}` | [{relative}]({link}) |")
    return rows


def figure_lines(record: StageReportRecord, stage_directory: Path) -> list[str]:
    """Render SVG plots inline and link all other declared plot formats."""
    raw = record.summary.get("artifacts", [])
    if not isinstance(raw, list):
        raise TypeError(f"job '{record.job_id}' summary.artifacts must be a list")
    lines: list[str] = []
    for relative in raw:
        if not isinstance(relative, str) or "/plots/" not in f"/{relative}" and not relative.startswith("plots/"):
            continue
        path = (record.artifact_directory / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"job '{record.job_id}' declared missing plot: {path}")
        link = path.relative_to(stage_directory.resolve()).as_posix()
        label = f"{record.job_id}: {Path(relative).stem}"
        lines.append(f"![{label}]({link})" if path.suffix.lower() == ".svg" else f"[{label}]({link})")
    return lines or ["No plot artifacts were declared."]


def describe_grid(values: Any, *, symbol: str = "x") -> str:
    """Describe a finite one-dimensional coordinate grid."""
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or not grid.size or np.any(~np.isfinite(grid)):
        return "not available"
    if grid.size == 1:
        return f"one point at ${symbol}={grid[0]:.6g}$"
    spacing = np.diff(grid)
    if np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-12):
        return f"{grid.size} points from ${symbol}={grid[0]:.6g}$ to ${symbol}={grid[-1]:.6g}$ with $\\Delta {symbol}={spacing[0]:.6g}$"
    return f"{grid.size} nonuniform points from ${symbol}={grid.min():.6g}$ to ${symbol}={grid.max():.6g}$"


def stage_overlay_lines(
    records: tuple[StageReportRecord, ...],
    artifact_directory: Path,
    *,
    coordinate: str,
    stem: str,
    ylabel: str,
) -> list[str]:
    """Create stage-level real/imaginary overlays from compatible 1D outputs."""
    from lamet_agent.plotting import configure_plot, errorbar, save_figure, start_plot

    usable = [record for record in records if getattr(record.output, "dims", None) == [coordinate]]
    if not usable:
        return ["No compatible one-dimensional outputs were available for a stage overlay."]
    complex_output = any(np.iscomplexobj(record.output.values) for record in usable)
    components = (("real", "Re"), ("imag", "Im")) if complex_output else (("real", ""),)
    lines: list[str] = []
    for component, suffix in components:
        start_plot()
        plotted = 0
        for record in usable:
            data = getattr(record.output, component) if np.iscomplexobj(record.output.values) else record.output
            mode = str(data.attrs.get("sample_error_mode", "covariance"))
            average = data.average(mode)
            errorbar(data.coords[coordinate], average, label=record.job_id)
            plotted += 1
        configure_plot(xlabel=coordinate, ylabel=f"{suffix} {ylabel}".strip(), legend=plotted <= 15)
        suffix_name = f"_{component}" if complex_output else ""
        pdf = artifact_directory / "plots" / f"{stem}{suffix_name}.pdf"
        svg = artifact_directory / "plots" / f"{stem}{suffix_name}.svg"
        save_figure(pdf, svg)
        lines.extend([
            f"![{suffix or ylabel} stage overlay](plots/{svg.name})",
            "",
            f"[{suffix or ylabel} stage overlay (PDF)](plots/{pdf.name})",
            "",
        ])
    return lines


def write_report(artifact_directory: Path, lines: list[str]) -> Path:
    """Write the one canonical stage report."""
    path = artifact_directory / "report.md"
    if path.exists():
        raise FileExistsError(f"stage report already exists: {path}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
