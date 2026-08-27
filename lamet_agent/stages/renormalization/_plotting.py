"""Deterministic renderers for renormalization diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import gvar
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.plotting import (
    configure_plot,
    continuous_color,
    errorline,
    hline,
    line,
    save_figure,
    series_color,
    start_plot,
)

Z_FM_LABEL = r"$z\,[\mathrm{fm}]$"
INVERSE_LATTICE_SPACING_LABEL = r"$a^{-1}\,[\mathrm{GeV}]$"
SELF_RENORMALIZATION_FACTOR_LABEL = r"$Z_R(z,a)$"
RENORMALIZED_MATRIX_ELEMENT_LABELS = {
    "real": r"$\mathrm{Re}\,h^R(z)$",
    "imag": r"$\mathrm{Im}\,h^R(z)$",
    "both": r"$h^R(z)$",
}


def lattice_spacing_label(spacing_fm: object, *, default: str | None = None) -> str:
    """Format one stage-owned lattice-spacing label."""
    if isinstance(spacing_fm, Real) and not isinstance(spacing_fm, bool):
        spacing = float(spacing_fm)
        if math.isfinite(spacing) and spacing > 0:
            return rf"$a={spacing:.4g}\,\mathrm{{fm}}$"
    if default is not None:
        return default
    raise ValueError("spacing_fm must be a finite positive real number")


def momentum_label(
    momentum_gev: object,
    *,
    momentum: object | None = None,
    default: str | None = None,
) -> str:
    """Format one stage-owned momentum label."""
    if isinstance(momentum_gev, Real) and not isinstance(momentum_gev, bool):
        value = float(momentum_gev)
        if math.isfinite(value):
            return rf"$P_z={round(value, 2):g}\,\mathrm{{GeV}}$"
    if isinstance(momentum, str):
        match = re.fullmatch(r"PX(-?\d+)PY(-?\d+)PZ(-?\d+)", momentum.upper())
        if match is not None:
            px, py, pz = (int(value) for value in match.groups())
            if px == 0 and py == 0:
                return rf"$P_z={pz}\,(2\pi/L)$"
            return rf"$\mathbf{{P}}=({px},{py},{pz})\,(2\pi/L)$"
    if default is not None:
        return default
    raise ValueError("momentum_gev must be a finite real number")

_FIT_CAPTIONS = {
    "factor": "Reusable self-renormalization factor",
    "fit_lnM_vs_inv_a": r"Reference $\ln|M|$ fit input",
    "fit_mR_zmsbar": r"Finite factor and $Z_{\overline{\mathrm{MS}}}$ comparison",
    "fit_m_over_zR": r"Self-renormalization consistency check",
    "fit_f1": r"Discretization coefficient $f_1(z)$",
}


def _output_paths(directory: Path, stem: str, formats: Sequence[str]) -> tuple[Path, ...]:
    normalized = tuple(str(value).lower().lstrip(".") for value in formats)
    if not normalized or any(value not in {"pdf", "svg"} for value in normalized):
        raise ValueError("renormalization plot formats must be a nonempty subset of pdf and svg")
    return tuple(directory / f"{stem}.{suffix}" for suffix in normalized)


def _gvars(mean: Any, sdev: Any) -> np.ndarray:
    return np.asarray(gvar.gvar(np.asarray(mean, dtype=float), np.asarray(sdev, dtype=float)), dtype=object)


def render_fit_diagnostics(
    plot_data: Mapping[str, Any],
    *,
    directory: Path,
    prefix: str = "",
    formats: Sequence[str] = ("pdf",),
) -> list[tuple[str, str]]:
    """Render the five fit diagnostics from serialized fit results."""
    z = np.asarray(plot_data["z_fm"], dtype=float)
    a = np.asarray(plot_data["a_fm"], dtype=float)
    inverse_a = np.asarray(plot_data["inverse_a_gev"], dtype=float)
    factor = _gvars(plot_data["factor_mean"], plot_data["factor_sdev"])
    lnm = _gvars(plot_data["lnm_mean"], plot_data["lnm_sdev"])
    m_r = _gvars(plot_data["mR_mean"], plot_data["mR_sdev"])
    m_r_ratio_mean = np.asarray(plot_data["mR_over_zmsbar_mean"], dtype=float)
    m_r_ratio_sdev = np.asarray(plot_data["mR_over_zmsbar_sdev"], dtype=float)
    m_over_zr = _gvars(plot_data["m_over_zR_mean"], plot_data["m_over_zR_sdev"])
    f1 = _gvars(plot_data["f1_mean"], plot_data["f1_sdev"])
    zmsbar = np.asarray(plot_data["zmsbar"], dtype=float)
    if factor.shape != (len(a), len(z)) or lnm.shape != factor.shape or m_over_zr.shape != factor.shape:
        raise ValueError("fit plot data has inconsistent (a,z) shapes")

    rendered: list[tuple[str, str]] = []

    stem = f"{prefix}factor"
    start_plot()
    for index, spacing in enumerate(a):
        errorline(z, factor[index], color=series_color(index), label=lattice_spacing_label(spacing))
    configure_plot(xlabel=Z_FM_LABEL, ylabel=SELF_RENORMALIZATION_FACTOR_LABEL, legend=True)
    save_figure(*_output_paths(directory, stem, formats))
    rendered.append((stem, _FIT_CAPTIONS["factor"]))

    stem = f"{prefix}fit_lnM_vs_inv_a"
    start_plot()
    highlighted = set(np.linspace(0, len(z) - 1, min(7, len(z)), dtype=int).tolist())
    for index, coordinate in enumerate(z):
        label = rf"$z={coordinate:.3g}\,\mathrm{{fm}}$" if index in highlighted else None
        errorline(inverse_a, lnm[:, index], color=continuous_color(index, len(z)), label=label)
    configure_plot(
        xlabel=INVERSE_LATTICE_SPACING_LABEL,
        ylabel=r"$\ln|M(z,a)|$",
        legend=bool(highlighted),
        legend_loc="best",
    )
    save_figure(*_output_paths(directory, stem, formats))
    rendered.append((stem, _FIT_CAPTIONS["fit_lnM_vs_inv_a"]))

    stem = f"{prefix}fit_mR_zmsbar"
    start_plot()
    finite = np.isfinite(zmsbar) & np.isfinite(m_r_ratio_mean) & np.isfinite(m_r_ratio_sdev)
    m_r_ratio = _gvars(m_r_ratio_mean[finite], m_r_ratio_sdev[finite])
    line(z[finite], zmsbar[finite], color="0.2", label=r"$Z_{\overline{\mathrm{MS}}}(z,\mu)$")
    errorline(z, m_r, color=series_color(0), label=r"$m_R(z)=\exp[g(z)-m_0z]$")
    errorline(
        z[finite], m_r_ratio[finite], color=series_color(1), marker="s", label=r"$m_R/Z_{\overline{\mathrm{MS}}}$"
    )
    configure_plot(xlabel=Z_FM_LABEL, ylabel="factor", legend=True)
    save_figure(*_output_paths(directory, stem, formats))
    rendered.append((stem, _FIT_CAPTIONS["fit_mR_zmsbar"]))

    stem = f"{prefix}fit_m_over_zR"
    start_plot()
    for index, spacing in enumerate(a):
        errorline(z, m_over_zr[index], color=series_color(index), label=lattice_spacing_label(spacing))
    errorline(z, m_r, color=series_color(len(a)), marker="x", label=r"$m_R(z)$")
    configure_plot(xlabel=Z_FM_LABEL, ylabel=r"$M_{\mathrm{bare}}(z,a)/Z_R(z,a)$", legend=True)
    save_figure(*_output_paths(directory, stem, formats))
    rendered.append((stem, _FIT_CAPTIONS["fit_m_over_zR"]))

    stem = f"{prefix}fit_f1"
    start_plot()
    errorline(z, f1, color=series_color(0), label=r"$f_1(z)$")
    hline(0.0, color="0.3", linestyle="dashed")
    configure_plot(xlabel=Z_FM_LABEL, ylabel=r"$f_1(z)\,[\mathrm{GeV}]$", legend=True)
    save_figure(*_output_paths(directory, stem, formats))
    rendered.append((stem, _FIT_CAPTIONS["fit_f1"]))
    return rendered


def render_result(
    data: EnsembleData,
    *,
    directory: Path,
    stem: str,
    formats: Sequence[str] = ("pdf",),
    sample_error_mode: str | None = None,
) -> tuple[str, str]:
    """Render one renormalized matrix element with explicit components."""
    mode = sample_error_mode or str(data.attrs.get("sample_error_mode", "covariance"))
    start_plot()
    if np.iscomplexobj(data.values):
        errorline(data.coords["z"], data.real.average(mode), color=series_color(0), label="real")
        errorline(data.coords["z"], data.imag.average(mode), color=series_color(1), marker="s", label="imaginary")
        ylabel = RENORMALIZED_MATRIX_ELEMENT_LABELS["both"]
        legend = True
    else:
        errorline(data.coords["z"], data.average(mode), color=series_color(0))
        ylabel = RENORMALIZED_MATRIX_ELEMENT_LABELS["real"]
        legend = False
    hline(0.0, color="0.3", linestyle="dashed")
    configure_plot(xlabel=Z_FM_LABEL, ylabel=ylabel, legend=legend)
    save_figure(*_output_paths(directory, stem, formats))
    return stem, "Renormalized matrix element"


def render_zmsbar_comparison(
    plot_data: Mapping[str, Any],
    *,
    directory: Path,
    stem: str,
    formats: Sequence[str] = ("pdf",),
) -> tuple[str, str]:
    """Render the apply-time H/Z_R and finite-kernel comparison."""
    z = np.asarray(plot_data["z_fm"], dtype=float)
    h_over_zr = _gvars(plot_data["h_over_zR_real_mean"], plot_data["h_over_zR_real_sdev"])
    zmsbar = np.asarray(plot_data["zmsbar"], dtype=float)
    if h_over_zr.shape != z.shape or zmsbar.shape != z.shape:
        raise ValueError("apply plot data has inconsistent z shapes")
    start_plot()
    errorline(z, h_over_zr, color=series_color(0), label=r"$\mathrm{Re}[H/Z_R]$")
    line(z, zmsbar, color=series_color(1), label=r"$Z_{\overline{\mathrm{MS}}}(z,\mu)$")
    hline(0.0, color="0.3", linestyle="dashed")
    configure_plot(xlabel=Z_FM_LABEL, ylabel=r"$\mathrm{Re}[H(z)/Z_R(z,a)]$", legend=True)
    save_figure(*_output_paths(directory, stem, formats))
    return stem, r"$H/Z_R$ and $Z_{\overline{\mathrm{MS}}}$ comparison"


__all__ = ["render_fit_diagnostics", "render_result", "render_zmsbar_comparison"]
