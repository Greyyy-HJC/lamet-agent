"""Markdown reporting helpers for the perturbative-matching stage.

The organizing rule: **nothing here is hardcoded to one kernel**. The same report
serves a Coulomb-gauge quark PDF and a gauge-invariant meson DA, which differ in
their factorization, their notation, and their source paper -- so every statement is
derived from the ``kernel_id`` the manifest chose, and the numbers themselves always
come from ``kernels.py`` alone (nothing here can change a result).

The functions fall into five groups:

1. Reading the kernel_id. ``_split_kernel_id`` breaks
   ``<gauge>_<operator>_<distribution>_<scheme>_<order>`` into its fields;
   ``_parse_kernel_id`` is the common two-field view; ``is_da_kernel`` and
   ``_kernel_description`` answer "what is this kernel?" from those fields plus the
   OPERATOR_TEXT / DISTRIBUTION_TEXT / PDF_POLARIZATION_TEXT tables at the top. Add a
   kernel with a new operator or distribution and those tables want a new row.

2. Reading the kernel's provenance. ``_kernel_reference`` returns the paper and
   equations the kernel function tags itself with (``@kernel_reference`` in
   kernels.py), so citations follow the kernel rather than a table kept in sync here.

3. The static tables: ``_settings_table``, ``_field_definitions``,
   ``_scheme_explanation``, ``_diagnostics``, ``_figure_block``, ``_outputs_table``.
   Plain formatting of the job record the stage produced.

4. The formula section -- the only part that calls an LLM. ``_kernel_source`` collects
   the kernel and everything it actually calls (following the call graph, so a DA
   kernel gets its ``V(x, y)`` and a PDF kernel its ``C(ksi)``); ``_fetch_paper_text``
   pulls the LaTeX of the tagged arXiv paper; ``_formula_prompt`` asks the model to
   write the closed form *and* cross-check code against paper; ``_llm_kernel_formula``
   makes the call and caches it; ``_matching_formula_text`` wraps that in the
   factorization. The factorization is NOT branched on here: ``_kernel_structure`` reads
   the ``matching_structure`` the kernel declares in ``kernels.py`` (its display equation,
   its notation guidance, and any all-orders resummation), and both the prompt and the
   rendered factorization follow it -- so a DA kernel, a PDF kernel, or a renormalon-resummed
   kernel each render themselves with no ``if is_da``/``if is_lrr`` in this module, and a new
   kernel needs no change here. Division of labour: the code is authoritative for WHICH
   terms exist, the paper for how they are WRITTEN, and the cross-check reports where the
   two disagree. ``FormulaLlm`` carries the run's LLM config down from the CLI. The formula
   helper raises rather than inventing a formula offline. Because generation costs a paper
   download plus a ~27k-token prompt, and cannot change while the kernel does not, the
   result is cached on disk: ``formulas/`` ships inside the wheel (fill it with
   ``lamet-agent precompute-formulas``) and the user cache directory catches the rest. The
   cache file header digests the kernel's own source, so an edited kernel misses, is
   regenerated rather than served a stale formula, and overwrites its own file.
   If an uncached report-only request fails, the report records that failure without
   changing or discarding the already-computed numerical result.

5. Assembly and writing: ``build_matching_report_markdown`` orders the sections,
   ``write_matching_report`` writes one job's file, and ``write_matching_stage_report``
   writes the one report per stage that the runner actually calls.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import inspect
import io
import os
import re
import ssl
import tarfile
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent import kernels
from lamet_agent.core.llm import PROVIDERS, provider_config, request_llm_text
from lamet_agent.core.reporting import (
    format_report_list as _fmt_list,
    format_report_value as _fmt,
    markdown_artifact_paths,
    resolve_report_target as _report_target,
    translate_markdown_report,
)


# Logical operator -> human text, keyed by the ``<operator>`` field of a
# ``<gauge>_<operator>_<distribution>_<scheme>_<order>`` kernel_id. The Dirac structure
# only; the distribution it measures comes from DISTRIBUTION_TEXT, since the same
# operator serves a PDF and a DA. Naming the polarization here too would say
# "helicity DA", which is not a thing.
OPERATOR_TEXT = {
    "gt": "$\\gamma^t$",
    "gtg5": "$\\gamma^t\\gamma_5$",
    "gz": "$\\gamma^z$",
    "gzg5": "$\\gamma^z\\gamma_5$",
    "gtgpg5": "$\\gamma^t\\gamma_\\perp\\gamma_5$",
}

# What the operator measures, keyed by the ``<distribution>`` field of the id. A PDF's
# polarization is the operator's; a DA has none, so it is not spelled out here.
DISTRIBUTION_TEXT = {
    "quark_PDF": "quark PDF",
    "gluon_PDF": "gluon PDF",
    "DA": "meson distribution amplitude",
    "qDA": "quark distribution amplitude",
    "gDA": "gluon distribution amplitude",
}

# The polarization a Dirac structure selects, but only for a PDF -- a DA's gamma^z gamma_5
# is not a "helicity DA".
PDF_POLARIZATION_TEXT = {
    "gt": "unpolarized",
    "gtg5": "helicity",
    "gz": "unpolarized",
    "gzg5": "helicity",
    "gtgpg5": "transversity",
}

# Scheme -> human text. The paper and equation numbers are NOT listed here: they are
# tagged on each kernel in kernels.py (@kernel_reference) and read back by
# _kernel_reference below, so kernels from different papers each cite their own.
SCHEME_TEXT = {
    "msbar": "MSbar",
    "ratio": "ratio",
    "hybrid": "hybrid",
}

MATCHING_ARTIFACT_DESCRIPTIONS = {
    "lightcone_artifact": "Matched light-cone PDF samples (EnsembleData NetCDF)",
    "matched_plot": "PDF plot comparing quasi and light-cone PDFs",
    "matched_plot_image": "SVG companion for Markdown embedding",
}

MATCHING_ARTIFACT_ORDER = ("lightcone_artifact", "matched_plot", "matched_plot_image")


DISTRIBUTION_TOKENS = ("quark_PDF", "gluon_PDF", "DA", "qDA", "gDA")

DA_TOKENS = frozenset({"DA", "qDA", "gDA"})


def is_da_kernel(kernel_id: str) -> bool:
    """True for a distribution-amplitude kernel, whose factorization has a different shape.

    A DA kernel's density is a genuine two-variable ``V(x, y)`` carrying its own poles and
    integrated with a plain ``dy``; a PDF kernel's is a coefficient of ``ksi = x/y`` alone,
    integrated with ``dy/|y|``. The two therefore diverge differently at the endpoints, so
    callers that treat them alike would misstate whichever kernel they were not written for.
    """
    return any(part in DA_TOKENS for part in str(kernel_id).split("_"))


def _parse_kernel_id(kernel_id: str) -> tuple[str, str]:
    """Split a ``<gauge>_<operator>_<distribution>_<scheme>_<order>`` id into (operator, scheme).

    See ``_split_kernel_id`` for the field layout; this is the two-field view most callers
    want. Falls back to ('', '') for an id that does not follow the convention so the
    report degrades gracefully instead of raising.
    """
    _gauge, operator, _distribution, scheme = _split_kernel_id(kernel_id)
    return operator, scheme


def _split_kernel_id(kernel_id: str) -> tuple[str, str, str, str]:
    """Split an id into (gauge, operator, distribution, scheme).

    The distribution token (quark_PDF/gluon_PDF for the quark/gluon PDF, DA for the meson
    distribution amplitude) separates the operator from the scheme, and the order (NLO)
    trails it. A token may itself span several ``_`` segments, so match on joined segments
    rather than on a single one. Every field is returned rather than hardcoded by callers:
    the same report serves a Coulomb-gauge quark PDF and a gauge-invariant DA, and naming
    either one's fields in prose would misdescribe the other. Falls back to empty strings
    for an id that does not follow the convention.
    """
    parts = str(kernel_id).split("_")
    # <gauge>, <op...>, quark_PDF|DA, <scheme>, <order>
    for idx in range(2, len(parts)):
        for token in DISTRIBUTION_TOKENS:
            width = len(token.split("_"))
            if idx + width < len(parts) and "_".join(parts[idx : idx + width]) == token:
                return parts[0], "_".join(parts[1:idx]), token, parts[idx + width]
    return "", "", "", ""


def _kernel_description(kernel_id: str, *, language: str) -> str:
    """Describe what a kernel matches, composed from the id's own fields.

    A PDF is named by its polarization and Dirac structure; a DA by its Dirac structure
    alone, since "helicity DA" would be a category error. Anything the id does not spell
    out is left out rather than guessed.
    """
    _gauge, operator, distribution, _scheme = _split_kernel_id(kernel_id)
    dirac = OPERATOR_TEXT.get(operator, operator)
    measured = DISTRIBUTION_TEXT.get(distribution, distribution)
    if not measured:
        return dirac or "not recorded"
    if distribution in DA_TOKENS:
        return f"{dirac} {measured}" if dirac else measured
    polarization = PDF_POLARIZATION_TEXT.get(operator, "")
    parts_text = [part for part in (polarization, dirac, measured) if part]
    return " ".join(parts_text)


def _kernel_structure(kernel_id: str) -> dict[str, Any]:
    """Return the render-structure the kernel declares (its ``matching_structure``).

    The kernel is the single source of truth for how its factorization is drawn (see
    ``kernels.py``): whether it is a PDF coefficient $C(x/y)$, a DA $V(x,y)$, or carries an
    all-orders resummation on top. This module only reads that description and renders it,
    so it never enumerates kernel families -- a new kernel needs no change here. A kernel
    with no declaration (or an unknown id) falls back to letting the formula LLM state the
    factorization from the source code alone.
    """
    fn = getattr(kernels, str(kernel_id), None)
    structure = getattr(fn, "matching_structure", None)
    if isinstance(structure, dict):
        return structure
    return {
        "factorization": None,
        "result_noun": "light-cone distribution",
        "source_noun": "quasi distribution",
        "notation": (
            "- Read the factorization and the coefficient off the code above; state the "
            "matching relation the kernel implements and the explicit coefficient, in the "
            "paper's notation.\n"
        ),
        "extra_structure": None,
        "extra_note": None,
    }


def _kernel_reference(kernel_id: str) -> tuple[str, str]:
    """Return the ``(arxiv_id, equations)`` tagged on the kernel the manifest selected.

    The manifest names the kernel, the registry name is the function name in kernels.py,
    and the function carries its own provenance (``@kernel_reference``) -- so the paper
    follows the kernel, with no table here to keep in sync and no default paper baked in.
    An unknown or untagged kernel_id yields ``("", "")``: the report then cites nothing
    and the formula is derived from the code alone, rather than pointing at some other
    paper's equations. Every registered kernel is tagged (a test enforces it).
    """
    fn = getattr(kernels, str(kernel_id), None)
    return getattr(fn, "arxiv_id", "") or "", getattr(fn, "equations", "") or ""


def _format_grid(x_grid: np.ndarray, *, language: str) -> str:
    if x_grid.size == 0:
        return "not recorded"
    if x_grid.size == 1:
        return f"one point at $x={_fmt(x_grid[0])}$"
    diffs = np.diff(x_grid)
    if np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        return f"from $x={_fmt(x_grid[0])}$ to $x={_fmt(x_grid[-1])}$ with spacing $\\Delta x={_fmt(diffs[0])}$, for {x_grid.size} points"
    return f"nonuniform grid with {x_grid.size} points; preview `{_fmt_list(x_grid)}`"


def _trapz_norm(
    x_grid: np.ndarray, values: np.ndarray, *, lo: float | None = None, hi: float | None = None
) -> float:
    """Integral of ``values`` over the x grid, optionally restricted to ``[lo, hi]``.

    The window is always passed in by the caller rather than fixed here: which x range
    carries the normalization is a property of the distribution and of the run (see
    ``_norm_summary``), not of the trapezoid rule.
    """
    if x_grid.size < 2 or values.size != x_grid.size:
        return float("nan")
    order = np.argsort(x_grid)
    x_sorted = x_grid[order]
    v_sorted = values[order]
    if lo is not None or hi is not None:
        mask = np.ones(x_sorted.shape, dtype=bool)
        if lo is not None:
            mask &= x_sorted >= lo
        if hi is not None:
            mask &= x_sorted <= hi
        x_sorted, v_sorted = x_sorted[mask], v_sorted[mask]
        if x_sorted.size < 2:
            return float("nan")
    # np.trapezoid is the NumPy 2.x name; fall back to np.trapz on older NumPy.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(v_sorted, x_sorted))


# The diagnostics section reports the integral over the range the job actually matched, and
# then two notes. It asserts no expected value and names no physics: whether the integral
# should be 1 is a convention set two stages upstream (whether the matrix element was
# normalized at z=0), and which combination a sector selects depends on the operator's charge
# conjugation, which this stage never sees. So the notes say only what is visible here -- how
# the stored array sits on the axis, and what to check the number against.


def _output_scale(data: dict[str, Any]) -> float:
    """The Fourier stage's ``output_scale``, which says how many times the integral counts.

    The Fourier stage transforms with the extended-distribution convention
    $h(\\lambda)=\\int dx\\,e^{ix\\lambda}q_{\\rm ext}(x)$ and then multiplies by a projection
    factor it records as ``output_scale``: 2 for the ``valence`` and ``singlet`` sectors, whose
    single-channel (Re or Im) transform mirrors the distribution about $x=0$, and 1 for
    ``full``/``sea`` and for a DA. So a factor of 2 means the integral over the whole matched
    range counts one physical side twice, and the one-sided integral is half of it. Reading
    the factor off the run rather than assuming keeps that statement right for whichever
    projection the manifest selected, including ones added later.

    The value travels on the data: the Fourier stage writes it into the quasi-PDF's attrs,
    ``apply_matching`` copies the quasi attrs onto the matched EnsembleData, and the runner
    spreads those attrs into the job record. A record without it (a hand-built one, or an
    older artifact) falls back to 1, the unscaled convention.
    """
    try:
        scale = float(data.get("output_scale", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _is_even_about_zero(x_grid: np.ndarray, values: np.ndarray) -> bool:
    """Whether the stored distribution actually repeats itself across $x=0$.

    The mirroring is a property of the projection, not of the physics: a single (Re) channel
    of $\\int d\\lambda\\,e^{-ix\\lambda}h(\\lambda)$ is even in $x$ whatever the hadron does,
    so a valence run's array literally holds each side twice (~1e-14 on a real run). But that
    is worth checking rather than inferring from ``output_scale``, which only records what the
    Fourier stage intended: a record whose factor and array disagree -- an externally built
    quasi-distribution, a projection added later that scales without mirroring -- would
    otherwise be told it has a doubled side it does not have.
    """
    if x_grid.size < 3 or float(np.min(x_grid)) >= 0.0 or float(np.max(x_grid)) <= 0.0:
        return False
    order = np.argsort(x_grid)
    x_sorted, v_sorted = x_grid[order], values[order]
    scale = float(np.max(np.abs(v_sorted))) if v_sorted.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return False
    # Points whose reflection falls outside the grid clamp to the edge value, so a window
    # that is not symmetric about 0 fails here too -- which is the honest answer: half of it
    # has nothing to be compared against.
    reflected = np.interp(-x_sorted, x_sorted, v_sorted)
    return bool(np.max(np.abs(v_sorted - reflected)) <= 1e-6 * scale)


def _has_interior_gap(x_grid: np.ndarray) -> bool:
    """True when the grid is missing an interior stretch of points.

    A non-uniform output grid can leave the two endpoints in place but nothing
    between them and the rest of the window. The trapezoid rule then bridges that
    hole with a straight line, so the integral over the matched range is part
    interpolation -- worth saying out loud rather than reporting as if every point
    in the window were data.
    """
    if x_grid.size < 4:
        return False
    diffs = np.diff(np.sort(x_grid))
    median = float(np.median(diffs))
    return bool(median > 0.0 and float(np.max(diffs)) > 2.0 * median)


def _norm_summary(data: dict[str, Any]) -> dict[str, Any] | None:
    """Integrate the quasi and matched distributions over the range matching produced.

    The window is the light-cone grid's own $[\\min x, \\max x]$ -- what this job actually
    matched -- rather than a fixed $[0, 1]$: a DA lives on
    $[0, 1]$, a PDF grid runs over both the quark and antiquark sides, and a narrower
    ``lc_x_ls`` window shortens whichever it is, so any hardcoded window is wrong for some kernel the manifest
    may select. Both distributions are integrated over that same window so the comparison is
    like for like, each on its own grid.

    Both the raw integral and the one-sided one (integral / ``output_scale``) are returned:
    where the projection mirrors the distribution about $x=0$, the second is the integral
    over one physical side -- computed as half the full-range integral rather than by masking
    $x\\ge 0$, which would drop half of the bin straddling $x=0$ (worth ~1% on a 100-point
    grid, since the distribution peaks right there). Neither is compared against anything
    here; what they are worth reading against is left to the notes.

    Returns ``None`` when the job record does not carry enough to integrate.
    """
    x_grid = np.asarray(data.get("x_grid", []), dtype=float)
    lc_mean = np.asarray(data.get("lightcone_mean", []), dtype=float)
    quasi_mean = np.asarray(data.get("quasi_mean", []), dtype=float)
    quasi_grid = np.asarray(data.get("quasi_x_grid", []), dtype=float)
    if quasi_grid.size != quasi_mean.size:
        # Older job records stored no separate quasi grid because the two coincided.
        quasi_grid = x_grid
    if x_grid.size < 2 or lc_mean.size != x_grid.size or quasi_mean.size != quasi_grid.size:
        return None
    lo, hi = float(np.min(x_grid)), float(np.max(x_grid))
    quasi_val = _trapz_norm(quasi_grid, quasi_mean, lo=lo, hi=hi)
    lc_val = _trapz_norm(x_grid, lc_mean, lo=lo, hi=hi)
    rel = abs(lc_val - quasi_val) / abs(quasi_val) if quasi_val else float("nan")
    scale = _output_scale(data)
    return {
        "lo": lo,
        "hi": hi,
        "quasi": quasi_val,
        "lightcone": lc_val,
        "rel_change": rel,
        "scale": scale,
        "part": str(data.get("component", data.get("part", "")) or ""),
        "unit_quasi": quasi_val / scale,
        "unit_lightcone": lc_val / scale,
        "interpolated_gap": _has_interior_gap(x_grid),
        # Mirroring is claimed only where the stored array really does repeat across x=0 and
        # the run recorded a factor to divide back out -- both, not either.
        "mirrored": bool(abs(scale - 1.0) > 1e-9 and _is_even_about_zero(x_grid, lc_mean)),
    }


def _norm_window_text(summary: dict[str, Any]) -> str:
    """The matched window as a display integral, with the run's own limits."""
    return f"$\\int_{{{_fmt(summary['lo'])}}}^{{{_fmt(summary['hi'])}}} f\\,dx$"


def _symmetry_note(summaries: list[dict[str, Any]]) -> str:
    """One line on how the stored distribution sits on the axis, and nothing beyond that.

    Deliberately no physics label. Which combination a sector selects is a question about the
    operator's charge conjugation as much as about the sector name -- the same ``valence``
    picks out $q-\\bar q$ for an unpolarized kernel and the other combination for a helicity
    one -- and this stage sees neither. What it can see is the array: whether it repeats
    itself across $x=0$, checked point by point (``_is_even_about_zero``). That statement is
    true for any observable, named or not, and needs no mapping to keep in step.
    """
    mirrored = [s for s in summaries if s["mirrored"]]
    if not mirrored:
        return (
            "- The matched distribution is stored once over the range above, so the matched "
            "integral is the whole of it."
        )
    scales = {round(float(s["scale"]), 12) for s in mirrored}
    factor = _fmt(scales.pop()) if len(scales) == 1 else "its recorded factor"
    note = (
        f"- The matched distribution is symmetric about $x=0$: it holds each side twice, so "
        f"`one-sided` is the matched integral divided by ${factor}$."
    )
    if len(mirrored) != len(summaries):
        note += " Rows with no `one-sided` value are stored once instead."
    return note


def _norm_notes(summaries: list[dict[str, Any]], *, interpolated: bool) -> list[str]:
    """The two things the numbers above cannot say, and nothing else.

    Neither is a verdict. Whether the integral should be 1 is decided two stages upstream, by
    whether the matrix element was normalized at $z=0$ -- so the report points back at that
    rather than comparing against 1 itself, which would be right only for the convention it
    was written for.
    """
    if not summaries:
        return ["- No job carried enough grid and distribution data to integrate."]
    notes = [_symmetry_note(summaries)]
    notes.append(
        "- Check the integral against the normalization fixed at the start of the run: it is "
        "1 when the matrix element was normalized at $z=0$ (the renormalization stage's "
        "`normalization`), and otherwise it reproduces whatever constant the input carries."
    )
    if interpolated:
        notes.append(
            "- One matched grid has an interior gap; the integral bridges "
            "it linearly, so that stretch is interpolation rather than matched data."
        )
    return notes


def _settings_table(data: dict[str, Any], *, language: str) -> list[str]:
    kernel_id = str(data.get("kernel_id", "not recorded"))
    _operator, scheme = _parse_kernel_id(kernel_id)
    op_en = _kernel_description(kernel_id, language="en")
    scheme_en = SCHEME_TEXT.get(scheme, scheme or "not recorded")
    # The `CG` prefix of the kernel_id marks the Coulomb-gauge (no Wilson line)
    # construction; anything else is the conventional gauge-invariant one.
    is_coulomb = kernel_id.upper().startswith("CG")
    gauge_en = "Coulomb gauge ($\\partial_i A_i=0$, no Wilson line)" if is_coulomb else "gauge-invariant (straight Wilson line)"
    # The paper is whatever the selected kernel declares in kernels.py -- the manifest
    # picks the kernel_id, and the citation follows it.
    arxiv_id, equations = _kernel_reference(kernel_id)
    reference_en = f"arXiv:{arxiv_id} {equations}".strip() if arxiv_id else "not declared by the kernel"
    x_grid = np.asarray(data.get("x_grid", []), dtype=float)
    # The quasi grid is only worth its own row when matching did not simply keep it:
    # normally it is the light-cone grid, and repeating it would be noise.
    quasi_x_grid = np.asarray(data.get("quasi_x_grid", []), dtype=float)
    separate_quasi_grid = quasi_x_grid.size > 0 and (
        quasi_x_grid.size != x_grid.size or not np.allclose(quasi_x_grid, x_grid)
    )
    zspz = data.get("zspz")
    pz_value = data.get("momentum_gev")
    try:
        pz_text = f"$P_z={_fmt(float(pz_value))}$ GeV"
    except (TypeError, ValueError):
        pz_text = str(pz_value or "not recorded")

    rows = [
        ("Operator / kernel", f"`{kernel_id}` ({op_en})"),
        ("Kernel reference", reference_en),
        ("Gauge convention", gauge_en),
        ("Matching scheme", f"`{scheme}` ({scheme_en})"),
        ("Component (re/im)", f"`{data.get('component', 'not recorded')}`"),
        ("Hadron momentum", pz_text),
        ("Renormalization scale", f"$\\mu={_fmt(data.get('mu'))}$ GeV"),
    ]
    if zspz is not None:
        rows.append(("Wilson-line scale", f"$z_sP_z={_fmt(zspz)}$"))
    rows.extend(
        [
            ("Resampling mode", f"`{data.get('resample', 'not recorded')}` with {data.get('n_sample', 'n/a')} samples"),
            ("x grid (light-cone output)", _format_grid(x_grid, language="en")),
        ]
    )
    if separate_quasi_grid:
        rows.append(("x grid (quasi input)", _format_grid(quasi_x_grid, language="en") + "; differs from the output grid, so the quasi data was linearly interpolated"))
    rows.append(("Quasi-PDF source", f"`{data.get('source', 'not recorded')}`"))
    header = "| Quantity | Value |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _field_definitions(*, language: str) -> list[str]:
    """Explain the report's rows, describing the id layout rather than one kernel's.

    Spelling a concrete id here (it used to say ``CG_<operator>_quark_PDF_<scheme>_NLO``)
    misnames every kernel that is not a Coulomb-gauge quark PDF -- a gauge-invariant DA
    matched none of those three fields.
    """
    return [
        "| Entry | Meaning |",
        "|---|---|",
        "| Operator / kernel | The selected matching kernel, whose id reads `<gauge>_<operator>_<distribution>_<scheme>_<order>`: the gauge is `CG` (Coulomb) or `GI` (gauge-invariant); the operator is the Dirac structure (gt, gtg5, ...); the distribution is `quark_PDF` or `DA`; the scheme sets the finite terms; the order is the perturbative order (NLO throughout). |",
        "| Matching scheme | `msbar` / `ratio` / `hybrid`, chosen by the kernel_id's scheme field; hybrid also needs the Wilson-line length $z_s$. |",
        "| Hadron momentum | $P_z$, which must match the Fourier stage and enters the kernel's logarithmic scale. |",
        "| Renormalization scale | MSbar renormalization scale $\\mu$ in GeV. |",
        "| Resampling mode | The resampling axis carried by the input (bootstrap/jackknife); matching is done sample by sample to preserve the correlation structure. |",
    ]


# --- LLM-derived kernel formula --------------------------------------------
# The whole section answers one question -- "what coefficient did this run apply?" --
# for whichever kernel_id the manifest named, and it splits that job three ways:
#
#   * the CODE decides WHICH terms exist. ``_kernel_source`` hands the model the kernel
#     function and everything it calls, and the prompt says the code is the single
#     source of truth for the terms: which logs, which branches, whether a delta term
#     is there, what the scheme correction is. No formula is hand-written here, so a
#     kernel and its report cannot drift apart.
#   * the PAPER decides HOW it is WRITTEN. ``_fetch_paper_text`` pulls the LaTeX of the
#     arXiv id the kernel tags itself with, and the prompt says the paper is the
#     authority for notation: xi = x/y versus V(x, y), the plus-prescription's bracket
#     structure, how the subtraction domain is marked.
#   * the CROSS-CHECK decides WHETHER THE TWO AGREE. The prompt requires a verdict
#     comparing code against paper term by term, listing every discrepancy. Handing
#     over both without demanding that verdict just lets the model quietly reconcile
#     them, and a mistranscribed kernel would read as agreement.
#
# Nothing in this section names a kernel or a paper. The manifest picks a kernel_id,
# the kernel carries its own @kernel_reference (arXiv id + equations), and
# _kernel_reference reads it back -- so a kernel from a new paper needs no change here.
# See kernels.py. Provider configs (base_url / default_model / key_env) come from
# ``core.llm.PROVIDERS`` so this module stays in sync with the rest of the agent.

# Generating a formula is a network round-trip; memoize so the per-job and the
# stage-level report reuse one call per kernel and language. The key is the kernel_id
# itself, not its parsed fields: CG_gt_quark_PDF_hybrid_NLO and GI_gt_quark_PDF_hybrid_NLO
# share an operator and a scheme but are different kernels from different papers, and
# keying on those fields would serve one of them the other's formula. The value is
# ``(markdown, paper_used)`` so the provenance note knows whether the paper text
# actually made it into the prompt.
_FORMULA_CACHE: dict[tuple[str, str], tuple[str, bool]] = {}
# Paper text fetched once per source (local path or arXiv id).
_PAPER_CACHE: dict[str, str | None] = {}
_PAPER_SOURCE_TIMEOUT_SECONDS = 10
_PAPER_HTML_TIMEOUT_SECONDS = 3
_REPORT_LLM_TIMEOUT_SECONDS = 30
_REPORT_LLM_ATTEMPTS = 2


# --- formula disk cache ------------------------------------------------------
# The in-process caches above die with the process, so every fresh run re-downloaded the
# paper and re-billed a ~27k-token prompt to regenerate a formula that cannot have
# changed. These two directories persist it across runs:
#
#   1. ``_BUNDLED_FORMULA_DIR`` ships inside the wheel (checked into git, populated by
#      ``lamet-agent precompute-formulas``), so a user's very first run is already a hit.
#   2. the user cache directory catches everything the wheel does not cover -- a kernel
#      someone added locally -- and is the only one ever written at run time.
#
# Each file HEADER carries a digest of the kernel's own source together with its paper tag,
# which is what makes shipping the text safe: edit a kernel by one sign and the digest no
# longer matches, so the stale formula is not served and the section is regenerated (and
# re-cross-checked against the paper) instead of silently going out of date. The digest
# deliberately excludes the paper text -- it must be computable *before* the network call,
# or looking up the cache would cost the download it is meant to avoid. The trade-off is
# that a new arXiv version of an unchanged paper is not noticed; delete the file (or point
# LAMET_FORMULA_CACHE_DIR elsewhere) to force a refresh.
_BUNDLED_FORMULA_DIR = Path(__file__).resolve().parent / "formulas"
_FORMULA_FILE_HEADER = "<!-- lamet-agent formula cache"


def user_formula_dir() -> Path | None:
    """Return the writable formula cache directory, or None when caching is disabled.

    ``LAMET_FORMULA_CACHE_DIR`` overrides the location; setting it to an empty string
    turns the writable layer off entirely (useful while iterating on a kernel, and what
    the tests use to stay hermetic).
    """
    override = os.environ.get("LAMET_FORMULA_CACHE_DIR")
    if override is not None:
        return Path(override).expanduser() if override.strip() else None
    root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(root).expanduser() / "lamet-agent" / "formulas"


def formula_cache_filename(kernel_id: str, language: str) -> str:
    """Return the cache file name for one kernel: one stable name, forever.

    The digest lives in the file's header rather than its name, so editing a kernel
    overwrites its formula in place instead of dropping a second file beside the first and
    leaving the first behind as an orphan. One kernel is always exactly one file, and git
    shows an edit as a modification -- readable as a diff -- rather than an add plus a
    delete.
    """
    return f"{kernel_id}.{language}.md"


def formula_digest(kernel_id: str, language: str) -> str:
    """Return the fingerprint deciding whether a cached formula still fits the kernel.

    Covers the kernel's own source together with its paper tag, and deliberately not the
    paper text: it has to be computable *before* the network call, or looking up the cache
    would cost the download it exists to avoid.
    """
    source = _kernel_source(kernel_id)
    arxiv_id, equations = _kernel_reference(kernel_id)
    return hashlib.sha256(
        "\0".join([kernel_id, language, arxiv_id, equations, source]).encode("utf-8")
    ).hexdigest()[:16]


def _read_formula_file(path: Path) -> tuple[str, bool, str] | None:
    """Return ``(markdown, paper_used, digest)`` from a cache file, or None if unusable.

    Whether the digest still matches the kernel is the caller's decision -- prune only
    needs to know the file is one of ours, while a lookup needs the comparison.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    header, _, body = raw.partition("\n")
    if not header.startswith(_FORMULA_FILE_HEADER) or not body.strip():
        return None  # not one of ours, or truncated: regenerate rather than trust it
    found = re.search(r"digest=([0-9a-f]+)", header)
    return body.strip(), "paper_used=true" in header, found.group(1) if found else ""


def _write_formula_file(
    path: Path,
    *,
    kernel_id: str,
    arxiv_id: str,
    equations: str,
    digest: str,
    markdown: str,
    paper_used: bool,
) -> None:
    """Persist one formula, overwriting any earlier version of the same kernel.

    Best-effort: a read-only cache dir must not fail the run.
    """
    header = (
        f"{_FORMULA_FILE_HEADER}; kernel={kernel_id}; arxiv={arxiv_id or 'none'}; "
        f"equations={equations or 'none'}; digest={digest}; "
        f"paper_used={'true' if paper_used else 'false'} -->"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{header}\n{markdown.strip()}\n", encoding="utf-8")
    except OSError:
        pass


@dataclass(frozen=True)
class FormulaLlm:
    """The LLM the report uses to write the kernel's closed form.

    Passed in explicitly, exactly like the review stage's tool arguments: the run's
    ``--backend`` and the provider/key/model the CLI already resolved are handed down as
    parameters. Reading them back out of the environment would mean the report could
    silently use a different model, or a different key, from the run itself.
    """

    backend: str = "api"
    provider: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    base_url: str | None = None

    def resolved(self) -> tuple[str, str | None, str | None, str | None, str | None]:
        """Validate and fill provider defaults, returning what request_llm_text needs."""
        if self.backend == "codex":
            return "codex", None, None, self.model_name, None
        if self.backend != "api":
            raise RuntimeError(
                f"The matching report's formula section needs an LLM, but this run used "
                f"backend={self.backend!r}. Run with --backend api (plus --model "
                f"provider/model_id) or --backend codex."
            )
        if not self.provider:
            raise RuntimeError(
                "The matching report's formula section needs --model provider/model_id "
                f"(one of {sorted(PROVIDERS)})."
            )
        config = provider_config(self.provider)
        if config is None:
            raise RuntimeError(
                f"Unknown provider {self.provider!r}; use one of {sorted(PROVIDERS)}."
            )
        if not self.api_key:
            raise RuntimeError(
                f"The matching report's formula section needs an API key for "
                f"provider={self.provider!r} (--api-key-file, or {config['key_env']})."
            )
        return (
            "api",
            self.provider,
            self.api_key,
            self.model_name or config["default_model"],
            self.base_url or config["base_url"],
        )


def _kernel_source(kernel_id: str) -> str:
    """Return the kernel and everything it actually calls, as LLM ground truth.

    The dependencies are followed from the kernel itself rather than listed here. A
    hardcoded list named only the PDF coefficients, so a DA kernel's prompt carried
    ``C_ratio`` and friends -- which it never calls -- while omitting the ``V(x, y)`` it
    integrates. The model then had no way to document the coefficient, and (correctly)
    said so instead of inventing one. Following the call graph keeps this honest for
    whatever kernel is added next, too.
    """
    builder = getattr(kernels, str(kernel_id), None)
    if builder is None:
        return inspect.getsource(kernels)

    # Walk the module-level functions the kernel reaches, transitively. co_names lists
    # every global name a code object mentions; the ones resolving to a function defined
    # in kernels.py are its real dependencies. Nested code objects have to be walked too
    # -- the PDF kernels pass their coefficient in as a lambda, so C_hybrid and friends
    # are named only inside that lambda's own code object, not the function's.
    collected: dict[str, Any] = {}

    def walk_code(code: Any) -> None:
        for name in code.co_names:
            if name in collected:
                continue
            dependency = getattr(kernels, name, None)
            if inspect.isfunction(dependency) and dependency.__module__ == kernels.__name__:
                collected[name] = dependency
                walk_code(dependency.__code__)
        for const in code.co_consts:
            if inspect.iscode(const):
                walk_code(const)

    walk_code(builder.__code__)
    pieces = [inspect.getsource(builder)]
    pieces.extend(inspect.getsource(fn) for fn in collected.values())
    return "\n\n".join(pieces)


def _strip_html(raw: str) -> str:
    """Crude HTML -> text so an arXiv HTML page is usable as LLM context."""
    no_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", no_scripts)
    return re.sub(r"[ \t\f\v]+", " ", html.unescape(text))


def _fetch_arxiv_source(arxiv_id: str) -> str | None:
    """Download the arXiv LaTeX e-print source and return its ``.tex`` text.

    The ``e-print`` endpoint returns a gzipped tar of the LaTeX source (sometimes a
    single gzipped ``.tex``). Extracting the ``.tex`` files gives the LLM the raw
    ``\\begin{equation}`` math -- the plus-prescription notation survives intact,
    unlike the HTML mirrors which mangle the formulas. Best-effort: any failure
    returns ``None``. The largest ``.tex`` (usually the main manuscript, where the
    matching coefficients live) is placed first so it survives truncation.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lamet-agent/1.0"})
        with urllib.request.urlopen(req, timeout=_PAPER_SOURCE_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
    except (TimeoutError, urllib.error.URLError, ssl.SSLError, ValueError):
        return None

    texts: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".tex"):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                texts.append(handle.read().decode("utf-8", errors="replace"))
    except (tarfile.TarError, OSError, EOFError):
        # Not a tar: try a single gzipped member, else treat the bytes as plain text.
        try:
            texts.append(gzip.decompress(raw).decode("utf-8", errors="replace"))
        except (OSError, EOFError):
            try:
                texts.append(raw.decode("utf-8", errors="replace"))
            except UnicodeDecodeError:
                return None

    texts = [t for t in texts if t.strip()]
    if not texts:
        return None
    texts.sort(key=len, reverse=True)
    return "\n\n".join(texts)


def _local_paper_path(arxiv_id: str) -> str | None:
    """Path to a local copy of *this* paper, from ``LAMET_FORMULA_PAPER_PATH_<arxiv_id>``.

    The variable is per paper (dots in the id become underscores, e.g.
    ``LAMET_FORMULA_PAPER_PATH_2412_20461``) precisely because one run can match several
    jobs with kernels from different papers -- a single global path would silently feed
    the wrong paper to every one of them.
    """
    return os.environ.get(f"LAMET_FORMULA_PAPER_PATH_{arxiv_id.replace('.', '_')}")


def _fetch_paper_text(paper_arxiv_id: str, *, max_chars: int = 80_000) -> str | None:
    """Return the paper text (local copy preferred, else arXiv LaTeX source), or None.

    ``paper_arxiv_id`` comes from the kernel the manifest selected (its
    ``@kernel_reference`` tag), so each kernel fetches its own paper -- nothing here
    knows or assumes a particular one. A local copy wins when
    ``LAMET_FORMULA_PAPER_PATH_<arxiv_id>`` points at a ``.txt``/``.md``/``.tex``/HTML
    file; otherwise the arXiv LaTeX e-print source is fetched so the LLM reads the real
    equations (the HTML mirrors are a last-resort fallback -- their math is mangled).
    The fetch is best-effort: any failure, or an untagged kernel, returns ``None`` and
    the formula is then generated from the kernel code alone.
    """
    if not paper_arxiv_id:
        return None  # untagged kernel: no paper to fetch, and none to invent
    arxiv_id = paper_arxiv_id
    local = _local_paper_path(arxiv_id)
    cache_key = local or f"arxiv:{arxiv_id}"
    if cache_key in _PAPER_CACHE:
        return _PAPER_CACHE[cache_key]

    text: str | None = None
    if local:
        candidate = Path(local).expanduser()
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8", errors="replace")
            text = raw if candidate.suffix.lower() in {".txt", ".md", ".tex"} else _strip_html(raw)
    if text is None:
        # Preferred: the LaTeX e-print source (clean math). Fall back to HTML only
        # if the source is unreachable.
        text = _fetch_arxiv_source(arxiv_id)
    if text is None:
        for url in (
            f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
            f"https://ar5iv.org/abs/{arxiv_id}",
            f"https://arxiv.org/abs/{arxiv_id}",
        ):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "lamet-agent/1.0"})
                with urllib.request.urlopen(req, timeout=_PAPER_HTML_TIMEOUT_SECONDS) as resp:
                    text = _strip_html(resp.read().decode("utf-8", errors="replace"))
                break
            except (TimeoutError, urllib.error.URLError, ssl.SSLError, ValueError):
                continue

    if text is not None:
        text = text.strip()[:max_chars] or None
    _PAPER_CACHE[cache_key] = text
    return text


def _formula_prompt(
    operator: str,
    scheme: str,
    language: str,
    *,
    notation: str,
    extra_note: str | None,
    source: str,
    paper_text: str | None,
    paper_arxiv_id: str,
    equations: str,
) -> str:
    lang_line = "Write the prose in English."
    paper_block = (
        f"LaTeX source of the paper (arXiv:{paper_arxiv_id}). It is the authority for the "
        "NOTATION: copy its symbols and, in particular, its exact plus-prescription convention "
        "for the matching coefficient verbatim.\n\"\"\"\n" + paper_text + "\n\"\"\"\n\n"
        if paper_text
        else "No paper text was available; rely on the code below as the source of truth and use "
        "the paper's own plus-prescription convention, writing the subtraction point and the "
        "domain explicitly.\n\n"
    )
    # The kernel is tagged with the exact equations it transcribes, so point the model
    # at them instead of making it search the paper for the right coefficient.
    equation_line = (
        f"The kernel implements {equations} of that paper -- document that coefficient.\n\n"
        if equations
        else ""
    )
    # The notation guidance and any extra-structure instruction are declared by the kernel
    # (its ``matching_structure`` in kernels.py) and passed in here: a PDF kernel supplies
    # its $\xi=x/y$ plus-prescription block, a DA kernel its $V(x,y)$ block, and a resummed
    # kernel adds the instruction to document the all-orders piece too. This module does not
    # know or branch on which family it is -- it just relays what the kernel asked for.
    notation_block = (
        "Requirements:\n"
        "- Use $...$ for inline math and $$...$$ for display equations (KaTeX/MathJax).\n"
        + notation
        + (extra_note or "")
    )
    # The point of handing over both the paper and the code is to let one check the other.
    # Without an explicit verdict the model silently reconciles them and a transcription
    # error in the kernel would read as agreement.
    crosscheck_block = (
        "- Then cross-check the code against the paper and report the result under a final "
        "`#### Consistency check` heading. Compare term by term: the regular coefficient, every "
        "logarithm and its argument, the plus-prescription and its domain, any delta term, and "
        "the scheme-specific correction. State plainly whether the code reproduces "
        f"{equations or 'the cited equations'} of arXiv:{paper_arxiv_id or 'the cited paper'}. "
        "List every discrepancy you find, however small (a sign, a factor, a missing term, a "
        "different argument inside a log), and say which side has it. If the paper text was not "
        "available to you, say that no check was possible instead of implying one passed. Do not "
        "smooth over a disagreement: reporting a real discrepancy is the most valuable thing "
        "this section can do.\n"
        if paper_text
        else "- Then state, under a final `#### Consistency check` heading, that the paper text "
        "could not be retrieved, so the closed form above was derived from the code alone and "
        "was NOT checked against the paper.\n"
    )
    return (
        "You are documenting one stage of a LaMET lattice-QCD analysis. Produce a Markdown "
        f"fragment giving the explicit matching coefficient for the `{operator}` operator "
        f"in the `{scheme}` scheme, exactly as the paper presents it.\n\n"
        f"{equation_line}"
        f"{paper_block}"
        "The number in the report was produced by this exact Python code -- it is the single "
        "source of truth for WHICH terms are present. Read it together with the paper and write "
        "the closed-form coefficient it implements: the splitting function, the logs, the "
        "arctan/arctanh branch, and any scheme-specific finite correction. If the paper and the "
        "code disagree on a term, follow the code; but for NOTATION always follow the paper.\n"
        f"```python\n{source}\n```\n\n"
        f"{notation_block}"
        "- State the explicit regular coefficient and any scheme-specific correction.\n"
        f"{crosscheck_block}"
        "- Be concise (a few sentences plus the equations); no headings other than the "
        "consistency-check one asked for above, no preamble like 'Here is'. Output only the "
        "Markdown fragment.\n"
        f"- {lang_line}"
    )


def _llm_kernel_formula(kernel_id: str, *, language: str, llm: FormulaLlm) -> tuple[str, bool]:
    """Generate the explicit kernel coefficient with an LLM, returning ``(md, paper_used)``.

    The model reads the exact ``kernels.py`` code that produced the number together with
    the LaTeX of the paper that kernel tags itself with (when reachable), and returns two
    things: the closed form, and a verdict on whether the code and the paper agree. The
    three-way split is the point (see the section comment above) -- the code is
    authoritative for which terms are present, the paper for how they are written, and
    the cross-check is what turns "the model saw both" into something a reader can rely
    on. The report stores no hand-written formula; this helper raises on LLM failure so
    its caller can emit an explicit provenance-only fallback. The boolean reports whether
    the paper text actually reached the prompt, which decides whether the cross-check
    could run at all.
    """
    operator, scheme = _parse_kernel_id(kernel_id)
    paper_arxiv_id, equations = _kernel_reference(kernel_id)
    cache_key = (kernel_id, language)  # the id already carries the scheme
    if cache_key in _FORMULA_CACHE:
        return _FORMULA_CACHE[cache_key]

    # Disk layers before the network: bundled (shipped in the wheel) then the user cache.
    # A hit here also means an unconfigured LLM is fine -- ``llm.resolved()`` below is only
    # reached when a formula genuinely has to be generated, so a --backend mock run can
    # still render a full matching report from the shipped formulas.
    filename = formula_cache_filename(kernel_id, language)
    digest = formula_digest(kernel_id, language)
    writable_dir = user_formula_dir()
    for directory in (_BUNDLED_FORMULA_DIR, writable_dir):
        if directory is None:
            continue
        cached = _read_formula_file(directory / filename)
        # A file whose digest has moved on describes a kernel that no longer exists in
        # this form; it is overwritten below rather than served.
        if cached is not None and cached[2] == digest:
            _FORMULA_CACHE[cache_key] = cached[:2]
            return cached[:2]

    backend, provider, api_key, model_name, base_url = llm.resolved()
    structure = _kernel_structure(kernel_id)
    source = _kernel_source(kernel_id)
    paper_text = _fetch_paper_text(paper_arxiv_id)
    prompt = _formula_prompt(
        operator,
        scheme,
        language,
        notation=structure["notation"],
        extra_note=structure.get("extra_note"),
        source=source,
        paper_text=paper_text,
        paper_arxiv_id=paper_arxiv_id,
        equations=equations,
    )
    # Reuse the shared LLM client (retries + error handling live in core.llm) instead
    # of a second hand-rolled HTTP call. The backend follows the run's --backend; the
    # api-only fields are empty strings under codex and ignored by that backend.
    text = request_llm_text(
        backend=backend,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        messages=[{"role": "user", "content": prompt}],
        request_timeout_seconds=_REPORT_LLM_TIMEOUT_SECONDS,
        request_attempts=_REPORT_LLM_ATTEMPTS,
    ).strip()
    if not text:
        raise RuntimeError("LLM returned an empty matching formula.")
    result = (text, paper_text is not None)
    if writable_dir is not None:
        _write_formula_file(
            writable_dir / filename,
            kernel_id=kernel_id,
            arxiv_id=paper_arxiv_id,
            equations=equations,
            digest=digest,
            markdown=text,
            paper_used=paper_text is not None,
        )
    _FORMULA_CACHE[cache_key] = result
    return result


def precompute_kernel_formulas(
    kernel_ids: Iterable[str],
    *,
    llm: FormulaLlm,
    language: str = "en",
    prune: bool = False,
) -> dict[str, Any]:
    """Fill the bundled formula directory so an installed wheel needs no LLM at run time.

    Run from ``lamet-agent precompute-formulas`` before cutting a release, and after adding
    or editing a kernel. Only missing entries are generated -- an already-current file is a
    digest hit and costs nothing -- so re-running it after touching one kernel regenerates
    that kernel alone. An edited kernel
    overwrites its own file, so ``prune`` is only about kernels renamed or deleted
    outright -- nothing accumulates from ordinary edits.
    """
    directory = _BUNDLED_FORMULA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, str] = {}
    generated: list[str] = []
    skipped: list[str] = []
    for kernel_id in kernel_ids:
        filename = formula_cache_filename(kernel_id, language)
        wanted[filename] = kernel_id
        current = _read_formula_file(directory / filename)
        if current is not None and current[2] == formula_digest(kernel_id, language):
            skipped.append(kernel_id)
            continue
        # Bypass both the in-process cache and the user cache: this must write the file
        # the wheel will ship, from a real generation, not copy whatever happens to be
        # lying around in ~/.cache.
        _FORMULA_CACHE.pop((kernel_id, language), None)
        env_key = "LAMET_FORMULA_CACHE_DIR"
        previous = os.environ.get(env_key)
        os.environ[env_key] = ""
        try:
            markdown, paper_used = _llm_kernel_formula(kernel_id, language=language, llm=llm)
        finally:
            if previous is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous
        arxiv_id, equations = _kernel_reference(kernel_id)
        _write_formula_file(
            directory / filename,
            kernel_id=kernel_id,
            arxiv_id=arxiv_id,
            equations=equations,
            digest=formula_digest(kernel_id, language),
            markdown=markdown,
            paper_used=paper_used,
        )
        generated.append(kernel_id)
    pruned: list[str] = []
    if prune:
        for path in sorted(directory.glob("*.md")):
            # Only ever delete files this module wrote: the header check keeps the
            # directory's README (and anything else a human put here) out of the sweep.
            if path.name not in wanted and _read_formula_file(path) is not None:
                path.unlink()
                pruned.append(path.name)
    return {
        "directory": str(directory),
        "generated": generated,
        "up_to_date": skipped,
        "pruned": pruned,
    }


def _matching_formula_text(data: dict[str, Any], *, language: str, llm: FormulaLlm) -> str:
    kernel_id = str(data.get("kernel_id", ""))
    # Provenance follows the manifest's kernel: whichever kernel_id was selected, its
    # @kernel_reference in kernels.py names the paper and equations cited here.
    paper_arxiv_id, equations = _kernel_reference(kernel_id)
    if paper_arxiv_id:
        reference = f"arXiv:{paper_arxiv_id} {equations}".strip()
    else:
        reference = "The kernel declares no paper reference"
    # The factorization follows the kernel: it declares its own display equation, the names
    # of the matched/source distributions, and any all-orders structure it carries. This
    # module renders whatever the kernel declared -- it does not know PDF from DA from LRR.
    structure = _kernel_structure(kernel_id)
    formula = structure.get("factorization")
    result_name = structure["result_noun"]
    source_name = structure["source_noun"]
    extra_structure = structure.get("extra_structure")
    discrete = r"f_i=\sum_j K_{ij}\,\tilde f_j,\qquad K=\text{(nx, ny) matching matrix}."
    # The explicit coefficient is generated at report time by an LLM that reads
    # the source paper together with the kernels.py code which produced the number
    # (no formula is hardcoded). A short note records the provenance so a reader
    # knows it was machine-derived and from which sources.
    try:
        generated, paper_used = _llm_kernel_formula(kernel_id, language="en", llm=llm)
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            "Matching report formula generation failed; the numerical matching output "
            "is complete and unaffected. Writing a provenance-only fallback. "
            f"Cause: {exc}",
            UserWarning,
            stacklevel=2,
        )
        generated = (
            "The explicit coefficient narrative was not generated because the report-only "
            "LLM request failed. The numerical matching matrix and output artifact were "
            "computed before report generation and are unaffected. Consult the cited "
            "equations and the selected implementation in `lamet_agent/kernels.py`."
        )
        note = "(the report-only formula request failed; the numerical result is unaffected)\n\n"
    else:
        source_en = (
            f"arXiv:{paper_arxiv_id} together with the `kernels.py` implementation"
            if paper_used
            else "the `kernels.py` implementation"
        )
        note = f"(the explicit form below was generated by the model from {source_en})\n\n"
    explicit = note + generated

    parts: list[str] = []
    lead = f"{reference}. The {result_name} is obtained from the {source_name} by inverting the matching kernel"
    parts.append(f"{lead}:\n\n" if formula else f"{lead} (its factorization is given in the explicit form below).\n\n")
    if formula:
        parts.append(f"$$\n{formula}\n$$\n\n")
    parts.append(
        "After discretization this is a matrix product (applied to every resampling sample independently, then the statistics are rebuilt):\n\n"
        f"$$\n{discrete}\n$$\n\n"
        f"$K$ is built directly as the NLO-truncated inverse (the LO term minus the NLO correction), so nothing is "
        f"numerically inverted. Its rows live on the {result_name}'s output grid and its columns on the "
        f"{source_name}'s input grid, so it is square only when the two grids coincide.\n\n"
    )
    if extra_structure:
        parts.append(
            "This kernel does not stop at fixed order: on top of the fixed-order matrix it resums the "
            "leading Wilson-line renormalon to all orders, so the matching matrix takes the matrix-exponential form\n\n"
            f"$$\n{extra_structure}\n$$\n\n"
        )
    parts.append("Here the LO part is the identity, and the explicit matching correction (including the structure above) is:\n\n")
    parts.append(explicit)
    return "".join(parts)


def _scheme_explanation(data: dict[str, Any], *, language: str) -> list[str]:
    """Describe the scheme, citing the equations the selected kernel is tagged with.

    What each scheme adds is a property of the scheme, but *where it is written down* is a
    property of the kernel's paper. Hardcoding equation numbers here cited the Coulomb-gauge
    PDF paper at every kernel, including DA kernels from an entirely different one, so the
    citation is read off ``@kernel_reference`` instead.
    """
    kernel_id = str(data.get("kernel_id", ""))
    _operator, scheme = _parse_kernel_id(kernel_id)
    arxiv_id, equations = _kernel_reference(kernel_id)
    cite = f"arXiv:{arxiv_id} {equations}".strip() if arxiv_id else ""
    notes = {
        "msbar": "The MSbar scheme adds a finite MSbar conversion on top of the bare ratio coefficient.",
        "ratio": "The ratio scheme uses the bare regular coefficient directly, with no extra finite terms.",
        "hybrid": "The hybrid scheme adds a Wilson-line sine-integral correction to the ratio coefficient and depends on $z_sP_z$.",
    }
    body = notes.get(scheme, "Unrecognized matching scheme; only the selected kernel_id is recorded.")
    if cite:
        body = f"{body} The kernel used here is transcribed from {cite}."
    return ["## Matching Scheme", body]


def _diagnostics(data: dict[str, Any], *, language: str) -> list[str]:
    # The integral is taken over the range this job matched (see _norm_summary), so the
    # numbers hold for a DA on [0, 1] as much as for a PDF grid spanning both signs of x, and
    # the limits printed are the run's own rather than a hardcoded window. What follows the
    # numbers is notes, not a verdict -- see _norm_notes.
    summary = _norm_summary(data)
    if summary is None:
        return ["- Matching diagnostics were not available."]
    lines = [
        f"- Integral over the matching range {_norm_window_text(summary)}: quasi "
        f"${_fmt(summary['quasi'])}$, light-cone ${_fmt(summary['lightcone'])}$ -- the range "
        "the matching actually produced, after any endpoint cut.",
    ]
    if summary["mirrored"]:
        lines.append(
            f"- One-sided: quasi ${_fmt(summary['unit_quasi'])}$, light-cone "
            f"${_fmt(summary['unit_lightcone'])}$."
        )
    lines.append(f"- Relative change from matching: {_fmt(100 * summary['rel_change'])}%.")
    lines.extend(_norm_notes([summary], interpolated=bool(summary["interpolated_gap"])))
    return lines


def _figure_block(artifacts: dict[str, Any], *, language: str) -> list[str]:
    heading = "## Figures and Visual Assessment"
    label = "Quasi vs light-cone comparison"
    image_value = artifacts.get("matched_plot_image")
    pdf_value = artifacts.get("matched_plot")
    lines = [heading, "", f"### {label}"]
    if image_value:
        lines.append(f"![{label}]({image_value})")
        if pdf_value:
            lines.append("")
            lines.append(f"[{label} (PDF, vector)]({pdf_value})")
    elif pdf_value:
        lines.append(f"[{label} (PDF)]({pdf_value})")
    else:
        lines.append("Not available.")
    return lines


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    header = "| File | Description |"
    lines = [header, "|---|---|"]
    for key in MATCHING_ARTIFACT_ORDER:
        value = artifacts.get(key)
        if not value:
            continue
        desc = MATCHING_ARTIFACT_DESCRIPTIONS[key]
        lines.append(f"| `{value}` | {desc} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def build_matching_report_markdown(
    *,
    result: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    language: str = "en",
    llm: FormulaLlm,
) -> str:
    artifacts = artifacts or {}
    kernel_id = str(result.get("kernel_id", "not recorded"))
    _operator, scheme = _parse_kernel_id(kernel_id)
    op_en = _kernel_description(kernel_id, language="en")
    scheme_en = SCHEME_TEXT.get(scheme, scheme or "not recorded")

    lines = [
        "# Perturbative Matching Analysis Report",
        "",
        "## Abstract",
        f"This report summarizes converting the `{kernel_id}` ({op_en}) quasi-PDF into the light-cone PDF using the `{scheme_en}`-scheme NLO matching kernel.",
        "",
        "## Analysis Setup",
        *_settings_table(result, language="en"),
        "",
        "### Field Definitions",
        *_field_definitions(language="en"),
        "",
        "## Matching Formula",
        _matching_formula_text(result, language="en", llm=llm),
        "",
        *_scheme_explanation(result, language="en"),
        "",
        "## Diagnostics and Consistency Checks",
        *_diagnostics(result, language="en"),
        "",
        *_figure_block(artifacts, language="en"),
        "",
        "## Output Artifacts",
        *_outputs_table(artifacts, language="en"),
    ]
    return "\n".join(lines) + "\n"


def write_matching_report(
    *,
    result: dict[str, Any],
    artifacts: dict[str, Any] | None,
    path: str | Path,
    report_language: str = "en",
    llm: FormulaLlm,
) -> dict[str, Path]:
    """Write one matching report and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_artifacts = markdown_artifact_paths(
        artifacts,
        base_dir=output.parent,
        path_keys=MATCHING_ARTIFACT_ORDER,
    )
    markdown = build_matching_report_markdown(result=result, artifacts=report_artifacts, language="en", llm=llm)
    output.write_text(markdown, encoding="utf-8")
    if report_language.lower() == "ch":
        backend, provider, api_key, model_name, base_url = llm.resolved()
        translated = translate_markdown_report(
            markdown,
            backend=backend,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )
        target, _language = _report_target(output, report_language)
        target.write_text(translated, encoding="utf-8")
        return {"report": target}
    return {"report": output}


def write_matching_stage_report(
    *,
    jobs: list[dict[str, Any]],
    systematics_jobs: list[dict[str, Any]] | None = None,
    path: str | Path,
    report_language: str = "en",
    llm: FormulaLlm,
) -> dict[str, Path]:
    """Write one report summarizing all matching jobs in a stage."""
    output = Path(path)
    target, language = _report_target(output, report_language)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output
    first = jobs[0]["result"]
    language = "en"
    kernel_id = str(first.get("kernel_id", "not recorded"))
    _operator, scheme = _parse_kernel_id(kernel_id)
    op_en = _kernel_description(kernel_id, language=language)
    scheme_en = SCHEME_TEXT.get(scheme, scheme or "not recorded")
    all_jobs = jobs + list(systematics_jobs or [])
    use_systematics_table = bool(systematics_jobs)
    lines = [
        "# Perturbative Matching Stage Report",
        "",
        f"This report summarizes all perturbative-matching jobs for `{kernel_id}` ({op_en}) using the `{scheme_en}` scheme.",
        "",
        "## Job Summary",
        (
            "| job | kernel | $P_z$ | $z_s$ [fm] | selected range | $\\mu$ [GeV] | output | plot |"
            if use_systematics_table
            else "| job | kernel | $P_z$ | output | plot |"
        ),
        (
            "|---|---|---:|---:|---|---:|---|---|"
            if use_systematics_table
            else "|---|---|---:|---|---|"
        ),
    ]
    for item in all_jobs:
        result = item["result"]
        artifacts = markdown_artifact_paths(
            item.get("artifacts", {}),
            base_dir=output.parent,
            path_keys=MATCHING_ARTIFACT_ORDER,
        )
        if use_systematics_table:
            lines.append(
                f"| `{item['job_id']}` | {result.get('kernel_id', 'n/a')} | "
                f"{_fmt(result.get('momentum_gev'))} | {_fmt(result.get('zs_fm'))} | "
                f"{result.get('selected_range_label', 'n/a')} | {_fmt(result.get('mu'))} | "
                f"{artifacts.get('lightcone_artifact', 'n/a')} | "
                f"{artifacts.get('matched_plot', 'n/a')} |"
            )
        else:
            lines.append(
                f"| `{item['job_id']}` | {result.get('kernel_id', 'n/a')} | "
                f"{_fmt(result.get('momentum_gev'))} | "
                f"{artifacts.get('lightcone_artifact', 'n/a')} | "
                f"{artifacts.get('matched_plot', 'n/a')} |"
            )
    setting_data = {**first, "momentum_gev": "see per-momentum table"}
    lines.extend(
        [
            "",
            "## Analysis Setup",
            *_settings_table(setting_data, language=language),
            "",
            "### Field Definitions",
            *_field_definitions(language=language),
            "",
            "## Matching Formula",
            _matching_formula_text(first, language=language, llm=llm),
            "",
            *_scheme_explanation(first, language=language),
            "",
            "## Diagnostics and Consistency Checks",
        ]
    )
    # Each job is integrated over the range it matched, so jobs with different endpoint
    # cuts or grids stay comparable to themselves; the window is printed per row rather
    # than fixed in the header. The summaries are collected before the table is emitted
    # because the header depends on them: the one-sided column exists only if some row is
    # actually mirrored.
    summaries: list[dict[str, Any]] = []
    rows: list[tuple[str, dict[str, Any] | None, Any]] = []
    interpolated = False
    for item in jobs:
        result = item["result"]
        summary = _norm_summary(result)
        rows.append((str(item["job_id"]), summary, result.get("momentum_gev")))
        if summary is None:
            continue
        summaries.append(summary)
        interpolated = interpolated or bool(summary["interpolated_gap"])
    # The one-sided column only says something the `matched` column does not where the
    # projection mirrors the distribution, so it appears only for the rows that have one.
    show_one_sided = any(s["mirrored"] for s in summaries)
    one_sided_header = " one-sided $\\int f\\,dx$ |" if show_one_sided else ""
    one_sided_align = "---:|" if show_one_sided else ""
    lines.extend(
        [
            f"| job | $P_z$ | matching range | quasi $\\int f\\,dx$ | "
            f"matched $\\int f\\,dx$ |{one_sided_header} change |",
            f"|---|---:|---|---:|---:|{one_sided_align}---:|",
        ]
    )
    for job_id, summary, momentum in rows:
        if summary is None:
            blanks = " n/a |" * (4 + int(show_one_sided))
            lines.append(f"| `{job_id}` | {_fmt(momentum)} |{blanks}")
            continue
        # A row whose projection does not mirror has no separate one-sided value; the dash
        # says so rather than repeating the matched integral under a heading it did not earn.
        one_sided_cell = ""
        if show_one_sided:
            mirrored = summary["mirrored"]
            one_sided_cell = f" {_fmt(summary['unit_lightcone']) if mirrored else '--'} |"
        lines.append(
            f"| `{job_id}` | {_fmt(momentum)} | "
            f"$[{_fmt(summary['lo'])}, {_fmt(summary['hi'])}]$ | "
            f"{_fmt(summary['quasi'])} | {_fmt(summary['lightcone'])} |{one_sided_cell} "
            f"{_fmt(100 * summary['rel_change'])}% |"
        )
    closing = [
        "",
        "Each job is integrated over the x range it actually matched (the `matching range` "
        "column), after any endpoint cut, rather than over a fixed window.",
        "",
        *_norm_notes(summaries, interpolated=interpolated),
    ]
    closing.extend(["", "## Figures and Visual Assessment"])
    lines.extend(closing)
    for item in jobs:
        result = item["result"]
        artifacts = markdown_artifact_paths(
            item.get("artifacts", {}),
            base_dir=output.parent,
            path_keys=MATCHING_ARTIFACT_ORDER,
        )
        image = artifacts.get("matched_plot_image")
        plot = artifacts.get("matched_plot")
        label = "Quasi vs light-cone comparison"
        lines.extend(["", f"### `{item['job_id']}`: $P_z={_fmt(result.get('momentum_gev'))}$ GeV"])
        if image:
            lines.append(f"![{label}]({image})")
            if plot:
                lines.append("")
                lines.append(f"[{label} (PDF, vector)]({plot})")
        elif plot:
            lines.append(f"[{label} (PDF)]({plot})")
        else:
            lines.append("Not available.")
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "| File | Description |",
            "|---|---|",
        ]
    )
    for item in jobs:
        artifacts = markdown_artifact_paths(
            item.get("artifacts", {}),
            base_dir=output.parent,
            path_keys=(*MATCHING_ARTIFACT_ORDER, *(key for key in item.get("artifacts", {}) if key.startswith("matching_overlay_"))),
        )
        for key in MATCHING_ARTIFACT_ORDER:
            value = artifacts.get(key)
            if value:
                desc = MATCHING_ARTIFACT_DESCRIPTIONS[key]
                lines.append(f"| [{Path(value).name}]({value}) | `{item['job_id']}`: {desc} |")
    stage_artifacts = markdown_artifact_paths(
        jobs[0].get("artifacts", {}),
        base_dir=output.parent,
        path_keys=(*(key for key in jobs[0].get("artifacts", {}) if key.startswith("matching_overlay_")),),
    )
    for key, value in sorted(stage_artifacts.items(), key=lambda item: ("_image_" in item[0], item[0])):
        if not key.startswith("matching_overlay_"):
            continue
        stem = Path(value).stem
        label = stem[3:] if stem.startswith("mt_") else stem
        lines.append(f"| [{Path(value).name}]({value}) | Matching overlay for ensemble {label} |")
    markdown = "\n".join(lines) + "\n"
    output.write_text(markdown, encoding="utf-8")
    if report_language.lower() == "ch":
        backend, provider, api_key, model_name, base_url = llm.resolved()
        translated = translate_markdown_report(
            markdown,
            backend=backend,
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )
        _target, _language = _report_target(output, report_language)
        _target.write_text(translated, encoding="utf-8")
        report_path = _target
    return {"report": report_path}
