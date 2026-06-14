"""Perturbative-matching stage tools.

Purpose:
- convert a quasi-PDF into the light-cone PDF via an NLO matching kernel
- support *multiple* kernels so each operator can use its own kernel

Design:
- ``KERNEL_REGISTRY`` maps a logical ``kernel_id`` (e.g. an operator label) to a
  builder callable from ``lamet_agent.kernels``. To add a new operator, drop its
  kernel function in ``kernels.py`` and register it here -- nothing else changes.
- Each tool takes the shared per-stage ``store`` plus JSON-friendly kwargs, writes
  its result under ``store[out]``, and returns a small summary dict.

Expected inputs:
- a quasi-PDF produced by the Fourier stage (loaded from an artifact on disk,
  since each stage starts with a fresh store)
- a momentum grid ``x_ls`` and the nucleon momentum ``pz_gev``

Expected outputs:
- the matching kernel matrix and the matched (light-cone) PDF as gvar arrays
- a quasi-vs-light-cone comparison PDF under artifacts/

Example usage:
- from lamet_agent.stages.matching.functions import STAGE_TOOLS
- store = {}
- STAGE_TOOLS["load_quasi_pdf"](store, path="artifacts/quasi_pdf.npz")
- STAGE_TOOLS["build_matching_kernel"](store, kernel_id="unpolarized_gT", pz_gev=1.5)
- STAGE_TOOLS["apply_matching"](store)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from lamet_agent.kernels import (
    helicity_matching_kernel_nlo_gTg5,
    unpolarized_gluon_matching_kernel_nlo,
    unpolarized_matching_kernel_nlo_gT,
)

try:
    import gvar as gv
except ModuleNotFoundError:  # pragma: no cover - depends on optional analysis deps
    gv = None  # type: ignore[assignment]


def _require_gvar():
    """Return the gvar module, or raise a user-facing error if it is not installed."""
    if gv is None:
        raise RuntimeError("The matching stage requires gvar. Install the analysis extras first.")
    return gv


# --- kernel registry --------------------------------------------------------
# This dict is the only place the matching stage knows "which kernels exist".
# The agent selects a kernel by passing one of these keys as kernel_id to
# build_matching_kernel.
#
# The kernels' math lives in kernels.py, not here -- this file only wires those
# functions into the agent. To add a kernel for a new operator, three steps:
#   1. write its kernel function (pure numpy) in kernels.py;
#   2. import it at the top of this file;
#   3. add a "kernel_id": function line below.
# Nothing else in this stage needs to change.
#
# Every registered kernel obeys the same signature (so build_matching_kernel can
# call them uniformly):
#   builder(x_ls, pz_gev, mu=2.0, y_ls=None, eps=1e-12) -> (nx, ny) float matrix
KERNEL_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "unpolarized_gT": unpolarized_matching_kernel_nlo_gT,
    "helicity_gTg5": helicity_matching_kernel_nlo_gTg5,
    "unpolarized_gluon": unpolarized_gluon_matching_kernel_nlo,
    # "transversity_gTg5gj": transversity_matching_kernel_nlo_...,  # TODO
}


# Each tool below follows the same pattern: take this stage's shared store, do
# one step, write the result under store[out], and return a small summary for the
# agent to decide the next step. The store is how adjacent tools pass data
# (load -> build -> apply -> plot).


def list_kernels(store: dict[str, Any]) -> dict[str, Any]:
    """Tell the agent which kernel_ids are available (reads no input)."""
    del store  # this tool needs no input; the registry itself is the answer
    return {"kernel_ids": sorted(KERNEL_REGISTRY)}


# --- load the quasi-PDF from the previous (Fourier) stage --------------------


def _read_fourier_artifact(raw: Any, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the quasi-PDF from the EnsembleData npz written by the Fourier stage.

    The Fourier stage does not store ``x_ls/quasi_mean/quasi_sdev`` directly;
    instead it uses ``EnsembleData.save_npz`` to write a npz with JSON metadata
    (see ``_save_fourier_npz`` in ``stages/fourier/functions.py``):
      - the momentum-fraction grid x lives in ``coords["x"]`` inside
        ``__ensemble_data_metadata__``;
      - the real-part mean/errors are in the extra arrays ``ft_re_mean`` /
        ``ft_re_stat_sdev`` / ``ft_re_sys_sdev`` (the imaginary part is ``ft_im_*``).

    The quasi-PDF is the real part of the Fourier result (``component="re"``); the
    total error combines statistical and systematic (model-average) errors in
    quadrature: sdev = sqrt(stat^2 + sys^2).
    """
    import json

    # The metadata is a JSON string; its coords["x"] is the x grid shared by the
    # quasi- and light-cone PDFs.
    metadata = json.loads(str(raw["__ensemble_data_metadata__"]))
    coords = metadata["coords"]
    grid_key = "x" if "x" in coords else metadata["dims"][0]
    x_ls = np.asarray(coords[grid_key], dtype=float)

    # Take the real (re) or imaginary (im) component; here the quasi-PDF is the
    # real part of the Fourier result.
    quasi_mean = np.asarray(raw[f"ft_{component}_mean"], dtype=float)
    stat_sdev = np.asarray(raw[f"ft_{component}_stat_sdev"], dtype=float)
    sys_sdev = np.asarray(raw[f"ft_{component}_sys_sdev"], dtype=float)
    # stat (+) sys errors in quadrature, used as the quasi-PDF total error for gvar.
    quasi_sdev = np.sqrt(stat_sdev**2 + sys_sdev**2)
    return x_ls, quasi_mean, quasi_sdev


def load_quasi_pdf(
    store: dict[str, Any],
    *,
    path: str,
    component: str = "re",
    quasi_out: str = "quasi_gv",
    grid_out: str = "x_ls",
) -> dict[str, Any]:
    """Load a quasi-PDF and its momentum grid from a Fourier-stage artifact.

    The store is fresh each stage, so cross-stage data is passed on disk.

    Two artifact layouts are accepted automatically:
    - the real Fourier-stage ``EnsembleData`` npz (default): grid from the JSON
      metadata ``coords["x"]``, quasi-PDF from ``ft_<component>_mean`` with error
      ``sqrt(stat^2 + sys^2)``;
    - a simple hand-made npz with ``x_ls``, ``quasi_mean``, ``quasi_sdev``.

    ``component`` selects the real (``"re"``) or imaginary (``"im"``) channel of
    the Fourier output; the unpolarized quasi-PDF lives in the real part.
    """
    # Each stage's store is fresh, so cross-stage data is passed via on-disk artifacts.
    raw = np.load(path, allow_pickle=True)

    # Auto-detect the artifact format: read the real Fourier output when the
    # EnsembleData metadata is present, otherwise fall back to the simple
    # hand-made format (x_ls / quasi_mean / quasi_sdev).
    if "__ensemble_data_metadata__" in raw:
        x_ls, quasi_mean, quasi_sdev = _read_fourier_artifact(raw, component)
    elif "x_ls" in raw:
        x_ls = np.asarray(raw["x_ls"], dtype=float)
        quasi_mean = np.asarray(raw["quasi_mean"], dtype=float)
        quasi_sdev = np.asarray(raw["quasi_sdev"], dtype=float)
    else:
        raise ValueError(
            f"Unrecognized quasi-PDF artifact '{path}': expected a Fourier-stage "
            "EnsembleData npz or an npz with x_ls/quasi_mean/quasi_sdev."
        )

    # Combine mean and sdev into a gvar array; the later matrix product
    # propagates the error automatically.
    gvar = _require_gvar()
    quasi_gv = gvar.gvar(quasi_mean, quasi_sdev)
    if quasi_gv.shape != x_ls.shape:
        raise ValueError(
            f"quasi-PDF shape {quasi_gv.shape} and x_ls shape {x_ls.shape} must match."
        )

    # The store is the temporary dict shared by this stage's tools; build/apply
    # read the data back from here.
    store[grid_out] = x_ls
    store[quasi_out] = quasi_gv
    return {
        "quasi_out": quasi_out,
        "grid_out": grid_out,
        "component": component,
        "n_points": int(x_ls.size),
    }


# --- build the kernel matrix ------------------------------------------------


def build_matching_kernel(
    store: dict[str, Any],
    *,
    kernel_id: str,
    pz_gev: float,
    mu: float = 2.0,
    grid: str = "x_ls",
    y_grid: str | None = None,
    out: str = "kernel_matrix",
) -> dict[str, Any]:
    """Build the (nx, ny) matching kernel for the chosen operator/kernel_id."""
    if kernel_id not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel_id '{kernel_id}'. Available: {sorted(KERNEL_REGISTRY)}"
        )
    if grid not in store:
        raise ValueError(f"Momentum grid '{grid}' not in store; run load_quasi_pdf first.")
    if y_grid is not None and y_grid not in store:
        raise ValueError(f"Y grid '{y_grid}' not in store.")

    # Look up the operator's kernel function by kernel_id. The formula and
    # discretization all live in kernels.py.
    builder = KERNEL_REGISTRY[kernel_id]

    # x_ls is the grid of the output light-cone PDF; by default y_ls=x_ls, i.e.
    # the quasi-PDF lives on the same grid.
    x_ls = np.asarray(store[grid], dtype=float)
    y_ls = None if y_grid is None else np.asarray(store[y_grid], dtype=float)

    # Build the already-discretized kernel matrix, typically shaped (len(x_ls), len(y_ls)).
    matrix = builder(x_ls, pz_gev=pz_gev, mu=mu, y_ls=y_ls)
    if matrix.ndim != 2:
        raise ValueError(f"Kernel builder returned a non-matrix object with ndim={matrix.ndim}.")
    if matrix.shape[0] != x_ls.size:
        raise ValueError(
            f"Kernel row count {matrix.shape[0]} does not match x grid size {x_ls.size}."
        )

    store[out] = matrix  # apply_matching reads it back under this name
    return {
        "out": out,
        "kernel_id": kernel_id,
        "shape": list(matrix.shape),
        "pz_gev": pz_gev,
        "mu": mu,
    }


# --- apply the kernel: quasi-PDF -> light-cone PDF --------------------------


def apply_matching(
    store: dict[str, Any],
    *,
    kernel: str = "kernel_matrix",
    quasi: str = "quasi_gv",
    out: str = "lightcone_gv",
) -> dict[str, Any]:
    """Convolve the kernel with the quasi-PDF: ``lightcone = K @ quasi``.

    gvar carries the uncertainty propagation through the matrix product.
    """
    if kernel not in store:
        raise ValueError(f"Kernel '{kernel}' not in store; run build_matching_kernel first.")
    if quasi not in store:
        raise ValueError(f"Quasi-PDF '{quasi}' not in store; run load_quasi_pdf first.")

    # This matrix product is the matching convolution itself. quasi_gv is a gvar
    # array, so @ automatically propagates the quasi-PDF error to the lightcone PDF.
    matrix = np.asarray(store[kernel], dtype=float)
    quasi_gv = store[quasi]
    if matrix.ndim != 2:
        raise ValueError(f"Kernel '{kernel}' must be a 2D matrix.")
    if matrix.shape[1] != np.size(quasi_gv):
        raise ValueError(
            f"Kernel columns ({matrix.shape[1]}) must match quasi-PDF size ({np.size(quasi_gv)})."
        )

    lightcone = matrix @ quasi_gv
    store[out] = lightcone
    return {
        "out": out,
        "n_points": int(np.size(lightcone)),
        "mean_sample": [float(_require_gvar().mean(v)) for v in lightcone[:3]],
    }


# --- plotting ---------------------------------------------------------------


def plot_matched_pdf(
    store: dict[str, Any],
    *,
    grid: str = "x_ls",
    quasi: str = "quasi_gv",
    lightcone: str = "lightcone_gv",
    save_path: str | None = None,
    artifacts_dir: str | None = None,
    xlim: list[float] | tuple[float, float] | None = None,
    ylim: list[float] | tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Plot quasi vs matched (light-cone) PDF and save a PDF artifact.

    ``save_path``/``artifacts_dir`` are injected by the harness for plot tools
    (add ``plot_matched_pdf`` to ``_PLOT_TOOLS`` in core/tools.py).
    """
    if grid not in store:
        raise ValueError(f"Momentum grid '{grid}' not in store.")
    if quasi not in store:
        raise ValueError(f"Quasi-PDF '{quasi}' not in store.")
    if lightcone not in store:
        raise ValueError(f"Light-cone PDF '{lightcone}' not in store; run apply_matching first.")

    # Import matplotlib only when actually plotting, so pure numerical matching
    # does not force-load the plotting library.
    import matplotlib.pyplot as plt

    from lamet_agent.core.plotting import BLUE, FONT_SIZE, LEGEND_SETS, ORANGE, default_plot

    gvar = _require_gvar()
    x_ls = np.asarray(store[grid], dtype=float)
    quasi_gv = store[quasi]
    lightcone_gv = store[lightcone]

    if x_ls.shape != np.shape(quasi_gv) or x_ls.shape != np.shape(lightcone_gv):
        raise ValueError("x grid, quasi-PDF, and light-cone PDF must have matching shapes.")

    # If the harness did not pass save_path, default to artifacts/matched_pdf.pdf.
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    resolved_save = Path(save_path) if save_path is not None else out_dir / "matched_pdf"
    if resolved_save.suffix.lower() != ".pdf":
        resolved_save = resolved_save.with_suffix(".pdf")
    resolved_save.parent.mkdir(parents=True, exist_ok=True)

    # Plot the quasi- and matched light-cone PDFs as a continuous line plus an
    # error band (fill_between) showing the gvar +/-1 sigma interval, rather than
    # discrete error points. Same style as the Fourier-stage plots.
    fig, ax = default_plot()

    def _band(values, *, label: str, color: str) -> None:
        # Center line is the mean; the translucent band is [mean - sdev, mean + sdev].
        mean = gvar.mean(values)
        sdev = gvar.sdev(values)
        ax.fill_between(x_ls, mean - sdev, mean + sdev, color=color, alpha=0.32, linewidth=0, label=label)
        ax.plot(x_ls, mean, color=color, linewidth=0.9, alpha=0.85)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    _band(quasi_gv, label="quasi", color=BLUE)
    _band(lightcone_gv, label="light-cone", color=ORANGE)
    ax.set_xlabel(r"$x$", **FONT_SIZE)
    ax.set_ylabel(r"$f(x)$", **FONT_SIZE)
    x_limits = (-2.2, 2.2) if xlim is None else (float(xlim[0]), float(xlim[1]))
    y_limits = (-0.1, 2.51) if ylim is None else (float(ylim[0]), float(ylim[1]))
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.legend(**LEGEND_SETS)
    fig.savefig(resolved_save, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return {"path": str(resolved_save), "n_points": int(x_ls.size)}


# --- tool registry exposed to the agent -------------------------------------
# Only functions registered here are visible and callable by the agent. Functions
# defined above but not registered here cannot be called. (core/tools.py's
# resolve_stage_tools reads this by name.)

STAGE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "list_kernels": list_kernels,
    "load_quasi_pdf": load_quasi_pdf,
    "build_matching_kernel": build_matching_kernel,
    "apply_matching": apply_matching,
    "plot_matched_pdf": plot_matched_pdf,
}
