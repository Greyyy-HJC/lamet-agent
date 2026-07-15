"""Perturbative-matching stage tools.

Purpose:
- convert a quasi-PDF into the light-cone PDF via an NLO matching kernel
- support *multiple* kernels so each operator can use its own kernel

Matching is done **sample by sample**: instead of propagating a single
central-value-plus-error gvar array through the convolution, the kernel is
applied to every resampling (bootstrap/jackknife/raw) sample of the quasi-PDF
independently. The statistics (mean, error, covariance) are then rebuilt from
the matched samples. This keeps the full sample-level correlation structure --
e.g. correlations between x bins and across analysis stages -- which a
gvar-only convolution would discard.

Design:
- ``KERNEL_REGISTRY`` maps each exact public function name from ``lamet_agent.kernels``
  to that builder callable. To add a new operator, drop its
  kernel function in ``kernels.py`` and register it here -- nothing else changes.
- Each tool takes the shared per-stage ``store`` plus JSON-friendly kwargs, writes
  its result under ``store[out]``, and returns a small summary dict.

Expected inputs:
- a quasi-PDF produced by the Fourier stage, passed in memory by job id or loaded
  from an external NetCDF source. It carries the full per-sample quasi-PDF (an
  ``EnsembleData`` with a leading resampling axis).
- a momentum grid ``x_ls`` and the nucleon momentum ``momentum_gev``

Expected outputs:
- the matching kernel matrix and the matched (light-cone) PDF, both kept as
  ``EnsembleData`` so downstream stages still see every sample
- a quasi-vs-light-cone comparison PDF under artifacts/

Example usage:
- from lamet_agent.stages.matching.functions import STAGE_TOOLS
- store = {}
- STAGE_TOOLS["load_quasi_pdf"](store, path="artifacts/fourier_result.nc")
- STAGE_TOOLS["build_matching_kernel"](store, kernel_id="CG_gt_qPDF_msbar_NLO", momentum_gev=1.5)
- STAGE_TOOLS["apply_matching"](store)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from lamet_agent.core.data import EnsembleData
from lamet_agent.stages.matching.reporting import FormulaLlm, write_matching_report

# All matching kernels live in the self-contained kernels.py.
from lamet_agent.kernels import (
    CG_gt_qPDF_hybrid_NLO,
    CG_gt_qPDF_msbar_NLO,
    CG_gt_qPDF_ratio_NLO,
    CG_gtg5_qPDF_hybrid_NLO,
    CG_gtg5_qPDF_msbar_NLO,
    CG_gtg5_qPDF_ratio_NLO,
    CG_gtgpg5_qPDF_hybrid_NLO,
    CG_gtgpg5_qPDF_msbar_NLO,
    CG_gtgpg5_qPDF_ratio_NLO,
    CG_gz_qPDF_hybrid_NLO,
    CG_gz_qPDF_msbar_NLO,
    CG_gz_qPDF_ratio_NLO,
    CG_gzg5_qPDF_hybrid_NLO,
    CG_gzg5_qPDF_msbar_NLO,
    CG_gzg5_qPDF_ratio_NLO,
    GI_gt_qPDF_hybrid_NLO,
    GI_gt_qPDF_ratio_NLO,
    GI_gtgpg5_qPDF_hybrid_NLO,
    GI_gtgpg5_qPDF_ratio_NLO,
    GI_gtg5_qPDF_hybrid_NLO,
    GI_gtg5_qPDF_ratio_NLO,
    GI_gz_qPDF_hybrid_NLO,
    GI_gz_qPDF_ratio_NLO,
    GI_gzg5_DA_hybrid_NLO,
    GI_gzg5_DA_ratio_NLO,
    GI_gzg5_qPDF_hybrid_NLO,
    GI_gzg5_qPDF_ratio_NLO,
)


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
#   builder(lightcone_x_ls, momentum_gev, mu=2.0, quasi_y_ls=None, eps=1e-12, zspz=None) -> (nx, ny) matrix
#
# Naming convention: <gauge>_<operator>_qPDF_<scheme>_<order>. <gauge> is the
# operator's gauge construction -- CG for Coulomb gauge (arXiv:2602.11283, no Wilson
# line), GI for gauge-invariant (straight Wilson line); <operator> is the Dirac
# structure (gt = gamma^t, gtg5 = gamma^t gamma5); "qPDF" is the parton species -- q for
# quark, so a gluon kernel would be gPDF; <scheme> is the matching scheme, always
# written out explicitly (msbar / ratio / hybrid); <order> is the perturbative order
# (NLO throughout). Only the hybrid kernels use the Wilson-line scale zspz (built
# from the zs_fm input below); the others ignore it.
KERNEL_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    # unpolarized gamma^t
    "CG_gt_qPDF_msbar_NLO": CG_gt_qPDF_msbar_NLO,
    "CG_gt_qPDF_ratio_NLO": CG_gt_qPDF_ratio_NLO,
    "CG_gt_qPDF_hybrid_NLO": CG_gt_qPDF_hybrid_NLO,
    # helicity gamma^t gamma5
    "CG_gtg5_qPDF_msbar_NLO": CG_gtg5_qPDF_msbar_NLO,
    "CG_gtg5_qPDF_ratio_NLO": CG_gtg5_qPDF_ratio_NLO,
    "CG_gtg5_qPDF_hybrid_NLO": CG_gtg5_qPDF_hybrid_NLO,
    # unpolarized / helicity gamma^z
    "CG_gz_qPDF_msbar_NLO": CG_gz_qPDF_msbar_NLO,
    "CG_gz_qPDF_ratio_NLO": CG_gz_qPDF_ratio_NLO,
    "CG_gz_qPDF_hybrid_NLO": CG_gz_qPDF_hybrid_NLO,
    "CG_gzg5_qPDF_msbar_NLO": CG_gzg5_qPDF_msbar_NLO,
    "CG_gzg5_qPDF_ratio_NLO": CG_gzg5_qPDF_ratio_NLO,
    "CG_gzg5_qPDF_hybrid_NLO": CG_gzg5_qPDF_hybrid_NLO,
    # transversity gamma^t gamma_perp gamma5
    "CG_gtgpg5_qPDF_msbar_NLO": CG_gtgpg5_qPDF_msbar_NLO,
    "CG_gtgpg5_qPDF_ratio_NLO": CG_gtgpg5_qPDF_ratio_NLO,
    "CG_gtgpg5_qPDF_hybrid_NLO": CG_gtgpg5_qPDF_hybrid_NLO,
    # gauge-invariant (straight Wilson line) gamma^t and gamma^t gamma5. Both operators
    # share one NLO coefficient, so unpolarized and helicity map to the same math.
    "GI_gt_qPDF_ratio_NLO": GI_gt_qPDF_ratio_NLO,
    "GI_gt_qPDF_hybrid_NLO": GI_gt_qPDF_hybrid_NLO,
    "GI_gtg5_qPDF_ratio_NLO": GI_gtg5_qPDF_ratio_NLO,
    "GI_gtg5_qPDF_hybrid_NLO": GI_gtg5_qPDF_hybrid_NLO,
    # gauge-invariant gamma^z and gamma^z gamma5 (arXiv:2604.00143). Same coefficient
    # as the GI gamma^t pair plus the 2(1-ksi) of Eq. (C7).
    "GI_gz_qPDF_ratio_NLO": GI_gz_qPDF_ratio_NLO,
    "GI_gz_qPDF_hybrid_NLO": GI_gz_qPDF_hybrid_NLO,
    "GI_gzg5_qPDF_ratio_NLO": GI_gzg5_qPDF_ratio_NLO,
    "GI_gzg5_qPDF_hybrid_NLO": GI_gzg5_qPDF_hybrid_NLO,
    # gauge-invariant transversity (arXiv:2208.08008). Unlike the Coulomb-gauge
    # transversity, the hybrid scheme here does differ from the ratio one.
    "GI_gtgpg5_qPDF_ratio_NLO": GI_gtgpg5_qPDF_ratio_NLO,
    "GI_gtgpg5_qPDF_hybrid_NLO": GI_gtgpg5_qPDF_hybrid_NLO,
    # meson distribution amplitude (arXiv:2212.14415 Eq. 4.15, V_qq,p). Its kernel is a
    # genuine V(x, y), not a function of x/y -- see the DA section of kernels.py.
    "GI_gzg5_DA_ratio_NLO": GI_gzg5_DA_ratio_NLO,
    "GI_gzg5_DA_hybrid_NLO": GI_gzg5_DA_hybrid_NLO,
}


def is_hybrid_kernel(kernel_id: str) -> bool:
    """True when ``kernel_id`` names a hybrid-scheme kernel (the only ones needing zspz).

    The scheme is a segment of the id (CG_<operator>_qPDF_<scheme>_<order>), not its
    tail, so match on the segment rather than on a suffix.
    """
    return "hybrid" in str(kernel_id).split("_")


def resolve_kernel_id(kernel_id: str, scheme: str) -> str:
    """Validate that ``kernel_id`` is a registered kernel and return it.

    Kernels are selected by their own registry name -- e.g. ``CG_gt_qPDF_ratio_NLO``,
    ``CG_gz_qPDF_msbar_NLO``, ``CG_gtgpg5_qPDF_hybrid_NLO`` -- so the manifest names the exact
    operator+scheme it wants; there is no logical alias to translate. ``scheme`` is
    still part of the manifest declaration (it also drives the renormalization stage)
    and is only used here to make the error message clearer.
    """
    if kernel_id in KERNEL_REGISTRY:
        return kernel_id
    raise ValueError(
        f"Unknown kernel_id {kernel_id!r} (scheme {scheme!r}). "
        f"Available: {sorted(KERNEL_REGISTRY)}"
    )


# --- hybrid-scheme scale -----------------------------------------------------
# The hybrid kernel needs the dimensionless Wilson-line scale zspz = z_s * P_z.
# Both z_s (the Wilson-line length, in fm) and P_z (the hadron momentum, in GeV)
# come from the matching job's effective stage parameters (zs_fm and momentum_gev);
# nothing ensemble-specific is hardcoded here. GEV_FM is the only constant -- the
# physical hbar*c used to make zspz dimensionless:
#     zspz = zs_fm * momentum_gev / GEV_FM
GEV_FM = 0.1973269631  # hbar*c in GeV*fm


# Each tool below follows the same pattern: take this stage's shared store, do
# one step, write the result under store[out], and return a small summary for the
# agent to decide the next step. The store is how adjacent tools pass data
# (load -> build -> apply -> plot).


def list_kernels(store: dict[str, Any]) -> dict[str, Any]:
    """Tell the agent which kernel_ids are available (reads no input)."""
    del store  # this tool needs no input; the registry itself is the answer
    return {"kernel_ids": sorted(KERNEL_REGISTRY)}


# --- load the quasi-PDF from the previous (Fourier) stage --------------------


def _component_ensemble_data(data: EnsembleData, component: str) -> EnsembleData:
    """Return the real or imaginary channel of a (possibly complex) EnsembleData."""
    if component == "re":
        return data.real
    if component == "im":
        return data.imag
    raise ValueError(f"component must be 're' or 'im', got '{component}'.")


def _samples_ensemble_data_from_npz(raw: Any) -> EnsembleData:
    """Build an EnsembleData from a simple hand-made npz of per-sample data.

    Accepts ``x_ls`` plus a 2D ``quasi_samples`` array shaped ``(n_sample, n_x)``
    and an optional scalar ``resample`` string (defaults to ``"bootstrap"``).
    """
    x_ls = np.asarray(raw["x_ls"], dtype=float)
    samples = np.asarray(raw["quasi_samples"], dtype=float)
    if samples.ndim != 2:
        raise ValueError("quasi_samples must be a 2D array shaped (n_sample, n_x).")
    if samples.shape[1] != x_ls.size:
        raise ValueError(
            f"quasi_samples second axis ({samples.shape[1]}) must match x_ls size ({x_ls.size})."
        )
    resample = str(raw["resample"]) if "resample" in raw else "bootstrap"
    return EnsembleData(
        ensemble=None,
        resample=resample,
        values=[samples[idx] for idx in range(samples.shape[0])],
        dims=("x",),
        coords={"x": x_ls.tolist()},
        name="quasi_pdf",
    )


def load_quasi_pdf(
    store: dict[str, Any],
    *,
    path: str | None = None,
    component: str = "re",
    quasi_out: str = "quasi_ed",
    grid_out: str = "x_ls",
) -> dict[str, Any]:
    """Load the per-sample quasi-PDF and its momentum grid from a Fourier artifact.

    Matching is done sample by sample, so this selects the **full resampling
    axis** from the in-memory job input or an external artifact, not just a
    central value with an error.

    Three artifact layouts are accepted automatically:
    - the Fourier-stage ``EnsembleData`` NetCDF artifact (default): the complex
      quasi-PDF samples live in the DataArray values and the x grid in coord ``x``;
    - the legacy Fourier-stage ``EnsembleData`` npz: the complex
      quasi-PDF samples live in ``values`` (shape ``(n_sample, n_x)``) and the x
      grid in the JSON metadata ``coords["x"]``;
    - a simple hand-made npz with ``x_ls`` and a 2D ``quasi_samples`` array
      (shape ``(n_sample, n_x)``) plus an optional ``resample`` string.

    ``component`` selects the real (``"re"``) or imaginary (``"im"``) channel of
    the Fourier output; the unpolarized quasi-PDF lives in the real part.
    """
    if path is None and isinstance(store.get("quasi"), EnsembleData):
        quasi_ed = _component_ensemble_data(store["quasi"], component)
    elif path is not None:
        try:
            data = EnsembleData.from_netcdf(path) if Path(path).suffix.lower() == ".nc" else EnsembleData.load_npz(path)[0]
            quasi_ed = _component_ensemble_data(data, component)
        except ValueError:
            raw = np.load(path, allow_pickle=False)
            if "quasi_samples" not in raw or "x_ls" not in raw:
                raise ValueError(
                    f"Unrecognized quasi-PDF artifact '{path}': expected a Fourier-stage "
                    "EnsembleData NetCDF artifact or an npz with x_ls/quasi_samples."
                )
            quasi_ed = _samples_ensemble_data_from_npz(raw)
    else:
        raise ValueError("load_quasi_pdf needs the job's quasi input or an artifact path.")

    if quasi_ed.resample == "gvar":
        raise ValueError(
            f"Artifact '{path}' stores a gvar (central value + error) quasi-PDF, which "
            "has no resampling axis; sample-by-sample matching needs the raw/bootstrap/"
            "jackknife samples. Re-export the quasi-PDF with its samples."
        )

    # The x grid (shared by the quasi- and light-cone PDFs) is the EnsembleData's
    # only physical coordinate.
    x_ls = np.asarray(quasi_ed.coords["x"], dtype=float)

    # The store is the temporary dict shared by this stage's tools; build/apply
    # read the data back from here.
    store[grid_out] = x_ls
    store[quasi_out] = quasi_ed
    store["matching_component"] = component
    return {
        "quasi_out": quasi_out,
        "grid_out": grid_out,
        "component": component,
        "resample": quasi_ed.resample,
        "n_sample": int(quasi_ed.n_sample),
        "n_points": int(x_ls.size),
    }


# --- build the kernel matrix ------------------------------------------------


def build_matching_kernel(
    store: dict[str, Any],
    *,
    kernel_id: str,
    momentum_gev: float,
    mu: float = 2.0,
    zs_fm: float | None = None,
    grid: str = "x_ls",
    y_grid: str | None = None,
    out: str = "kernel_matrix",
) -> dict[str, Any]:
    """Build the (nx, ny) matching kernel for the chosen operator/kernel_id.

    Hybrid kernels also need the Wilson-line length ``zs_fm`` (z_s in fm); together
    with ``momentum_gev`` it sets the dimensionless scale ``zspz = zs_fm * momentum_gev / GEV_FM``.
    Both ``zs_fm`` and ``momentum_gev`` come from the matching job's effective stage
    parameters. MSbar and ratio kernels ignore ``zs_fm``.
    """
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

    # Hybrid kernels need the Wilson-line scale zspz = z_s * P_z, built from the
    # matching job parameters zs_fm (fm) and momentum_gev (GeV). Only the hybrid builders accept a
    # zspz keyword (msbar/ratio do not), so it is computed and passed only for
    # the hybrid ids.
    zspz: float | None = None
    builder_kwargs: dict[str, Any] = {"momentum_gev": momentum_gev, "mu": mu, "quasi_y_ls": y_ls}
    if is_hybrid_kernel(kernel_id):
        if zs_fm is None:
            raise ValueError(
                f"Kernel '{kernel_id}' is a hybrid kernel and needs zs_fm (z_s in fm) "
                "from perturbative_matching defaults or job params to build "
                "zspz = zs_fm * momentum_gev / GEV_FM."
            )
        zspz = float(zs_fm) * momentum_gev / GEV_FM
        builder_kwargs["zspz"] = zspz

    # Build the already-discretized kernel matrix, typically shaped (len(x_ls), len(y_ls)).
    matrix = builder(x_ls, **builder_kwargs)
    if matrix.ndim != 2:
        raise ValueError(f"Kernel builder returned a non-matrix object with ndim={matrix.ndim}.")
    if matrix.shape[0] != x_ls.size:
        raise ValueError(
            f"Kernel row count {matrix.shape[0]} does not match x grid size {x_ls.size}."
        )

    store[out] = matrix  # apply_matching reads it back under this name
    store["matching_kernel_info"] = {"kernel_id": kernel_id, "momentum_gev": momentum_gev, "mu": mu, "zspz": zspz}
    return {
        "out": out,
        "kernel_id": kernel_id,
        "shape": list(matrix.shape),
        "momentum_gev": momentum_gev,
        "mu": mu,
        "zspz": zspz,  # None for MSbar/ratio; zs_fm * momentum_gev / GEV_FM for hybrid
    }


# --- apply the kernel: quasi-PDF -> light-cone PDF --------------------------


def apply_matching(
    store: dict[str, Any],
    *,
    kernel: str = "kernel_matrix",
    quasi: str = "quasi_ed",
    grid: str = "x_ls",
    out: str = "lightcone_ed",
    save_path: str | None = None,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Apply the matching kernel sample by sample: ``lightcone_i = K @ quasi_i``.

    Each resampling sample of the quasi-PDF is convolved with the kernel
    independently, so the full sample-level correlation structure is preserved.
    The matched samples are stored as an ``EnsembleData`` (same resampling mode
    as the input) under ``store[out]``; its mean/error/covariance can be rebuilt
    from the samples by any downstream stage.
    """
    if kernel not in store:
        raise ValueError(f"Kernel '{kernel}' not in store; run build_matching_kernel first.")
    if quasi not in store:
        raise ValueError(f"Quasi-PDF '{quasi}' not in store; run load_quasi_pdf first.")
    if grid not in store:
        raise ValueError(f"Momentum grid '{grid}' not in store; run load_quasi_pdf first.")

    matrix = np.asarray(store[kernel], dtype=float)
    quasi_ed: EnsembleData = store[quasi]
    if matrix.ndim != 2:
        raise ValueError(f"Kernel '{kernel}' must be a 2D matrix.")

    # quasi_ed.values is shaped (n_sample, n_y); apply the kernel to every sample
    # at once as samples @ K^T, giving (n_sample, n_x) matched samples. This is
    # mathematically identical to looping lightcone_i = K @ quasi_i.
    quasi_samples = np.asarray(quasi_ed.values, dtype=float)
    if matrix.shape[1] != quasi_samples.shape[1]:
        raise ValueError(
            f"Kernel columns ({matrix.shape[1]}) must match quasi-PDF y points "
            f"({quasi_samples.shape[1]})."
        )

    lightcone_samples = quasi_samples @ matrix.T  # (n_sample, n_x)

    # The kernel rows define the output (light-cone) x grid.
    x_out = np.asarray(store[grid], dtype=float)
    if matrix.shape[0] != x_out.size:
        raise ValueError(
            f"Kernel rows ({matrix.shape[0]}) must match output x grid size ({x_out.size})."
        )

    lightcone_ed = EnsembleData(
        ensemble=quasi_ed.ensemble,
        resample=quasi_ed.resample,
        values=[lightcone_samples[idx] for idx in range(lightcone_samples.shape[0])],
        dims=("x",),
        coords={"x": x_out.tolist()},
        attrs=quasi_ed.attrs,
        name="lightcone_pdf",
    )
    store[out] = lightcone_ed
    output = Path(save_path) if save_path is not None else Path(artifacts_dir or "artifacts") / "matched_pdf"
    artifact = output.with_suffix(".nc")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    lightcone_ed.to_netcdf(artifact)
    store["output"] = lightcone_ed
    store["matching_artifact"] = str(artifact)

    # Summarize with the sample-built central value (mean over samples) so the
    # agent can sanity-check the result without re-reading the store.
    return {
        "out": out,
        "resample": lightcone_ed.resample,
        "n_sample": int(lightcone_ed.n_sample),
        "n_points": int(x_out.size),
        "artifact": str(artifact),
        "mean_sample": [float(v) for v in np.asarray(lightcone_ed.mean)[:3]],
    }


# --- plotting ---------------------------------------------------------------


def plot_matched_pdf(
    store: dict[str, Any],
    *,
    grid: str = "x_ls",
    quasi: str = "quasi_ed",
    lightcone: str = "lightcone_ed",
    save_path: str | None = None,
    artifacts_dir: str | None = None,
    xlim: list[float] | tuple[float, float] | None = None,
    ylim: list[float] | tuple[float, float] | None = None,
    sector: str | None = None,
) -> dict[str, Any]:
    """Plot quasi vs matched (light-cone) PDF and save a PDF artifact.

    Both PDFs are ``EnsembleData``; the band is their sample-built mean +/- error.

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

    x_ls = np.asarray(store[grid], dtype=float)
    quasi_ed: EnsembleData = store[quasi]
    lightcone_ed: EnsembleData = store[lightcone]

    if x_ls.shape != np.shape(quasi_ed.mean) or x_ls.shape != np.shape(lightcone_ed.mean):
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

    plot_min = np.inf
    plot_max = -np.inf

    def _band(data: EnsembleData, *, label: str, color: str) -> None:
        # Center line is the sample mean; the band is the sample-built +/- error,
        # using the resampling-aware mean/sdev of EnsembleData.
        nonlocal plot_min, plot_max
        mean = np.asarray(data.mean, dtype=float)
        sdev = np.asarray(data.sdev, dtype=float)
        plot_min = min(plot_min, float(np.min(mean - sdev)))
        plot_max = max(plot_max, float(np.max(mean + sdev)))
        ax.fill_between(x_ls, mean - sdev, mean + sdev, color=color, alpha=0.32, linewidth=0, label=label)
        ax.plot(x_ls, mean, color=color, linewidth=0.9, alpha=0.85)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    _band(quasi_ed, label="quasi", color=BLUE)
    _band(lightcone_ed, label="light-cone", color=ORANGE)
    ax.set_xlabel(r"$x$", **FONT_SIZE)
    ax.set_ylabel(r"$f(x)$", **FONT_SIZE)
    sector_name = str(sector or quasi_ed.attrs.get("sector", "")).lower()
    x_limits = ((-0.01, 1.01) if sector_name == "valence" else (-1.01, 1.01)) if xlim is None else (float(xlim[0]), float(xlim[1]))
    y_limits = (plot_min - 0.2, plot_max + 1.0) if ylim is None else (float(ylim[0]), float(ylim[1]))
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.legend(**LEGEND_SETS)
    svg_save = resolved_save.with_suffix(".svg")
    fig.savefig(resolved_save, bbox_inches="tight", transparent=True)
    fig.savefig(svg_save, bbox_inches="tight")
    plt.close(fig)

    result = {
        "path": str(resolved_save),
        "plot_image": str(svg_save),
        "n_points": int(x_ls.size),
        "xlim": [float(x_limits[0]), float(x_limits[1])],
        "ylim": [float(y_limits[0]), float(y_limits[1])],
    }
    # Leave the plot path in the store so the report can link it.
    store["matching_plot"] = result
    return result


# --- reporting --------------------------------------------------------------


def _report_path(raw: str | None, *, default_name: str, artifacts_dir: str | None = None) -> Path:
    """Resolve the report output path under ``artifacts_dir`` (mirrors the Fourier stage)."""
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw:
        path = Path(raw).expanduser()
        if path.is_absolute():
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return out_dir / path
    return out_dir / default_name


def report_matching_result(
    store: dict[str, Any],
    *,
    kernel_id: str | None = None,
    momentum_gev: float | None = None,
    mu: float = 2.0,
    zs_fm: float | None = None,
    component: str | None = None,
    save_path: str | None = None,
    artifacts_dir: str | None = None,
    report_language: str = "en",
    backend: str = "",
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Write a Markdown report for the matching stage.

    Physics parameters come from the matching job and kernel declaration; the x
    grid, quasi-PDF, and matched PDF come from the current job store.
    """
    if "quasi_ed" not in store or "lightcone_ed" not in store or "x_ls" not in store:
        raise ValueError(
            "report_matching_result needs the quasi-PDF, matched PDF, and x grid in "
            "the store; run load_quasi_pdf, build_matching_kernel and apply_matching first."
        )
    quasi_ed: EnsembleData = store["quasi_ed"]
    lightcone_ed: EnsembleData = store["lightcone_ed"]
    x_ls = np.asarray(store["x_ls"], dtype=float)

    # Hybrid kernels carry the dimensionless Wilson-line scale; recompute it here
    # from the manifest inputs (same formula as build_matching_kernel).
    zspz = None
    if kernel_id is not None and is_hybrid_kernel(kernel_id) and zs_fm is not None and momentum_gev is not None:
        zspz = float(zs_fm) * float(momentum_gev) / GEV_FM

    result = {
        "kernel_id": kernel_id,
        "momentum_gev": momentum_gev,
        "mu": mu,
        "zspz": zspz,
        "component": component,
        "source": "job input",
        "resample": lightcone_ed.resample,
        "n_sample": int(lightcone_ed.n_sample),
        "n_points": int(x_ls.size),
        "x_grid": x_ls.tolist(),
        "quasi_mean": [float(v) for v in np.asarray(quasi_ed.mean)],
        "lightcone_mean": [float(v) for v in np.asarray(lightcone_ed.mean)],
    }

    artifacts: dict[str, Any] = {}
    if isinstance(store.get("matching_artifact"), str):
        artifacts["lightcone_artifact"] = store["matching_artifact"]
    if isinstance(store.get("matching_plot"), dict):
        artifacts["matched_plot"] = store["matching_plot"].get("path")
        artifacts["matched_plot_image"] = store["matching_plot"].get("plot_image")

    output = _report_path(save_path, default_name="report_matching.md", artifacts_dir=artifacts_dir)
    # The LLM that writes the formula section is the run's own -- the agent injects the
    # resolved backend/provider/key here, exactly as it does for the review stage's tool.
    llm = FormulaLlm(backend=backend or "api", provider=provider, api_key=api_key,
                     model_name=model_name, base_url=base_url)
    paths = write_matching_report(
        result=result, artifacts=artifacts, path=output, report_language=report_language, llm=llm
    )
    report = {"report": str(paths["report"])}
    store["matching_report"] = report
    return report


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
    "report_matching_result": report_matching_result,
}
