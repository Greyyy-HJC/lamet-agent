"""Read and plot converted CG PDF matrix-element samples.

Purpose:
- load converted two-column qTMDPDF sample text files from ``data_cg_pdf``
- compare the HISQa060_X ensemble at P=0 and P=5 as a function of z
- show separate P=0 and P=5 plots for quick inspection of the converted matrix elements

Expected inputs:
- ``data_cg_pdf/bare_matrix_elements/*.txt`` files generated from raw
  matrix-element bootstrap samples

Expected outputs:
- two matplotlib figures shown interactively

Example usage:
- PYTHONPATH=src .venv/bin/python examples/cg_pdf_data/read_cg_pdf_matrix_elements.py
"""
# %%
from __future__ import annotations

from pathlib import Path

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np

from lamet_agent.core.plotting import COLOR_CYCLE, ERRORBAR_STYLE, FONT_SIZE, default_plot, LEGEND_SETS
from lamet_agent.stages.correlator.functions import bs_ls_avg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_ELEMENT_DIR = PROJECT_ROOT / "data_cg_pdf" / "bare_matrix_elements"
DS_STAGE1_P0_DIR = PROJECT_ROOT / "runs" / "ds_pdf_stage1" / "artifacts" / "bare_qpdf"
Z_VALUES_A060_X = range(25)

SERIES = {
    "P=0": {
        "tag": "CG52bxp00_CG52bxp00",
        "momentum": "PX0PY0PZ0",
        "color": COLOR_CYCLE[0],
    },
    "P=5": {
        "tag": "CG52bxp30_CG52bxp30",
        "momentum": "PX5PY0PZ0",
        "color": COLOR_CYCLE[1],
    },
}


def matrix_element_path(directory: Path, *, tag: str, momentum: str, z: int) -> Path:
    """Build the complete HISQa060_X matrix-element filename."""
    filename = f"HISQa060_X_{tag}_free_X_{momentum}_b0_z{z}.txt"
    return directory / filename


def read_sample_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return real and imaginary bootstrap samples from one converted text file."""
    arr = np.loadtxt(path)
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 2:
        raise ValueError(f"expected two columns in {path}, got shape {arr.shape}")
    return arr[:, 0], arr[:, 1]


def collect_series(
    *,
    directory: Path = MATRIX_ELEMENT_DIR,
    tag: str,
    momentum: str,
    z_values: range = Z_VALUES_A060_X,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one HISQa060_X momentum series and return z, real gvar, imag gvar."""
    if not directory.is_dir():
        raise FileNotFoundError(f"matrix-element directory does not exist: {directory}")

    real_samples: list[np.ndarray] = []
    imag_samples: list[np.ndarray] = []
    loaded_z_values: list[int] = []
    for z in z_values:
        path = matrix_element_path(directory, tag=tag, momentum=momentum, z=z)
        if not path.exists():
            continue
        real, imag = read_sample_file(path)
        real_samples.append(real)
        imag_samples.append(imag)
        loaded_z_values.append(z)

    if not loaded_z_values:
        raise FileNotFoundError(f"no converted samples found in {directory} for HISQa060_X {tag} {momentum}")

    z_array = np.asarray(loaded_z_values, dtype=int)
    real_array = np.asarray(real_samples, dtype=float)
    imag_array = np.asarray(imag_samples, dtype=float)
    real_gv = bs_ls_avg(real_array.T)
    imag_gv = bs_ls_avg(imag_array.T)
    return z_array, real_gv, imag_gv


def plot_momentum(label: str, *, tag: str, momentum: str, color: str) -> None:
    """Plot real and imaginary converted matrix elements for one momentum."""
    fig, ax = default_plot()
    z_array, real_gv, imag_gv = collect_series(tag=tag, momentum=momentum)
    ax.errorbar(
        z_array,
        gv.mean(real_gv),
        gv.sdev(real_gv),
        label="Re",
        color=color,
        **ERRORBAR_STYLE,
    )
    ax.errorbar(
        z_array,
        gv.mean(imag_gv),
        gv.sdev(imag_gv),
        label="Im",
        color=color,
        marker="s",
        **ERRORBAR_STYLE,
    )

    ax.set_xlabel(r"$z/a$", **FONT_SIZE)
    ax.set_ylabel("Bare qPDF sample value", **FONT_SIZE)
    ax.set_title(f"HISQa060_X converted qPDF samples, {label}", **FONT_SIZE)
    ax.legend(**LEGEND_SETS)
    fig.tight_layout()


def plot_a060_x_p0_p5() -> None:
    """Show separate P=0 and P=5 converted matrix-element plots."""
    for label, config in SERIES.items():
        plot_momentum(
            label,
            tag=config["tag"],
            momentum=config["momentum"],
            color=config["color"],
        )


if __name__ == "__main__":
    plot_a060_x_p0_p5()
    plt.show()

# %%
