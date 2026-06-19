"""Reusable plotting defaults for lattice-QCD analysis."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import gvar as gv

# Modern, publication-oriented palette (Nature-style inspired).
GREY = "#7F7F7F"
RED = "#D62728"
PEACH = "#FFBE7A"
ORANGE = "#E69F00"
SUNKIST = "#F2C12E"
YELLOW = "#FFD54F"
LIME = "#B2DF8A"
GREEN = "#2CA02C"
TURQUOISE = "#1B9E77"
BLUE = "#4E79A7"
GRAPE = "#6A3D9A"
VIOLET = "#7B6FD0"
FUCHSIA = "#CC79A7"
BROWN = "#8C564B"
EMERALD = "#009E73"
SKY = "#56B4E9"
GOLD = "#F0E442"
ROYAL_BLUE = "#0072B2"
VERMILION = "#D55E00"
SILVER = "#999999"
OCHRE = "#A6761D"
LEAF = "#66A61E"
AZURE = "#1F78B4"
CRIMSON = "#E31A1C"
ROSE = "#FB9A99"
LAVENDER = "#CAB2D6"
UMBER = "#B15928"

COLOR_CYCLE = [
    BLUE,
    ORANGE,
    GREEN,
    RED,
    VIOLET,
    FUCHSIA,
    TURQUOISE,
    GRAPE,
    LIME,
    PEACH,
    SUNKIST,
    YELLOW,
    BROWN,
    EMERALD,
    SKY,
    GOLD,
    ROYAL_BLUE,
    VERMILION,
    SILVER,
    OCHRE,
    LEAF,
    AZURE,
    CRIMSON,
    ROSE,
    LAVENDER,
    UMBER,
]

def darken_color(color, factor=0.65):
    """
    Darken a hex color by multiplying RGB channels with factor.
    factor < 1 => darker
    """
    rgb = mcolors.to_rgb(color)
    dark_rgb = tuple(max(min(c * factor, 1), 0) for c in rgb)
    return mcolors.to_hex(dark_rgb)

EDGE_COLOR_CYCLE = [
    darken_color(c, factor=0.55)
    for c in COLOR_CYCLE
]

MARKER_CYCLE = [
    ".",
    "o",
    "s",
    "P",
    "X",
    "*",
    "p",
    "D",
    "<",
    ">",
    "^",
    "v",
    "1",
    "2",
    "3",
    "4",
    "+",
    "x",
    "h",
    "H",
    "d",
    "|",
    "_",
    ",",
]

FONT_CONFIG = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
}

FIG_WIDTH = 6.75
GOLDEN_RATIO = 1.618034333
FIG_SIZE = (FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO)

PLOT_AXES = [0.15, 0.15, 0.8, 0.8]
FONT_SIZE = {"fontsize": 18}
LEGEND_SIZE = {"fontsize": 14}
LABEL_SIZE = {"labelsize": 18}

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

TSEP = r"$t_{\mathrm{sep}}$"

TMIN_LABEL = r"$t_{\mathrm{min}}~/~a$"
TMAX_LABEL = r"$t_{\mathrm{max}}~/~a$"
TAU_CENTER_LABEL = r"$(\tau - t_{\rm{sep}}/2)~/~a$"
TSEP_LABEL = r"${t_{\mathrm{sep}}~/~a}$"
Z_LABEL = r"${z~/~a}$"
LAMBDA_LABEL = r"$\lambda = z P^z$"
MEFF_LABEL = r"${m}_{\mathrm{eff}}$"

RATIO_REAL_LABEL = r"$\Re\left[\mathcal{R}(t_{\mathrm{sep}},\tau)\right]$"
RATIO_IMAG_LABEL = r"$\Im\left[\mathcal{R}(t_{\mathrm{sep}},\tau)\right]$"


def apply_plot_style() -> None:
    """Apply package default font settings to matplotlib rcParams."""
    rcParams.update(FONT_CONFIG)


def auto_ylim(
    y_data: Sequence[np.ndarray], yerr_data: Sequence[np.ndarray], y_range_ratio: float = 4.0
) -> tuple[float, float]:
    """Compute y-limits from data and uncertainties with symmetric margin."""
    all_y = np.concatenate(
        [y + yerr for y, yerr in zip(y_data, yerr_data)]
        + [y - yerr for y, yerr in zip(y_data, yerr_data)]
    )
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    y_range = y_max - y_min
    return y_min - y_range / y_range_ratio, y_max + y_range / y_range_ratio


def default_plot() -> tuple[Figure, Axes]:
    """Create a default single-panel plot."""
    apply_plot_style()
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.axes()
    ax.tick_params(direction="in", top=True, right=True, **LABEL_SIZE)
    ax.grid(linestyle=":")
    return fig, ax


def default_sub_plot(height_ratio: int = 3) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create default 2-row subplots with a shared x-axis."""
    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=FIG_SIZE,
        gridspec_kw={"height_ratios": [height_ratio, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)

    for ax in (ax1, ax2):
        ax.tick_params(direction="in", top=True, right=True, **LABEL_SIZE)
        ax.grid(linestyle=":")

    return fig, (ax1, ax2)


from collections.abc import Sequence
from typing import Literal


ResamplingMode = Literal["none", "jk", "bs"]
SampleCovarianceMode = Literal["jk", "bs"]


def bad_point_filter(data: np.ndarray, threshold: float = 1) -> np.ndarray:
    """Replace entries with absolute value above threshold by random signs."""
    filtered = np.array(data, copy=True)
    mask = np.abs(filtered) > threshold
    bad_loc = np.argwhere(mask)

    for loc in bad_loc:
        filtered[tuple(loc)] = np.random.choice([-1, 1])

    return filtered


def bin_data(data: np.ndarray, bin_size: int, axis: int = 0) -> np.ndarray:
    """Average adjacent configurations into bins.

    Parameters
    ----------
    data:
        Input ensemble data.
    bin_size:
        Number of configurations per bin.
    axis:
        Configuration axis.

    Returns
    -------
    numpy.ndarray
        Binned data.
    """
    data = np.asarray(data)

    if bin_size < 1:
        raise ValueError("bin_size must be a positive integer")

    data = np.moveaxis(data, axis, 0)
    n_bins = data.shape[0] // bin_size
    data = data[: n_bins * bin_size]
    data = data.reshape(n_bins, bin_size, *data.shape[1:]).mean(axis=1)

    return np.moveaxis(data, 0, axis)


def bootstrap(
    data: np.ndarray,
    n_samples: int,
    sample_size: int | None = None,
    axis: int = 0,
    bin_size: int = 1,
    seed: int | None = 1984,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate bootstrap samples from ensemble data.

    Parameters
    ----------
    data:
        Input ensemble data.
    n_samples:
        Number of bootstrap samples to generate.
    sample_size:
        Number of configurations drawn per bootstrap sample. Defaults to the
        number of configurations.
    axis:
        Configuration axis.
    bin_size:
        Optional bin size applied before resampling.
    seed:
        Random seed for reproducible sampling.
    return_indices:
        Whether to return the sampled configuration indices with the samples.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Bootstrap sample averages, optionally with sampled indices.
    """
    data = np.asarray(data)

    if bin_size > 1:
        data = bin_data(data, bin_size, axis=axis)

    n_conf = data.shape[axis]

    if sample_size is None:
        sample_size = n_conf

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_conf, (n_samples, sample_size), replace=True)
    samples = np.take(data, indices, axis=axis).mean(axis=axis + 1)

    if return_indices:
        return samples, indices

    return samples


def jackknife(data: np.ndarray, axis: int = 0, bin_size: int = 1) -> np.ndarray:
    """Generate leave-one-bin-out jackknife samples.

    Parameters
    ----------
    data:
        Input ensemble data.
    axis:
        Configuration axis.
    bin_size:
        Optional bin size applied before jackknife resampling.

    Returns
    -------
    numpy.ndarray
        Jackknife sample averages.
    """
    data = np.asarray(data)

    if bin_size > 1:
        data = bin_data(data, bin_size, axis=axis)

    n_conf = data.shape[axis]

    if n_conf < 2:
        raise ValueError("jackknife needs at least two samples")

    total = data.sum(axis=axis, keepdims=True)

    return (total - data) / (n_conf - 1)


def apply_resampling(
    data: np.ndarray,
    mode: ResamplingMode = "none",
    *,
    sample_axis: int = 0,
    n_samples: int = 200,
    bin_size: int = 5,
    seed: int | None = 1984,
) -> np.ndarray:
    """Apply optional resampling on a correlator array."""
    data = np.asarray(data)
    if mode == "none":
        return data
    if mode == "jk":
        return jackknife(data, axis=sample_axis, bin_size=bin_size)
    if mode == "bs":
        return bootstrap(
            data,
            n_samples=n_samples,
            axis=sample_axis,
            bin_size=bin_size,
            seed=seed,
        )
    raise ValueError(f"unsupported resampling mode: {mode!r}")


def jk_ls_avg(jk_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average jackknife samples into gvar values."""
    jk_arr = np.asarray(jk_ls)
    assert np.isrealobj(jk_arr), "jk_ls must contain real-valued samples"
    if axis != 0:
        jk_arr = np.swapaxes(jk_arr, 0, axis)

    shape = jk_arr.shape
    jk_flat = jk_arr.reshape(shape[0], -1)
    n_sample = jk_flat.shape[0]
    mean = np.mean(jk_flat, axis=0)

    if jk_flat.shape[1] == 1:
        sdev = np.std(jk_flat, axis=0) * np.sqrt(n_sample - 1)
        return gv.gvar(mean, sdev)

    cov = np.cov(jk_flat, rowvar=False) * (n_sample - 1)
    out = gv.gvar(mean, cov)
    return out.reshape(shape[1:])


def bs_ls_avg(bs_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average bootstrap samples into gvar values."""
    bs_arr = np.asarray(bs_ls)
    assert np.isrealobj(bs_arr), "bs_ls must contain real-valued samples"
    if axis != 0:
        bs_arr = np.swapaxes(bs_arr, 0, axis)

    shape = bs_arr.shape
    bs_flat = bs_arr.reshape(shape[0], -1)
    mean = np.mean(bs_flat, axis=0)

    if bs_flat.shape[1] == 1:
        sdev = np.std(bs_flat, axis=0)
        return gv.gvar(mean, sdev)

    cov = np.cov(bs_flat, rowvar=False)
    out = gv.gvar(mean, cov)
    return out.reshape(shape[1:])


def bs_ls_avg_percentile(bs_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average bootstrap samples into gvar values without cross-correlations.

    The central value is the per-entry median across samples and the symmetric
    error is half the 16-84 percentile range, i.e. the 1-sigma width of a
    normal distribution. No covariance is propagated between entries.

    Parameters
    ----------
    bs_ls:
        Bootstrap samples.
    axis:
        Axis indexing independent bootstrap samples.

    Returns
    -------
    numpy.ndarray
        ``gvar`` array shaped like ``bs_ls`` with ``axis`` removed.
    """
    bs_arr = np.asarray(bs_ls)
    assert np.isrealobj(bs_arr), "bs_ls must contain real-valued samples"
    if axis != 0:
        bs_arr = np.swapaxes(bs_arr, 0, axis)

    shape = bs_arr.shape
    bs_flat = bs_arr.reshape(shape[0], -1)

    mid = np.median(bs_flat, axis=0)
    p16, p84 = np.percentile(bs_flat, [16, 84], axis=0)
    sdev = 0.5 * (p84 - p16)

    out = gv.gvar(mid, sdev)
    return out.reshape(shape[1:])


def jk_dict_avg(data: dict[str, np.ndarray]) -> dict[str, list[gv.GVar]]:
    """Average a dict of jackknife arrays into a dict of gvar lists."""
    key_order = list(data.keys())
    lengths = {key: len(data[key][0]) for key in key_order}
    n_sample = len(data[key_order[0]])

    merged: list[list[float]] = []
    for idx in range(n_sample):
        row: list[float] = []
        for key in key_order:
            row.extend(list(data[key][idx]))
        merged.append(row)

    gv_ls = list(jk_ls_avg(np.asarray(merged)))
    out: dict[str, list[gv.GVar]] = {}
    for key in key_order:
        out[key] = [gv_ls.pop(0) for _ in range(lengths[key])]
    return out


def bs_dict_avg(data: dict[str, np.ndarray]) -> dict[str, list[gv.GVar]]:
    """Average a dict of bootstrap arrays into a dict of gvar lists."""
    key_order = list(data.keys())
    lengths = {key: len(data[key][0]) for key in key_order}
    n_sample = len(data[key_order[0]])

    merged: list[list[float]] = []
    for idx in range(n_sample):
        row: list[float] = []
        for key in key_order:
            row.extend(list(data[key][idx]))
        merged.append(row)

    gv_ls = list(bs_ls_avg(np.asarray(merged)))
    out: dict[str, list[gv.GVar]] = {}
    for key in key_order:
        out[key] = [gv_ls.pop(0) for _ in range(lengths[key])]
    return out


def plot_fourier_artifact(
    path,
    *,
    save_path=None,
    title=None,
    show=False,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot real and imaginary momentum-space distributions from a Fourier artifact.

    Expected input is the NetCDF artifact written by the Fourier stage. Legacy
    NPZ artifacts remain accepted for old examples.

    Example
    -------
    ``plot_fourier_artifact("artifacts/fourier_result.nc", save_path="fourier.pdf")``
    """
    from lamet_agent.core.plotting import plot_fourier_artifact as _plot_fourier_artifact

    return _plot_fourier_artifact(
        path,
        save_path=save_path,
        title=title,
        show=show,
    )
