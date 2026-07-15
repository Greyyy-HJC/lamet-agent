"""Renormalization stage tools.

Purpose:
- load bare coordinate-space matrix-element bootstrap samples as EnsembleData
- apply sample-preserving hybrid-ratio or self-renormalization
- fit a self-renormalization factor zR from zero-momentum reference data

Expected inputs:
- correlator-stage bare matrix-element NetCDF files
- NPZ/NetCDF reference with ``z`` (fm) and samples on ``z`` or ``(a, z)``
- tool arguments supplied by the agent as JSON-compatible values

Expected outputs:
- renormalized complex EnsembleData on ``z`` for downstream Fourier tools
- ``reference``: bootstrap/jackknife EnsembleData on ``z`` or ``(a, z)``
- ``zR``: bootstrap EnsembleData on ``(a, z)`` with one sample equal to mean zR

Example usage:
- from lamet_agent.stages.renorm.functions import STAGE_TOOLS
- store = {}
- STAGE_TOOLS["load_bare_matrix_element"](store, path="reference.nc")
- STAGE_TOOLS["fit_self_renormalization_factor"](store, kernel_id="ZMSbar_da", d=-0.08183)
- STAGE_TOOLS["apply_self_renormalization"](store, kernel_id="ZMSbar_da")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt
import numpy as np

from lamet_agent import kernels
from lamet_agent.core.data import EnsembleData, EnsembleInfo
from lamet_agent.core.plotting import COLOR_CYCLE, ERRORBAR_STYLE, FONT_SIZE, LEGEND_SETS, default_plot
from lamet_agent.core.resampling import sample_mean_and_sdev
from lamet_agent.core.tools import resolve_plot_save_path

GEV_FM = 0.1973269631
_ZMSBAR_KERNELS = {
    "ZMSbar_pdf": kernels.ZMSbar_pdf,
    "ZMSbar_da": kernels.ZMSbar_da,
    "pdf": kernels.ZMSbar_pdf,
    "da": kernels.ZMSbar_da,
}



def _resample_name(value: str | None) -> Literal["bootstrap", "jackknife", "raw"]:
    mode = (value or "bootstrap").lower()
    if mode in {"bs", "bootstrap"}:
        return "bootstrap"
    if mode in {"jk", "jackknife"}:
        return "jackknife"
    if mode == "raw":
        return "raw"
    raise ValueError("resample must be one of 'bootstrap', 'jackknife', 'bs', 'jk', or 'raw'")


def _resample_mode(data: EnsembleData) -> str:
    if data.resample == "bootstrap":
        return "bs"
    if data.resample == "jackknife":
        return "jk"
    return data.resample


def _bare_grid_paths_from_dir(
    txt_dir: str | Path,
    *,
    filename_glob: str,
    z_regex: str,
) -> tuple[list[tuple[float, Path]], dict[str, Any]]:
    directory = Path(txt_dir)
    paths: list[tuple[float, Path]] = []
    pattern = re.compile(z_regex)
    for path in sorted(directory.glob(filename_glob)):
        match = pattern.search(path.name)
        if match is None:
            continue
        paths.append((float(match.group(1)), path))
    if not paths:
        raise ValueError(f"no bare matrix txt files matched {filename_glob!r} in {txt_dir}")
    return paths, {"output_subdir": str(directory), "resample_mode": "bootstrap"}


def _load_complex_txt_grid(paths: list[tuple[float, Path]]) -> tuple[np.ndarray, np.ndarray]:
    z_values: list[float] = []
    samples_by_z: list[np.ndarray] = []
    n_sample: int | None = None
    for z_value, path in sorted(paths, key=lambda item: item[0]):
        raw = np.loadtxt(path, dtype=float)
        arr = np.atleast_2d(raw)
        if arr.shape[1] < 2:
            raise ValueError(f"bare matrix txt file must have at least two columns: {path}")
        complex_samples = arr[:, 0] + 1j * arr[:, 1]
        if n_sample is None:
            n_sample = int(complex_samples.shape[0])
        elif complex_samples.shape[0] != n_sample:
            raise ValueError(f"sample count mismatch in {path}: {complex_samples.shape[0]} != {n_sample}")
        z_values.append(float(z_value))
        samples_by_z.append(complex_samples)
    if not samples_by_z:
        raise ValueError("no bare matrix-element samples were loaded")
    return np.asarray(z_values, dtype=float), np.stack(samples_by_z, axis=1)


def _matrix_to_ensemble(
    *,
    z_values: np.ndarray,
    samples: np.ndarray,
    resample: Literal["bootstrap", "jackknife", "raw"],
    attrs: dict[str, Any] | None = None,
    name: str,
) -> EnsembleData:
    values = [np.asarray(samples[idx], dtype=complex) for idx in range(samples.shape[0])]
    ensemble_id = "" if attrs is None else str(attrs.get("ensemble", ""))
    return EnsembleData(
        ensemble=EnsembleInfo("", ensemble_id, 1.0, 1.0, 1, 1, 0.0),
        resample=resample,
        values=values,
        dims=("z",),
        coords={"z": np.asarray(z_values, dtype=float).tolist()},
        attrs={key: str(value) for key, value in (attrs or {}).items() if value is not None},
        name=name,
    )


def _require_matrix_data(store: dict[str, Any], key: str) -> EnsembleData:
    data = store.get(key)
    if not isinstance(data, EnsembleData):
        raise ValueError(f"store[{key!r}] does not contain EnsembleData")
    if data.dims != ["z"]:
        raise ValueError(f"store[{key!r}] must have physical dimension ['z']")
    values = np.asarray(data.values)
    if values.ndim != 2:
        raise ValueError(f"store[{key!r}] values must be shaped (resample,z)")
    return data


def _z_index(z_values: np.ndarray, target: float, *, label: str) -> int:
    matches = np.flatnonzero(np.isclose(z_values, float(target), rtol=0.0, atol=1e-10))
    if matches.size == 0:
        raise ValueError(f"{label} z={target} is not present in coordinate grid")
    return int(matches[0])


def _artifact_stem(raw: str | None, *, artifacts_dir: str | Path | None, default_stem: str) -> Path:
    out_dir = Path(artifacts_dir) if artifacts_dir is not None else Path.cwd() / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    return Path(resolve_plot_save_path(raw, artifacts_dir=out_dir, default_stem=default_stem))


def _resolve_zmsbar(kernel_id: str | None = None, zms_kind: Literal["pdf", "da"] | None = None):
    key = kernel_id or zms_kind or "da"
    if key not in _ZMSBAR_KERNELS:
        raise ValueError(f"unsupported ZMSbar kernel_id/zms_kind: {key!r}")
    return key if key.startswith("ZMSbar_") else f"ZMSbar_{key}", _ZMSBAR_KERNELS[key]


def normalize_bare_matrix_element_at_z0(data: EnsembleData) -> EnsembleData:
    """Divide each resampled matrix element by its lattice ``z=0`` value."""
    z_values = np.asarray(data.coords["z"], dtype=float)
    z0_idx = _z_index(z_values, 0.0, label="normalization")
    samples = np.asarray(data.values, dtype=complex)
    if data.dims == ["z"]:
        normalized = samples / samples[:, z0_idx : z0_idx + 1]
    elif data.dims == ["a", "z"]:
        normalized = samples / samples[:, :, z0_idx : z0_idx + 1]
    else:
        raise ValueError(f"unsupported dims for z=0 normalization: {data.dims}")
    attrs = {**data.attrs, "normalized_at_z0": "true"}
    if data.dims == ["z"]:
        resample = data.resample if data.resample in {"bootstrap", "jackknife", "raw"} else "bootstrap"
        return _matrix_to_ensemble(
            z_values=z_values,
            samples=normalized,
            resample=resample,
            attrs=attrs,
            name=data.name or "bare_matrix_element",
        )
    values = [np.asarray(normalized[idx], dtype=complex) for idx in range(normalized.shape[0])]
    return EnsembleData(
        data.ensemble,
        data.resample,
        values,
        dims=data.dims,
        coords=data.coords,
        attrs={key: str(value) for key, value in attrs.items() if value is not None},
        name=data.name,
    )


def load_bare_matrix_element_grid(
    store: dict[str, Any],
    *,
    netcdf_path: str | None = None,
    txt_dir: str | None = None,
    filename_glob: str = "*.txt",
    z_regex: str = r"_z([+-]?\d+(?:\.\d+)?)\.txt$",
    resample: Literal["bootstrap", "jackknife", "raw", "bs", "jk"] | None = None,
    out: str = "bare_matrix_element",
) -> dict[str, Any]:
    """Load correlator-stage bare matrix elements into complex EnsembleData."""
    source: str
    if netcdf_path is not None:
        data = EnsembleData.from_netcdf(netcdf_path)
        source = netcdf_path
    elif txt_dir is None:
        existing = store.get("bare_matrix_element_data")
        if isinstance(existing, EnsembleData):
            data = existing
            source = "bare_matrix_element_data"
        elif isinstance(store.get("bare_matrix_element_netcdf"), str):
            source = str(store["bare_matrix_element_netcdf"])
            data = EnsembleData.from_netcdf(source)
        else:
            raise ValueError("provide netcdf_path or txt_dir, or run fit_bare_matrix_grid first")
    else:
        assert txt_dir is not None
        paths, metadata = _bare_grid_paths_from_dir(txt_dir, filename_glob=filename_glob, z_regex=z_regex)
        z_values, samples = _load_complex_txt_grid(paths)
        resample_name = _resample_name(resample or str(metadata.get("resample_mode", "bootstrap")))
        data = _matrix_to_ensemble(
            z_values=z_values,
            samples=samples,
            resample=resample_name,
            attrs={"source": txt_dir, "resample_mode": metadata.get("resample_mode", resample_name)},
            name="bare_matrix_element",
        )
        source = txt_dir

    store[out] = data
    loaded = _require_matrix_data(store, out)
    z_values = np.asarray(loaded.coords["z"], dtype=float)
    samples = np.asarray(loaded.values, dtype=complex)
    store[f"{out}_arrays"] = {
        "coord": z_values,
        "re_samples": np.real(samples),
        "im_samples": np.imag(samples),
    }
    return {
        "out": out,
        "data": out,
        "n_z": int(len(z_values)),
        "n_sample": int(samples.shape[0]),
        "z_values": z_values.tolist(),
        "resample": loaded.resample,
        "source": source,
    }


def apply_ratio_scheme_renormalization(
    store: dict[str, Any],
    *,
    target: str = "target_bare_matrix_element",
    denominator: str = "denominator_bare_matrix_element",
    scheme: str = "hybrid_ratio",
    scheme_parameters: dict[str, float] | None = None,
    out: str = "matrix_element_data",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    job_id: str | None = None,
    sample_error_mode: str = "covariance",
) -> dict[str, Any]:
    """Apply hybrid-ratio renormalization and preserve all samples."""
    if scheme != "hybrid_ratio":
        raise ValueError(f"unsupported renormalization scheme: {scheme!r}")
    target_data = _require_matrix_data(store, target)
    denom_data = _require_matrix_data(store, denominator)
    if target_data.resample != denom_data.resample:
        raise ValueError(f"target and denominator resampling must match: {target_data.resample} != {denom_data.resample}")

    z_target = np.asarray(target_data.coords["z"], dtype=float)
    z_denom = np.asarray(denom_data.coords["z"], dtype=float)
    if z_target.shape != z_denom.shape or not np.allclose(z_target, z_denom, rtol=0.0, atol=1e-10):
        raise ValueError("target and denominator z grids must match exactly")
    target_values = np.asarray(target_data.values, dtype=complex)
    denom_values = np.asarray(denom_data.values, dtype=complex)
    if target_values.shape != denom_values.shape:
        raise ValueError("target and denominator sample arrays must have matching shape")

    params = scheme_parameters or {}
    zs_fm = float(params["zs_fm"])
    m0_gev = float(params.get("m0_gev", 0.0))
    delta_m_gev = float(params.get("delta_m_gev", 0.0))
    lattice_spacing_fm = float(target_data.attrs["lattice_spacing_fm"])
    zs_lattice = zs_fm / lattice_spacing_fm
    zs_idx = int(np.argmin(np.abs(np.abs(z_denom) - zs_lattice)))
    z_fm = np.abs(z_target) * lattice_spacing_fm
    mass_scale = (delta_m_gev + m0_gev) / GEV_FM
    exponent = np.exp(mass_scale * (z_fm - zs_fm))
    short = target_values / denom_values
    long = exponent[None, :] * target_values / denom_values[:, zs_idx : zs_idx + 1]
    renorm_values = np.where((np.abs(z_target) * lattice_spacing_fm)[None, :] <= zs_fm, short, long)

    attrs = {
        **target_data.attrs,
        "scheme": scheme,
        "zs_fm": str(zs_fm),
        "zs_lattice": str(zs_lattice),
        "zs_grid": str(float(z_denom[zs_idx])),
        "delta_m_gev": str(delta_m_gev),
        "m0_gev": str(m0_gev),
        "target": target,
        "denominator": denominator,
        "job_id": job_id,
        "sample_error_mode": sample_error_mode,
        "average_method": sample_error_mode,
    }
    result = _matrix_to_ensemble(
        z_values=z_target,
        samples=renorm_values,
        resample=target_data.resample,
        attrs=attrs,
        name="renormalized_matrix_element",
    )
    store[out] = result
    store["matrix_element_data"] = result
    store["output"] = result
    store["matrix_element"] = {
        "coord": z_target,
        "re_samples": np.real(renorm_values),
        "im_samples": np.imag(renorm_values),
        "scheme": scheme,
    }

    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="renormalized_matrix_element")
    artifact = stem.with_suffix(".nc")
    result.to_netcdf(artifact)
    store["matrix_element_netcdf"] = str(artifact)
    return {
        "out": out,
        "data": "matrix_element_data",
        "artifact": str(artifact),
        "n_z": int(len(z_target)),
        "n_sample": int(renorm_values.shape[0]),
        "zs_fm": zs_fm,
        "zs_lattice": zs_lattice,
        "zs_grid": float(z_denom[zs_idx]),
        "delta_m_gev": float(delta_m_gev),
        "m0_gev": float(m0_gev),
    }


def plot_renormalized_matrix_element(
    store: dict[str, Any],
    *,
    data: str = "matrix_element_data",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    title: str | None = None,
    sample_error_mode: str = "covariance",
) -> dict[str, Any]:
    """Plot sample-averaged renormalized matrix elements to PDF."""
    matrix = _require_matrix_data(store, data)
    z_values = np.asarray(matrix.coords["z"], dtype=float)
    values = np.asarray(matrix.values, dtype=complex)
    if not np.all(np.isfinite(values)):
        raise ValueError("renormalized matrix-element samples contain non-finite values")
    mode = _resample_mode(matrix)
    re_mean, re_err = sample_mean_and_sdev(np.real(values), mode=mode, sample_error_mode=sample_error_mode, axis=0)
    im_mean, im_err = sample_mean_and_sdev(np.imag(values), mode=mode, sample_error_mode=sample_error_mode, axis=0)

    fig, ax = default_plot()
    ax.errorbar(z_values, re_mean, re_err, label="Re", color=COLOR_CYCLE[0], **ERRORBAR_STYLE)
    ax.errorbar(z_values, im_mean, im_err, label="Im", color=COLOR_CYCLE[1], marker="s", **ERRORBAR_STYLE)
    ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
    ax.set_ylabel(r"Renormalized matrix element", **FONT_SIZE)
    if title is None:
        ensemble = matrix.ensemble.id if matrix.ensemble is not None and matrix.ensemble.id else ""
        momentum = matrix.attrs.get("momentum_gev")
        if momentum is not None:
            title = rf"{ensemble} $p={float(momentum):.2f}\,\mathrm{{GeV}}$ renormalized matrix elements"
        else:
            title = "Renormalized matrix elements"
    ax.set_title(title, **FONT_SIZE)
    ax.legend(**LEGEND_SETS)
    fig.tight_layout()
    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="renormalized_matrix_element")
    plot_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(plot_path, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot": str(plot_path),
        "plot_image": str(svg_path),
        "data": data,
        "n_z": int(len(z_values)),
        "n_sample": int(values.shape[0]),
    }


def load_bare_matrix_element(
    store: dict[str, Any],
    *,
    path: str | None = None,
    netcdf_path: str | None = None,
    resample: Literal["bootstrap", "jackknife"] = "bootstrap",
    a: float | list[float] | None = None,
    z_key: str = "z",
    samples_key: str = "samples",
    out: str = "reference",
) -> dict[str, Any]:
    """Load bare matrix-element samples from NetCDF or NPZ into EnsembleData."""
    source = path or netcdf_path
    if source is None:
        raise ValueError("provide path or netcdf_path")
    source_path = Path(source)
    if source_path.suffix.lower() in {".nc", ".netcdf"}:
        reference = EnsembleData.from_netcdf(source_path)
        store[out] = reference
        return {
            "out": out,
            "resample": reference.resample,
            "dims": list(reference.dims),
            "n_sample": reference.n_sample,
            "z_values": reference.coords["z"],
            "a_values": reference.coords.get("a", [reference.ensemble.a_s]),
            "source": str(source_path),
        }

    data = np.load(source_path)
    z = np.asarray(data[z_key], dtype=float)
    samples = np.asarray(data[samples_key], dtype=float)
    if a is None:
        a_list = [float(data["a"][0])] if "a" in data else [1.0]
    else:
        a_list = [float(a)] if isinstance(a, (int, float)) else [float(x) for x in a]

    a_s = a_list[0]
    ensemble = EnsembleInfo("", "", a_s, a_s, 96, 96, 0.0)
    values = [samples[i] for i in range(samples.shape[0])]
    if samples.ndim == 2:
        reference = EnsembleData(ensemble, resample, values, dims=("z",), coords={"z": z.tolist()})
    else:
        reference = EnsembleData(
            ensemble,
            resample,
            values,
            dims=("a", "z"),
            coords={"a": a_list, "z": z.tolist()},
        )

    store[out] = reference
    return {
        "out": out,
        "resample": resample,
        "dims": list(reference.dims),
        "n_sample": reference.n_sample,
        "z_values": reference.coords["z"],
        "a_values": reference.coords.get("a", [reference.ensemble.a_s]),
        "source": str(source_path),
    }


def fit_self_renormalization_factor(
    store: dict[str, Any],
    *,
    reference: str = "reference",
    out: str = "zR",
    kernel_id: str | None = None,
    zms_kind: Literal["pdf", "da"] | None = None,
    m0_gev: float | None = None,
    k: float = 3.320,
    lqcd: float = 0.1,
    mu: float = 2.0,
    d: float | None = None,
    cf: float = 4.0 / 3.0,
    b0: float = 11.0 - 2.0 / 3.0 * 3.0,
    svdcut: float = 1e-12,
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fit self-renormalization factor zR from zero-momentum reference EnsembleData.

    ``d`` is required (fixed; enters both the gz continuum/discretization fit
    and zR construction). ``m0_gev`` is optional: if provided, freeze that
    slope; if omitted, fit ``m0`` from short-distance ``g(z)`` vs
    ``ZMSbar_pdf`` (first three z points).

    Fits once on sample-averaged ``ln|M|``. Stored ``zR`` is a one-sample
    EnsembleData holding that mean zR grid; apply jobs divide each target
    sample by this mean.
    """
    if d is None:
        raise ValueError("fit_self_renormalization_factor requires d (fixed; never fitted)")
    d_val = float(d)
    ref: EnsembleData = store[reference]
    if ref.resample not in {"bootstrap", "jackknife"}:
        raise ValueError(
            f"fit_self_renormalization_factor requires bootstrap/jackknife reference samples; got resample={ref.resample!r}"
        )
    resolved_kernel_id, _zms_apply = _resolve_zmsbar(kernel_id, zms_kind)
    z_coords = list(ref.coords["z"])
    if "a" in ref.dims:
        a_coords = list(ref.coords["a"])
    else:
        a_coords = [ref.ensemble.a_s]

    z_arr = np.asarray(z_coords, dtype=float)
    z0_matches = np.flatnonzero(np.isclose(z_arr, 0.0, rtol=0.0, atol=1e-10))
    z0_idx = int(z0_matches[0]) if z0_matches.size else None
    skip_z0 = ref.attrs.get("normalized_at_z0") == "true"

    # Sample-averaged ln|M| gvars (pipeline stays sample-based on disk).
    samples = ref.array.values
    ln_values = [np.log(np.abs(s)) for s in samples]
    ln_m = EnsembleData(
        ref.ensemble,
        ref.resample,
        ln_values,
        dims=ref.dims,
        coords=ref.coords,
        attrs=ref.attrs,
        name=ref.name,
    )
    ln_gv = ln_m.gvar
    if "a" not in ref.dims:
        ln_gv = ln_gv.reshape(1, -1)
    n_a = len(a_coords)
    n_z = len(z_coords)

    z_x: dict[str, list[float]] = {"z": [], "x": []}
    lnm: list[Any] = []
    for ia, a_val in enumerate(a_coords):
        x = GEV_FM / float(a_val)
        for iz, z_val in enumerate(z_coords):
            if skip_z0 and z0_idx is not None and iz == z0_idx:
                continue
            z_x["z"].append(float(z_val))
            z_x["x"].append(x)
            lnm.append(ln_gv[ia, iz])

    priors = gv.BufferDict()
    for z_val in z_coords:
        priors[f"g{z_val}"] = gv.gvar(0, 20)
        priors[f"f1{z_val}"] = gv.gvar(0, 5)

    def fcn(z_x_in, p):
        out_vals = []
        for zm, xm in zip(z_x_in["z"], z_x_in["x"]):
            out_vals.append(
                k * zm * xm / gv.log(lqcd / xm)
                + p[f"g{zm}"]
                + p[f"f1{zm}"] / xm
                + 3 * cf / b0 * gv.log(gv.log(xm / lqcd) / gv.log(mu / lqcd))
                + gv.log(1 + d_val / gv.log(lqcd / xm))
            )
        return out_vals

    gz_fit = lsf.nonlinear_fit(
        data=(z_x, lnm),
        prior=priors,
        fcn=fcn,
        maxit=10000,
        svdcut=svdcut,
        fitter="scipy_least_squares",
    )

    p = gz_fit.p
    g_post = [p[f"g{z}"] for z in z_coords]
    if m0_gev is not None:
        m0 = gv.gvar(float(m0_gev), 0.0)
        m0_source = "fixed"
    else:
        if n_z < 3:
            raise ValueError(
                "fit_self_renormalization_factor needs at least 3 z points to fit m0_gev when it is omitted"
            )
        z_m0 = [float(z) for z in z_coords[:3]]
        g_m0 = g_post[:3]

        def m0_fcn(x, p_m0):
            z_arr_m0 = np.asarray(x, dtype=float)
            zms = np.asarray(kernels.ZMSbar_pdf(z_arr_m0, mu=mu), dtype=float)
            return np.log(zms) + p_m0["m0"] * z_arr_m0 + p_m0["b"]

        m0_priors = gv.BufferDict()
        m0_priors["m0"] = gv.gvar(0, 20)
        m0_priors["b"] = gv.gvar(0, 100)
        m0_fit = lsf.nonlinear_fit(
            data=(z_m0, g_m0),
            prior=m0_priors,
            fcn=m0_fcn,
            maxit=10000,
            svdcut=svdcut,
            fitter="scipy_least_squares",
        )
        m0 = m0_fit.p["m0"]
        m0_source = "fit"

    g_means = np.asarray([float(gv.mean(g)) for g in g_post], dtype=float)
    g_sdevs = np.asarray([float(gv.sdev(g)) for g in g_post], dtype=float)
    f1_means = np.asarray([float(gv.mean(p[f"f1{z}"])) for z in z_coords], dtype=float)
    f1_sdevs = np.asarray([float(gv.sdev(p[f"f1{z}"])) for z in z_coords], dtype=float)
    fit_lnm_mean = np.empty((n_a, n_z), dtype=float)
    fit_lnm_sdev = np.empty((n_a, n_z), dtype=float)
    zr_mean = np.empty((n_a, n_z), dtype=float)
    for ia, a_val in enumerate(a_coords):
        xm = GEV_FM / float(a_val)
        for iz, z_val in enumerate(z_coords):
            fit_ln = (
                k * z_val * xm / gv.log(lqcd / xm)
                + p[f"g{z_val}"]
                + p[f"f1{z_val}"] / xm
                + 3 * cf / b0 * gv.log(gv.log(xm / lqcd) / gv.log(mu / lqcd))
                + gv.log(1 + d_val / gv.log(lqcd / xm))
            )
            fit_lnm_mean[ia, iz] = float(gv.mean(fit_ln))
            fit_lnm_sdev[ia, iz] = float(gv.sdev(fit_ln))
            temp = (
                k * z_val * xm / gv.log(lqcd / xm)
                + p[f"f1{z_val}"] / xm
                + 3 * cf / b0 * gv.log(gv.log(xm / lqcd) / gv.log(mu / lqcd))
                + gv.log(1 + d_val / gv.log(lqcd / xm))
                + m0 * z_val
            )
            zr_mean[ia, iz] = float(gv.mean(np.exp(temp)))

    lnm_mean = np.asarray(
        [[float(gv.mean(ln_gv[ia, iz])) for iz in range(n_z)] for ia in range(n_a)],
        dtype=float,
    )
    lnm_sdev = np.asarray(
        [[float(gv.sdev(ln_gv[ia, iz])) for iz in range(n_z)] for ia in range(n_a)],
        dtype=float,
    )

    # One-sample EnsembleData holding the mean zR (sample-based NetCDF contract).
    resample_name = "bootstrap" if ref.resample == "bootstrap" else "jackknife"
    m0_mean = float(gv.mean(m0))
    m0_sdev = float(gv.sdev(m0))
    zR = EnsembleData(
        ref.ensemble,
        resample_name,
        [np.asarray(zr_mean, dtype=complex)],
        dims=("a", "z"),
        coords={"a": a_coords, "z": z_coords},
        attrs={
            "kernel_id": resolved_kernel_id,
            "mu": str(mu),
            "m0_gev": str(m0_mean),
            "d": str(d_val),
            "m0_source": m0_source,
            "resample_mode": resample_name,
            "sample_construction": "mean_from_averaged_fit",
        },
        name="zR",
    )
    store[out] = zR
    store["output"] = zR

    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="zR")
    artifact = stem.with_suffix(".nc")
    zR.to_netcdf(artifact)
    store["zR_netcdf"] = str(artifact)

    mR = np.exp(g_means - m0_mean * np.asarray(z_coords, dtype=float))
    store["self_renorm_fit"] = {
        "z": [float(z) for z in z_coords],
        "a": [float(a) for a in a_coords],
        "lnm_mean": lnm_mean,
        "lnm_sdev": lnm_sdev,
        "fit_lnm_mean": fit_lnm_mean,
        "fit_lnm_sdev": fit_lnm_sdev,
        "g_mean": g_means,
        "g_sdev": g_sdevs,
        "f1_mean": f1_means,
        "f1_sdev": f1_sdevs,
        "zR_mean": zr_mean,
        "mR": mR,
        "m0": m0_mean,
        "m0_sdev": m0_sdev,
        "m0_source": m0_source,
        "kernel_id": resolved_kernel_id,
        "mu": float(mu),
        "d": d_val,
        "svdcut": float(svdcut),
        "skip_z0": bool(skip_z0),
    }
    return {
        "out": out,
        "artifact": str(artifact),
        "kernel_id": resolved_kernel_id,
        "m0": m0_mean,
        "m0_sdev": m0_sdev,
        "m0_source": m0_source,
        "mu": float(mu),
        "d": d_val,
        "svdcut": float(svdcut),
        "z_values": z_coords,
        "a_values": a_coords,
        "n_z": n_z,
        "n_a": n_a,
        "n_sample": 1,
    }


def _remap_zr_values(
    zr_vals: np.ndarray,
    *,
    z_vals: np.ndarray,
    lattice_spacing_fm: float,
    d_from: float,
    d_to: float,
    m0_from: float,
    m0_to: float,
    lqcd: float = 0.1,
) -> np.ndarray:
    """Remap mean zR from (d_from, m0_from) to (d_to, m0_to).

    Continuum/discretization pieces cancel; only the ``d`` log term and
    ``m0*z`` slope differ between operators (legacy PDF→DA replacement).
    """
    x = GEV_FM / float(lattice_spacing_fm)
    log_term = float(np.log(lqcd / x))
    if abs(log_term) < 1e-30:
        raise ValueError(f"invalid log(lqcd/x) for lattice_spacing_fm={lattice_spacing_fm}")
    scale = (1.0 + d_to / log_term) / (1.0 + d_from / log_term)
    return np.asarray(zr_vals, dtype=float) * scale * np.exp((m0_to - m0_from) * np.asarray(z_vals, dtype=float))


def apply_self_renormalization(
    store: dict[str, Any],
    *,
    target: str = "target",
    zR: str = "zR",
    kernel_id: str | None = None,
    zms_kind: Literal["pdf", "da"] | None = None,
    mu: float = 2.0,
    d: float | None = None,
    m0_gev: float | None = None,
    lqcd: float = 0.1,
    out: str = "matrix_element_data",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    job_id: str | None = None,
    sample_error_mode: str = "covariance",
) -> dict[str, Any]:
    """Apply self-renormalization: H / (zR * ZMSbar), preserving all samples.

    Optional ``d`` / ``m0_gev`` remap upstream zR from the fit-job operator
    parameters onto this apply job (e.g. PDF-fit zR → DA ``d``/``m0``).
    """
    target_data = _require_matrix_data(store, target)
    zR_data = store[zR]
    if not isinstance(zR_data, EnsembleData):
        raise ValueError(f"store[{zR!r}] does not contain EnsembleData")
    resolved_kernel_id, zms_fn = _resolve_zmsbar(kernel_id or zR_data.attrs.get("kernel_id"), zms_kind)

    z_target = np.asarray(target_data.coords["z"], dtype=float)
    z_zr = np.asarray(zR_data.coords["z"], dtype=float)
    a_coords = list(zR_data.coords.get("a", [zR_data.ensemble.a_s]))
    lattice_spacing_fm = float(target_data.attrs.get("lattice_spacing_fm", a_coords[0]))
    ia = int(np.argmin([abs(float(a) - lattice_spacing_fm) for a in a_coords]))
    a_used = float(a_coords[ia])

    # Mean zR on the fit grid (zR is bootstrap EnsembleData on (a,z) or (z)).
    zr_arr = np.asarray(zR_data.values)
    if zr_arr.ndim == 3:
        zr_grid = np.mean(np.real(zr_arr), axis=0)  # (a, z)
    elif zr_arr.ndim == 2:
        zr_grid = np.mean(np.real(zr_arr), axis=0)  # (z,)
    else:
        raise ValueError(f"store[{zR!r}] values must be shaped (resample,a,z) or (resample,z)")

    d_from_raw = zR_data.attrs.get("d", "")
    m0_from_raw = zR_data.attrs.get("m0_gev", "")
    d_from = float(d_from_raw) if d_from_raw not in {None, ""} else None
    m0_from = float(m0_from_raw) if m0_from_raw not in {None, ""} else None
    remap = d is not None or m0_gev is not None
    if remap:
        if d_from is None or m0_from is None:
            raise ValueError(
                "apply_self_renormalization d/m0_gev override requires upstream zR attrs "
                "'d' and 'm0_gev' from the fit job"
            )
        d_to = float(d) if d is not None else d_from
        m0_to = float(m0_gev) if m0_gev is not None else m0_from
        if zr_grid.ndim == 2:
            remapped = np.empty_like(zr_grid, dtype=float)
            for ia_all, a_val in enumerate(a_coords):
                remapped[ia_all] = _remap_zr_values(
                    zr_grid[ia_all],
                    z_vals=z_zr,
                    lattice_spacing_fm=float(a_val),
                    d_from=d_from,
                    d_to=d_to,
                    m0_from=m0_from,
                    m0_to=m0_to,
                    lqcd=lqcd,
                )
            zr_grid = remapped
        else:
            zr_grid = _remap_zr_values(
                zr_grid,
                z_vals=z_zr,
                lattice_spacing_fm=a_used,
                d_from=d_from,
                d_to=d_to,
                m0_from=m0_from,
                m0_to=m0_to,
                lqcd=lqcd,
            )
        # Keep diagnostics on the remapped factor for this apply job.
        remapped_zR = EnsembleData(
            zR_data.ensemble,
            zR_data.resample if zR_data.resample in {"bootstrap", "jackknife"} else "bootstrap",
            [np.asarray(zr_grid, dtype=complex)],
            dims=tuple(zR_data.dims),
            coords={dim: list(zR_data.coords[dim]) for dim in zR_data.dims},
            attrs={
                **zR_data.attrs,
                "d": str(d_to),
                "m0_gev": str(m0_to),
                "d_from": str(d_from),
                "m0_from": str(m0_from),
                "sample_construction": "remapped_from_upstream_zR",
            },
            name="zR",
        )
        store[zR] = remapped_zR
        zR_data = remapped_zR
    else:
        d_to = d_from
        m0_to = m0_from

    if zr_grid.ndim == 2:
        zr_vals = zr_grid[ia]
    else:
        zr_vals = zr_grid
    if z_zr.shape == z_target.shape and np.allclose(z_zr, z_target, rtol=0.0, atol=1e-10):
        zr_on_target = np.asarray(zr_vals, dtype=float)
    else:
        zr_on_target = np.interp(z_target, z_zr, zr_vals)

    zms = np.asarray(zms_fn(z_target, mu=mu), dtype=float)
    target_values = np.asarray(target_data.values, dtype=complex)
    renorm_values = target_values / (zr_on_target[None, :] * zms[None, :])

    attrs = {
        **target_data.attrs,
        "scheme": "self_renormalization",
        "kernel_id": resolved_kernel_id,
        "mu": str(mu),
        "m0_gev": "" if m0_to is None else str(m0_to),
        "d": "" if d_to is None else str(d_to),
        "lattice_spacing_fm_used": str(a_used),
        "target": target,
        "job_id": job_id,
        "sample_error_mode": sample_error_mode,
        "average_method": sample_error_mode,
    }
    if remap:
        attrs["d_from"] = str(d_from)
        attrs["m0_from"] = str(m0_from)
    result = _matrix_to_ensemble(
        z_values=z_target,
        samples=renorm_values,
        resample=target_data.resample,
        attrs=attrs,
        name="renormalized_matrix_element",
    )
    store[out] = result
    store["matrix_element_data"] = result
    store["output"] = result
    store["matrix_element"] = {
        "coord": z_target,
        "re_samples": np.real(renorm_values),
        "im_samples": np.imag(renorm_values),
        "scheme": "self_renormalization",
    }

    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="renormalized_matrix_element")
    artifact = stem.with_suffix(".nc")
    result.to_netcdf(artifact)
    store["matrix_element_netcdf"] = str(artifact)
    return {
        "out": out,
        "data": "matrix_element_data",
        "artifact": str(artifact),
        "scheme": "self_renormalization",
        "kernel_id": resolved_kernel_id,
        "mu": float(mu),
        "m0_gev": m0_to,
        "d": d_to,
        "remapped": bool(remap),
        "n_z": int(len(z_target)),
        "n_sample": int(renorm_values.shape[0]),
        "lattice_spacing_fm": lattice_spacing_fm,
    }


def _save_plot_pair(fig, stem: Path) -> tuple[str, str]:
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    fig.savefig(pdf, bbox_inches="tight", transparent=True)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(pdf), str(svg)


def plot_self_renormalization_diagnostics(
    store: dict[str, Any],
    *,
    mode: Literal["fit", "apply"] = "fit",
    target: str = "target",
    zR: str = "zR",
    fit: str = "self_renorm_fit",
    sibling_artifacts: list[str] | None = None,
    include_discrete_effect: bool = False,
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    sample_error_mode: str = "covariance",
    kernel_id: str | None = None,
    mu: float | None = None,
) -> dict[str, Any]:
    """Plot self-renorm diagnostics.

    ``mode='fit'`` writes fit-only panels once (no ``fit_vs_data`` / no m0 panel).
    ``mode='apply'`` writes per-target ``zmsbar_compare``; when
    ``include_discrete_effect`` is true and sibling NetCDFs exist, also writes
    one multi-a discrete-effect overlay under stage-level names
    ``discrete_effect_re`` / ``discrete_effect_im`` (no job-id prefix).
    """
    zR_data = store.get(zR)
    if not isinstance(zR_data, EnsembleData):
        raise ValueError(f"store[{zR!r}] does not contain EnsembleData")
    fit_data = store.get(fit)
    if mode == "fit" and not isinstance(fit_data, dict):
        raise ValueError(f"store[{fit!r}] must contain the self-renorm fit diagnostics dict")
    if not isinstance(fit_data, dict):
        fit_data = {}

    resolved_kernel_id, zms_fn = _resolve_zmsbar(
        kernel_id or fit_data.get("kernel_id") or zR_data.attrs.get("kernel_id"),
        None,
    )
    # Fit-check panels compare mR against ZMSbar_pdf.
    zms_fit_fn = kernels.ZMSbar_pdf
    mu_val = float(mu if mu is not None else fit_data.get("mu", zR_data.attrs.get("mu", 2.0)))
    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="self_renorm")
    plots: dict[str, str] = {}

    if mode == "fit":
        z_fit = np.asarray(fit_data["z"], dtype=float)
        a_fit = np.asarray(fit_data["a"], dtype=float)
        x_fit = GEV_FM / a_fit
        lnm_mean = np.asarray(fit_data["lnm_mean"], dtype=float)
        lnm_sdev = np.asarray(fit_data["lnm_sdev"], dtype=float)
        f1_mean = np.asarray(fit_data["f1_mean"], dtype=float)
        f1_sdev = np.asarray(fit_data["f1_sdev"], dtype=float)
        zr_mean = np.asarray(fit_data["zR_mean"], dtype=float)
        mR = np.asarray(fit_data["mR"], dtype=float)

        fig, ax = default_plot()
        highlight_indices = list(range(0, len(z_fit), max(1, len(z_fit) // 6)))
        if len(z_fit) - 1 not in highlight_indices:
            highlight_indices.append(len(z_fit) - 1)
        for iz, z_val in enumerate(z_fit):
            label = rf"$z={z_val:.2f}\,\mathrm{{fm}}$" if iz in highlight_indices else None
            ax.errorbar(
                x_fit,
                lnm_mean[:, iz],
                lnm_sdev[:, iz],
                label=label,
                color=plt.cm.viridis(iz / max(1, len(z_fit) - 1)),
                **ERRORBAR_STYLE,
            )
        ax.set_xlabel(r"$1/a$ [GeV]", **FONT_SIZE)
        ax.set_ylabel(r"$\ln|M|$", **FONT_SIZE)
        ax.set_title("Reference matrix element after interpolation", **FONT_SIZE)
        ax.legend(fontsize=12, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        fig.tight_layout()
        pdf, svg = _save_plot_pair(fig, stem.with_name(stem.name + "_fit_lnM_vs_inv_a"))
        plots["fit_lnM_vs_inv_a"] = pdf
        plots["fit_lnM_vs_inv_a_image"] = svg

        zms = np.asarray(zms_fit_fn(z_fit, mu=mu_val), dtype=float)
        fig, ax = default_plot()
        ax.plot(z_fit, zms, color="k", label=r"$Z_{\overline{\mathrm{MS}}}$")
        ax.errorbar(z_fit, mR, np.zeros_like(mR), color=COLOR_CYCLE[0], label=r"$m_R=\exp(g(z)-m_0 z)$", **ERRORBAR_STYLE)
        ax.errorbar(z_fit, mR / zms, np.zeros_like(mR), color=COLOR_CYCLE[1], label="ratio", marker="s", **ERRORBAR_STYLE)
        ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
        ax.set_ylabel("factor", **FONT_SIZE)
        ax.set_title(r"$m_R$ vs $Z_{\overline{\mathrm{MS}}}$", **FONT_SIZE)
        ax.legend(**LEGEND_SETS)
        fig.tight_layout()
        pdf, svg = _save_plot_pair(fig, stem.with_name(stem.name + "_fit_mR_zmsbar"))
        plots["fit_mR_zmsbar"] = pdf
        plots["fit_mR_zmsbar_image"] = svg

        fig, ax = default_plot()
        for ia, a_val in enumerate(a_fit):
            ratio = np.exp(lnm_mean[ia]) / zr_mean[ia]
            ax.errorbar(
                z_fit,
                ratio,
                np.zeros_like(ratio),
                label=rf"$a={a_val:.4f}\,\mathrm{{fm}}$",
                color=COLOR_CYCLE[ia % len(COLOR_CYCLE)],
                **ERRORBAR_STYLE,
            )
        ax.errorbar(z_fit, mR, np.zeros_like(mR), color=COLOR_CYCLE[len(a_fit) % len(COLOR_CYCLE)], label=r"$m_R=\exp(g(z)-m_0 z)$", marker="x", **ERRORBAR_STYLE)
        ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
        ax.set_ylabel(r"$M_{\mathrm{bare}}/z_R$", **FONT_SIZE)
        ax.set_title("PDF self-renormalization check", **FONT_SIZE)
        ax.legend(**LEGEND_SETS)
        fig.tight_layout()
        pdf, svg = _save_plot_pair(fig, stem.with_name(stem.name + "_fit_m_over_zR"))
        plots["fit_m_over_zR"] = pdf
        plots["fit_m_over_zR_image"] = svg

        fig, ax = default_plot()
        ax.errorbar(z_fit, f1_mean, f1_sdev, color=COLOR_CYCLE[0], **ERRORBAR_STYLE)
        ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
        ax.set_ylabel(r"$f_1(z)$", **FONT_SIZE)
        ax.set_title("Discretization coefficient $f_1(z)$", **FONT_SIZE)
        fig.tight_layout()
        pdf, svg = _save_plot_pair(fig, stem.with_name(stem.name + "_fit_f1"))
        plots["fit_f1"] = pdf
        plots["fit_f1_image"] = svg

        store["self_renorm_plots"] = plots
        return {
            "plots": plots,
            "mode": mode,
            "kernel_id": resolved_kernel_id,
            "mu": mu_val,
            "n_sibling": 0,
        }

    # apply mode
    target_data = _require_matrix_data(store, target)
    z_target = np.asarray(target_data.coords["z"], dtype=float)
    a_coords = list(zR_data.coords.get("a", [zR_data.ensemble.a_s]))
    lattice_spacing_fm = float(target_data.attrs.get("lattice_spacing_fm", a_coords[0]))
    ia = int(np.argmin([abs(float(a) - lattice_spacing_fm) for a in a_coords]))
    zr_arr = np.asarray(zR_data.values)
    if zr_arr.ndim == 3:
        zr_vals = np.mean(np.real(zr_arr[:, ia, :]), axis=0)
    elif zr_arr.ndim == 2:
        zr_vals = np.mean(np.real(zr_arr), axis=0)
    else:
        raise ValueError(f"store[{zR!r}] values must be shaped (resample,a,z) or (resample,z)")
    z_zr = np.asarray(zR_data.coords["z"], dtype=float)
    zr_on_target = zr_vals if (z_zr.shape == z_target.shape and np.allclose(z_zr, z_target)) else np.interp(z_target, z_zr, zr_vals)
    zms_target = np.asarray(zms_fn(z_target, mu=mu_val), dtype=float)
    target_values = np.asarray(target_data.values, dtype=complex)
    mode_rs = _resample_mode(target_data)
    h_over_zr = target_values / zr_on_target[None, :]
    re_hzr, re_hzr_err = sample_mean_and_sdev(np.real(h_over_zr), mode=mode_rs, sample_error_mode=sample_error_mode, axis=0)

    fig, ax = default_plot()
    ax.errorbar(z_target, re_hzr, re_hzr_err, color=COLOR_CYCLE[0], label=rf"$H/z_R$ ($a={lattice_spacing_fm:.4f}\,\mathrm{{fm}}$)", **ERRORBAR_STYLE)
    ax.plot(z_target, zms_target, color=COLOR_CYCLE[1], label=r"$Z_{\overline{\mathrm{MS}}}$")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
    ax.set_ylabel(r"Re$[H/z_R]$", **FONT_SIZE)
    ax.set_title(r"Compare $H/z_R$ with $Z_{\overline{\mathrm{MS}}}$", **FONT_SIZE)
    ax.legend(**LEGEND_SETS)
    fig.tight_layout()
    pdf, svg = _save_plot_pair(fig, stem.with_name(stem.name + "_zmsbar_compare"))
    plots["zmsbar_compare"] = pdf
    plots["zmsbar_compare_image"] = svg

    if include_discrete_effect:
        series: list[tuple[float, np.ndarray]] = []
        for path in sibling_artifacts or []:
            sibling_path = Path(path)
            if not sibling_path.is_file():
                continue
            sibling = EnsembleData.from_netcdf(sibling_path)
            series.append(
                (
                    float(sibling.attrs.get("lattice_spacing_fm", sibling.ensemble.a_s)),
                    np.asarray(sibling.values, dtype=complex),
                )
            )

        if len(series) >= 2:
            fig_re, ax_re = default_plot()
            fig_im, ax_im = default_plot()
            for idx, (a_val, values) in enumerate(sorted(series, key=lambda item: item[0])):
                re_m, re_e = sample_mean_and_sdev(np.real(values), mode="bs", sample_error_mode=sample_error_mode, axis=0)
                im_m, im_e = sample_mean_and_sdev(np.imag(values), mode="bs", sample_error_mode=sample_error_mode, axis=0)
                z_axis = z_target if values.shape[1] == len(z_target) else np.arange(values.shape[1], dtype=float)
                color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                ax_re.errorbar(z_axis, re_m, re_e, color=color, label=rf"$a={a_val:.4f}\,\mathrm{{fm}}$", **ERRORBAR_STYLE)
                ax_im.errorbar(z_axis, im_m, im_e, color=color, label=rf"$a={a_val:.4f}\,\mathrm{{fm}}$", **ERRORBAR_STYLE)
            for ax, ylabel, title, key in (
                (ax_re, r"Re$[H/(z_R Z_{\overline{\mathrm{MS}}})]$", "Discrete-effect overlay (Re)", "discrete_effect_re"),
                (ax_im, r"Im$[H/(z_R Z_{\overline{\mathrm{MS}}})]$", "Discrete-effect overlay (Im)", "discrete_effect_im"),
            ):
                ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
                ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
                ax.set_ylabel(ylabel, **FONT_SIZE)
                ax.set_title(title, **FONT_SIZE)
                ax.legend(**LEGEND_SETS)
            fig_re.tight_layout()
            fig_im.tight_layout()
            # Stage-level names (no job-id prefix) under the renormalization artifacts dir.
            stage_dir = Path(artifacts_dir) if artifacts_dir is not None else stem.parent
            stage_dir.mkdir(parents=True, exist_ok=True)
            pdf, svg = _save_plot_pair(fig_re, stage_dir / "discrete_effect_re")
            plots["discrete_effect_re"] = pdf
            plots["discrete_effect_re_image"] = svg
            pdf, svg = _save_plot_pair(fig_im, stage_dir / "discrete_effect_im")
            plots["discrete_effect_im"] = pdf
            plots["discrete_effect_im_image"] = svg

    store["self_renorm_plots"] = plots
    return {
        "plots": plots,
        "mode": mode,
        "kernel_id": resolved_kernel_id,
        "mu": mu_val,
        "n_sibling": len(sibling_artifacts or []),
        "lattice_spacing_fm": lattice_spacing_fm,
        "include_discrete_effect": bool(include_discrete_effect),
    }


STAGE_TOOLS = {
    "load_bare_matrix_element_grid": load_bare_matrix_element_grid,
    "apply_ratio_scheme_renormalization": apply_ratio_scheme_renormalization,
    "apply_self_renormalization": apply_self_renormalization,
    "plot_renormalized_matrix_element": plot_renormalized_matrix_element,
    "plot_self_renormalization_diagnostics": plot_self_renormalization_diagnostics,
    "load_bare_matrix_element": load_bare_matrix_element,
    "fit_self_renormalization_factor": fit_self_renormalization_factor,
}
