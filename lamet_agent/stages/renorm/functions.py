"""Renormalization stage tools.

Purpose:
- load bare coordinate-space matrix-element bootstrap samples as EnsembleData
- apply sample-preserving ratio, hybrid-ratio, or hybrid self-renormalization
- fit a full-range zR factor with short-distance MSbar finite matching

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
- STAGE_TOOLS["fit_self_renormalization_factor"](
      store, kernel_id="ZMSbar_da", LambdaQCD_gev=0.1, d=-0.08183)
- STAGE_TOOLS["apply_self_renormalization"](
      store, kernel_id="ZMSbar_da", LambdaQCD_gev=0.1)
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
SELF_RENORM_K = 3.320
SELF_RENORM_CF = 4.0 / 3.0
SELF_RENORM_B0 = 11.0 - 2.0 / 3.0 * 3.0
_ZMSBAR_KERNELS = {
    "ZMSbar_pdf": kernels.ZMSbar_pdf,
    "ZMSbar_da": kernels.ZMSbar_da,
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


def _resolve_zmsbar(kernel_id: str | None = None):
    key = kernel_id or "ZMSbar_da"
    if key not in _ZMSBAR_KERNELS:
        raise ValueError(f"unsupported ZMSbar kernel_id: {key!r}")
    return key, _ZMSBAR_KERNELS[key]


def _resolve_lambdaqcd(LambdaQCD_gev: float) -> float:
    """Validate the required LambdaQCD ansatz scale in GeV."""
    value = float(LambdaQCD_gev)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("LambdaQCD_gev must be a finite positive value in GeV")
    return value


def _target_z_mask(
    z_target: np.ndarray,
    z_zr: np.ndarray,
    *,
    policy: Literal["strict", "intersection", "extrapolate"],
) -> np.ndarray:
    if policy not in {"strict", "intersection", "extrapolate"}:
        raise ValueError("z_coverage_policy must be 'strict', 'intersection', or 'extrapolate'")
    tolerance = 1e-10
    covered = (z_target >= float(np.min(z_zr)) - tolerance) & (
        z_target <= float(np.max(z_zr)) + tolerance
    )
    if policy == "strict" and not np.all(covered):
        raise ValueError(
            "target z grid lies outside the fitted zR range: "
            f"target=[{float(np.min(z_target))}, {float(np.max(z_target))}], "
            f"zR=[{float(np.min(z_zr))}, {float(np.max(z_zr))}]"
        )
    if not np.any(covered):
        raise ValueError("target and zR grids have no overlapping z range")
    if policy == "extrapolate":
        if np.any(z_target < float(np.min(z_zr)) - tolerance):
            raise ValueError(
                "zR extrapolation only supports the long-distance upper end; "
                f"target minimum {float(np.min(z_target))} is below zR minimum {float(np.min(z_zr))}"
            )
        return np.ones_like(z_target, dtype=bool)
    return covered


def _self_renorm_target_coordinates(
    target_data: EnsembleData,
    *,
    lattice_spacing_fm: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return target coordinates in fm and the nonzero mask for hybrid self renorm."""
    z_input = np.asarray(target_data.coords["z"], dtype=float)
    raw_unit = target_data.attrs.get("coord_unit")
    input_unit = "fm" if raw_unit in {None, ""} else str(raw_unit).lower()
    if input_unit == "lattice":
        if not np.isfinite(lattice_spacing_fm) or lattice_spacing_fm <= 0.0:
            raise ValueError(
                "hybrid self-renormalization requires positive lattice_spacing_fm "
                "when target coord_unit='lattice'"
            )
        z_fm = np.abs(z_input) * float(lattice_spacing_fm)
    elif input_unit == "fm":
        z_fm = z_input.copy()
    else:
        raise ValueError(
            "hybrid self-renormalization target coord_unit must be 'lattice' or 'fm'"
        )
    nonzero = ~np.isclose(z_fm, 0.0, rtol=0.0, atol=1e-10)
    if not np.any(nonzero):
        raise ValueError(
            "hybrid self-renormalization target has no nonzero z points after unit conversion"
        )
    return z_fm, nonzero, input_unit


def _match_lattice_spacing(a_coords: list[float], lattice_spacing_fm: float) -> int:
    matches = np.flatnonzero(
        np.isclose(np.asarray(a_coords, dtype=float), float(lattice_spacing_fm), rtol=0.0, atol=1e-10)
    )
    if matches.size == 0:
        raise ValueError(
            f"zR has no exact lattice-spacing match for a={lattice_spacing_fm}; available a values are {a_coords}"
        )
    return int(matches[0])


def _interpolate_zr(z_target: np.ndarray, z_zr: np.ndarray, zr_vals: np.ndarray) -> np.ndarray:
    tolerance = 1e-10
    _target_z_mask(z_target, z_zr, policy="strict")
    if z_zr.shape == z_target.shape and np.allclose(z_zr, z_target, rtol=0.0, atol=tolerance):
        return np.asarray(zr_vals, dtype=float)
    return np.interp(z_target, z_zr, zr_vals)


def _self_renorm_zr_from_f1(
    z_values: np.ndarray,
    f1_values: np.ndarray,
    *,
    lattice_spacing_fm: float,
    d: float,
    m0_gev: float,
    mu: float,
    LambdaQCD_gev: float,
) -> np.ndarray:
    """Construct zR from the single-family self-renormalization ansatz."""
    z_arr = np.asarray(z_values, dtype=float)
    f1_arr = np.asarray(f1_values, dtype=float)
    x = GEV_FM / float(lattice_spacing_fm)
    lambdaqcd_gev = _resolve_lambdaqcd(LambdaQCD_gev)
    log_lqcd_over_x = float(np.log(lambdaqcd_gev / x))
    scale_term = 1.0 + float(d) / log_lqcd_over_x
    if scale_term <= 0.0:
        raise ValueError("self-renormalization d term must remain positive")
    constant = (
        3.0
        * SELF_RENORM_CF
        / SELF_RENORM_B0
        * np.log(
            np.log(x / lambdaqcd_gev)
            / np.log(float(mu) / lambdaqcd_gev)
        )
        + np.log(scale_term)
    )
    log_zr = (
        SELF_RENORM_K * z_arr * x / log_lqcd_over_x
        + f1_arr / x
        + constant
        + float(m0_gev) * z_arr
    )
    return np.exp(log_zr)


def _extrapolate_zr_long_distance(
    z_target: np.ndarray,
    z_zr: np.ndarray,
    zr_vals: np.ndarray,
    *,
    lattice_spacing_fm: float,
    d: float | None,
    m0_gev: float | None,
    mu: float,
    LambdaQCD_gev: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extend zR through a quadratic fit to the inferred long-distance f1."""
    z_target_arr = np.asarray(z_target, dtype=float)
    z_zr_arr = np.asarray(z_zr, dtype=float)
    zr_arr = np.asarray(zr_vals, dtype=float)
    if np.any(zr_arr <= 0.0):
        raise ValueError("zR extrapolation requires positive fitted zR values")

    zmax = float(np.max(z_zr_arr))
    extrapolated = z_target_arr > zmax + 1e-10
    if not np.any(extrapolated):
        return _interpolate_zr(z_target_arr, z_zr_arr, zr_arr), {
            "n_z_extrapolated": 0,
            "z_extrapolation_method": "none",
            "f1_tail_zmin_fm": None,
        }
    if d is None or m0_gev is None:
        raise ValueError("zR extrapolation requires d and m0_gev provenance")

    x = GEV_FM / float(lattice_spacing_fm)
    baseline = _self_renorm_zr_from_f1(
        z_zr_arr,
        np.zeros_like(z_zr_arr),
        lattice_spacing_fm=lattice_spacing_fm,
        d=float(d),
        m0_gev=float(m0_gev),
        mu=float(mu),
        LambdaQCD_gev=float(LambdaQCD_gev),
    )
    f1_values = x * (np.log(zr_arr) - np.log(baseline))
    tail_zmin = 0.4 * zmax
    tail_mask = z_zr_arr >= tail_zmin - 1e-10
    if int(np.count_nonzero(tail_mask)) < 3:
        tail_mask = np.zeros_like(z_zr_arr, dtype=bool)
        tail_mask[-min(3, len(z_zr_arr)) :] = True
    if int(np.count_nonzero(tail_mask)) < 3:
        raise ValueError("zR extrapolation requires at least three fitted z points")

    coefficients = np.polyfit(z_zr_arr[tail_mask], f1_values[tail_mask], deg=2)
    result = np.interp(np.minimum(z_target_arr, zmax), z_zr_arr, zr_arr)
    z_extra = z_target_arr[extrapolated]
    result[extrapolated] = _self_renorm_zr_from_f1(
        z_extra,
        np.polyval(coefficients, z_extra),
        lattice_spacing_fm=lattice_spacing_fm,
        d=float(d),
        m0_gev=float(m0_gev),
        mu=float(mu),
        LambdaQCD_gev=float(LambdaQCD_gev),
    )
    return result, {
        "n_z_extrapolated": int(np.count_nonzero(extrapolated)),
        "z_extrapolation_method": "quadratic_f1_tail",
        "f1_tail_zmin_fm": float(np.min(z_zr_arr[tail_mask])),
    }


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
    target: str,
    denominator: str,
    scheme: str,
    strategy: str,
    scheme_parameters: dict[str, float],
    out: str = "matrix_element_data",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    job_id: str | None = None,
    ensemble: str | None = None,
    sample_error_mode: str,
) -> dict[str, Any]:
    """Apply the external_denominator strategy in the ratio or hybrid scheme."""
    if strategy != "external_denominator":
        raise ValueError(f"unsupported renormalization strategy: {strategy!r}")
    if scheme not in {"ratio", "hybrid"}:
        raise ValueError(f"unsupported renormalization scheme: {scheme!r}")
    target_data = _require_matrix_data(store, target)
    denom_data = _require_matrix_data(store, denominator)
    if target_data.resample != denom_data.resample:
        raise ValueError(f"target and denominator resampling must match: {target_data.resample} != {denom_data.resample}")

    z_target = np.asarray(target_data.coords["z"], dtype=float)
    z_denom = np.asarray(denom_data.coords["z"], dtype=float)
    if z_target.shape != z_denom.shape or not np.allclose(z_target, z_denom, rtol=0.0, atol=1e-10):
        raise ValueError("target and denominator z grids must match exactly")
    lattice_spacing_raw = target_data.attrs.get("lattice_spacing_fm")
    if lattice_spacing_raw in {None, ""}:
        raise ValueError(
            "external_denominator renormalization requires target lattice_spacing_fm "
            "to convert output z coordinates to fm"
        )
    try:
        lattice_spacing_fm = float(lattice_spacing_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "external_denominator renormalization requires target lattice_spacing_fm "
            "to be a finite positive value"
        ) from exc
    if not np.isfinite(lattice_spacing_fm) or lattice_spacing_fm <= 0.0:
        raise ValueError(
            "external_denominator renormalization requires target lattice_spacing_fm "
            "to be a finite positive value"
        )
    z_output_fm = z_target * lattice_spacing_fm
    target_values = np.asarray(target_data.values, dtype=complex)
    denom_values = np.asarray(denom_data.values, dtype=complex)
    if target_values.shape != denom_values.shape:
        raise ValueError("target and denominator sample arrays must have matching shape")

    renorm_values = target_values / denom_values
    hybrid_metadata: dict[str, float] = {}
    if scheme == "hybrid":
        params = scheme_parameters
        zs_fm = float(params["zs_fm"])
        m0_gev = float(params["m0_gev"])
        delta_m_gev = float(params["delta_m_gev"])
        zs_lattice = zs_fm / lattice_spacing_fm
        zs_idx = int(np.argmin(np.abs(np.abs(z_denom) - zs_lattice)))
        z_abs_fm = np.abs(z_output_fm)
        mass_scale = (delta_m_gev + m0_gev) / GEV_FM
        exponent = np.exp(mass_scale * (z_abs_fm - zs_fm))
        long = exponent[None, :] * target_values / denom_values[:, zs_idx : zs_idx + 1]
        renorm_values = np.where(z_abs_fm[None, :] <= zs_fm, renorm_values, long)
        hybrid_metadata = {
            "zs_fm": zs_fm,
            "zs_lattice": zs_lattice,
            "zs_grid": float(z_denom[zs_idx]),
            "delta_m_gev": delta_m_gev,
            "m0_gev": m0_gev,
        }

    attrs = {
        **target_data.attrs,
        "scheme": scheme,
        "strategy": strategy,
        "target": target,
        "denominator": denominator,
        "job_id": job_id,
        "sample_error_mode": sample_error_mode,
        "average_method": sample_error_mode,
        "coord_unit": "fm",
        "input_coord_unit": "lattice",
    }
    if ensemble:
        attrs["ensemble"] = ensemble
    attrs.update({key: str(value) for key, value in hybrid_metadata.items()})
    result = _matrix_to_ensemble(
        z_values=z_output_fm,
        samples=renorm_values,
        resample=target_data.resample,
        attrs=attrs,
        name="renormalized_matrix_element",
    )
    store[out] = result
    store["matrix_element_data"] = result
    store["output"] = result
    store["matrix_element"] = {
        "coord": z_output_fm,
        "re_samples": np.real(renorm_values),
        "im_samples": np.imag(renorm_values),
        "scheme": scheme,
        "strategy": strategy,
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
        "scheme": scheme,
        "strategy": strategy,
        **hybrid_metadata,
    }


def plot_renormalized_matrix_element(
    store: dict[str, Any],
    *,
    data: str = "matrix_element_data",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    title: str | None = None,
    sample_error_mode: str,
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
    scheme: str = "ratio",
    strategy: str = "self_renormalization",
    kernel_id: str | None = None,
    mu: float,
    LambdaQCD_gev: float,
    d: float | None = None,
    svdcut: float,
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fit the zR factor for the self-renormalization strategy.

    ``d`` is required and fixed in the continuum/discretization fit and zR
    construction. The finite slope ``m0`` is always fitted from the first
    three short-distance ``g(z)`` points against ``ZMSbar_pdf``. The fit uses
    one discretization coefficient ``f1(z)`` and never extrapolates beyond the
    reference z grid.
    """
    if scheme not in {"ratio", "hybrid", "msbar"}:
        raise ValueError(f"unsupported renormalization scheme: {scheme!r}")
    if strategy != "self_renormalization":
        raise ValueError(f"unsupported renormalization strategy: {strategy!r}")
    if d is None:
        raise ValueError("fit_self_renormalization_factor requires d (fixed; never fitted)")
    d_val = float(d)
    lambdaqcd_gev = _resolve_lambdaqcd(LambdaQCD_gev)
    ref = store.get(reference)
    if not isinstance(ref, EnsembleData):
        raise ValueError(f"store[{reference!r}] does not contain EnsembleData")
    if ref.resample not in {"bootstrap", "jackknife"}:
        raise ValueError(
            f"fit_self_renormalization_factor requires bootstrap/jackknife reference samples; got resample={ref.resample!r}"
        )
    if ref.dims not in (["z"], ["a", "z"]):
        raise ValueError("hybrid self-renormalization reference must have dimensions ['z'] or ['a', 'z']")
    if "discretization_groups" in ref.attrs:
        raise ValueError(
            "reference discretization_groups metadata is no longer supported; "
            "provide one discretization family"
        )
    resolved_kernel_id, _zms_apply = _resolve_zmsbar(kernel_id)
    z_coords = [float(z) for z in ref.coords["z"]]
    if "a" in ref.dims:
        a_coords = [float(a) for a in ref.coords["a"]]
    else:
        a_coords = [float(ref.ensemble.a_s)]
    z_arr = np.asarray(z_coords, dtype=float)
    if z_arr.size == 0 or np.any(np.diff(z_arr) <= 0.0):
        raise ValueError("reference z coordinates must be a non-empty, strictly increasing grid")
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

    z_x: dict[str, list[Any]] = {"z": [], "x": []}
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
                SELF_RENORM_K * zm * xm / gv.log(lambdaqcd_gev / xm)
                + p[f"g{zm}"]
                + p[f"f1{zm}"] / xm
                + 3 * SELF_RENORM_CF / SELF_RENORM_B0
                * gv.log(
                    gv.log(xm / lambdaqcd_gev)
                    / gv.log(mu / lambdaqcd_gev)
                )
                + gv.log(1 + d_val / gv.log(lambdaqcd_gev / xm))
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
    if n_z < 3:
        raise ValueError("fit_self_renormalization_factor needs at least 3 z points to fit m0")
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
    f1_post = [p[f"f1{z}"] for z in z_coords]
    f1_mean = np.asarray([float(gv.mean(value)) for value in f1_post], dtype=float)
    f1_sdev = np.asarray([float(gv.sdev(value)) for value in f1_post], dtype=float)

    fit_lnm_mean = np.empty((n_a, n_z), dtype=float)
    fit_lnm_sdev = np.empty((n_a, n_z), dtype=float)
    zr_mean = np.empty((n_a, n_z), dtype=float)
    for ia, a_val in enumerate(a_coords):
        xm = GEV_FM / float(a_val)
        for iz, z_val in enumerate(z_coords):
            fit_ln = (
                SELF_RENORM_K * z_val * xm / gv.log(lambdaqcd_gev / xm)
                + p[f"g{z_val}"]
                + p[f"f1{z_val}"] / xm
                + 3 * SELF_RENORM_CF / SELF_RENORM_B0
                * gv.log(
                    gv.log(xm / lambdaqcd_gev)
                    / gv.log(mu / lambdaqcd_gev)
                )
                + gv.log(1 + d_val / gv.log(lambdaqcd_gev / xm))
            )
            fit_lnm_mean[ia, iz] = float(gv.mean(fit_ln))
            fit_lnm_sdev[ia, iz] = float(gv.sdev(fit_ln))
            temp = (
                SELF_RENORM_K * z_val * xm / gv.log(lambdaqcd_gev / xm)
                + p[f"f1{z_val}"] / xm
                + 3 * SELF_RENORM_CF / SELF_RENORM_B0
                * gv.log(
                    gv.log(xm / lambdaqcd_gev)
                    / gv.log(mu / lambdaqcd_gev)
                )
                + gv.log(1 + d_val / gv.log(lambdaqcd_gev / xm))
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
    alpha_s_derived = float(kernels.alphas_nloop(mu))
    zR = EnsembleData(
        ref.ensemble,
        resample_name,
        [np.asarray(zr_mean, dtype=complex)],
        dims=("a", "z"),
        coords={"a": a_coords, "z": z_arr.tolist()},
        attrs={
            "scheme": scheme,
            "strategy": strategy,
            "kernel_id": resolved_kernel_id,
            "mu": str(mu),
            "LambdaQCD_gev": str(lambdaqcd_gev),
            "alpha_s_derived": str(alpha_s_derived),
            "alpha_s_source": "alphas_nloop",
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
        "f1_mean": f1_mean,
        "f1_sdev": f1_sdev,
        "zR_mean": zr_mean,
        "mR": mR,
        "m0": m0_mean,
        "m0_sdev": m0_sdev,
        "m0_source": m0_source,
        "kernel_id": resolved_kernel_id,
        "mu": float(mu),
        "LambdaQCD_gev": lambdaqcd_gev,
        "alpha_s_derived": alpha_s_derived,
        "alpha_s_source": "alphas_nloop",
        "d": d_val,
        "svdcut": float(svdcut),
        "skip_z0": bool(skip_z0),
    }
    return {
        "out": out,
        "artifact": str(artifact),
        "scheme": scheme,
        "strategy": strategy,
        "job_kind": "fit",
        "kernel_id": resolved_kernel_id,
        "m0": m0_mean,
        "m0_sdev": m0_sdev,
        "m0_source": m0_source,
        "mu": float(mu),
        "LambdaQCD_gev": lambdaqcd_gev,
        "alpha_s_derived": alpha_s_derived,
        "alpha_s_source": "alphas_nloop",
        "d": d_val,
        "svdcut": float(svdcut),
        "z_values": z_arr.tolist(),
        "a_values": a_coords,
        "n_z": int(n_z),
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
    LambdaQCD_gev: float,
) -> np.ndarray:
    """Remap mean zR from (d_from, m0_from) to (d_to, m0_to).

    Continuum/discretization pieces cancel; only the ``d`` log term and
    ``m0*z`` slope differ between reference and target operators.
    """
    x = GEV_FM / float(lattice_spacing_fm)
    lambdaqcd_gev = _resolve_lambdaqcd(LambdaQCD_gev)
    log_term = float(np.log(lambdaqcd_gev / x))
    if abs(log_term) < 1e-30:
        raise ValueError(f"invalid self-renormalization log term for lattice_spacing_fm={lattice_spacing_fm}")
    scale = (1.0 + d_to / log_term) / (1.0 + d_from / log_term)
    return np.asarray(zr_vals, dtype=float) * scale * np.exp((m0_to - m0_from) * np.asarray(z_vals, dtype=float))


def apply_self_renormalization(
    store: dict[str, Any],
    *,
    target: str = "target",
    denominator: str | None = None,
    zR: str = "zR",
    scheme: str = "ratio",
    strategy: str = "self_renormalization",
    zs_fm: float | None = None,
    kernel_id: str | None = None,
    mu: float,
    LambdaQCD_gev: float,
    d: float | None = None,
    m0_gev: float | None = None,
    z_coverage_policy: Literal["strict", "intersection", "extrapolate"],
    out: str = "matrix_element_data",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    job_id: str | None = None,
    ensemble: str | None = None,
    metadata: dict[str, Any] | None = None,
    sample_error_mode: str,
) -> dict[str, Any]:
    """Apply the self-renormalization strategy in ratio, hybrid, or MSbar scheme.

    Optional ``d`` / ``m0_gev`` remap upstream zR from the fit-job operator
    parameters onto this apply job (e.g. PDF-fit zR → DA ``d``/``m0``).
    ``LambdaQCD_gev`` is this apply job's scale; it need not match the fit job.
    """
    if scheme not in {"ratio", "hybrid", "msbar"}:
        raise ValueError(f"unsupported renormalization scheme: {scheme!r}")
    if strategy != "self_renormalization":
        raise ValueError(f"unsupported renormalization strategy: {strategy!r}")
    if scheme == "hybrid" and (denominator is None or zs_fm is None):
        raise ValueError("hybrid+self_renormalization requires denominator and zs_fm")
    target_data = _require_matrix_data(store, target)
    denominator_data = _require_matrix_data(store, denominator) if denominator is not None else None
    zR_data = store[zR]
    if not isinstance(zR_data, EnsembleData):
        raise ValueError(f"store[{zR!r}] does not contain EnsembleData")
    lambdaqcd_gev = _resolve_lambdaqcd(LambdaQCD_gev)
    resolved_kernel_id, zms_fn = _resolve_zmsbar(kernel_id or zR_data.attrs.get("kernel_id"))
    alpha_s_derived = float(kernels.alphas_nloop(mu))

    z_zr = np.asarray(zR_data.coords["z"], dtype=float)
    a_coords = list(zR_data.coords.get("a", [zR_data.ensemble.a_s]))
    metadata_fields = {
        "ensemble",
        "momentum",
        "volume",
        "bz_direction",
        "lattice_spacing_fm",
        "momentum_gev",
        "initial_momentum",
        "final_momentum",
        "initial_momentum_gev",
        "final_momentum_gev",
        "t_gev2",
        "xi",
        "fitting_form",
        "hadron",
        "gfix",
    }
    metadata_out = {
        key: value
        for key, value in (metadata or {}).items()
        if key in metadata_fields and value is not None
    }
    if ensemble:
        metadata_out["ensemble"] = ensemble
    elif "ensemble" not in metadata_out and target_data.ensemble is not None:
        metadata_out["ensemble"] = target_data.ensemble.id
    lattice_spacing_fm = float(
        metadata_out.get("lattice_spacing_fm", target_data.attrs.get("lattice_spacing_fm", a_coords[0]))
    )
    ia = _match_lattice_spacing([float(a) for a in a_coords], lattice_spacing_fm)
    a_used = float(a_coords[ia])
    z_target_input, nonzero_mask, input_coord_unit = _self_renorm_target_coordinates(
        target_data, lattice_spacing_fm=lattice_spacing_fm
    )
    if scheme == "hybrid":
        assert denominator_data is not None
        if target_data.resample != denominator_data.resample:
            raise ValueError(
                "hybrid+self_renormalization target and denominator resampling must match: "
                f"{target_data.resample} != {denominator_data.resample}"
            )
        z_denominator_input, _, denominator_coord_unit = _self_renorm_target_coordinates(
            denominator_data, lattice_spacing_fm=lattice_spacing_fm
        )
        if (
            z_denominator_input.shape != z_target_input.shape
            or not np.allclose(z_denominator_input, z_target_input, rtol=0.0, atol=1e-10)
        ):
            raise ValueError(
                "hybrid+self_renormalization target and denominator z grids must match exactly"
            )
        if np.asarray(denominator_data.values).shape != np.asarray(target_data.values).shape:
            raise ValueError(
                "hybrid+self_renormalization target and denominator sample arrays must have matching shape"
            )
    else:
        denominator_coord_unit = None
    nonzero_indices = np.flatnonzero(nonzero_mask)
    coverage_mask = _target_z_mask(
        z_target_input[nonzero_mask], z_zr, policy=z_coverage_policy
    )
    target_indices = nonzero_indices[coverage_mask]
    z_target = z_target_input[target_indices]
    zero_indices = np.flatnonzero(~nonzero_mask)
    output_indices = np.sort(np.concatenate((zero_indices, target_indices)))
    z_output = z_target_input[output_indices]
    n_z_zero_passthrough = int(len(zero_indices))
    n_z_coverage_dropped = int(len(nonzero_indices) - len(target_indices))
    n_z_dropped = n_z_coverage_dropped

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
                    LambdaQCD_gev=lambdaqcd_gev,
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
                LambdaQCD_gev=lambdaqcd_gev,
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
                "scheme": scheme,
                "strategy": strategy,
                "d": str(d_to),
                "m0_gev": str(m0_to),
                "d_from": str(d_from),
                "m0_from": str(m0_from),
                "LambdaQCD_gev": str(lambdaqcd_gev),
                "sample_construction": "remapped_from_upstream_zR",
                "alpha_s_derived": str(alpha_s_derived),
                "alpha_s_source": "alphas_nloop",
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
    if z_coverage_policy == "extrapolate":
        zr_on_target, extrapolation = _extrapolate_zr_long_distance(
            z_target,
            z_zr,
            np.asarray(zr_vals, dtype=float),
            lattice_spacing_fm=a_used,
            d=d_to,
            m0_gev=m0_to,
            mu=float(zR_data.attrs.get("mu", mu)),
            LambdaQCD_gev=lambdaqcd_gev,
        )
    else:
        zr_on_target = _interpolate_zr(z_target, z_zr, np.asarray(zr_vals, dtype=float))
        extrapolation = {
            "n_z_extrapolated": 0,
            "z_extrapolation_method": "none",
            "f1_tail_zmin_fm": None,
        }

    all_target_values = np.asarray(target_data.values, dtype=complex)
    target_values = all_target_values[:, target_indices]
    hybrid_metadata: dict[str, str] = {}
    if scheme == "ratio":
        zms = np.asarray(zms_fn(z_target, mu=mu), dtype=float)
        renorm_nonzero = target_values / (zr_on_target[None, :] * zms[None, :])
    elif scheme == "msbar":
        renorm_nonzero = target_values / zr_on_target[None, :]
    else:
        assert denominator_data is not None and zs_fm is not None
        all_denominator_values = np.asarray(denominator_data.values, dtype=complex)
        denominator_values = all_denominator_values[:, target_indices]
        switch_position = int(np.argmin(np.abs(np.abs(z_target) - float(zs_fm))))
        zt = denominator_values[:, switch_position] / zr_on_target[switch_position]
        if np.any(np.isclose(np.abs(zt), 0.0, rtol=0.0, atol=1e-30)):
            raise ValueError("hybrid+self_renormalization produced zero Z_T at the switch point")
        short_values = target_values / denominator_values
        long_values = target_values / (zr_on_target[None, :] * zt[:, None])
        short_mask = np.abs(z_target) <= float(zs_fm)
        renorm_nonzero = np.where(short_mask[None, :], short_values, long_values)
        hybrid_metadata = {
            "denominator": str(denominator),
            "denominator_coord_unit": str(denominator_coord_unit),
            "zs_fm": str(float(zs_fm)),
            "zs_grid_fm": str(float(z_target[switch_position])),
            "ZT_re_mean": str(float(np.mean(np.real(zt)))),
            "ZT_im_mean": str(float(np.mean(np.imag(zt)))),
        }
    renorm_by_index = {
        int(input_index): renorm_nonzero[:, position]
        for position, input_index in enumerate(target_indices)
    }
    renorm_values = np.stack(
        [
            (
                all_target_values[:, input_index]
                / np.asarray(denominator_data.values, dtype=complex)[:, input_index]
                if scheme == "hybrid" and denominator_data is not None
                else all_target_values[:, input_index]
            )
            if not nonzero_mask[input_index]
            else renorm_by_index[int(input_index)]
            for input_index in output_indices
        ],
        axis=1,
    )
    attrs = {
        **target_data.attrs,
        "scheme": scheme,
        "strategy": strategy,
        "kernel_id": resolved_kernel_id,
        "mu": str(mu),
        "LambdaQCD_gev": str(lambdaqcd_gev),
        "alpha_s_derived": str(alpha_s_derived),
        "alpha_s_source": "alphas_nloop",
        "m0_gev": "" if m0_to is None else str(m0_to),
        "d": "" if d_to is None else str(d_to),
        "lattice_spacing_fm_used": str(a_used),
        "target": target,
        "job_id": job_id,
        "sample_error_mode": sample_error_mode,
        "average_method": sample_error_mode,
        "coord_unit": "fm",
        "input_coord_unit": input_coord_unit,
        "z_coverage_policy": z_coverage_policy,
        "z_input_min_fm": str(float(np.min(z_target_input))),
        "z_input_max_fm": str(float(np.max(z_target_input))),
        "z_output_min_fm": str(float(np.min(z_output))),
        "z_output_max_fm": str(float(np.max(z_output))),
        "n_z_dropped": str(n_z_dropped),
        "n_z_zero_passthrough": str(n_z_zero_passthrough),
        "n_z_coverage_dropped": str(n_z_coverage_dropped),
        "z0_treatment": (
            "target_over_denominator" if scheme == "hybrid" else "passthrough_without_self_renormalization"
        ),
        "n_z_extrapolated": str(extrapolation["n_z_extrapolated"]),
        "z_extrapolation_method": str(extrapolation["z_extrapolation_method"]),
        "f1_tail_zmin_fm": (
            "" if extrapolation["f1_tail_zmin_fm"] is None else str(extrapolation["f1_tail_zmin_fm"])
        ),
        "zR_input_min_fm": str(float(np.min(z_zr))),
        "zR_input_max_fm": str(float(np.max(z_zr))),
    }
    attrs.update(hybrid_metadata)
    attrs.update(metadata_out)
    if remap:
        attrs["d_from"] = str(d_from)
        attrs["m0_from"] = str(m0_from)
    result = _matrix_to_ensemble(
        z_values=z_output,
        samples=renorm_values,
        resample=target_data.resample,
        attrs=attrs,
        name="renormalized_matrix_element",
    )
    store[out] = result
    store["matrix_element_data"] = result
    store["output"] = result
    store["matrix_element"] = {
        "coord": z_output,
        "re_samples": np.real(renorm_values),
        "im_samples": np.imag(renorm_values),
        "scheme": scheme,
        "strategy": strategy,
    }

    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="renormalized_matrix_element")
    artifact = stem.with_suffix(".nc")
    result.to_netcdf(artifact)
    store["matrix_element_netcdf"] = str(artifact)
    return {
        "out": out,
        "data": "matrix_element_data",
        "artifact": str(artifact),
        "scheme": scheme,
        "strategy": strategy,
        "kernel_id": resolved_kernel_id,
        "mu": float(mu),
        "LambdaQCD_gev": lambdaqcd_gev,
        "alpha_s_derived": alpha_s_derived,
        "alpha_s_source": "alphas_nloop",
        "m0_gev": m0_to,
        "d": d_to,
        **{key: value for key, value in metadata_out.items() if key != "lattice_spacing_fm"},
        "remapped": bool(remap),
        "n_z": int(len(z_output)),
        "n_sample": int(renorm_values.shape[0]),
        "lattice_spacing_fm": lattice_spacing_fm,
        "z_coverage_policy": z_coverage_policy,
        "n_z_input": int(len(z_target_input)),
        "n_z_dropped": n_z_dropped,
        "n_z_zero_passthrough": n_z_zero_passthrough,
        "n_z_coverage_dropped": n_z_coverage_dropped,
        "input_coord_unit": input_coord_unit,
        **hybrid_metadata,
        **extrapolation,
        "z_input_range_fm": [float(np.min(z_target_input)), float(np.max(z_target_input))],
        "z_output_range_fm": [float(np.min(z_output)), float(np.max(z_output))],
        "zR_input_range_fm": [float(np.min(z_zr)), float(np.max(z_zr))],
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
    sample_error_mode: str,
    kernel_id: str | None = None,
    mu: float,
    LambdaQCD_gev: float,
    z_coverage_policy: Literal["strict", "intersection", "extrapolate"],
) -> dict[str, Any]:
    """Plot hybrid-self-renormalization diagnostics.

    ``mode='fit'`` writes fit-only panels once (no ``fit_vs_data`` / no m0 panel).
    ``mode='apply'`` writes per-target ``zmsbar_compare``; when
    ``include_discrete_effect`` is true and sibling NetCDFs exist, also writes
    one multi-a overlay per momentum under stage-level names
    ``discrete_effect_<momentum>_re/im`` (no job-id prefix).
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
        kernel_id or fit_data.get("kernel_id") or zR_data.attrs.get("kernel_id")
    )
    # Fit-check panels compare mR against ZMSbar_pdf.
    zms_fit_fn = kernels.ZMSbar_pdf
    mu_val = float(mu)
    lambdaqcd_gev = _resolve_lambdaqcd(LambdaQCD_gev)
    alpha_s_derived = float(kernels.alphas_nloop(mu_val))
    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="self_renorm")
    plots: dict[str, str] = {}

    if mode == "fit":
        z_fit = np.asarray(fit_data["z"], dtype=float)
        a_fit = np.asarray(fit_data["a"], dtype=float)
        x_fit = GEV_FM / a_fit
        lnm_mean = np.asarray(fit_data["lnm_mean"], dtype=float)
        lnm_sdev = np.asarray(fit_data["lnm_sdev"], dtype=float)
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
        ax.set_title("Reference matrix element fit input", **FONT_SIZE)
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
        ax.set_ylabel(r"$M_{\mathrm{bare}}/Z_R$", **FONT_SIZE)
        ax.set_title("PDF self-renormalization check", **FONT_SIZE)
        ax.legend(**LEGEND_SETS)
        fig.tight_layout()
        pdf, svg = _save_plot_pair(fig, stem.with_name(stem.name + "_fit_m_over_zR"))
        plots["fit_m_over_zR"] = pdf
        plots["fit_m_over_zR_image"] = svg

        fig, ax = default_plot()
        ax.errorbar(
            z_fit,
            np.asarray(fit_data["f1_mean"], dtype=float),
            np.asarray(fit_data["f1_sdev"], dtype=float),
            color=COLOR_CYCLE[0],
            label=r"$f_1$",
            **ERRORBAR_STYLE,
        )
        ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
        ax.set_ylabel(r"$f_1(z)$", **FONT_SIZE)
        ax.set_title("Discretization coefficient $f_1(z)$", **FONT_SIZE)
        ax.legend(**LEGEND_SETS)
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
            "LambdaQCD_gev": lambdaqcd_gev,
            "alpha_s_derived": alpha_s_derived,
            "alpha_s_source": "alphas_nloop",
            "n_sibling": 0,
        }

    # apply mode
    target_data = _require_matrix_data(store, target)
    a_coords = list(zR_data.coords.get("a", [zR_data.ensemble.a_s]))
    lattice_spacing_fm = float(target_data.attrs.get("lattice_spacing_fm", a_coords[0]))
    ia = _match_lattice_spacing([float(a) for a in a_coords], lattice_spacing_fm)
    zr_arr = np.asarray(zR_data.values)
    if zr_arr.ndim == 3:
        zr_vals = np.mean(np.real(zr_arr[:, ia, :]), axis=0)
    elif zr_arr.ndim == 2:
        zr_vals = np.mean(np.real(zr_arr), axis=0)
    else:
        raise ValueError(f"store[{zR!r}] values must be shaped (resample,a,z) or (resample,z)")
    z_zr = np.asarray(zR_data.coords["z"], dtype=float)
    z_target_input, nonzero_mask, input_coord_unit = _self_renorm_target_coordinates(
        target_data, lattice_spacing_fm=lattice_spacing_fm
    )
    nonzero_indices = np.flatnonzero(nonzero_mask)
    coverage_mask = _target_z_mask(
        z_target_input[nonzero_mask], z_zr, policy=z_coverage_policy
    )
    target_indices = nonzero_indices[coverage_mask]
    z_target = z_target_input[target_indices]
    if z_coverage_policy == "extrapolate":
        zr_on_target, extrapolation = _extrapolate_zr_long_distance(
            z_target,
            z_zr,
            np.asarray(zr_vals, dtype=float),
            lattice_spacing_fm=lattice_spacing_fm,
            d=float(zR_data.attrs["d"]) if zR_data.attrs.get("d") not in {None, ""} else None,
            m0_gev=(
                float(zR_data.attrs["m0_gev"])
                if zR_data.attrs.get("m0_gev") not in {None, ""}
                else None
            ),
            mu=float(zR_data.attrs.get("mu", mu_val)),
            LambdaQCD_gev=lambdaqcd_gev,
        )
    else:
        zr_on_target = _interpolate_zr(z_target, z_zr, np.asarray(zr_vals, dtype=float))
        extrapolation = {
            "n_z_extrapolated": 0,
            "z_extrapolation_method": "none",
            "f1_tail_zmin_fm": None,
        }
    zms_target = np.asarray(zms_fn(z_target, mu=mu_val), dtype=float)
    target_values = np.asarray(target_data.values, dtype=complex)[:, target_indices]
    mode_rs = _resample_mode(target_data)
    h_over_zr = target_values / zr_on_target[None, :]
    re_hzr, re_hzr_err = sample_mean_and_sdev(np.real(h_over_zr), mode=mode_rs, sample_error_mode=sample_error_mode, axis=0)

    fig, ax = default_plot()
    ax.errorbar(z_target, re_hzr, re_hzr_err, color=COLOR_CYCLE[0], label=rf"$H/Z_R$ ($a={lattice_spacing_fm:.4f}\,\mathrm{{fm}}$)", **ERRORBAR_STYLE)
    ax.plot(z_target, zms_target, color=COLOR_CYCLE[1], label=r"$Z_{\overline{\mathrm{MS}}}$")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
    ax.set_ylabel(r"Re$[H/Z_R]$", **FONT_SIZE)
    ax.set_title(r"Compare $H/Z_R$ with $Z_{\overline{\mathrm{MS}}}$", **FONT_SIZE)
    ax.legend(**LEGEND_SETS)
    fig.tight_layout()
    pdf, svg = _save_plot_pair(fig, stem.with_name(stem.name + "_zmsbar_compare"))
    plots["zmsbar_compare"] = pdf
    plots["zmsbar_compare_image"] = svg

    if include_discrete_effect:
        series_by_momentum: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {}
        for path in sibling_artifacts or []:
            sibling_path = Path(path)
            if not sibling_path.is_file():
                continue
            sibling = EnsembleData.from_netcdf(sibling_path)
            momentum = str(sibling.attrs.get("momentum") or "momentum_unknown")
            series_by_momentum.setdefault(momentum, []).append(
                (
                    float(sibling.attrs.get("lattice_spacing_fm", sibling.ensemble.a_s)),
                    np.asarray(sibling.values, dtype=complex),
                    np.asarray(sibling.coords["z"], dtype=float),
                )
            )

        for momentum, series in sorted(series_by_momentum.items()):
            if len(series) < 2:
                continue
            momentum_slug = re.sub(r"[^A-Za-z0-9]+", "_", momentum).strip("_").lower()
            fig_re, ax_re = default_plot()
            fig_im, ax_im = default_plot()
            for idx, (a_val, values, z_axis) in enumerate(sorted(series, key=lambda item: item[0])):
                re_m, re_e = sample_mean_and_sdev(np.real(values), mode="bs", sample_error_mode=sample_error_mode, axis=0)
                im_m, im_e = sample_mean_and_sdev(np.imag(values), mode="bs", sample_error_mode=sample_error_mode, axis=0)
                if values.shape[1] != len(z_axis):
                    raise ValueError(f"discrete-effect artifact for {momentum} has inconsistent z coordinates")
                color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                ax_re.errorbar(z_axis, re_m, re_e, color=color, label=rf"$a={a_val:.4f}\,\mathrm{{fm}}$", **ERRORBAR_STYLE)
                ax_im.errorbar(z_axis, im_m, im_e, color=color, label=rf"$a={a_val:.4f}\,\mathrm{{fm}}$", **ERRORBAR_STYLE)
            for ax, ylabel, title in (
                (ax_re, r"Re$[H/(Z_R Z_{\overline{\mathrm{MS}}})]$", "Discrete-effect overlay (Re)"),
                (ax_im, r"Im$[H/(Z_R Z_{\overline{\mathrm{MS}}})]$", "Discrete-effect overlay (Im)"),
            ):
                ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
                ax.set_xlabel(r"$z$ [fm]", **FONT_SIZE)
                ax.set_ylabel(ylabel, **FONT_SIZE)
                ax.set_title(f"{title}: {momentum}", **FONT_SIZE)
                ax.legend(**LEGEND_SETS)
            fig_re.tight_layout()
            fig_im.tight_layout()
            # Stage-level, momentum-specific names under the renormalization artifacts dir.
            stage_dir = Path(artifacts_dir) if artifacts_dir is not None else stem.parent
            stage_dir.mkdir(parents=True, exist_ok=True)
            re_key = f"discrete_effect_{momentum_slug}_re"
            pdf, svg = _save_plot_pair(fig_re, stage_dir / re_key)
            plots[re_key] = pdf
            plots[f"{re_key}_image"] = svg
            im_key = f"discrete_effect_{momentum_slug}_im"
            pdf, svg = _save_plot_pair(fig_im, stage_dir / im_key)
            plots[im_key] = pdf
            plots[f"{im_key}_image"] = svg

    store["self_renorm_plots"] = plots
    return {
        "plots": plots,
        "mode": mode,
        "kernel_id": resolved_kernel_id,
        "mu": mu_val,
        "LambdaQCD_gev": lambdaqcd_gev,
        "alpha_s_derived": alpha_s_derived,
        "alpha_s_source": "alphas_nloop",
        "n_sibling": len(sibling_artifacts or []),
        "lattice_spacing_fm": lattice_spacing_fm,
        "include_discrete_effect": bool(include_discrete_effect),
        "z_coverage_policy": z_coverage_policy,
        "n_z_dropped": int(len(nonzero_indices) - len(target_indices)),
        "n_z_zero_skipped": int(np.count_nonzero(~nonzero_mask)),
        "n_z_coverage_dropped": int(len(nonzero_indices) - len(target_indices)),
        "input_coord_unit": input_coord_unit,
        **extrapolation,
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
