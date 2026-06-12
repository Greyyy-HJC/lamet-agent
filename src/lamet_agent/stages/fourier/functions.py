"""Fourier-transform stage tools.

Purpose:
- load coordinate-space real/imaginary samples
- call the local sample-preserving extrapolation and Fourier workflow
- keep large arrays in the stage store and write compact `.npz` artifacts

Expected inputs:
- an `.npz` file with `coord`, `re_samples`, and `im_samples`, or
  an HDF5 file with group datasets such as `Pz=4/z_ary`, `Pz=4/Re`, `Pz=4/Im`
- tool arguments supplied by the agent as JSON-compatible values

Expected outputs:
- Fourier samples stored in the per-stage store
- summary arrays written under `artifacts/`
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from lamet_agent.core.data import EnsembleData
from lamet_agent.core.plotting import plot_fourier_extension_quality, plot_fourier_npz
from lamet_agent.stages.fourier.workflow import (
    _coord_scale,
    _normalise_resample_mode,
    _n_fit_parameters,
    _sample_sdev,
    fit_tail_quality_for_mean,
    run_fourier_workflow,
)


def _samples_axis_zero(values: np.ndarray, coord: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix-element samples must be a 2D array")
    n_coord = len(coord)
    if arr.shape[1] == n_coord:
        return arr
    if arr.shape[0] == n_coord:
        return np.moveaxis(arr, 1, 0)
    raise ValueError("matrix-element samples must have one axis matching the coordinate length")


def matrix_element_to_ensemble_data(
    *,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    resample: str = "bootstrap",
    attrs: dict[str, Any] | None = None,
    name: str = "renormalized_matrix_element",
) -> EnsembleData:
    """Build a complex EnsembleData matrix element with dimension z."""
    coord_arr = np.asarray(coord, dtype=float)
    re_axis0 = _samples_axis_zero(np.asarray(re_samples, dtype=float), coord_arr)
    im_axis0 = _samples_axis_zero(np.asarray(im_samples, dtype=float), coord_arr)
    if re_axis0.shape != im_axis0.shape:
        raise ValueError("real and imaginary matrix-element samples must have matching shapes")
    values = [re_axis0[idx] + 1j * im_axis0[idx] for idx in range(re_axis0.shape[0])]
    return EnsembleData(
        ensemble=None,
        resample=_normalise_resample_mode(resample),
        values=values,
        dims=("z",),
        coords={"z": coord_arr.tolist()},
        attrs={key: str(value) for key, value in (attrs or {}).items() if value is not None},
        name=name,
    )


def ensemble_data_to_legacy_arrays(data: EnsembleData) -> dict[str, np.ndarray]:
    """Convert EnsembleData(z) back to legacy coord/re_samples/im_samples arrays."""
    if not isinstance(data, EnsembleData):
        raise TypeError("matrix_element_data must be an EnsembleData")
    if data.dims != ["z"]:
        raise ValueError("Fourier matrix_element_data must have physical dimension ['z']")
    values = np.asarray(data.values)
    if values.ndim != 2:
        raise ValueError("Fourier matrix_element_data values must be shaped (resample,z)")
    return {
        "coord": np.asarray(data.coords["z"], dtype=float),
        "re_samples": np.asarray(np.real(values), dtype=float),
        "im_samples": np.asarray(np.imag(values), dtype=float),
        "resample_mode": data.resample,
    }


def fourier_result_to_ensemble_data(result: dict[str, Any]) -> EnsembleData:
    """Build a complex EnsembleData(x) from Fourier workflow samples."""
    ft_re = np.asarray(result["ft_re_samples"], dtype=float)
    ft_im = np.asarray(result["ft_im_samples"], dtype=float)
    weights = np.asarray(result.get("scheme_weights", []), dtype=float)
    if weights.shape == (ft_re.shape[0],):
        re_samples = np.sum(weights[:, None, None] * ft_re, axis=0)
        im_samples = np.sum(weights[:, None, None] * ft_im, axis=0)
    else:
        re_samples = np.mean(ft_re, axis=0)
        im_samples = np.mean(ft_im, axis=0)
    values = [re_samples[idx] + 1j * im_samples[idx] for idx in range(re_samples.shape[0])]
    attrs = {
        "method": str(result.get("method", "")),
        "order": str(result.get("order", "")),
        "observable": str(result.get("observable", "")),
        "coord_unit": str(result.get("coord_unit", "")),
    }
    for key in ("pz_gev", "pz_prime_gev", "a_fm"):
        value = result.get(key)
        if value is not None:
            attrs[key] = str(value)
    return EnsembleData(
        ensemble=None,
        resample=_normalise_resample_mode(str(result.get("resample_mode", "bootstrap"))),
        values=values,
        dims=("x",),
        coords={"x": np.asarray(result["k_grid"], dtype=float).tolist()},
        attrs=attrs,
        name="fourier_transform",
    )


def load_renormalized_matrix_element_samples(
    store: dict[str, Any],
    *,
    path: str,
    input_format: str | None = None,
    h5_group: str | None = None,
    coord_key: str = "coord",
    re_key: str = "re_samples",
    im_key: str = "im_samples",
    resample_mode: str = "bootstrap",
) -> dict[str, Any]:
    """Load renormalized coordinate-space matrix-element samples from NPZ or HDF5."""
    matrix_element_data, fmt, group_name = _load_matrix_element_data(
        path=path,
        input_format=input_format,
        h5_group=h5_group,
        coord_key=coord_key,
        re_key=re_key,
        im_key=im_key,
        resample_mode=resample_mode,
    )
    legacy = ensemble_data_to_legacy_arrays(matrix_element_data)
    out = "matrix_element"
    store["matrix_element_data"] = matrix_element_data
    store[out] = {
        **legacy,
        "path": str(path),
        "input_format": fmt,
        "resample_mode": matrix_element_data.resample,
    }
    if group_name is not None:
        store[out]["h5_group"] = group_name
    return {
        "out": out,
        "data": "matrix_element_data",
        "input_format": fmt,
        "h5_group": group_name,
        "resample_mode": matrix_element_data.resample,
        "n_coord": int(len(legacy["coord"])),
        "n_sample": int(legacy["re_samples"].shape[0]),
        "re_shape": list(legacy["re_samples"].shape),
        "im_shape": list(legacy["im_samples"].shape),
    }


def _load_matrix_element_data(
    *,
    path: str,
    input_format: str | None,
    h5_group: str | None,
    coord_key: str,
    re_key: str,
    im_key: str,
    resample_mode: str,
) -> tuple[EnsembleData, str, str | None]:
    """Load NPZ/HDF5 matrix-element samples and normalize them to EnsembleData."""
    if input_format is not None:
        fmt = input_format.lower()
    else:
        suffix = Path(path).suffix.lower()
        fmt = "h5" if suffix in {".h5", ".hdf5"} else suffix.lstrip(".")
    if fmt == "hdf5":
        fmt = "h5"
    resample = _normalise_resample_mode(resample_mode)

    if fmt == "npz":
        try:
            data, _extras = EnsembleData.load_npz(path)
        except ValueError:
            with np.load(path, allow_pickle=False) as npz:
                data = matrix_element_to_ensemble_data(
                    coord=np.asarray(npz[coord_key], dtype=float),
                    re_samples=np.asarray(npz[re_key], dtype=float),
                    im_samples=np.asarray(npz[im_key], dtype=float),
                    resample=resample,
                    attrs={"input_format": fmt, "path": str(path)},
                )
        if data.dims != ["z"]:
            raise ValueError("Fourier NPZ EnsembleData input must have physical dimension ['z']")
        values = np.asarray(data.values)
        if values.ndim != 2:
            raise ValueError("Fourier NPZ EnsembleData input must be shaped (resample,z)")
        return data, fmt, None

    if fmt == "h5":
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Reading HDF5 Fourier inputs requires installing lamet-agent with the analysis extra"
            ) from exc

        use_coord_key = "z_ary" if coord_key == "coord" else coord_key
        use_re_key = "Re" if re_key == "re_samples" else re_key
        use_im_key = "Im" if im_key == "im_samples" else im_key
        with h5py.File(path, "r") as h5f:
            group_names = [name for name, item in h5f.items() if isinstance(item, h5py.Group)]
            group_name = h5_group or _infer_h5_group(path, group_names)
            if group_name not in h5f:
                raise ValueError(f"HDF5 group {group_name!r} not found; available groups: {group_names}")
            group = h5f[group_name]
            data = matrix_element_to_ensemble_data(
                coord=np.asarray(group[use_coord_key], dtype=float),
                re_samples=np.asarray(group[use_re_key], dtype=float),
                im_samples=np.asarray(group[use_im_key], dtype=float),
                resample=resample,
                attrs={"input_format": fmt, "h5_group": group_name, "path": str(path)},
            )
        return data, fmt, group_name

    raise ValueError("input_format must be 'npz', 'h5', or 'hdf5'")


def _infer_h5_group(path: str, group_names: list[str]) -> str:
    match = re.search(r"(?:^|_)pz([+-]?\d+)(?:\.|_|$)", Path(path).name, flags=re.IGNORECASE)
    if match:
        group = f"Pz={match.group(1)}"
        if group in group_names:
            return group

    if len(group_names) == 1:
        return group_names[0]
    raise ValueError("h5_group is required when the HDF5 file has multiple groups and no pz can be inferred")


def _artifact_path(raw: str | None, *, default_name: str) -> Path:
    out_dir = Path.cwd() / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw:
        return out_dir / Path(raw).name
    return out_dir / default_name


def _save_fourier_npz(path: Path, result: dict[str, Any]) -> None:
    data = fourier_result_to_ensemble_data(result)
    data.save_npz(
        path,
        ft_re_samples=result["ft_re_samples"],
        ft_im_samples=result["ft_im_samples"],
        ft_re_mean=result["ft_re_mean"],
        ft_im_mean=result["ft_im_mean"],
        ft_re_stat_sdev=result["ft_re_stat_sdev"],
        ft_im_stat_sdev=result["ft_im_stat_sdev"],
        ft_re_sys_sdev=result["ft_re_sys_sdev"],
        ft_im_sys_sdev=result["ft_im_sys_sdev"],
        scheme_labels=np.asarray(result["scheme_labels"]),
        fit_failures=np.asarray(result["fit_failures"], dtype=int),
        scheme_weights=np.asarray(result.get("scheme_weights", []), dtype=float),
        scheme_fit_chi2_dof=np.asarray(result.get("scheme_fit_chi2_dof", []), dtype=float),
        scheme_roughness=np.asarray(result.get("scheme_roughness", []), dtype=float),
        scheme_scores=np.asarray(result.get("scheme_scores", []), dtype=float),
        best_scheme_index=np.asarray(result.get("best_scheme_index", -1), dtype=int),
        pz_gev=np.asarray(result.get("pz_gev", np.nan), dtype=float),
        pz_prime_gev=np.asarray(result.get("pz_prime_gev", np.nan), dtype=float),
        a_fm=np.asarray(result.get("a_fm", np.nan), dtype=float),
        method=np.asarray(result.get("method", "")),
        order=np.asarray(result.get("order", "")),
        observable=np.asarray(result.get("observable", "")),
    )


def _save_fourier_fit_info_npz(path: Path, result: dict[str, Any]) -> None:
    schemes = result["scheme_results"]
    fit_params = np.asarray([item["fit_params"] for item in schemes], dtype=float)
    fit_chi2 = np.asarray([item["fit_chi2"] for item in schemes], dtype=float)
    fit_dof = np.asarray([item["fit_dof"] for item in schemes], dtype=int)
    fit_q = np.asarray([item["fit_q"] for item in schemes], dtype=float)
    mean_fit_params = np.asarray([item["mean_fit_params"] for item in schemes], dtype=float)
    mean_fit_chi2 = np.asarray([item["mean_fit_chi2"] for item in schemes], dtype=float)
    mean_fit_dof = np.asarray([item["mean_fit_dof"] for item in schemes], dtype=int)
    mean_fit_q = np.asarray([item["mean_fit_q"] for item in schemes], dtype=float)

    resample_mode = _normalise_resample_mode(str(result.get("resample_mode", "bootstrap")))
    fit_chi2_dof = fit_chi2 / np.maximum(fit_dof, 1)
    if fit_params.shape[1] < 2:
        fit_param_sdev = np.zeros((fit_params.shape[0], fit_params.shape[2]), dtype=float)
    else:
        fit_param_sdev = np.asarray([_sample_sdev(item, resample_mode=resample_mode) for item in fit_params])

    scheme_labels = np.asarray(result["scheme_labels"])
    fit_param_labels = np.asarray(schemes[0]["fit_param_labels"])
    param_samples = np.moveaxis(fit_params, 1, 0)
    fit_info_data = EnsembleData(
        ensemble=None,
        resample=resample_mode,
        values=[param_samples[idx] for idx in range(param_samples.shape[0])],
        dims=("scheme", "parameter"),
        coords={"scheme": scheme_labels.tolist(), "parameter": fit_param_labels.tolist()},
        attrs={
            "method": str(result.get("method", "")),
            "order": str(result.get("order", "")),
            "observable": str(result.get("observable", "")),
        },
        name="fourier_fit_parameters",
    )
    fit_info_data.save_npz(
        path,
        scheme_labels=scheme_labels,
        fit_param_labels=fit_param_labels,
        fit_params=fit_params,
        fit_param_center=np.mean(fit_params, axis=1),
        fit_param_sdev=fit_param_sdev,
        fit_chi2=fit_chi2,
        fit_dof=fit_dof,
        fit_q=fit_q,
        fit_chi2_dof=fit_chi2_dof,
        fit_chi2_center=np.mean(fit_chi2, axis=1),
        fit_chi2_dof_center=np.mean(fit_chi2_dof, axis=1),
        fit_q_center=np.mean(fit_q, axis=1),
        mean_fit_params=mean_fit_params,
        mean_fit_chi2=mean_fit_chi2,
        mean_fit_dof=mean_fit_dof,
        mean_fit_q=mean_fit_q,
        scheme_weights=np.asarray(result.get("scheme_weights", []), dtype=float),
        scheme_fit_chi2_dof=np.asarray(result.get("scheme_fit_chi2_dof", []), dtype=float),
        scheme_roughness=np.asarray(result.get("scheme_roughness", []), dtype=float),
        scheme_scores=np.asarray(result.get("scheme_scores", []), dtype=float),
        best_scheme_index=np.asarray(result.get("best_scheme_index", -1), dtype=int),
    )


def _scan_values(spec: dict[str, Any], key: str) -> list[float]:
    values_key = f"{key}_values"
    if values_key in spec:
        return [float(item) for item in spec[values_key]]
    start = float(spec[f"{key}_start"])
    stop = float(spec[f"{key}_stop"])
    step = float(spec.get(f"{key}_step", spec.get("step", 1.0)))
    if step <= 0:
        raise ValueError(f"{key}_step must be positive")
    values = []
    current = start
    while current <= stop + 0.5 * step:
        values.append(round(current, 12))
        current += step
    return values


def _positive_grid(coord: np.ndarray) -> np.ndarray:
    positive = np.asarray(coord, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) < 4:
        raise ValueError("automatic scheme_scan needs at least four positive coordinate points")
    return positive


def _last_stable_z_index(
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    *,
    resample_mode: str,
) -> int:
    """Return the last positive-grid index before large-z data become unreliable."""
    positive_mask = np.asarray(coord, dtype=float) > 0
    re = np.asarray(re_samples, dtype=float)[:, positive_mask]
    im = np.asarray(im_samples, dtype=float)[:, positive_mask]
    re_mean = np.mean(re, axis=0)
    im_mean = np.mean(im, axis=0)
    re_sdev = _sample_sdev(re, resample_mode=resample_mode)
    im_sdev = _sample_sdev(im, resample_mode=resample_mode)

    magnitude = np.hypot(re_mean, im_mean)
    uncertainty = np.hypot(re_sdev, im_sdev)
    scale = max(float(np.max(magnitude)), 1e-12)
    rel_uncertainty = uncertainty / np.maximum(magnitude, 0.05 * scale)
    baseline = float(np.median(rel_uncertainty[: min(4, len(rel_uncertainty))]))
    uncertainty_limit = min(1.0, max(0.35, 3.0 * baseline + 0.05))

    signal = re_mean + 1j * im_mean
    signal_scale = max(float(np.max(np.abs(signal))), 1e-12)
    jitter_unstable = np.zeros_like(rel_uncertainty, dtype=bool)
    if len(signal) >= 5:
        curvature = np.abs(signal[2:] - 2.0 * signal[1:-1] + signal[:-2])
        curvature = curvature / np.maximum(np.abs(signal[1:-1]), 0.05 * signal_scale)
        curvature_baseline = float(np.median(curvature[: min(4, len(curvature))]))
        curvature_limit = min(2.0, max(0.5, 4.0 * curvature_baseline + 0.05))
        jitter_unstable[1:-1] = curvature > curvature_limit

    unstable = (rel_uncertainty > uncertainty_limit) | jitter_unstable
    min_points = min(5, len(rel_uncertainty))
    for idx in range(min_points, len(rel_uncertainty) - 1):
        if unstable[idx] and unstable[idx + 1]:
            return max(min_points - 1, idx - 1)
    return len(rel_uncertainty) - 1


def _pick_four_tail_values(grid: np.ndarray, *, end_index: int) -> list[float]:
    end_index = int(np.clip(end_index, 0, len(grid) - 1))
    start_index = max(0, end_index - 3)
    values = grid[start_index : end_index + 1]
    if len(values) < 4:
        values = grid[: min(len(grid), 4)]
    return [float(item) for item in values[-4:]]


def _first_four_zmin_values(positive: np.ndarray, *, zmax_reference: float, min_width: float) -> list[float]:
    max_allowed = float(zmax_reference) - float(min_width)
    candidates = positive[positive <= max_allowed]
    if len(candidates) < 4:
        candidates = positive[positive < float(zmax_reference)]
    return [float(item) for item in candidates[:4]]


def _tail_quality_stable_start(qualities: list[dict[str, Any]]) -> int:
    finite = [
        item
        for item in qualities
        if item["ok"] and np.isfinite(item["chi2_dof"]) and item["n_points"] >= 2
    ]
    if not finite:
        return 0

    chi = np.asarray([item["chi2_dof"] for item in finite], dtype=float)
    q_values = np.asarray([item["q_value"] for item in finite], dtype=float)
    best = float(np.min(chi))
    chi_limit = max(best * 1.25, best + 0.15, 1.0)
    for idx, item in enumerate(qualities):
        if not item["ok"] or not np.isfinite(item["chi2_dof"]):
            continue
        if item["chi2_dof"] > chi_limit:
            continue
        if item["q_value"] < 0.05 and np.nanmax(q_values) >= 0.05:
            continue
        later = [
            later_item["chi2_dof"]
            for later_item in qualities[idx : min(len(qualities), idx + 3)]
            if later_item["ok"] and np.isfinite(later_item["chi2_dof"])
        ]
        if later and max(later) <= max(chi_limit * 1.1, chi_limit + 0.1):
            return idx
    return int(np.nanargmin([item["chi2_dof"] if item["ok"] else np.inf for item in qualities]))


def _pick_four_zmin_values_by_tail_fit(
    positive: np.ndarray,
    *,
    zmax_values: list[float],
    min_width: float,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    method: str,
    order: str,
    observable: str,
    coord_unit: str,
    pz_gev: float | None,
    pz_prime_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    Lambda0: float,
    min_fit_points: int,
) -> list[float]:
    stable_starts = []
    for zmax in zmax_values:
        candidates = positive[(positive < float(zmax)) & ((float(zmax) - positive) >= min_width)]
        candidates = np.asarray(
            [candidate for candidate in candidates if np.count_nonzero((positive >= candidate) & (positive <= zmax)) >= min_fit_points],
            dtype=float,
        )
        if len(candidates) == 0:
            continue
        qualities = [
            fit_tail_quality_for_mean(
                coord,
                re_samples,
                im_samples,
                zmin=float(candidate),
                zmax=float(zmax),
                method=method,
                order=order,
                observable=observable,
                coord_unit=coord_unit,
                pz_gev=pz_gev,
                pz_prime_gev=pz_prime_gev,
                a_fm=a_fm,
                resample_mode=resample_mode,
                Lambda0=Lambda0,
                min_fit_points=min_fit_points,
            )
            for candidate in candidates
        ]
        stable_starts.append(float(candidates[_tail_quality_stable_start(qualities)]))

    if not stable_starts:
        return _first_four_zmin_values(positive, zmax_reference=max(zmax_values), min_width=min_width)

    anchor = min(stable_starts)
    max_allowed = max(zmax_values) - min_width
    candidates = positive[(positive >= anchor) & (positive <= max_allowed)]
    if len(candidates) < 4:
        candidates = positive[positive <= max_allowed]
    if len(candidates) <= 4:
        return [float(item) for item in candidates]
    indices = np.linspace(0, len(candidates) - 1, 4)
    return [float(candidates[int(round(item))]) for item in indices]


def _default_z_ext_max(
    coord: np.ndarray,
    *,
    coord_unit: str,
    pz_gev: float | None,
    a_fm: float | None,
) -> float:
    """Return the coordinate value whose lambda is eight units past the data."""
    _fit_scale, ft_scale = _coord_scale(coord_unit, pz_gev=pz_gev, a_fm=a_fm)
    return float(np.max(coord) + 8.0 / ft_scale)


def _auto_fill_scheme_scan(
    spec: dict[str, Any],
    *,
    coord: np.ndarray,
    positive: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    stable_idx: int,
    coord_unit: str,
    method: str,
    order: str,
    observable: str,
    pz_gev: float | None,
    pz_prime_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    Lambda0: float,
    min_fit_points: int,
) -> dict[str, Any]:
    """Fill missing scan keys with stable zmax values and tail-fit zmin diagnostics."""
    dz = float(np.median(np.diff(positive)))
    if "zmax_values" not in spec and "zmax_start" not in spec:
        spec["zmax_values"] = _pick_four_tail_values(positive, end_index=stable_idx)
    if "zmax_values" in spec:
        zmax_values = [float(item) for item in spec["zmax_values"]]
    else:
        zmax_values = _scan_values(spec, "zmax")
    zmax_reference = max(zmax_values)

    if "min_width" not in spec:
        spec["min_width"] = float(max((min_fit_points - 1) * dz, 3.0 * dz, 0.25 * zmax_reference))
    min_width = float(spec["min_width"])

    if "zmin_values" not in spec and "zmin_start" not in spec:
        spec["zmin_values"] = _pick_four_zmin_values_by_tail_fit(
            positive,
            zmax_values=zmax_values,
            min_width=min_width,
            coord=coord,
            re_samples=re_samples,
            im_samples=im_samples,
            method=method,
            order=order,
            observable=observable,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            pz_prime_gev=pz_prime_gev,
            a_fm=a_fm,
            resample_mode=resample_mode,
            Lambda0=Lambda0,
            min_fit_points=min_fit_points,
        )
    if "z_ext_max" not in spec:
        spec["z_ext_max"] = _default_z_ext_max(
            coord,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            a_fm=a_fm,
        )
    if "smooth" not in spec:
        spec["smooth"] = "linear"
    spec["min_fit_points"] = int(min_fit_points)
    return spec


def _scan_has_all_range_keys(spec: dict[str, Any] | None) -> bool:
    if spec is None:
        return False
    has_zmin = "zmin_values" in spec or "zmin_start" in spec
    has_zmax = "zmax_values" in spec or "zmax_start" in spec
    return has_zmin and has_zmax and "min_width" in spec and "z_ext_max" in spec and "smooth" in spec


def _fill_scheme_defaults(spec: dict[str, Any]) -> dict[str, Any]:
    spec.setdefault("y_range", [-2.0, 2.0])
    spec.setdefault("roughness_weight", 1.0)
    spec.setdefault("model_average", True)
    return spec


def _auto_scheme_scan(
    *,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    coord_unit: str,
    method: str,
    order: str,
    observable: str,
    pz_gev: float | None,
    pz_prime_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
    Lambda0: float,
    min_fit_points: int,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a conservative scan from stable zmax and tail-fit zmin diagnostics."""
    spec = dict(existing or {})
    positive = _positive_grid(coord)
    re_axis0 = np.asarray(re_samples, dtype=float)
    im_axis0 = np.asarray(im_samples, dtype=float)
    stable_idx = _last_stable_z_index(coord, re_axis0, im_axis0, resample_mode=resample_mode)
    spec = _auto_fill_scheme_scan(
        spec,
        coord=np.asarray(coord, dtype=float),
        positive=positive,
        re_samples=re_axis0,
        im_samples=im_axis0,
        stable_idx=stable_idx,
        coord_unit=coord_unit,
        method=method,
        order=order,
        observable=observable,
        pz_gev=pz_gev,
        pz_prime_gev=pz_prime_gev,
        a_fm=a_fm,
        resample_mode=resample_mode,
        Lambda0=Lambda0,
        min_fit_points=min_fit_points,
    )
    spec["auto_generated"] = True
    return spec


def _generate_scan_schemes(spec: dict[str, Any], *, coord: np.ndarray | None = None) -> list[dict[str, Any]]:
    zmin_values = _scan_values(spec, "zmin")
    zmax_values = _scan_values(spec, "zmax")
    min_width = float(spec.get("min_width", 0.0))
    z_ext_max = float(spec["z_ext_max"])
    smooth = str(spec.get("smooth", "linear"))
    max_schemes = int(spec.get("max_schemes", 200))
    min_fit_points = int(spec.get("min_fit_points", 0))
    coord_arr = None if coord is None else np.asarray(coord, dtype=float)

    schemes = []
    for zmin in zmin_values:
        for zmax in zmax_values:
            if zmax <= zmin or zmax - zmin < min_width:
                continue
            if coord_arr is not None and min_fit_points > 0:
                fit_count = np.count_nonzero((coord_arr >= float(zmin)) & (coord_arr <= float(zmax)) & (coord_arr > 0))
                if fit_count < min_fit_points:
                    continue
            scheme = {
                "label": f"zmin_{zmin:g}_zmax_{zmax:g}".replace(".", "p"),
                "zmin": zmin,
                "zmax": zmax,
                "z_ext_max": z_ext_max,
                "smooth": smooth,
            }
            schemes.append(scheme)
            if len(schemes) >= max_schemes:
                return schemes
    if not schemes:
        raise ValueError("scheme_scan produced no valid zmin/zmax combinations")
    return schemes


def _resolve_k_grid(k_grid: list[float] | dict[str, Any]) -> list[float]:
    if isinstance(k_grid, dict):
        start = float(k_grid["start"])
        stop = float(k_grid["stop"])
        if "num" in k_grid:
            num = int(k_grid["num"])
            if num < 2:
                raise ValueError("k_grid num must be at least 2")
            return np.linspace(start, stop, num).tolist()
        step = float(k_grid["step"])
        if step <= 0:
            raise ValueError("k_grid step must be positive")
        return np.arange(start, stop + 0.5 * step, step).tolist()
    return [float(item) for item in k_grid]


def _scheme_fit_chi2_dof(
    *,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    scheme_result: dict[str, Any],
    coord_unit: str,
    pz_gev: float | None,
    a_fm: float | None,
    resample_mode: str,
) -> float:
    scale, _ = _coord_scale(coord_unit, pz_gev=pz_gev, a_fm=a_fm)
    fit_coord = coord * scale
    zmin, zmax = scheme_result["fit_range"]
    fit_mask = (coord >= float(zmin)) & (coord <= float(zmax)) & (coord > 0)
    if np.count_nonzero(fit_mask) == 0:
        return float("inf")

    target_z = fit_coord[fit_mask]
    data_re = np.mean(re_samples[:, fit_mask], axis=0)
    data_im = np.mean(im_samples[:, fit_mask], axis=0)
    sigma_re = _sample_sdev(re_samples[:, fit_mask], resample_mode=resample_mode)
    sigma_im = _sample_sdev(im_samples[:, fit_mask], resample_mode=resample_mode)
    floor = max(1e-8, 0.02 * max(float(np.max(np.abs(data_re))), float(np.max(np.abs(data_im))), 1.0))
    sigma_re = np.maximum(sigma_re, floor)
    sigma_im = np.maximum(sigma_im, floor)

    z_ext = np.asarray(scheme_result["z_ext"], dtype=float)
    fit_re_mean = np.mean(np.asarray(scheme_result["fit_re_samples"], dtype=float), axis=0)
    fit_im_mean = np.mean(np.asarray(scheme_result["fit_im_samples"], dtype=float), axis=0)
    fit_re = np.interp(target_z, z_ext, fit_re_mean)
    fit_im = np.interp(target_z, z_ext, fit_im_mean)

    chi2 = np.sum(((fit_re - data_re) / sigma_re) ** 2) + np.sum(((fit_im - data_im) / sigma_im) ** 2)
    n_params = int(np.asarray(scheme_result["fit_params"]).shape[1])
    dof = max(1, 2 * int(np.count_nonzero(fit_mask)) - n_params)
    return float(chi2 / dof)


def _roughness_score(k_grid: np.ndarray, ft_mean: np.ndarray, y_range: list[float] | tuple[float, float] | None) -> float:
    if y_range is None:
        mask = np.ones_like(k_grid, dtype=bool)
    else:
        low, high = float(y_range[0]), float(y_range[1])
        mask = (k_grid >= low) & (k_grid <= high)
    y = k_grid[mask]
    curve = ft_mean[mask]
    if len(y) < 3:
        return 0.0
    order = np.argsort(y)
    curve = curve[order]
    second = np.diff(curve, n=2)
    scale = max(float(np.sqrt(np.mean(curve**2))), float(np.max(np.abs(curve))), 1e-8)
    return float(np.sqrt(np.mean(second**2)) / scale)


def _apply_scheme_model_average(
    result: dict[str, Any],
    *,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    coord_unit: str,
    pz_gev: float | None,
    a_fm: float | None,
    y_range: list[float] | tuple[float, float] | None,
    roughness_weight: float,
    resample_mode: str,
) -> None:
    ft_re = np.asarray(result["ft_re_samples"], dtype=float)
    ft_im = np.asarray(result["ft_im_samples"], dtype=float)
    k_grid = np.asarray(result["k_grid"], dtype=float)
    re_mean_by_scheme = np.mean(ft_re, axis=1)
    im_mean_by_scheme = np.mean(ft_im, axis=1)
    re_stat_by_scheme = np.asarray([_sample_sdev(item, resample_mode=resample_mode) for item in ft_re])
    im_stat_by_scheme = np.asarray([_sample_sdev(item, resample_mode=resample_mode) for item in ft_im])

    fit_chi2 = []
    roughness = []
    for idx, scheme_result in enumerate(result["scheme_results"]):
        fit_chi2.append(
            _scheme_fit_chi2_dof(
                coord=coord,
                re_samples=re_samples,
                im_samples=im_samples,
                scheme_result=scheme_result,
                coord_unit=coord_unit,
                pz_gev=pz_gev,
                a_fm=a_fm,
                resample_mode=resample_mode,
            )
        )
        roughness.append(_roughness_score(k_grid, re_mean_by_scheme[idx], y_range))

    fit_arr = np.asarray(fit_chi2, dtype=float)
    rough_arr = np.asarray(roughness, dtype=float)
    failures = np.asarray(result["fit_failures"], dtype=float)
    scores = fit_arr + float(roughness_weight) * rough_arr + 100.0 * failures
    finite = np.isfinite(scores)
    if not np.any(finite):
        weights = np.full(len(scores), 1.0 / len(scores))
    else:
        shifted = np.where(finite, scores - np.nanmin(scores[finite]), np.inf)
        weights = np.exp(-0.5 * shifted)
        weights[~np.isfinite(weights)] = 0.0
        total = float(np.sum(weights))
        weights = weights / total if total > 0 else np.full(len(scores), 1.0 / len(scores))

    best = int(np.argmax(weights))
    re_mean = np.sum(weights[:, None] * re_mean_by_scheme, axis=0)
    im_mean = np.sum(weights[:, None] * im_mean_by_scheme, axis=0)
    re_stat = np.sqrt(np.sum(weights[:, None] * re_stat_by_scheme**2, axis=0))
    im_stat = np.sqrt(np.sum(weights[:, None] * im_stat_by_scheme**2, axis=0))
    re_sys = np.sqrt(np.sum(weights[:, None] * (re_mean_by_scheme - re_mean) ** 2, axis=0))
    im_sys = np.sqrt(np.sum(weights[:, None] * (im_mean_by_scheme - im_mean) ** 2, axis=0))

    result["ft_re_mean"] = re_mean
    result["ft_im_mean"] = im_mean
    result["ft_re_stat_sdev"] = re_stat
    result["ft_im_stat_sdev"] = im_stat
    result["ft_re_sys_sdev"] = re_sys
    result["ft_im_sys_sdev"] = im_sys
    result["scheme_weights"] = weights.tolist()
    result["scheme_fit_chi2_dof"] = fit_arr.tolist()
    result["scheme_roughness"] = rough_arr.tolist()
    result["scheme_scores"] = scores.tolist()
    result["best_scheme_index"] = best
    result["best_scheme_label"] = result["scheme_labels"][best]


def run_fourier_transform(
    store: dict[str, Any],
    *,
    k_grid: list[float] | dict[str, Any],
    scheme_scan: dict[str, Any] | None = None,
    method: str = "GI",
    order: str = "NLA",
    observable: str = "nucleon_quark_transversity_quasi_pdf",
    coord_unit: str = "lambda",
    pz_gev: float | None = None,
    pz_prime_gev: float | None = None,
    a_fm: float | None = None,
    im_flip_for_ft: bool = False,
    Lambda0: float = 0.1,
    posterior_prior_error_scale: float = 3.0,
    fit_error_mode: str = "diagonal",
    save_path: str | None = None,
) -> dict[str, Any]:
    """Run local extrapolation and Fourier transform for loaded samples."""
    out = "fourier_result"
    matrix_element_data = store["matrix_element_data"]
    resample_mode = _normalise_resample_mode(getattr(matrix_element_data, "resample", "bootstrap"))
    matrix_element = ensemble_data_to_legacy_arrays(matrix_element_data)
    auto_scheme_scan = None
    scan_spec = _fill_scheme_defaults(dict(scheme_scan or {}))
    min_fit_points = max(
        _n_fit_parameters(method, order, observable),
        int(scan_spec.get("min_fit_points", 0) or 0),
    )
    scan_spec["min_fit_points"] = min_fit_points
    if not _scan_has_all_range_keys(scan_spec):
        scan_spec = _auto_scheme_scan(
            coord=np.asarray(matrix_element["coord"], dtype=float),
            re_samples=np.asarray(matrix_element["re_samples"], dtype=float),
            im_samples=np.asarray(matrix_element["im_samples"], dtype=float),
            coord_unit=coord_unit,
            method=method,
            order=order,
            observable=observable,
            pz_gev=pz_gev,
            pz_prime_gev=pz_prime_gev,
            a_fm=a_fm,
            resample_mode=resample_mode,
            Lambda0=float(Lambda0),
            min_fit_points=min_fit_points,
            existing=scan_spec,
        )
        auto_scheme_scan = scan_spec
    else:
        scan_spec = _fill_scheme_defaults(scan_spec)
    scheme_scan = scan_spec
    schemes = _generate_scan_schemes(scheme_scan, coord=np.asarray(matrix_element["coord"], dtype=float))
    k_values = _resolve_k_grid(k_grid)
    result = run_fourier_workflow(
        matrix_element["coord"],
        matrix_element["re_samples"],
        matrix_element["im_samples"],
        k_values,
        schemes=schemes,
        method=method,
        order=order,
        observable=observable,
        coord_unit=coord_unit,
        pz_gev=pz_gev,
        pz_prime_gev=pz_prime_gev,
        a_fm=a_fm,
        im_flip_for_ft=im_flip_for_ft,
        resample_mode=resample_mode,
        Lambda0=float(Lambda0),
        min_fit_points=min_fit_points,
        posterior_prior_error_scale=float(posterior_prior_error_scale),
        fit_error_mode=fit_error_mode,
    )
    result["resample_mode"] = resample_mode
    result["pz_gev"] = pz_gev
    result["pz_prime_gev"] = pz_prime_gev
    result["a_fm"] = a_fm
    result["Lambda0"] = float(Lambda0)
    result["posterior_prior_error_scale"] = float(posterior_prior_error_scale)
    result["fit_error_mode"] = str(fit_error_mode)
    if auto_scheme_scan is not None:
        result["auto_scheme_scan"] = auto_scheme_scan
    model_average = bool((scheme_scan or {}).get("model_average", True)) if scheme_scan is not None else True
    if model_average:
        _apply_scheme_model_average(
            result,
            coord=np.asarray(matrix_element["coord"], dtype=float),
            re_samples=np.asarray(matrix_element["re_samples"], dtype=float),
            im_samples=np.asarray(matrix_element["im_samples"], dtype=float),
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            a_fm=a_fm,
            y_range=(scheme_scan or {}).get("y_range"),
            roughness_weight=float((scheme_scan or {}).get("roughness_weight", 5.0)),
            resample_mode=resample_mode,
        )
    store["fourier_result_data"] = fourier_result_to_ensemble_data(result)
    store[out] = result
    artifact = _artifact_path(save_path, default_name=f"{out}.npz")
    fit_info_artifact = _artifact_path(None, default_name="fourier_fit_info.npz")
    _save_fourier_npz(artifact, result)
    _save_fourier_fit_info_npz(fit_info_artifact, result)
    result["fit_info_artifact"] = str(fit_info_artifact)
    return {
        "out": out,
        "artifact": str(artifact),
        "fit_info_artifact": str(fit_info_artifact),
        "n_schemes": int(result["ft_re_samples"].shape[0]),
        "n_samples": int(result["ft_re_samples"].shape[1]),
        "n_k": int(result["ft_re_samples"].shape[2]),
        "scheme_labels": result["scheme_labels"],
        "fit_failures": result["fit_failures"],
        "best_scheme_index": result.get("best_scheme_index"),
        "best_scheme_label": result.get("best_scheme_label"),
        "auto_scheme_scan": auto_scheme_scan,
    }


def summarize_fourier_result(
    store: dict[str, Any],
) -> dict[str, Any]:
    """Store and return a compact numerical summary of the Fourier result."""
    out = "fourier_summary"
    data = store["fourier_result"]
    summary = {
        "k_grid": np.asarray(data["k_grid"]).tolist(),
        "ft_re_mean": np.asarray(data["ft_re_mean"]).tolist(),
        "ft_im_mean": np.asarray(data["ft_im_mean"]).tolist(),
        "ft_re_stat_sdev": np.asarray(data["ft_re_stat_sdev"]).tolist(),
        "ft_im_stat_sdev": np.asarray(data["ft_im_stat_sdev"]).tolist(),
        "ft_re_sys_sdev": np.asarray(data["ft_re_sys_sdev"]).tolist(),
        "ft_im_sys_sdev": np.asarray(data["ft_im_sys_sdev"]).tolist(),
        "scheme_labels": list(data["scheme_labels"]),
        "fit_failures": list(data["fit_failures"]),
        "scheme_weights": list(data.get("scheme_weights", [])),
        "scheme_fit_chi2_dof": list(data.get("scheme_fit_chi2_dof", [])),
        "scheme_roughness": list(data.get("scheme_roughness", [])),
        "scheme_scores": list(data.get("scheme_scores", [])),
        "best_scheme_index": data.get("best_scheme_index"),
        "best_scheme_label": data.get("best_scheme_label"),
        "fit_info_artifact": data.get("fit_info_artifact"),
    }
    store[out] = summary
    return {"out": out, **summary}


def plot_fourier_result(
    store: dict[str, Any],
    *,
    npz_path: str | None = None,
    save_path: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot the Fourier-stage NPZ artifact and store the figure path."""
    source = npz_path
    if source is None:
        source = str(_artifact_path(None, default_name="fourier_result.npz"))
    output = _artifact_path(save_path, default_name="fourier_result.pdf")
    if title is not None and title.strip().lower() in {"fourier result", "fourier transform"}:
        title = None
    fig, _ = plot_fourier_npz(source, save_path=output, title=title)
    fig.clf()
    return {"plot": str(output), "source": str(source)}


def plot_fourier_extension_quality_result(
    store: dict[str, Any],
    *,
    scheme_index: int | None = None,
    save_path: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot data and smoothed real-part extension for one Fourier scheme."""
    matrix_element = ensemble_data_to_legacy_arrays(store["matrix_element_data"])
    data = store["fourier_result"]
    if scheme_index is None:
        scheme_index = int(data.get("best_scheme_index", 0) or 0)
    if title is not None and title.strip().lower() in {"fourier extension quality", "lambda extrapolation"}:
        title = None
    re_output = _artifact_path(save_path, default_name="fourier_extension_re.pdf")
    im_output = _artifact_path(None, default_name="fourier_extension_im.pdf")
    fig, _ = plot_fourier_extension_quality(
        matrix_element["coord"],
        matrix_element["re_samples"],
        data,
        scheme_index=scheme_index,
        component="re",
        pz_gev=data.get("pz_gev"),
        a_fm=data.get("a_fm"),
        save_path=re_output,
        title=title,
    )
    fig.clf()
    fig, _ = plot_fourier_extension_quality(
        matrix_element["coord"],
        matrix_element["im_samples"],
        data,
        scheme_index=scheme_index,
        component="im",
        pz_gev=data.get("pz_gev"),
        a_fm=data.get("a_fm"),
        save_path=im_output,
        title=title,
    )
    fig.clf()
    scheme_label = data["scheme_labels"][scheme_index]
    return {"plot_re": str(re_output), "plot_im": str(im_output), "scheme_label": scheme_label}


STAGE_TOOLS = {
    "load_renormalized_matrix_element_samples": load_renormalized_matrix_element_samples,
    "run_fourier_transform": run_fourier_transform,
    "summarize_fourier_result": summarize_fourier_result,
    "plot_fourier_result": plot_fourier_result,
    "plot_fourier_extension_quality_result": plot_fourier_extension_quality_result,
}
