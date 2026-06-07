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

from lamet_agent.core.plotting import plot_fourier_extension_quality, plot_fourier_npz
from lamet_agent.stages.fourier.workflow import run_fourier_workflow


def load_renormalized_matrix_element_samples(
    store: dict[str, Any],
    *,
    path: str,
    input_format: str | None = None,
    h5_group: str | None = None,
    h5_pz: int | str | None = None,
    coord_key: str = "coord",
    re_key: str = "re_samples",
    im_key: str = "im_samples",
    out: str = "matrix_element",
) -> dict[str, Any]:
    """Load renormalized coordinate-space matrix-element samples from NPZ or HDF5."""
    fmt = _resolve_input_format(path, input_format)
    if fmt == "npz":
        coord, re_samples, im_samples, group_name = _load_npz_matrix_element(path, coord_key, re_key, im_key)
    elif fmt == "h5":
        coord, re_samples, im_samples, group_name = _load_h5_matrix_element(
            path,
            h5_group=h5_group,
            h5_pz=h5_pz,
            coord_key="z_ary" if coord_key == "coord" else coord_key,
            re_key="Re" if re_key == "re_samples" else re_key,
            im_key="Im" if im_key == "im_samples" else im_key,
        )
    else:
        raise ValueError("input_format must be 'npz', 'h5', or 'hdf5'")
    store[out] = {
        "coord": coord,
        "re_samples": re_samples,
        "im_samples": im_samples,
        "path": str(path),
        "input_format": fmt,
    }
    if group_name is not None:
        store[out]["h5_group"] = group_name
    return {
        "out": out,
        "input_format": fmt,
        "h5_group": group_name,
        "n_coord": int(len(coord)),
        "re_shape": list(re_samples.shape),
        "im_shape": list(im_samples.shape),
    }


def load_matrix_element_samples(
    store: dict[str, Any],
    *,
    path: str,
    input_format: str | None = None,
    h5_group: str | None = None,
    h5_pz: int | str | None = None,
    coord_key: str = "coord",
    re_key: str = "re_samples",
    im_key: str = "im_samples",
    out: str = "matrix_element",
) -> dict[str, Any]:
    """Compatibility alias for load_renormalized_matrix_element_samples."""
    return load_renormalized_matrix_element_samples(
        store,
        path=path,
        input_format=input_format,
        h5_group=h5_group,
        h5_pz=h5_pz,
        coord_key=coord_key,
        re_key=re_key,
        im_key=im_key,
        out=out,
    )


def _resolve_input_format(path: str, input_format: str | None) -> str:
    if input_format is not None:
        fmt = input_format.lower()
    else:
        suffix = Path(path).suffix.lower()
        fmt = "h5" if suffix in {".h5", ".hdf5"} else suffix.lstrip(".")
    if fmt == "hdf5":
        fmt = "h5"
    return fmt


def _load_npz_matrix_element(
    path: str,
    coord_key: str,
    re_key: str,
    im_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    data = np.load(path)
    return (
        np.asarray(data[coord_key], dtype=float),
        np.asarray(data[re_key], dtype=float),
        np.asarray(data[im_key], dtype=float),
        None,
    )


def _infer_h5_group(path: str, h5_pz: int | str | None, group_names: list[str]) -> str:
    if h5_pz is not None:
        group = f"Pz={h5_pz}"
        if group not in group_names:
            raise ValueError(f"HDF5 group {group!r} not found; available groups: {group_names}")
        return group

    match = re.search(r"(?:^|_)pz([+-]?\d+)(?:\.|_|$)", Path(path).name, flags=re.IGNORECASE)
    if match:
        group = f"Pz={match.group(1)}"
        if group in group_names:
            return group

    if len(group_names) == 1:
        return group_names[0]
    raise ValueError("h5_group is required when the HDF5 file has multiple groups and no pz can be inferred")


def _load_h5_matrix_element(
    path: str,
    *,
    h5_group: str | None,
    h5_pz: int | str | None,
    coord_key: str,
    re_key: str,
    im_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Reading HDF5 Fourier inputs requires installing lamet-agent with the analysis extra") from exc

    with h5py.File(path, "r") as h5f:
        group_names = [name for name, item in h5f.items() if isinstance(item, h5py.Group)]
        group_name = h5_group or _infer_h5_group(path, h5_pz, group_names)
        if group_name not in h5f:
            raise ValueError(f"HDF5 group {group_name!r} not found; available groups: {group_names}")
        group = h5f[group_name]
        coord = np.asarray(group[coord_key], dtype=float)
        re_samples = np.asarray(group[re_key], dtype=float)
        im_samples = np.asarray(group[im_key], dtype=float)
    return coord, re_samples, im_samples, group_name


def _artifact_path(raw: str | None, *, default_name: str) -> Path:
    out_dir = Path.cwd() / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw:
        return out_dir / Path(raw).name
    return out_dir / default_name


def _save_fourier_npz(path: Path, result: dict[str, Any]) -> None:
    np.savez(
        path,
        k_grid=result["k_grid"],
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

    fit_chi2_dof = fit_chi2 / np.maximum(fit_dof, 1)
    if fit_params.shape[1] < 2:
        fit_param_sdev = np.zeros((fit_params.shape[0], fit_params.shape[2]), dtype=float)
    else:
        fit_param_sdev = np.std(fit_params, axis=1, ddof=1)

    np.savez(
        path,
        scheme_labels=np.asarray(result["scheme_labels"]),
        fit_param_labels=np.asarray(schemes[0]["fit_param_labels"]),
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


def _generate_scan_schemes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    zmin_values = _scan_values(spec, "zmin")
    zmax_values = _scan_values(spec, "zmax")
    min_width = float(spec.get("min_width", 0.0))
    z_ext_max = float(spec["z_ext_max"])
    smooth = str(spec.get("smooth", "linear"))
    max_schemes = int(spec.get("max_schemes", 200))

    schemes = []
    for zmin in zmin_values:
        for zmax in zmax_values:
            if zmax <= zmin or zmax - zmin < min_width:
                continue
            scheme = {
                "label": f"zmin_{zmin:g}_zmax_{zmax:g}".replace(".", "p"),
                "zmin": zmin,
                "zmax": zmax,
                "z_ext_max": z_ext_max,
                "smooth": smooth,
            }
            if "blend_start" in spec:
                scheme["blend_start"] = float(spec["blend_start"])
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


def _samples_axis_zero(values: np.ndarray, sample_axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if sample_axis in (1, -1):
        return np.moveaxis(arr, sample_axis, 0)
    if sample_axis == 0:
        return arr
    raise ValueError("sample_axis must be 0 or 1")


def _fit_scale(coord_unit: str, *, pz_gev: float | None, a_fm: float | None) -> float:
    unit = coord_unit.lower()
    fm_to_gev_inv = 5.067731237
    if unit == "lambda":
        return 1.0
    if unit == "gev_inv":
        return 1.0
    if unit == "fm":
        return fm_to_gev_inv
    if unit == "lattice":
        if a_fm is None:
            raise ValueError("a_fm is required when coord_unit='lattice'")
        return float(a_fm) * fm_to_gev_inv
    raise ValueError("coord_unit must be 'lambda', 'gev_inv', 'fm', or 'lattice'")


def _sample_sdev(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] < 2:
        return np.zeros(samples.shape[1], dtype=float)
    return np.std(samples, axis=0, ddof=1)


def _scheme_fit_chi2_dof(
    *,
    coord: np.ndarray,
    re_samples: np.ndarray,
    im_samples: np.ndarray,
    scheme_result: dict[str, Any],
    method: str,
    order: str,
    coord_unit: str,
    pz_gev: float | None,
    a_fm: float | None,
) -> float:
    scale = _fit_scale(coord_unit, pz_gev=pz_gev, a_fm=a_fm)
    fit_coord = coord * scale
    zmin, zmax = scheme_result["fit_range"]
    fit_mask = (coord >= float(zmin)) & (coord <= float(zmax)) & (coord > 0)
    if np.count_nonzero(fit_mask) == 0:
        return float("inf")

    target_z = fit_coord[fit_mask]
    data_re = np.mean(re_samples[:, fit_mask], axis=0)
    data_im = np.mean(im_samples[:, fit_mask], axis=0)
    sigma_re = _sample_sdev(re_samples[:, fit_mask])
    sigma_im = _sample_sdev(im_samples[:, fit_mask])
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
    method: str,
    order: str,
    coord_unit: str,
    pz_gev: float | None,
    a_fm: float | None,
    y_range: list[float] | tuple[float, float] | None,
    roughness_weight: float,
) -> None:
    ft_re = np.asarray(result["ft_re_samples"], dtype=float)
    ft_im = np.asarray(result["ft_im_samples"], dtype=float)
    k_grid = np.asarray(result["k_grid"], dtype=float)
    re_mean_by_scheme = np.mean(ft_re, axis=1)
    im_mean_by_scheme = np.mean(ft_im, axis=1)
    re_stat_by_scheme = np.asarray([_sample_sdev(item) for item in ft_re])
    im_stat_by_scheme = np.asarray([_sample_sdev(item) for item in ft_im])

    fit_chi2 = []
    roughness = []
    for idx, scheme_result in enumerate(result["scheme_results"]):
        fit_chi2.append(
            _scheme_fit_chi2_dof(
                coord=coord,
                re_samples=re_samples,
                im_samples=im_samples,
                scheme_result=scheme_result,
                method=method,
                order=order,
                coord_unit=coord_unit,
                pz_gev=pz_gev,
                a_fm=a_fm,
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
    samples: str = "matrix_element",
    k_grid: list[float] | dict[str, Any],
    schemes: list[dict[str, Any]] | None = None,
    scheme_scan: dict[str, Any] | None = None,
    method: str = "GI",
    order: str = "NLA",
    observable: str = "nucleon_quark_transversity_quasi_pdf",
    coord_unit: str = "lambda",
    pz_gev: float | None = None,
    pz_prime_gev: float | None = None,
    a_fm: float | None = None,
    sample_axis: int = 0,
    im_flip_for_ft: bool = False,
    save_path: str | None = None,
    out: str = "fourier_result",
) -> dict[str, Any]:
    """Run local extrapolation and Fourier transform for loaded samples."""
    matrix_element = store[samples]
    if scheme_scan is not None:
        schemes = _generate_scan_schemes(scheme_scan)
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
        sample_axis=sample_axis,
        im_flip_for_ft=im_flip_for_ft,
    )
    result["pz_gev"] = pz_gev
    result["pz_prime_gev"] = pz_prime_gev
    result["a_fm"] = a_fm
    result["sample_axis"] = sample_axis
    model_average = bool((scheme_scan or {}).get("model_average", True)) if scheme_scan is not None else True
    if model_average:
        score_re_samples = _samples_axis_zero(matrix_element["re_samples"], sample_axis)
        score_im_samples = _samples_axis_zero(matrix_element["im_samples"], sample_axis)
        _apply_scheme_model_average(
            result,
            coord=np.asarray(matrix_element["coord"], dtype=float),
            re_samples=score_re_samples,
            im_samples=score_im_samples,
            method=method,
            order=order,
            coord_unit=coord_unit,
            pz_gev=pz_gev,
            a_fm=a_fm,
            y_range=(scheme_scan or {}).get("y_range"),
            roughness_weight=float((scheme_scan or {}).get("roughness_weight", 5.0)),
        )
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
    }


def summarize_fourier_result(
    store: dict[str, Any],
    *,
    result: str = "fourier_result",
    out: str = "fourier_summary",
) -> dict[str, Any]:
    """Store and return a compact numerical summary of the Fourier result."""
    data = store[result]
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
    result: str = "fourier_result",
    npz_path: str | None = None,
    save_path: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot the Fourier-stage NPZ artifact and store the figure path."""
    source = npz_path
    if source is None:
        source = str(_artifact_path(None, default_name=f"{result}.npz"))
    output = _artifact_path(save_path, default_name=f"{result}.pdf")
    if title is not None and title.strip().lower() in {"fourier result", "fourier transform"}:
        title = None
    fig, _ = plot_fourier_npz(source, save_path=output, title=title)
    fig.clf()
    return {"plot": str(output), "source": str(source)}


def plot_fourier_extension_quality_result(
    store: dict[str, Any],
    *,
    samples: str = "matrix_element",
    result: str = "fourier_result",
    scheme_index: int | None = None,
    save_path: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot data and smoothed real-part extension for one Fourier scheme."""
    matrix_element = store[samples]
    data = store[result]
    if scheme_index is None:
        scheme_index = int(data.get("best_scheme_index", 0) or 0)
    if title is not None and title.strip().lower() in {"fourier extension quality", "lambda extrapolation"}:
        title = None
    re_samples = _samples_axis_zero(matrix_element["re_samples"], int(data.get("sample_axis", 0)))
    im_samples = _samples_axis_zero(matrix_element["im_samples"], int(data.get("sample_axis", 0)))
    re_output = _artifact_path(save_path, default_name="fourier_extension_re.pdf")
    im_output = _artifact_path(None, default_name="fourier_extension_im.pdf")
    fig, _ = plot_fourier_extension_quality(
        matrix_element["coord"],
        re_samples,
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
        im_samples,
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
    "load_matrix_element_samples": load_matrix_element_samples,
    "run_fourier_transform": run_fourier_transform,
    "summarize_fourier_result": summarize_fourier_result,
    "plot_fourier_result": plot_fourier_result,
    "plot_fourier_extension_quality_result": plot_fourier_extension_quality_result,
}
