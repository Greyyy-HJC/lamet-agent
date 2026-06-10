"""Renormalization stage tools.

Purpose:
- load bare coordinate-space matrix-element bootstrap samples as EnsembleData
- apply sample-preserving ratio/hybrid-scheme renormalization
- fit a self-renormalization factor zR from zero-momentum reference data

Expected inputs:
- correlator-stage bare matrix-element txt grids or report JSON files
- NPZ with ``z`` (fm) and ``samples`` (n_sample x n_z or n_sample x n_a x n_z)
- tool arguments supplied by the agent as JSON-compatible values

Expected outputs:
- renormalized complex EnsembleData on ``z`` for downstream Fourier tools
- ``reference``: bootstrap/jackknife EnsembleData on ``z`` or ``(a, z)``
- ``zR``: gvar EnsembleData on ``(a, z)``

Example usage:
- from lamet_agent.stages.renorm.functions import STAGE_TOOLS
- store = {}
- STAGE_TOOLS["load_bare_matrix_element"](store, path="reference.npz", a=0.0574)
- STAGE_TOOLS["fit_self_renormalization_factor"](store)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt
import numpy as np

from lamet_agent.core.data import EnsembleData, EnsembleInfo
from lamet_agent.core.plotting import COLOR_CYCLE, ERRORBAR_STYLE, FONT_SIZE, LEGEND_SETS, default_plot
from lamet_agent.core.resampling import sample_mean_err
from lamet_agent.core.tools import resolve_plot_save_path

GEV_FM = 0.1973269631



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


def _bare_grid_paths_from_report(report_json: str | Path) -> tuple[list[tuple[float, Path]], dict[str, Any]]:
    report_path = Path(report_json)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = report.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"bare matrix report has no outputs list: {report_json}")
    paths: list[tuple[float, Path]] = []
    for item in outputs:
        if not isinstance(item, dict) or "path" not in item or "z" not in item:
            raise ValueError("each bare matrix report output must contain path and z")
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = (report_path.parent / path).resolve()
        paths.append((float(item["z"]), path))
    return paths, report


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
    return EnsembleData(
        ensemble=None,
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


def load_bare_matrix_element_grid(
    store: dict[str, Any],
    *,
    report_json: str | None = None,
    txt_dir: str | None = None,
    filename_glob: str = "*.txt",
    z_regex: str = r"_z([+-]?\d+(?:\.\d+)?)\.txt$",
    resample: Literal["bootstrap", "jackknife", "raw", "bs", "jk"] | None = None,
    out: str = "bare_matrix_element",
) -> dict[str, Any]:
    """Load correlator-stage bare matrix-element txt grid into complex EnsembleData."""
    if report_json is None and txt_dir is None:
        report = store.get("bare_matrix_grid_report")
        if isinstance(report, dict) and isinstance(report.get("outputs"), list):
            paths = [(float(item["z"]), Path(str(item["path"]))) for item in report["outputs"]]
            metadata = report
        else:
            raise ValueError("provide report_json or txt_dir, or run fit_bare_matrix_grid first")
    elif report_json is not None:
        paths, metadata = _bare_grid_paths_from_report(report_json)
    else:
        assert txt_dir is not None
        paths, metadata = _bare_grid_paths_from_dir(txt_dir, filename_glob=filename_glob, z_regex=z_regex)

    z_values, samples = _load_complex_txt_grid(paths)
    resample_name = _resample_name(resample or str(metadata.get("resample_mode", "bootstrap")))
    data = _matrix_to_ensemble(
        z_values=z_values,
        samples=samples,
        resample=resample_name,
        attrs={
            "source": report_json or txt_dir or "bare_matrix_grid_report",
            "resample_mode": metadata.get("resample_mode", resample_name),
            "ensemble": metadata.get("ensemble"),
            "momentum": metadata.get("momentum"),
        },
        name="bare_matrix_element",
    )
    store[out] = data
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
        "resample": data.resample,
    }


def apply_ratio_scheme_renormalization(
    store: dict[str, Any],
    *,
    target: str = "target_bare_matrix_element",
    denominator: str = "denominator_bare_matrix_element",
    zs: float = 4.0,
    delta_m: float = 0.0,
    m0: float = 0.0,
    z0: float = 0.0,
    out: str = "matrix_element_data",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Apply Eq. 15 ratio/hybrid renormalization and preserve all samples."""
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

    z0_idx = _z_index(z_target, z0, label="normalization")
    zs_idx = _z_index(z_denom, zs, label="long-distance denominator")
    norm = denom_values[:, z0_idx] / target_values[:, z0_idx] 
    exponent = np.exp((float(delta_m) + float(m0)) * (np.abs(z_target) - float(zs)))
    short = norm[:, None] * target_values / denom_values
    long = norm[:, None] * exponent[None, :] * target_values / denom_values[:, zs_idx : zs_idx + 1]
    renorm_values = np.where(np.abs(z_target)[None, :] <= float(zs), short, long)

    attrs = {
        "scheme": "ratio_hybrid_eq15",
        "zs": str(float(zs)),
        "delta_m": str(float(delta_m)),
        "m0": str(float(m0)),
        "z0": str(float(z0)),
        "target": target,
        "denominator": denominator,
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
    store["matrix_element"] = {
        "coord": z_target,
        "re_samples": np.real(renorm_values),
        "im_samples": np.imag(renorm_values),
        "scheme": "ratio_hybrid_eq15",
    }

    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="renormalized_matrix_element")
    artifact = stem.with_suffix(".npz")
    result.save_npz(
        artifact,
        coord=z_target,
        re_samples=np.real(renorm_values),
        im_samples=np.imag(renorm_values),
    )
    return {
        "out": out,
        "data": "matrix_element_data",
        "artifact": str(artifact),
        "n_z": int(len(z_target)),
        "n_sample": int(renorm_values.shape[0]),
        "zs": float(zs),
        "delta_m": float(delta_m),
        "m0": float(m0),
        "normalization_z": float(z0),
    }


def plot_renormalized_matrix_element(
    store: dict[str, Any],
    *,
    data: str = "matrix_element_data",
    save_path: str | None = None,
    artifacts_dir: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot sample-averaged renormalized matrix elements to PDF."""
    matrix = _require_matrix_data(store, data)
    z_values = np.asarray(matrix.coords["z"], dtype=float)
    values = np.asarray(matrix.values, dtype=complex)
    mode = _resample_mode(matrix)
    re_mean: list[float] = []
    re_err: list[float] = []
    im_mean: list[float] = []
    im_err: list[float] = []
    for iz in range(values.shape[1]):
        r_mean, r_sdev = sample_mean_err(np.real(values[:, iz]), mode=mode)
        i_mean, i_sdev = sample_mean_err(np.imag(values[:, iz]), mode=mode)
        re_mean.append(r_mean)
        re_err.append(r_sdev)
        im_mean.append(i_mean)
        im_err.append(i_sdev)

    fig, ax = default_plot()
    ax.errorbar(z_values, re_mean, re_err, label="Re", color=COLOR_CYCLE[0], **ERRORBAR_STYLE)
    ax.errorbar(z_values, im_mean, im_err, label="Im", color=COLOR_CYCLE[1], marker="s", **ERRORBAR_STYLE)
    ax.set_xlabel(r"$z/a$", **FONT_SIZE)
    ax.set_ylabel(r"Renormalized matrix element", **FONT_SIZE)
    ax.set_title(title or "Ratio-scheme renormalized matrix elements", **FONT_SIZE)
    ax.legend(**LEGEND_SETS)
    fig.tight_layout()
    stem = _artifact_stem(save_path, artifacts_dir=artifacts_dir, default_stem="renormalized_matrix_element")
    plot_path = stem.with_suffix(".pdf")
    fig.savefig(plot_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return {
        "plot": str(plot_path),
        "data": data,
        "n_z": int(len(z_values)),
        "n_sample": int(values.shape[0]),
    }


def load_bare_matrix_element(
    store: dict[str, Any],
    *,
    path: str,
    resample: Literal["bootstrap", "jackknife"] = "bootstrap",
    a: float | list[float] | None = None,
    z_key: str = "z",
    samples_key: str = "samples",
    out: str = "reference",
) -> dict[str, Any]:
    """Load bare matrix-element bootstrap samples from NPZ into EnsembleData."""
    data = np.load(path)
    z = np.asarray(data[z_key], dtype=float)
    samples = np.asarray(data[samples_key], dtype=float)
    a_list = [float(a)] if isinstance(a, (int, float)) else [float(x) for x in a]

    if samples.ndim == 2:
        a_s = a_list[0]
        ensemble = EnsembleInfo("", "", a_s, a_s, 96, 96, 0.0)
        values = [samples[i] for i in range(samples.shape[0])]
        reference = EnsembleData(ensemble, resample, values, dims=("z",), coords={"z": z.tolist()})
    else:
        a_s = a_list[0]
        ensemble = EnsembleInfo("", "", a_s, a_s, 96, 96, 0.0)
        values = [samples[i] for i in range(samples.shape[0])]
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
    }


def fit_self_renormalization_factor(
    store: dict[str, Any],
    *,
    reference: str = "reference",
    out: str = "zR",
    normalize_z0: bool = True,
    n_m0: int = 3,
    zms_kind: Literal["pdf", "da"] = "da",
    k: float = 3.320,
    lqcd: float = 0.1,
    mu: float = 2.0,
    d: float = -0.08183,
    cf: float = 4.0 / 3.0,
    b0: float = 11.0 - 2.0 / 3.0 * 3.0,
) -> dict[str, Any]:
    """Fit self-renormalization factor zR from zero-momentum reference EnsembleData."""
    ref: EnsembleData = store[reference]
    z_coords = list(ref.coords["z"])
    if "a" in ref.dims:
        a_coords = list(ref.coords["a"])
    else:
        a_coords = [ref.ensemble.a_s]

    samples = ref.array.values
    z0_idx = int(np.argmin(z_coords))
    if normalize_z0:
        if "a" in ref.dims:
            norm = samples[:, :, z0_idx : z0_idx + 1]
        else:
            norm = samples[:, z0_idx : z0_idx + 1]
        samples = samples / norm
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

    z_x = {"z": [], "x": []}
    lnm: list[Any] = []
    for ia, a_val in enumerate(a_coords):
        x = GEV_FM / a_val
        for iz, z_val in enumerate(z_coords):
            if normalize_z0 and iz == z0_idx:
                continue
            z_x["z"].append(z_val)
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
                + gv.log(1 + d / gv.log(lqcd / xm))
            )
        return out_vals

    gz_fit = lsf.nonlinear_fit(
        data=(z_x, lnm),
        prior=priors,
        fcn=fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    lms = 0.24451721864451428
    alphas = 2 * np.pi / (b0 * np.log(mu / lms))

    def zms(z_arr):
        z_arr = np.asarray(z_arr, dtype=float)
        log_term = np.log(mu**2 * (z_arr / GEV_FM) ** 2 * np.exp(2 * np.euler_gamma) / 4)
        offset = 5.0 / 2.0 if zms_kind == "pdf" else 7.0 / 2.0
        return 1 + alphas * cf / (2 * np.pi) * (3.0 / 2.0 * log_term + offset)

    z_m0 = [z for z in z_coords if z > 0][:n_m0]
    g_m0 = [gz_fit.p[f"g{z}"] for z in z_m0]

    def m0_fcn(x, p):
        return np.log(zms(x)) + p["m0"] * np.array(x) + p["b"]

    m0_prior = gv.BufferDict()
    m0_prior["m0"] = gv.gvar(0, 20)
    m0_prior["b"] = gv.gvar(0, 100)
    m0_fit = lsf.nonlinear_fit(
        data=(z_m0, g_m0),
        prior=m0_prior,
        fcn=m0_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )
    m0 = m0_fit.p["m0"]

    zR_grid = np.empty((len(a_coords), len(z_coords)), dtype=object)
    p = gz_fit.p
    for ia, a_val in enumerate(a_coords):
        xm = GEV_FM / a_val
        for iz, z_val in enumerate(z_coords):
            temp = (
                k * z_val * xm / gv.log(lqcd / xm)
                + p[f"g{z_val}"]
                + p[f"f1{z_val}"] / xm
                + 3 * cf / b0 * gv.log(gv.log(xm / lqcd) / gv.log(mu / lqcd))
                + gv.log(1 + d / gv.log(lqcd / xm))
                - p[f"g{z_val}"]
                + m0 * z_val
            )
            zR_grid[ia, iz] = np.exp(temp)

    zR = EnsembleData(
        ref.ensemble,
        "gvar",
        zR_grid,
        dims=("a", "z"),
        coords={"a": a_coords, "z": z_coords},
        name="zR",
    )
    store[out] = zR
    return {
        "out": out,
        "m0": float(gv.mean(m0)),
        "m0_sdev": float(gv.sdev(m0)),
        "z_values": z_coords,
        "a_values": a_coords,
        "n_z": len(z_coords),
        "n_a": len(a_coords),
    }


STAGE_TOOLS = {
    "load_bare_matrix_element_grid": load_bare_matrix_element_grid,
    "apply_ratio_scheme_renormalization": apply_ratio_scheme_renormalization,
    "plot_renormalized_matrix_element": plot_renormalized_matrix_element,
    "load_bare_matrix_element": load_bare_matrix_element,
    "fit_self_renormalization_factor": fit_self_renormalization_factor,
}
