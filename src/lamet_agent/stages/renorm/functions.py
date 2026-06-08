"""Renormalization stage tools (self-renormalization v1).

Purpose:
- load bare coordinate-space matrix-element bootstrap samples as EnsembleData
- fit a self-renormalization factor zR from zero-momentum reference data

Expected inputs:
- NPZ with ``z`` (fm) and ``samples`` (n_sample x n_z or n_sample x n_a x n_z)
- tool arguments supplied by the agent as JSON-compatible values

Expected outputs:
- ``reference``: bootstrap/jackknife EnsembleData on ``z`` or ``(a, z)``
- ``zR``: gvar EnsembleData on ``(a, z)``

Example usage:
- from lamet_agent.stages.renorm.functions import STAGE_TOOLS
- store = {}
- STAGE_TOOLS["load_bare_matrix_element"](store, path="reference.npz", a=0.0574)
- STAGE_TOOLS["fit_self_renormalization_factor"](store)
"""

from __future__ import annotations

from typing import Any, Literal

import gvar as gv
import lsqfit as lsf
import numpy as np

from lamet_agent.core.data import EnsembleData, EnsembleInfo

GEV_FM = 0.1973269631


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
    "load_bare_matrix_element": load_bare_matrix_element,
    "fit_self_renormalization_factor": fit_self_renormalization_factor,
}
