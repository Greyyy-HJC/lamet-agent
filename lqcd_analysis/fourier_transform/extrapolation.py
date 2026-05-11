"""Long-distance extrapolation helpers for Fourier-transform workflows."""

from __future__ import annotations

import lsqfit as lsf
import numpy as np

from ..utils.logger import log_nonlinear_fit_quality

from .asymptotic_form import (
    asym_priors,
    nucleon_cg_qpdf_nla_im,
    nucleon_cg_qpdf_nla_re,
)


def weight_linear(num_points: int, weight_ini: float = 0.0) -> np.ndarray:
    """Return linearly increasing transition weights."""
    return np.linspace(weight_ini, 1, num_points)


def weight_logistic(num_points: int, weight_ini: float = 0.0) -> float:
    """Return the logistic transition weight used by legacy notebooks."""
    return 1 / (1 + np.exp(-num_points * (weight_ini - 0.5)))


def weight_exponential(num_points: int, weight_ini: float = 0.0) -> float:
    """Return the exponential transition weight used by legacy notebooks."""
    return np.exp(-num_points * weight_ini)


def extrapolate_nucleon_cg_qpdf_nla(
    lam_ls,
    re_gv,
    im_gv,
    fit_idx_range: tuple[int, int],
    extrapolated_length: float = 200,
    weight_ini: float = 0.0,
    m0: float = 0.0,
    label: str | None = None,
) -> tuple[
    list,
    list,
    list,
    lsf.nonlinear_fit,
    lsf.nonlinear_fit,
]:
    """Extrapolate nucleon CG qPDF data with the next-to-leading asymptotic form."""
    fit_start, fit_stop = fit_idx_range
    lam_gap = abs(lam_ls[1] - lam_ls[0])

    lam_fit = lam_ls[fit_start:fit_stop]
    lam_dic = {"re": np.array(lam_fit), "im": np.array(lam_fit)}
    pdf_dic = {
        "re": re_gv[fit_start:fit_stop],
        "im": im_gv[fit_start:fit_stop],
    }

    priors = asym_priors()
    re_fcn = nucleon_cg_qpdf_nla_re(m0)
    im_fcn = nucleon_cg_qpdf_nla_im(m0)

    def fcn(x, p):
        return {
            "re": re_fcn(x["re"], p),
            "im": im_fcn(x["im"], p),
        }

    fit_result = lsf.nonlinear_fit(
        data=(lam_dic, pdf_dic),
        prior=priors,
        fcn=fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )
    log_nonlinear_fit_quality(fit_result, kind="extrapolation", label=label)

    lam_ls_part1 = lam_ls[:fit_start]
    re_gv_part1 = re_gv[:fit_start]
    im_gv_part1 = im_gv[:fit_start]

    lam_ls_part2 = np.arange(lam_ls[fit_start], extrapolated_length, lam_gap)

    fit_val_part2 = fcn({"re": lam_ls_part2, "im": lam_ls_part2}, fit_result.p)
    re_gv_part2 = fit_val_part2["re"]
    im_gv_part2 = fit_val_part2["im"]

    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)

    num_gradual_points = fit_stop - fit_start
    weights = weight_linear(num_gradual_points, weight_ini)

    weighted_re = []
    weighted_im = []
    for idx, weight in enumerate(weights):
        data_idx = fit_start + idx
        weighted_re.append(
            weight * re_gv_part2[idx] + (1 - weight) * re_gv[data_idx]
        )
        weighted_im.append(
            weight * im_gv_part2[idx] + (1 - weight) * im_gv[data_idx]
        )

    extrapolated_re_gv = (
        list(re_gv_part1) + weighted_re + list(re_gv_part2[num_gradual_points:])
    )
    extrapolated_im_gv = (
        list(im_gv_part1) + weighted_im + list(im_gv_part2[num_gradual_points:])
    )

    return (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result,
        fit_result,
    )
