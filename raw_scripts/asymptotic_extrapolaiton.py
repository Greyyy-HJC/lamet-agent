import numpy as np
import lsqfit as lsf
import gvar as gv

import logging
my_logger = logging.getLogger("my_logger")

def exp_asym_prior():
    priors = gv.BufferDict()
    priors["a"] = gv.gvar(1, 10) # gv.gvar(0, 1) # gv.gvar(0, 10)
    priors["b"] = gv.gvar(0, 10) # gv.gvar(0, 1) # gv.gvar(0, 10)
    priors["c"] = gv.gvar(0, 10) # gv.gvar(0, 1) # gv.gvar(0, 10)
    priors["d"] = gv.gvar(0, 10) # gv.gvar(0, 1) # gv.gvar(0, 10)
    priors["e"] = gv.gvar(0, 10) # gv.gvar(0, 1) # gv.gvar(0, 10)
    priors["log(n)"] = gv.gvar(0.7, 1) # gv.gvar(0.7, 1)
    priors["log(m)"] = gv.gvar(-2, 2) # gv.gvar(-2.4, 10)
    
    return priors

def exp_asym_re_fcn(m0=0):
    """
    Asymptotic form from https://arxiv.org/pdf/2601.12189, Eq.(2.4), the 1 / lambda^n term is for CG cases.

    Args:
        m0 (int, optional): minimum value of meff, usually set to 0.1 GeV / P_mom.
    """
    def fcn(lam_ls, p):
        return ( p["b"] * np.cos(p["c"]) + p["d"] * np.cos(p["e"]) / abs(lam_ls) ) * np.exp( -lam_ls * (p["m"] + m0) ) / ( lam_ls**p["n"] )
    
    return fcn


def exp_asym_im_fcn(m0=0):
    """
    Asymptotic form from https://arxiv.org/pdf/2601.12189, Eq.(2.4), the 1 / lambda^n term is for CG cases.

    Args:
        m0 (int, optional): minimum value of meff, usually set to 0.1 GeV / P_mom.
    """
    def fcn(lam_ls, p):
        return ( p["b"] * np.sin(p["c"]) + p["d"] * np.sin(p["e"]) / abs(lam_ls) ) * np.exp( -lam_ls * (p["m"] + m0) ) / ( lam_ls**p["n"] )
    
    return fcn



def extrapolate_asymptotic_proton_pdf_joint(lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length=200, weight_ini=0, m0=0):
    """
    Fit real and imag parts jointly and extrapolate the quasi distribution at large lambda
    using asymptotic form.

    check https://arxiv.org/pdf/2601.12189, Eq.(2.4)
    """
    exp_fcn_re = exp_asym_re_fcn(m0)
    exp_fcn_im = exp_asym_im_fcn(m0)

    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda

    lam_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    lam_dic = {"re": np.array(lam_fit), "im": np.array(lam_fit)}
    pdf_dic = {
        "re": re_gv[fit_idx_range[0] : fit_idx_range[1]],
        "im": im_gv[fit_idx_range[0] : fit_idx_range[1]],
    }

    priors = exp_asym_prior()

    def fcn(x, p):
        val = {}
        val["re"] = exp_fcn_re(x["re"], p)
        val["im"] = exp_fcn_im(x["im"], p)
        return val

    fit_result = lsf.nonlinear_fit(
        data=(lam_dic, pdf_dic),
        prior=priors,
        fcn=fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    if fit_result.Q < 0.05:
        my_logger.warning(
            f">>> Bad joint extrapolation fit with Q = {fit_result.Q:.3f}, Chi2/dof = {fit_result.chi2/fit_result.dof:.3f}, meff = {fit_result.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good joint extrapolation fit with Q = {fit_result.Q:.3f}, Chi2/dof = {fit_result.chi2/fit_result.dof:.3f}, meff = {fit_result.p['m']:.3f}"
        )

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    lam_dic_read = {"re": lam_ls_part2, "im": lam_ls_part2}
    fit_val_part2 = fcn(lam_dic_read, fit_result.p)
    re_gv_part2 = fit_val_part2["re"]
    im_gv_part2 = fit_val_part2["im"]

    # *: standardize the way to do the extrapolation
    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    extrapolated_re_gv = list(re_gv_part1) + list(re_gv_part2)
    extrapolated_im_gv = list(im_gv_part1) + list(im_gv_part2)

    # *: Smooth the connection point
    # Define the number of points for gradual weighting
    num_gradual_points = fit_idx_range[1] - fit_idx_range[0]

    # Calculate weights for gradual transition
    weights = np.linspace(weight_ini, 1, num_gradual_points)

    # Prepare lists for the weighted averages
    weighted_re = []
    weighted_im = []

    for i in range(num_gradual_points):
        w = weights[i]
        weighted_re.append(w * re_gv_part2[i] + (1 - w) * re_gv[fit_idx_range[0] + i])
        weighted_im.append(w * im_gv_part2[i] + (1 - w) * im_gv[fit_idx_range[0] + i])

    # Combine the parts
    extrapolated_re_gv = list(re_gv_part1) + weighted_re + list(re_gv_part2[num_gradual_points:])
    extrapolated_im_gv = list(im_gv_part1) + weighted_im + list(im_gv_part2[num_gradual_points:])

    return (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result,
        fit_result,
    )