"""Asymptotic forms based on arXiv:2601.12189."""

from __future__ import annotations

import gvar as gv
import numpy as np


def asym_priors() -> gv.BufferDict:
    """Return broad priors for the asymptotic-form parameters."""
    priors = gv.BufferDict()

    priors["A2"] = gv.gvar(1, 10)
    priors["phi2"] = gv.gvar(0, np.pi)
    priors["A2p"] = gv.gvar(1, 10)
    priors["phi2p"] = gv.gvar(0, np.pi)
    priors["log(n)"] = gv.gvar(0.7, 1)
    priors["log(m)"] = gv.gvar(-2, 2)

    return priors


def nucleon_gi_qpdf_la_re(m0: float = 0.0):
    """Return the leading asymptotic GI qPDF fit function."""

    def fcn(lam_array, p):
        return p["A2"] * np.cos(p["phi2"]) * np.exp(-lam_array * (p["m"] + m0))

    return fcn

def nucleon_gi_qpdf_la_im(m0: float = 0.0):
    """Return the leading asymptotic GI qPDF fit function."""

    def fcn(lam_array, p):
        return p["A2"] * np.sin(p["phi2"]) * np.exp(-lam_array * (p["m"] + m0))

    return fcn


def nucleon_cg_qpdf_la_re(m0: float = 0.0):
    """Return the leading asymptotic CG qPDF fit function."""

    def fcn(lam_array, p):
        return (
            p["A2"]
            * np.cos(p["phi2"])
            * np.exp(-lam_array * (p["m"] + m0))
            / lam_array**p["n"]
        )

    return fcn

def nucleon_cg_qpdf_la_im(m0: float = 0.0):
    """Return the leading asymptotic CG qPDF fit function."""

    def fcn(lam_array, p):
        return (
            p["A2"]
            * np.sin(p["phi2"])
            * np.exp(-lam_array * (p["m"] + m0))
            / lam_array**p["n"]
        )

    return fcn


def nucleon_gi_qpdf_nla_re(m0: float = 0.0):
    """Return the next-to-leading asymptotic GI qPDF fit function."""

    def fcn(lam_array, p):
        numerator = (
            p["A2"] * np.cos(p["phi2"])
            + p["A2p"] * np.cos(p["phi2p"]) / abs(lam_array)
        )
        return numerator * np.exp(-lam_array * (p["m"] + m0))

    return fcn

def nucleon_gi_qpdf_nla_im(m0: float = 0.0):
    """Return the next-to-leading asymptotic GI qPDF fit function."""

    def fcn(lam_array, p):
        numerator = (
            p["A2"] * np.sin(p["phi2"])
            + p["A2p"] * np.sin(p["phi2p"]) / abs(lam_array)
        )
        return numerator * np.exp(-lam_array * (p["m"] + m0))

    return fcn

def nucleon_cg_qpdf_nla_re(m0: float = 0.0):
    """Return the next-to-leading asymptotic CG qPDF fit function."""

    def fcn(lam_array, p):
        numerator = (
            p["A2"] * np.cos(p["phi2"])
            + p["A2p"] * np.cos(p["phi2p"]) / abs(lam_array)
        )
        return numerator * np.exp(-lam_array * (p["m"] + m0)) / lam_array**p["n"]

    return fcn

def nucleon_cg_qpdf_nla_im(m0: float = 0.0):
    """Return the next-to-leading asymptotic CG qPDF fit function."""

    def fcn(lam_array, p):
        numerator = (
            p["A2"] * np.sin(p["phi2"])
            + p["A2p"] * np.sin(p["phi2p"]) / abs(lam_array)
        )
        return numerator * np.exp(-lam_array * (p["m"] + m0)) / lam_array**p["n"]

    return fcn