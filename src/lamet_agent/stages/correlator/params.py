"""Manifest parameter contract for correlator analysis."""

from lamet_agent.manifest_params import ListItems


MANIFEST_PARAM_SCHEMA = {
    "component": None,
    "correlator_rescale": None,
    "final_momentum": None,
    "fit_scope": None,
    "fit_strategy": None,
    "fitting_form": None,
    "initial_momentum": None,
    "model_average": None,
    "momentum": None,
    "nstate": None,
    "posterior_prior_error_scale": None,
    "prior_width": None,
    "pt2_windows": ListItems({"tmin": None, "tmax": None}),
    "pt3_tau_cuts": None,
    "pt3_windows": ListItems({"tau_cut": None, "tsep_ls": None}),
    "q_min": None,
    "svdcut": None,
}

REMOVED_MANIFEST_PARAMS: dict[str, str] = {}
