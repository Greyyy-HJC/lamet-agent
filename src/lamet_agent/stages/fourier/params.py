"""Manifest parameter contract for Fourier transformation."""


_GRID_SCHEMA = {"num": None, "start": None, "step": None, "stop": None}

MANIFEST_PARAM_SCHEMA = {
    "Lambda0_gev": None,
    "component": None,
    "coord_key": None,
    "coord_unit": None,
    "gfix": None,
    "h5_group": None,
    "hadron": None,
    "im_flip_for_ft": None,
    "im_key": None,
    "input_format": None,
    "method": None,
    "observable": None,
    "order": None,
    "output_scale": None,
    "part": None,
    "phase_shift": None,
    "plot_extension": {
        "save_path": None,
        "scheme_index": None,
        "title": None,
    },
    "plot_fourier": {
        "save_path": None,
        "title": None,
    },
    "posterior_prior_error_scale": None,
    "psi1_flavor_class": None,
    "psi2_flavor_class": None,
    "re_key": None,
    "report": {
        "enabled": None,
        "report_language": None,
        "save_path": None,
    },
    "scheme_scan": {
        "max_schemes": None,
        "model_average": None,
        "smooth": None,
        "step": None,
        "z_ext_max": None,
        "zmax_start": None,
        "zmax_step": None,
        "zmax_stop": None,
        "zmax_values": None,
        "zmin_start": None,
        "zmin_step": None,
        "zmin_stop": None,
        "zmin_values": None,
    },
    "sector": None,
    "target_observable": None,
    "y_grid": _GRID_SCHEMA,
}

REMOVED_MANIFEST_PARAMS = {
    "Lambda0": "is no longer supported; use Lambda0_gev.",
}
