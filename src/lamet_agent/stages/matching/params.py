"""Manifest parameter contract for perturbative matching."""


_GRID_SCHEMA = {"num": None, "start": None, "step": None, "stop": None}

MANIFEST_PARAM_SCHEMA = {
    "component": None,
    "endpoint_cut": None,
    "kernel_id": None,
    "lc_x_ls": _GRID_SCHEMA,
    "mu": None,
    "plot": {"xlim": None, "ylim": None},
    "quasi_y_ls": _GRID_SCHEMA,
    "sector": None,
    "xlim": None,
    "ylim": None,
    "zs_fm": None,
}

REMOVED_MANIFEST_PARAMS: dict[str, str] = {}
