"""Manifest parameter contract for renormalization."""


MANIFEST_PARAM_SCHEMA = {
    "b0": None,
    "cf": None,
    "d": None,
    "k": None,
    "kernel_id": None,
    "lqcd": None,
    "m0_gev": None,
    "mu": None,
    "normalization": None,
    "scheme": None,
    "scheme_parameters": {
        "delta_m_gev": None,
        "m0_gev": None,
    },
    "svdcut": None,
    "zms_kind": None,
    "zs_fm": None,
}

REMOVED_MANIFEST_PARAMS = {
    "scheme_parameters.zs_fm": (
        "is no longer supported; use flat stages.renormalization.defaults.zs_fm "
        "or the corresponding jobs[].params.zs_fm."
    ),
}
