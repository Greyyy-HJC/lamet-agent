"""Recursive contracts for user-authored stage manifest parameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

ParamSchema = dict[str, Any]


@dataclass(frozen=True)
class ListItems:
    """Apply a nested parameter schema to mapping items in a list."""

    schema: ParamSchema


@dataclass(frozen=True)
class StageParamContract:
    """Allowed parameter shape and path-specific migration messages for a stage."""

    schema: ParamSchema
    removed: dict[str, str]


_GRID_SCHEMA = {"num": None, "start": None, "step": None, "stop": None}


def merge_stage_params(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge job parameter overrides onto stage defaults.

    Nested mappings merge recursively; all other values, including lists, are
    replaced as complete values by the job override.
    """
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_stage_params(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


STAGE_PARAM_CONTRACTS = {
    "correlator_analysis": StageParamContract(
        schema={
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
            "tune_z": None,
        },
        removed={},
    ),
    "renormalization": StageParamContract(
        schema={
            "ensemble": None,
            "kernel_id": None,
            "mu": None,
            "normalization": None,
            "scheme": None,
            "strategy": None,
            "scheme_parameters": {
                "LambdaQCD_gev": None,
                "d": None,
                "delta_m_gev": None,
                "m0_gev": None,
                "svdcut": None,
                "z_coverage_policy": None,
            },
            "zs_fm": None,
        },
        removed={
            "LambdaQCD": (
                "was renamed; use scheme_parameters.LambdaQCD_gev and specify the value explicitly."
            ),
            "d": "belongs to strategy='self_renormalization'; use scheme_parameters.d.",
            "m0_gev": (
                "is scheme/strategy-specific; use scheme_parameters.m0_gev."
            ),
            "svdcut": (
                "belongs to strategy='self_renormalization'; use scheme_parameters.svdcut."
            ),
            "z_coverage_policy": (
                "belongs to strategy='self_renormalization'; use scheme_parameters.z_coverage_policy."
            ),
            "alpha_s": "is derived from mu by alphas_nloop and cannot be specified.",
            "Nf": "is not configurable for renormalization; self-renormalization uses alphas_nloop(mu).",
            "order": "is not configurable for renormalization; self-renormalization uses alphas_nloop(mu).",
            "b0": "is an internal hybrid-self-renormalization ansatz constant and cannot be overridden.",
            "cf": "is an internal hybrid-self-renormalization ansatz constant and cannot be overridden.",
            "f1_extension_zmin_fm": (
                "is no longer supported; apply-time extension is automatic with "
                "z_coverage_policy='extrapolate'."
            ),
            "k": "is an internal hybrid-self-renormalization ansatz constant and cannot be overridden.",
            "lqcd": "was renamed; use scheme_parameters.LambdaQCD_gev and specify the value explicitly.",
            "scheme_parameters.zs_fm": (
                "is no longer supported; use flat stages.renormalization.defaults.zs_fm "
                "or the corresponding jobs[].params.zs_fm."
            ),
            "zms_kind": "is no longer supported; select a declared ZMSbar_pdf or ZMSbar_da kernel_id.",
            "zr_zmax_fm": (
                "is no longer supported; the target grid determines automatic apply-time extension with "
                "z_coverage_policy='extrapolate'."
            ),
        },
    ),
    "fourier_transform": StageParamContract(
        schema={
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
            "symmetry_guarantee": None,
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
            "zmin_shift": None,
            "y_grid": _GRID_SCHEMA,
        },
        removed={"Lambda0": "is no longer supported; use Lambda0_gev."},
    ),
    "perturbative_matching": StageParamContract(
        schema={
            "component": None,
            "endpoint_cut": None,
            "kernel_id": None,
            "lc_x_ls": _GRID_SCHEMA,
            "mu": None,
            "plot": {"xlim": None, "ylim": None},
            "quasi_y_ls": _GRID_SCHEMA,
            "r": None,
            "scheme": None,
            "sector": None,
            "xlim": None,
            "ylim": None,
            "zs_fm": None,
        },
        removed={},
    ),
    "extrapolation": StageParamContract(
        schema={
            "allow_order_a": None,
            "allow_order_1overp": None,
            "allow_order_ap": None,
            "allow_order_a_sym": None,
            "allow_order_1overp_sym": None,
            "allow_order_ap_sym": None,
            "fitting_param_xdep": None,
            "posterior_prior_error_scale": None,
            "pdep_gev": None,
            "sample_error_mode": None,
            "workers": None,
        },
        removed={
            "lattice_spacing_allow_order": "was replaced by allow_order_a, for example [2].",
            "momentum_allow_order": "was replaced by allow_order_1overp, for example [2] or [2, 4].",
        },
    ),
    "review": StageParamContract(schema={"literature": None, "literature_max_papers": None}, removed={}),
}


_DERIVED_KINEMATICS_MESSAGE = (
    "is runner-derived from upstream discrete momentum, volume, and lattice_spacing_fm; "
    "remove it from stage defaults/params. For a partial run, declare momentum, volume, "
    "and lattice_spacing_fm on inputs.artifacts[]."
)
_COMMON_PARAMETER_MESSAGES = {
    key: _DERIVED_KINEMATICS_MESSAGE
    for key in (
        "a_fm",
        "bz_direction",
        "final_momentum",
        "final_momentum_gev",
        "initial_momentum",
        "initial_momentum_gev",
        "lattice_spacing_fm",
        "momentum",
        "momentum_gev",
        "pz_gev",
        "pz_out_gev",
        "volume",
    )
}
_COMMON_PARAMETER_MESSAGES.update(
    {
        "bin_size": "is run-wide; use metadata.bin_size.",
        "bs_samples": "is run-wide; use metadata.bs_samples.",
        "n_boot": "is run-wide; use metadata.bs_samples when metadata.resample_mode is 'bs'.",
        "random_seed": "is run-wide; use metadata.random_seed.",
        "resample_mode": "is run-wide; use metadata.resample_mode.",
        "sample_error_mode": "is run-wide; use metadata.sample_error_mode.",
        "seed": "is run-wide; use metadata.random_seed.",
        "workers": "is run-wide; use metadata.workers.",
    }
)


def _contract_for_stage(stage: str) -> tuple[ParamSchema, dict[str, str]]:
    contract = STAGE_PARAM_CONTRACTS.get(stage)
    if contract is None:
        raise ValueError(f"Stage {stage!r} must be registered in STAGE_PARAM_CONTRACTS.")
    return contract.schema, contract.removed


def _unknown_parameter_message(
    *,
    key: str,
    relative_path: str,
    full_path: str,
    schema: ParamSchema,
    removed: dict[str, str],
) -> str:
    migration = removed.get(relative_path) or removed.get(key) or _COMMON_PARAMETER_MESSAGES.get(key)
    if migration:
        return f"{full_path} {migration}"
    candidates = [candidate for candidate in schema if candidate != key]
    matches = get_close_matches(key, candidates, n=1, cutoff=0.72)
    suggestion = f"; did you mean {matches[0]!r}?" if matches else ""
    return f"{full_path} is not a supported stage parameter{suggestion}"


def _collect_parameter_issues(
    value: dict[str, Any],
    schema: ParamSchema,
    *,
    full_path: str,
    relative_path: str,
    removed: dict[str, str],
) -> list[str]:
    issues: list[str] = []
    for key, item in value.items():
        item_path = f"{full_path}.{key}"
        item_relative_path = f"{relative_path}.{key}" if relative_path else key
        if key not in schema:
            issues.append(
                _unknown_parameter_message(
                    key=key,
                    relative_path=item_relative_path,
                    full_path=item_path,
                    schema=schema,
                    removed=removed,
                )
            )
            continue
        child_schema = schema[key]
        if isinstance(child_schema, dict) and isinstance(item, dict):
            issues.extend(
                _collect_parameter_issues(
                    item,
                    child_schema,
                    full_path=item_path,
                    relative_path=item_relative_path,
                    removed=removed,
                )
            )
        elif isinstance(child_schema, ListItems) and isinstance(item, list):
            for index, child in enumerate(item):
                if not isinstance(child, dict):
                    continue
                issues.extend(
                    _collect_parameter_issues(
                        child,
                        child_schema.schema,
                        full_path=f"{item_path}[{index}]",
                        relative_path=f"{item_relative_path}[]",
                        removed=removed,
                    )
                )
    return issues


def validate_stage_parameter_mapping(
    stage: str,
    value: dict[str, Any],
    *,
    path: str,
) -> list[str]:
    """Return unknown-key issues for one stage defaults or params mapping."""
    schema, removed = _contract_for_stage(stage)
    return _collect_parameter_issues(
        value,
        schema,
        full_path=path,
        relative_path="",
        removed=removed,
    )
