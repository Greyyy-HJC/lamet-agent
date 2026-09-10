"""Manifest contract for perturbative matching."""

from __future__ import annotations

import inspect
import math
import re
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import numpy as np

from lamet_agent.contract import (
    CheckContext,
    Depends,
    Issue,
    List,
    Provides,
    Recommends,
    Source,
    Suggests,
    Value,
    stage_job_rules,
)


_DATA_KERNEL_ARGUMENTS = frozenset({"x_out", "x_in"})
_CONTEXT_KERNEL_ARGUMENTS = frozenset({"momentum_gev", "scale_gev", "zs_fm"})


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _increasing(values: list[object]) -> bool:
    return all(
        isinstance(left, (int, float)) and isinstance(right, (int, float)) and right > left
        for left, right in zip(values, values[1:])
    )


def _valid_lc_x_ls(value: object) -> bool:
    if isinstance(value, list):
        return bool(value) and _increasing(value)
    if not isinstance(value, dict) or set(value) != {"start", "stop"}:
        return False
    start, stop = value["start"], value["stop"]
    return (
        all(
            isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
            for item in (start, stop)
        )
        and start < stop
    )


def _valid_kernel_id(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value))


def _safe_systematics_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", value))


def _valid_mu_factor(value: int | float) -> bool:
    return _positive(value) and not math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-15)


def _annotation_accepts(annotation: Any, value: Any) -> bool:
    """Return whether one JSON value matches a supported kernel annotation."""
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        return any(_annotation_accepts(candidate, value) for candidate in arguments)
    if origin is Literal:
        return any(type(value) is type(choice) and value == choice for choice in arguments)
    if annotation is np.ndarray:
        return isinstance(value, list) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        )
    if annotation is str:
        return isinstance(value, str)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is list:
        return isinstance(value, list)
    if annotation is dict:
        return isinstance(value, dict)
    if annotation is type(None):
        return value is None
    if origin is list:
        return isinstance(value, list) and all(_annotation_accepts(arguments[0], item) for item in value)
    if origin is dict:
        return (
            isinstance(value, dict)
            and all(isinstance(key, str) for key in value)
            and all(_annotation_accepts(arguments[1], item) for item in value.values())
        )
    return False


def _kernel_parameter_issues(kernel: Any, values: dict[str, Any]) -> list[Issue]:
    """Validate authored parameters directly against one kernel signature."""
    physics = "Kernel parameters must match the selected kernel() signature; stage-owned arguments are implicit."
    signature = inspect.signature(kernel)
    parameters = list(signature.parameters.values())
    if (
        len(parameters) < 4
        or [parameter.name for parameter in parameters[:4]] != ["x_out", "x_in", "momentum_gev", "scale_gev"]
        or any(parameter.kind is not inspect.Parameter.KEYWORD_ONLY for parameter in parameters[2:])
        or any(
            parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for parameter in parameters
        )
    ):
        return [
            Issue(
                "kernel_id",
                "kernel must have signature kernel(x_out, x_in, *, momentum_gev, scale_gev, ...)",
                physics,
            )
        ]

    configurable = {
        parameter.name: parameter for parameter in parameters if parameter.name not in _DATA_KERNEL_ARGUMENTS
    }
    issues = [
        Issue(
            f"kernel_parameters.{name}",
            "is supplied by input/output data and cannot be overridden",
            physics,
        )
        for name in values
        if name in _DATA_KERNEL_ARGUMENTS
    ]
    issues.extend(
        Issue(
            f"kernel_parameters.{name}",
            "is not accepted by the selected kernel signature",
            physics,
        )
        for name in values
        if name not in configurable and name not in _DATA_KERNEL_ARGUMENTS
    )
    issues.extend(
        Issue(
            f"kernel_parameters.{name}",
            "is required by the selected kernel signature",
            physics,
        )
        for name, parameter in configurable.items()
        if parameter.default is inspect.Parameter.empty and name not in _CONTEXT_KERNEL_ARGUMENTS and name not in values
    )
    try:
        annotations = get_type_hints(kernel)
    except (NameError, TypeError) as exc:
        return [*issues, Issue("kernel_id", f"kernel annotations cannot be resolved: {exc}", physics)]
    for name, value in values.items():
        if name not in configurable:
            continue
        annotation = annotations.get(name, configurable[name].annotation)
        if annotation is inspect.Parameter.empty:
            issues.append(Issue(f"kernel_parameters.{name}", "has no type annotation in the kernel signature", physics))
        elif not _annotation_accepts(annotation, value):
            expected = getattr(annotation, "__name__", str(annotation).replace("typing.", ""))
            issues.append(
                Issue(
                    f"kernel_parameters.{name}",
                    f"must match kernel annotation {expected}",
                    physics,
                )
            )
    return issues


# ruff: disable[E501]
# fmt: off
PARAM_RULES = (
    Depends("", "scheme", physics="The matching scheme is explicit and selects the coefficient function and upstream renormalization convention."),
    Value("scheme", Literal["ratio", "hybrid", "msbar"], physics="The three schemes differ by their coefficient function: MSbar adds 0.5/|1-xi| to the ratio kernel, and hybrid instead adds the Wilson-line term set by the switching distance."),
    Depends("", "order", physics="The perturbative order remains explicit in the manifest and is encoded in the runtime-derived kernel filename."),
    Value("order", Literal["nlo"], physics="Only next-to-leading order matching kernels are currently available."),
    Recommends("", "resummation", physics="Matching uses fixed-order NLO unless an explicit resummation is selected.", default=""),
    Value("resummation", Literal["", "rgr", "lrr"], physics="Empty selects fixed-order NLO; rgr resums the running coupling and lrr resums the leading renormalon."),
    Recommends("", "resummation_part", physics="The component suffix is only used by RGR kernels.", default=""),
    Value("resummation_part", Literal["", "re", "im"], physics="RGR selects either the real or imaginary component; fixed-order and LRR kernels have no component suffix."),
    Depends("", "mu", physics="The matching scale is the MS-bar scale of the published light-cone distribution in GeV, conventionally 2 GeV, and enters every coefficient function through the logarithm ln(4 y^2 Pz^2 / mu^2)."),
    Depends("", "lc_x_ls", physics="A list is the exact light-cone output grid; a start/stop mapping instead keeps the quasi-grid points inside the closed window, and never interpolates."),
    Recommends("", "kernel_parameters", physics="Kernel-specific controls are explicit and are validated against the selected kernel signature; every kernel accepts eps, the regulator keeping plus-prescription denominators finite, and nlo_rgr_* kernels add kappa and mu_min_gev, which build row x at mu0=2*kappa*x*Pz and zero every row with mu0 below mu_min_gev, so together they impose the cutoff x_min=mu_min_gev/(2*kappa*Pz) and keep mu0 above the Landau pole.", default={}),
    Provides("", "hybrid", "scheme", physics="Only hybrid matching owns a Wilson-line switching distance, because only its coefficient function contains that term."),
    Depends("hybrid", "zs_fm", physics="The switching distance in fm is where the ratio scheme gives way to Wilson-line subtraction; the kernel uses the dimensionless zs*Pz, and it is the same physical distance the hybrid renormalization applied to this input."),
    Value("mu", (int, float), physics="The matching scale is finite and positive, and must stay far enough above LambdaQCD for a perturbative coupling to exist.", validator=_positive),
    Value("lc_x_ls", (list, dict), physics="The light-cone grid is increasing or has finite start/stop bounds.", validator=_valid_lc_x_ls),
    Value("kernel_parameters", dict, physics="Kernel parameters are an explicit mapping."),
    Value("hybrid.zs_fm", (int, float), physics="Hybrid switch distance is finite and positive.", validator=_positive),
)

INPUT_RULES = (
    Depends("", "quasi", physics="Matching consumes exactly one quasi distribution, whose attrs supply the momentum Pz and the provenance tokens the kernel filename must reproduce."),
    Source("quasi", physics="The quasi input is one prior job or external file source."),
)

SYSTEMATICS_RULES = (
    Recommends("", "defaults", physics="Matching systematics defaults are optional.", default={}),
    Recommends("", "variants", physics="Matching systematic variants are optional.", default=[]),
    Value("defaults", dict, physics="Matching systematics defaults form an object."),
    List("variants", "variant", physics="Matching systematic variants preserve authored order."),
    Suggests("", "defaults", "variants.variant", physics="Systematics defaults fill each matching variant."),
    Depends("variants.variant", "id", physics="Every matching variation has one safe label, which becomes the suffix of every generated job id."),
    Depends("variants.variant", "mu_factor", physics="Every matching variation multiplies the central mu by one noncentral factor, conventionally sqrt(2) and 1/sqrt(2), and the generated jobs become the lamet_scale component of the systematics budget."),
    Value("variants.variant.id", str, physics="Matching variation labels are safe identifiers.", validator=_safe_systematics_id),
    Value("variants.variant.mu_factor", (int, float), physics="Matching scale multipliers are finite, positive, and not one.", validator=_valid_mu_factor),
)
# fmt: on
# ruff: enable[E501]


def check_kernel_shape(context: CheckContext) -> Issue | None:
    resummation = context.params.get("resummation", "")
    part = context.params.get("resummation_part", "")
    if resummation == "" and part:
        return Issue("resummation_part", "requires resummation='rgr'", "Only RGR kernels select a real or imaginary component.")
    if resummation == "rgr" and part not in {"re", "im"}:
        return Issue("resummation_part", "must be 're' or 'im' for RGR", "RGR kernels are component-specific.")
    if resummation == "lrr" and part:
        return Issue("resummation_part", "must be empty for LRR", "LRR kernels have no component suffix.")
    return None


def check_x_output(context: CheckContext) -> Issue | None:
    window = context.params.get("lc_x_ls")
    if isinstance(window, dict) and window["start"] >= window["stop"]:
        return Issue("lc_x_ls", "must have start smaller than stop", "The matching interval must be ordered.")
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = (check_kernel_shape, check_x_output)


def check_systematics(context: CheckContext) -> Issue | None:
    labels = [variant["id"] for variant in context.params["variants"]]
    if len(set(labels)) != len(labels):
        return Issue("variants", "ids must be unique", "Every variation creates one job suffix.")
    return None


SYSTEMATICS_CHECKS = (check_systematics,)
