"""Manifest contract for perturbative matching."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
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
from lamet_agent.kernels import load_kernel
from lamet_agent.ui import warning


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
    Depends("", "kernel_id", physics="The matching kernel is selected by one public filename stem."),
    Depends("", "scheme", physics="The matching scheme is explicit and must agree with the selected kernel."),
    Value("scheme", Literal["ratio", "hybrid", "msbar"], physics="Scheme is ratio, hybrid, or MSbar."),
    Depends("", "mu", physics="The matching scale is an explicit physical choice."),
    Depends("", "lc_x_ls", physics="The output grid is an explicit list or a start/stop window on the quasi grid."),
    Recommends("", "kernel_parameters", physics="Kernels without scheme-specific controls receive an empty explicit parameter mapping.", default={}),
    Provides("", "hybrid", "scheme", physics="Hybrid matching owns its Wilson-line switching distance."),
    Depends("hybrid", "zs_fm", physics="Hybrid matching switch is a per-job parameter."),
    Value("kernel_id", str, physics="Kernel ids are safe public filename stems.", validator=_valid_kernel_id),
    Value("mu", (int, float), physics="The matching scale is finite and positive.", validator=_positive),
    Value("lc_x_ls", (list, dict), physics="The light-cone grid is increasing or has finite start/stop bounds.", validator=_valid_lc_x_ls),
    Value("kernel_parameters", dict, physics="Kernel parameters are an explicit mapping."),
    Value("hybrid.zs_fm", (int, float), physics="Hybrid switch distance is finite and positive.", validator=_positive),
)

INPUT_RULES = (
    Depends("", "quasi", physics="Matching consumes one quasi-distribution source."),
    Source("quasi", physics="The quasi input is one prior job or external file source."),
)

SYSTEMATICS_RULES = (
    Recommends("", "defaults", physics="Matching systematics defaults are optional.", default={}),
    Recommends("", "variants", physics="Matching systematic variants are optional.", default=[]),
    Value("defaults", dict, physics="Matching systematics defaults form an object."),
    List("variants", "variant", physics="Matching systematic variants preserve authored order."),
    Suggests("", "defaults", "variants.variant", physics="Systematics defaults fill each matching variant."),
    Depends("variants.variant", "id", physics="Every matching variation has one safe label."),
    Depends("variants.variant", "mu_factor", physics="Every matching variation has one noncentral scale multiplier."),
    Value("variants.variant.id", str, physics="Matching variation labels are safe identifiers.", validator=_safe_systematics_id),
    Value("variants.variant.mu_factor", (int, float), physics="Matching scale multipliers are finite, positive, and not one.", validator=_valid_mu_factor),
)
# fmt: on
# ruff: enable[E501]


def check_kernel_shape(context: CheckContext) -> Issue | None:
    kernel_id = context.params["kernel_id"]
    if kernel_id.startswith("_") or "/" in kernel_id or "\\" in kernel_id:
        return Issue(
            "kernel_id", "must be a public filename stem", "Kernel selection is lexical and has no alias registry."
        )
    tokens = kernel_id.split("_")
    schemes = [scheme for scheme in ("ratio", "hybrid", "msbar") if scheme in tokens]
    if len(schemes) != 1:
        return Issue(
            "kernel_id", "must contain exactly one scheme token", "The filename carries the kernel's physical scheme."
        )
    if context.params.get("scheme") != schemes[0]:
        return Issue(
            "scheme",
            f"must equal {schemes[0]!r} for kernel {kernel_id!r}",
            "The stage scheme and filename scheme are the same physical choice.",
        )
    return None


def check_x_output(context: CheckContext) -> Issue | None:
    window = context.params.get("lc_x_ls")
    if isinstance(window, dict) and window["start"] >= window["stop"]:
        return Issue("lc_x_ls", "must have start smaller than stop", "The matching interval must be ordered.")
    return None


def check_kernel_resources(context: CheckContext) -> Issue | None:
    kernel_id = context.params.get("kernel_id")
    if not isinstance(kernel_id, str) or not _valid_kernel_id(kernel_id):
        return None
    root = Path(__file__).parents[2] / "kernels"
    if not (root / f"{kernel_id}.py").is_file():
        return Issue(
            "kernel_id",
            f"kernel implementation does not exist: {kernel_id}.py",
            "Every selected kernel has one shipped implementation module.",
        )
    if not (root / f"{kernel_id}.md").is_file():
        return Issue(
            "kernel_id",
            f"kernel formula document does not exist: {kernel_id}.md",
            "Every selected kernel ships its formula provenance.",
        )
    return None


def check_kernel_parameters(context: CheckContext) -> list[Issue] | Issue | None:
    """Validate the explicit parameter mapping against the selected kernel."""
    values = context.params.get("kernel_parameters")
    kernel_id = context.params.get("kernel_id")
    if not isinstance(values, dict) or not isinstance(kernel_id, str) or not _valid_kernel_id(kernel_id):
        return None
    if check_kernel_shape(context) is not None or check_kernel_resources(context) is not None:
        return None
    try:
        kernel = load_kernel(kernel_id)
    except Exception as exc:
        return Issue(
            "kernel_id",
            f"cannot load kernel signature: {exc}",
            "The selected kernel must expose an inspectable kernel() callable.",
        )
    try:
        signature = inspect.signature(kernel)
        kernel_uses_zs = "zs_fm" in signature.parameters
        stage_supplies_zs = context.params.get("scheme") == "hybrid"
        if kernel_uses_zs != stage_supplies_zs:
            expected = "include" if stage_supplies_zs else "omit"
            return Issue(
                "kernel_id",
                f"kernel signature must {expected} stage-managed zs_fm for scheme {context.params.get('scheme')!r}",
                "The matching scheme determines whether the stage injects a Wilson-line switching distance.",
            )
        issues = _kernel_parameter_issues(kernel, values)
        overridden = sorted(_CONTEXT_KERNEL_ARGUMENTS.intersection(values))
        if not issues and overridden:
            warning(f"matching kernel_parameters overrides stage context: {overridden}")
        return issues
    except (TypeError, ValueError) as exc:
        return Issue(
            "kernel_id",
            f"cannot inspect kernel signature: {exc}",
            "The selected kernel must expose an inspectable kernel() callable.",
        )


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = (check_kernel_shape, check_x_output, check_kernel_resources, check_kernel_parameters)


def check_systematics(context: CheckContext) -> Issue | None:
    labels = [variant["id"] for variant in context.params["variants"]]
    if len(set(labels)) != len(labels):
        return Issue("variants", "ids must be unique", "Every variation creates one job suffix.")
    return None


SYSTEMATICS_CHECKS = (check_systematics,)
