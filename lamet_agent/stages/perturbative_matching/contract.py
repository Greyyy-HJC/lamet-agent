"""Manifest contract for perturbative matching."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, Provides, Recommends, Source, Value, stage_job_rules


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _increasing(values: list[object]) -> bool:
    return all(isinstance(left, (int, float)) and isinstance(right, (int, float)) and right > left for left, right in zip(values, values[1:]))


def _valid_lc_x_ls(value: object) -> bool:
    if isinstance(value, list):
        return bool(value) and _increasing(value)
    if not isinstance(value, dict) or set(value) != {"start", "stop"}:
        return False
    start, stop = value["start"], value["stop"]
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) for item in (start, stop)) and start < stop


def _valid_kernel_id(value: object) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value)
    )


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


def check_kernel_shape(context: CheckContext) -> Issue | None:
    kernel_id = context.params["kernel_id"]
    if kernel_id.startswith("_") or "/" in kernel_id or "\\" in kernel_id:
        return Issue("kernel_id", "must be a public filename stem", "Kernel selection is lexical and has no alias registry.")
    tokens = kernel_id.split("_")
    schemes = [scheme for scheme in ("ratio", "hybrid", "msbar") if scheme in tokens]
    if len(schemes) != 1:
        return Issue("kernel_id", "must contain exactly one scheme token", "The filename carries the kernel's physical scheme.")
    if context.params.get("scheme") != schemes[0]:
        return Issue("scheme", f"must equal {schemes[0]!r} for kernel {kernel_id!r}", "The stage scheme and filename scheme are the same physical choice.")
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
        return Issue("kernel_id", f"kernel implementation does not exist: {kernel_id}.py", "Every selected kernel has one shipped implementation module.")
    if not (root / f"{kernel_id}.md").is_file():
        return Issue("kernel_id", f"kernel formula document does not exist: {kernel_id}.md", "Every selected kernel ships its formula provenance.")
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = (check_kernel_shape, check_x_output, check_kernel_resources)
