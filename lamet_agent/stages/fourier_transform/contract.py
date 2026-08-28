"""Manifest contract for Fourier transformation."""

from __future__ import annotations

import math
import re
from typing import Literal

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
from lamet_agent.stages.fourier_transform.ask import zmax_fm as recommend_zmax_fm
from lamet_agent.stages.fourier_transform.ask import zmin_fm as recommend_zmin_fm


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _finite(value: int | float) -> bool:
    return math.isfinite(value)


def _nonnegative(value: int | float) -> bool:
    return math.isfinite(value) and value >= 0


def _valid_grid(value: object) -> bool:
    if isinstance(value, list):
        return bool(value) and all(
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and right > left
            for left, right in zip(value, value[1:])
        )
    if not isinstance(value, dict) or set(value) != {"start", "stop", "num"}:
        return False
    return (
        isinstance(value["start"], (int, float))
        and not isinstance(value["start"], bool)
        and isinstance(value["stop"], (int, float))
        and not isinstance(value["stop"], bool)
        and value["stop"] > value["start"]
        and isinstance(value["num"], int)
        and not isinstance(value["num"], bool)
        and value["num"] > 1
    )


def _nonempty(value: list[object]) -> bool:
    return len(value) > 0


def _unit_interval(value: int | float) -> bool:
    return math.isfinite(value) and 0 <= value <= 1


def _safe_systematics_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", value))


def _nonzero(value: int) -> bool:
    return value != 0


# ruff: disable[E501]
# fmt: off
PARAM_RULES = (
    Depends("", "quasi_y_ls", physics="The output grid is explicit and dimensionless."),
    Depends("", "zmin_fm", physics="Tail lower ranges are authored or data-recommended candidate values.", null_hook=recommend_zmin_fm),
    Depends("", "zmax_fm", physics="Tail upper ranges are authored or data-recommended candidate values.", null_hook=recommend_zmax_fm),
    Recommends("", "tail_window_step_offset", physics="Central Fourier jobs do not shift their lower tail-window candidates.", default=0),
    Depends("", "smooth", physics="Tail/data connection uses a declared prescription."),
    Depends("", "zmax_ext_fm", physics="The finite transform extent is explicit."),
    Depends("", "scheme_scan", physics="The complete native LA/NLA candidate scan is explicit."),
    Provides("", "da", "$.metadata.target_observable", physics="DA jobs own midpoint projection and endpoint flavor controls."),
    Depends("da", "phase_transfer_da", physics="A meson DA explicitly selects whether to project about its midpoint before tail fitting."),
    Depends("da", "psi1_flavor_class", physics="The first DA endpoint flavor class fixes the allowed tail term."),
    Depends("da", "psi2_flavor_class", physics="The second DA endpoint flavor class fixes the allowed tail term."),
    Depends("scheme_scan", "order", physics="Tail orders are explicit."),
    Depends("scheme_scan", "sector", physics="The distribution sector is explicit."),
    Depends("scheme_scan", "Lambda0_gev", physics="The fixed decay offset is explicit."),
    Depends("scheme_scan", "posterior_prior_error_scale", physics="Tail-prior scales are explicit."),
    Depends("scheme_scan", "model_average", physics="Model averaging is explicit."),
    Recommends("scheme_scan", "q_min", physics="The original Fourier range rule uses a 0.05 fit-quality threshold.", default=0.05),
    Recommends("scheme_scan", "max_schemes", physics="The reference scan bounds the number of range candidates at 200.", default=200),
    List("zmin_fm", "zmin", physics="Tail lower candidates are a nonempty ordered list.", validator=_nonempty),
    List("zmax_fm", "zmax", physics="Tail upper candidates are a nonempty ordered list.", validator=_nonempty),
    List("scheme_scan.order", "order", physics="At least one LA/NLA order is required.", validator=_nonempty),
    List("scheme_scan.posterior_prior_error_scale", "width", physics="At least one tail-prior scale is required.", validator=_nonempty),
    Value("quasi_y_ls", (list, dict), physics="The x grid is an increasing list or an explicit start/stop/count mapping.", validator=_valid_grid),
    Value("zmin_fm.zmin", (int, float), physics="Tail lower candidates are finite and nonnegative.", validator=_nonnegative),
    Value("zmax_fm.zmax", (int, float), physics="Tail upper candidates are finite and positive.", validator=_positive),
    Value("tail_window_step_offset", int, physics="Tail-window shifts are integer lattice-site offsets."),
    Value("smooth", Literal["linear", "none"], physics="The reference tail connection is linear across the selected fit interval or switches after it."),
    Value("zmax_ext_fm", (int, float), physics="Tail extent is finite and positive.", validator=_positive),
    Value("scheme_scan.order.order", Literal["LA", "NLA"], physics="The order selects the asymptotic tail terms."),
    Value("scheme_scan.sector", Literal["valence", "singlet", "full"], physics="The output records one explicit distribution sector."),
    Value("scheme_scan.Lambda0_gev", (int, float), physics="The fixed decay offset is finite and nonnegative.", validator=_nonnegative),
    Value("scheme_scan.posterior_prior_error_scale.width", (int, float), physics="Each tail-prior scale is finite and positive.", validator=_positive),
    Value("scheme_scan.model_average", bool, physics="Model averaging is an explicit selection policy."),
    Value("scheme_scan.q_min", (int, float), physics="The Fourier fit-quality threshold is a probability.", validator=_unit_interval),
    Value("scheme_scan.max_schemes", int, physics="The authored scan has a positive candidate bound.", validator=_positive),
    Value("da.phase_transfer_da", bool, physics="The DA midpoint phase projection is explicit."),
    Value("da.psi1_flavor_class", Literal["light", "heavy"], physics="The first DA endpoint is light or heavy."),
    Value("da.psi2_flavor_class", Literal["light", "heavy"], physics="The second DA endpoint is light or heavy."),
)

INPUT_RULES = (
    Depends("", "input", physics="Fourier transformation consumes one renormalized coordinate-space input."),
    Source("input", physics="The Fourier input is one prior job or external file source."),
)

SYSTEMATICS_RULES = (
    Recommends("", "defaults", physics="Fourier systematics defaults are optional.", default={}),
    Recommends("", "variants", physics="Fourier systematic variants are optional.", default=[]),
    Value("defaults", dict, physics="Fourier systematics defaults form an object."),
    List("variants", "variant", physics="Fourier systematic variants preserve authored order."),
    Suggests("", "defaults", "variants.variant", physics="Systematics defaults fill each Fourier variant."),
    Depends("variants.variant", "id", physics="Every Fourier variation has one safe label."),
    Depends("variants.variant", "tail_window_step_offset", physics="Every Fourier variation shifts the tail window by a nonzero lattice step."),
    Value("variants.variant.id", str, physics="Fourier variation labels are safe identifiers.", validator=_safe_systematics_id),
    Value("variants.variant.tail_window_step_offset", int, physics="Tail-window offsets are nonzero integers.", validator=_nonzero),
)
# fmt: on
# ruff: enable[E501]


def check_tail_ranges(context: CheckContext) -> Issue | None:
    physics = "Tail ranges must be positive, ordered, and contained in the transform extent."
    lower = context.params.get("zmin_fm")
    upper = context.params.get("zmax_fm")
    if not isinstance(lower, list) or not isinstance(upper, list):
        return None
    extent = context.params["zmax_ext_fm"]
    if any(value > extent for value in [*lower, *upper]):
        return Issue("zmax_ext_fm", "must contain every tail candidate range", physics)
    if lower and upper and any(left >= max(upper) for left in lower):
        return Issue("zmin_fm", "every lower candidate needs an authored upper candidate above it", physics)
    return None


def check_da_sector(context: CheckContext) -> Issue | None:
    scan = context.params.get("scheme_scan")
    if scan is None:
        return None
    if context.manifest["metadata"]["target_observable"] == "da" and scan["sector"] != "full":
        return Issue(
            "scheme_scan.sector",
            "must be full for a DA",
            "The migrated DA transform retains the full complex distribution.",
        )
    return None


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = (check_tail_ranges, check_da_sector)


def check_systematics(context: CheckContext) -> list[Issue]:
    variants = context.params["variants"]
    labels = [variant["id"] for variant in variants]
    offsets = [variant["tail_window_step_offset"] for variant in variants]
    issues = []
    if len(set(labels)) != len(labels):
        issues.append(Issue("variants", "ids must be unique", "Every variation creates one job suffix."))
    if len(set(offsets)) != len(offsets):
        issues.append(
            Issue("variants", "tail-window offsets must be unique", "Duplicate shifts create duplicate analyses.")
        )
    return issues


SYSTEMATICS_CHECKS = (check_systematics,)
