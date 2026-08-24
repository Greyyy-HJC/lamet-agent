"""Manifest contract for Fourier transformation."""

from __future__ import annotations

import math
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, List, Value


def _positive(value: int | float) -> bool:
    return math.isfinite(value) and value > 0


def _finite(value: int | float) -> bool:
    return math.isfinite(value)


def _nonnegative(value: int | float) -> bool:
    return math.isfinite(value) and value >= 0


def _valid_grid(value: object) -> bool:
    if isinstance(value, list):
        return bool(value) and all(isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool) and right > left for left, right in zip(value, value[1:]))
    if not isinstance(value, dict) or set(value) != {"start", "stop", "num"}:
        return False
    return isinstance(value["start"], (int, float)) and not isinstance(value["start"], bool) and isinstance(value["stop"], (int, float)) and not isinstance(value["stop"], bool) and value["stop"] > value["start"] and isinstance(value["num"], int) and not isinstance(value["num"], bool) and value["num"] > 1


def _nonempty(value: list[object]) -> bool:
    return len(value) > 0


def _unit_interval(value: int | float) -> bool:
    return math.isfinite(value) and 0 <= value <= 1


PARAM_RULES = (
    Depends("", "parton", physics="The parton type is explicit."),
    Depends("", "gfix", physics="The gauge-link construction is inherited or authored explicitly."),
    Depends("", "symmetry", physics="Negative-z completion is an explicit component convention."),
    Depends("", "transform", physics="Fourier phase and normalization are explicit."),
    Depends("", "quasi_y_ls", physics="The output grid is explicit and dimensionless."),
    Depends("", "tail_models", physics="Tail candidates are bounded by an authored list."),
    Depends("", "zmin_fm", physics="Tail lower ranges are authored candidate values."),
    Depends("", "zmax_fm", physics="Tail upper ranges are authored candidate values."),
    Depends("", "smoothing", physics="Tail/data connection uses a declared prescription."),
    Depends("", "zmax_ext_fm", physics="The finite transform extent is explicit."),
    Depends("", "scheme_scan", physics="The complete native LA/NLA candidate scan remains one stage-owned mapping.", required=False),
    Depends("", "phase_transfer_da", physics="A meson DA may be projected about its midpoint before tail fitting.", required=False),
    Depends("", "psi1_flavor_class", physics="The first DA endpoint flavor class fixes the allowed tail term.", required=False),
    Depends("", "psi2_flavor_class", physics="The second DA endpoint flavor class fixes the allowed tail term.", required=False),
    Depends("symmetry", "real", physics="The real-part signed-z convention is explicit."),
    Depends("symmetry", "imag", physics="The imaginary-part signed-z convention is explicit."),
    Depends("transform", "phase_sign", physics="The Fourier phase sign is explicit."),
    Depends("transform", "x_shift", physics="The Fourier x shift is explicit."),
    Depends("transform", "prefactor", physics="The Fourier normalization is explicit."),
    Depends("scheme_scan", "order", physics="Tail orders are explicit."),
    Depends("scheme_scan", "sector", physics="The distribution sector is explicit."),
    Depends("scheme_scan", "Lambda0_gev", physics="The fixed decay offset is explicit."),
    Depends("scheme_scan", "posterior_prior_error_scale", physics="Tail-prior scales are explicit."),
    Depends("scheme_scan", "model_average", physics="Model averaging is explicit."),
    Depends("scheme_scan", "max_schemes", physics="The range scan has an explicit bound."),
    Depends("scheme_scan", "component", physics="The fitted complex component is explicit."),
    Depends("scheme_scan", "output_scale", physics="The final distribution scale is explicit."),
    Depends("scheme_scan", "q_min", physics="The fit-quality threshold is explicit."),
    List("tail_models", "model", physics="At least one tail model is allowed.", validator=_nonempty),
    List("zmin_fm", "zmin", physics="Tail lower candidates are a nonempty ordered list.", validator=_nonempty),
    List("zmax_fm", "zmax", physics="Tail upper candidates are a nonempty ordered list.", validator=_nonempty),
    List("smoothing.smooth", "method", physics="Smoothing method choices are traversed explicitly."),
    List("smoothing.widths_fm", "width", physics="Smoothing widths are traversed explicitly."),
    List("scheme_scan.order", "order", physics="At least one LA/NLA order is required.", validator=_nonempty),
    List("scheme_scan.posterior_prior_error_scale", "width", physics="At least one tail-prior scale is required.", validator=_nonempty),
    Value("parton", Literal["quark"], physics="The available matching kernels are quark kernels."),
    Value("gfix", Literal["GI", "CG"], physics="Construction is GI or CG."),
    Value("symmetry.real", Literal["even", "odd", "explicit"], physics="The real-part convention is even, odd, or explicit."),
    Value("symmetry.imag", Literal["even", "odd", "explicit"], physics="The imaginary-part convention is even, odd, or explicit."),
    Value("transform.phase_sign", Literal[-1, 1], physics="The Fourier phase sign is plus or minus one."),
    Value("transform.x_shift", (int, float), physics="The Fourier x shift is finite.", validator=_finite),
    Value("transform.prefactor", Literal["pz_over_2pi", "one_over_2pi", "none"], physics="The Fourier prefactor is controlled."),
    Value("quasi_y_ls", (list, dict), physics="The x grid is an increasing list or an explicit start/stop/count mapping.", validator=_valid_grid),
    Value("tail_models.model", Literal["gi_nla", "cg_nla"], physics="Tail model ids are the locked GI/CG NLA families."),
    Value("zmin_fm.zmin", (int, float), physics="Tail lower candidates are finite and nonnegative.", validator=_nonnegative),
    Value("zmax_fm.zmax", (int, float), physics="Tail upper candidates are finite and positive.", validator=_positive),
    Depends("smoothing", "smooth", physics="Smoothing methods are explicit."),
    Depends("smoothing", "widths_fm", physics="Smoothing widths are explicit."),
    Value("smoothing.smooth.method", Literal["linear", "cosine"], physics="Smoothing is linear or cosine."),
    Value("smoothing.widths_fm.width", (int, float), physics="Smoothing widths are finite and positive.", validator=_positive),
    Value("zmax_ext_fm", (int, float), physics="Tail extent is finite and positive.", validator=_positive),
    Value("scheme_scan.order.order", Literal["LA", "NLA"], physics="The order selects the asymptotic tail terms."),
    Value("scheme_scan.sector", Literal["valence", "singlet", "full"], physics="The output records one explicit distribution sector."),
    Value("scheme_scan.Lambda0_gev", (int, float), physics="The fixed decay offset is finite and nonnegative.", validator=_nonnegative),
    Value("scheme_scan.posterior_prior_error_scale.width", (int, float), physics="Each tail-prior scale is finite and positive.", validator=_positive),
    Value("scheme_scan.model_average", bool, physics="Model averaging is an explicit selection policy."),
    Value("scheme_scan.max_schemes", int, physics="The authored scan has a positive candidate bound.", validator=_positive),
    Value("scheme_scan.component", Literal["re", "im", "both"], physics="The component controls both tail fitting and transformation."),
    Value("scheme_scan.output_scale", (int, float), physics="The final distribution scale is finite and positive.", validator=_positive),
    Value("scheme_scan.q_min", (int, float), physics="The candidate quality threshold is a probability.", validator=_unit_interval),
    Value("phase_transfer_da", bool, physics="The DA midpoint phase projection is explicit."),
    Value("psi1_flavor_class", Literal["light", "heavy"], physics="The first DA endpoint is light or heavy."),
    Value("psi2_flavor_class", Literal["light", "heavy"], physics="The second DA endpoint is light or heavy."),
)

INPUT_RULES = (
    Depends("", "input", physics="Fourier transformation consumes one renormalized coordinate-space input."),
    Value("input", dict, physics="The Fourier input is exactly one source object."),
)


def check_tail_ranges(context: CheckContext) -> Issue | None:
    physics = "Tail ranges must be positive, ordered, and contained in the transform extent."
    lower = context.params["zmin_fm"]
    upper = context.params["zmax_fm"]
    extent = context.params["zmax_ext_fm"]
    if any(value > extent for value in [*lower, *upper]):
        return Issue("params.zmax_ext_fm", "must contain every tail candidate range", physics)
    if lower and upper and any(left >= max(upper) for left in lower):
        return Issue("params.zmin_fm", "every lower candidate needs an authored upper candidate above it", physics)
    widths = context.params["smoothing"]["widths_fm"]
    if lower and upper and widths and any(left + max(widths) > max(upper) for left in lower):
        return Issue("params.smoothing.widths_fm", "overlaps must fit inside an authored upper range", physics)
    return None


def check_tail_family(context: CheckContext) -> Issue | None:
    construction = context.params["gfix"]
    models = context.params["tail_models"]
    prefix = "gi_" if construction == "GI" else "cg_"
    invalid = [model for model in models if not model.startswith(prefix)]
    return Issue("params.tail_models", f"models {invalid} do not match construction={construction!r}", "GI constructions use GI tail families and CG constructions use CG families.") if invalid else None


def check_da_sector(context: CheckContext) -> Issue | None:
    scan = context.params.get("scheme_scan")
    if scan is None:
        return None
    if context.manifest["metadata"]["target_observable"] == "da" and scan["sector"] != "full":
        return Issue("params.scheme_scan.sector", "must be full for a DA", "The migrated DA transform retains the full complex distribution.")
    return None


def check_da_conventions(context: CheckContext) -> Issue | None:
    names = {"phase_transfer_da", "psi1_flavor_class", "psi2_flavor_class"}
    present = names & set(context.params)
    if context.manifest["metadata"]["target_observable"] == "da" and present != names:
        return Issue("params", f"DA Fourier jobs require exactly {sorted(names)}", "The midpoint projection and two endpoint flavor classes define the DA tail physics.")
    if context.manifest["metadata"]["target_observable"] != "da" and present:
        return Issue("params", f"PDF Fourier jobs must not declare {sorted(present)}", "DA endpoint conventions do not apply to a PDF.")
    return None


CHECKS = (check_tail_ranges, check_tail_family, check_da_sector, check_da_conventions)
