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
    Depends("", "quasi_y_ls", physics="Dimensionless quasi momentum-fraction grid y for P_z/(2 pi hbar c) sum_z w_z exp(+i y P_z z/(hbar c)) h(z), with z and w_z in fm and full endpoint weights on a uniform grid; it becomes the downstream matching input grid, so its range and spacing set numerical coverage but add no lattice information."),
    Depends("", "zmin_fm", physics="Candidate lower separations for the long-distance tail fit in fm; positive-z data with zmin <= z <= zmax constrain the tail, and one common range is selected from the sample-average data before resample fits.", null_hook=recommend_zmin_fm),
    Depends("", "zmax_fm", physics="Candidate upper separations of measured data entering the tail fit in fm; varying zmax tests the balance between large-|z| asymptotic validity and deteriorating lattice signal, independently of the later extension endpoint.", null_hook=recommend_zmax_fm),
    Recommends("", "tail_window_step_offset", physics="Shifts every zmin_fm by this integer times the ensemble spatial spacing a_s while keeping zmax_fm fixed; zero is the unshifted default, while nonzero signed offsets produce tail-range variations in lattice-site units.", default=0),
    Depends("", "smooth", physics="Connection of measured/interpolated data to the fitted tail: linear ramps the data weight from one at selected zmin to zero at selected zmax; none assigns unit data weight through requested zmax_ext_fm and tail weight beyond it, so only a rounded-up outer grid point can use the tail."),
    Depends("", "zmax_ext_fm", physics="Requested maximum |z| in fm for the finite Fourier sum; it must cover the input and fit boundaries, while the actual endpoint is rounded to the nearest input-grid step."),
    Depends("", "scheme_scan", physics="Tail-model scan combining LA/NLA orders and internal-coordinate posterior-prior scales after one sample-average range is selected; q_min guides range and best-model choices, while model averaging uses all numerically valid finite-logGBF candidates."),
    Provides("", "da", "$.metadata.target_observable", physics="For target_observable=da, enables midpoint phase/symmetry projection and ordered endpoint flavor classes used to constrain meson-DA tail amplitudes; these controls do not apply to PDF jobs."),
    Depends("da", "phase_transfer_da", physics="For a meson DA, optionally rotates h(z) by exp(+i P_z z/(2 hbar c)), projects onto the real midpoint-symmetric channel, and rotates back before tail fitting and Fourier transformation; false retains the input complex samples."),
    Depends("da", "psi1_flavor_class", physics="Light/heavy class of the first ordered meson endpoint psi1; together with psi2 it selects charge-conjugation relations for light-light tails or the amplitude removed by a light-heavy ordering."),
    Depends("da", "psi2_flavor_class", physics="Light/heavy class of the second ordered endpoint psi2; swapping psi1 and psi2 swaps which endpoint amplitude is constrained for an asymmetric heavy-light meson."),
    Depends("scheme_scan", "order", physics="Asymptotic truncations compared at the selected range: LA retains the leading large-|z| structure, while NLA adds the next inverse-distance contribution and its fit parameters."),
    Depends("scheme_scan", "sector", physics="Hard-coded component and normalization projection: under the authored negative-x and operator conventions, quark-PDF valence and singlet correspond to q-antiquark and q+antiquark, while full keeps both complex channels; the code infers real/imaginary channel from polarization but does not verify those physics conventions, and DA permits only full."),
    Depends("scheme_scan", "Lambda0_gev", physics="Fixed nonnegative infrared offset in exp[-(Lambda+Lambda0)|z|/(hbar c)], with Lambda fitted; it changes the tail decay parameterization, not the perturbative matching scale."),
    Depends("scheme_scan", "posterior_prior_error_scale", physics="Positive multipliers of sample-average posterior widths in the bounded internal fit coordinates, with a 1e-8 floor, used to define resample-fit priors; multiple values form separate tail candidates and test regularization sensitivity."),
    Depends("scheme_scan", "model_average", physics="At fixed range, false chooses the largest-logGBF model above q_min per resample and falls back to maximum Q; true forms normalized exp(logGBF-max(logGBF)) weighted means over all numerically valid candidates, without Q filtering or an explicit between-model variance term."),
    Recommends("scheme_scan", "q_min", physics="Preferred lsqfit-Q threshold for sample-average range selection and non-averaged per-resample model choice, with maximum-Q fallback; publication still requires at least one center model at or above this threshold.", default=0.05),
    Recommends("scheme_scan", "max_schemes", physics="Computational cap on the deterministic model-zmin-zmax candidates used for sample-average range selection; it does not limit order/prior models refitted after the range is fixed.", default=200),
    List("zmin_fm", "zmin", physics="Nonempty ordered lower-bound candidates in fm; authored order is retained for reproducible Cartesian pairing, while the measured coordinate grid and zmin < zmax determine admissibility.", validator=_nonempty),
    List("zmax_fm", "zmax", physics="Nonempty ordered upper-bound candidates in fm, paired with admissible lower bounds to test how much measured long-distance information constrains the tail.", validator=_nonempty),
    List("scheme_scan.order", "order", physics="At least one asymptotic truncation is required; authored order fixes reproducible construction and labeling of the LA/NLA candidate set.", validator=_nonempty),
    List("scheme_scan.posterior_prior_error_scale", "width", physics="At least one posterior-width multiplier is required; authored order is retained when constructing and labeling tail-model candidates.", validator=_nonempty),
    Value("quasi_y_ls", (list, dict), physics="A strictly increasing dimensionless y grid, given explicitly or by start/stop/num; density controls Fourier and matching quadrature resolution but does not add lattice information.", validator=_valid_grid),
    Value("zmin_fm.zmin", (int, float), physics="Each lower tail boundary is a finite nonnegative distance in fm and must match a measured nonnegative z coordinate after any lattice-step offset.", validator=_nonnegative),
    Value("zmax_fm.zmax", (int, float), physics="Each upper tail boundary is a finite positive distance in fm on the measured z grid and must exceed the retained lower boundary.", validator=_positive),
    Value("tail_window_step_offset", int, physics="Dimensionless signed count of spatial lattice steps; the physical shift is exactly tail_window_step_offset times a_s."),
    Value("smooth", Literal["linear", "none"], physics="linear blends data and tail over [selected zmin,selected zmax]; none keeps data weight one for |z| <= requested zmax_ext_fm and uses the tail only at any rounded-up outer endpoint."),
    Value("zmax_ext_fm", (int, float), physics="Finite positive requested extent in fm that covers the input and all fit candidates; the represented largest |z| is the nearest input-spacing multiple.", validator=_positive),
    Value("scheme_scan.order.order", Literal["LA", "NLA"], physics="LA/NLA truncation within the GI or CG tail family selected by upstream gfix provenance: LA keeps the leading structure and NLA adds the next inverse-|z| term."),
    Value("scheme_scan.sector", Literal["valence", "singlet", "full"], physics="Under the assumed negative-x/operator convention, valence and singlet use polarization-dependent real/imaginary channels with factor two; full transforms both channels with unit normalization, and DA allows only full."),
    Value("scheme_scan.Lambda0_gev", (int, float), physics="Finite nonnegative offset in GeV added to the fitted decay rate, making the suppression exponent (Lambda+Lambda0_gev)|z|/(hbar c).", validator=_nonnegative),
    Value("scheme_scan.posterior_prior_error_scale.width", (int, float), physics="Finite positive multiplier of bounded-internal-coordinate posterior widths; zero and negative values are rejected.", validator=_positive),
    Value("scheme_scan.model_average", bool, physics="Whether to form per-resample evidence-weighted means over valid models or choose one model using q_min, logGBF, and maximum-Q fallback; no separate between-model variance is added."),
    Value("scheme_scan.q_min", (int, float), physics="Preferred threshold in [0,1] for the chi-square upper-tail probability Q and the final center-model publication gate; selection can fall back below it, and evidence averaging does not apply it.", validator=_unit_interval),
    Value("scheme_scan.max_schemes", int, physics="Positive number of ordered model-and-range candidates retained for sample-average range selection; it bounds cost without changing any evaluated tail formula.", validator=_positive),
    Value("da.phase_transfer_da", bool, physics="true applies midpoint phase rotation, real-channel symmetry projection, and inverse rotation before extension; false leaves the complex coordinate-space samples unchanged."),
    Value("da.psi1_flavor_class", Literal["light", "heavy"], physics="Ordered light/heavy mass class of psi1; it changes allowed DA tail amplitudes, not the Fourier kernel."),
    Value("da.psi2_flavor_class", Literal["light", "heavy"], physics="Independent light/heavy mass class of psi2, completing the ordered endpoint assignment for light-light, light-heavy, heavy-light, or heavy-heavy tail constraints."),
)

INPUT_RULES = (
    Depends("", "input", physics="One coordinate-space matrix element on a nonnegative-z or signed-z grid supplies resamples, fm coordinates, momentum, parton, gfix, and for PDF only polarization metadata; renormalization is expected from workflow provenance but not verified, and nonnegative input is completed as real-even/imaginary-odd."),
    Source("input", physics="The input may reference an upstream renormalization job or an external file obeying the same coordinate-space data and metadata contract."),
)

SYSTEMATICS_RULES = (
    Recommends("", "defaults", physics="Optional common values inherited by Fourier systematic variants, avoiding repeated authoring without changing the central jobs.", default={}),
    Recommends("", "variants", physics="Optional tail-window variations; each clones every central Fourier job with a distinct nonzero shift of zmin in lattice-site units.", default=[]),
    Value("defaults", dict, physics="Object of shared variant fields applied before variant-specific values, so an explicit variant value remains authoritative."),
    List("variants", "variant", physics="For each central job, authored variant order fixes the relative order of its clones; central-job order remains outermost, and each id determines its suffix and downstream label order."),
    Suggests("", "defaults", "variants.variant", physics="Missing fields in each Fourier variant are supplied from defaults before the concrete shifted-window job is generated."),
    Depends("variants.variant", "id", physics="Stable label identifying the tail-window variation in generated job IDs, artifacts, and systematics provenance."),
    Depends("variants.variant", "tail_window_step_offset", physics="Nonzero signed shift applied to every central zmin_fm as offset times a_s, while zmax_fm and all other Fourier parameters remain unchanged."),
    Value("variants.variant.id", str, physics="Lowercase identifier beginning with a letter and containing only letters, digits, or underscores, suitable for reproducible job suffixes.", validator=_safe_systematics_id),
    Value("variants.variant.tail_window_step_offset", int, physics="Nonzero integer count of spatial lattice steps; positive and negative values move the tail-fit onset later or earlier in |z|.", validator=_nonzero),
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
