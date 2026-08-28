"""Manifest contract for final review."""

from typing import Literal

from lamet_agent.contract import Depends, List, Source, Value, stage_job_rules


def _positive(value: int) -> bool:
    return value > 0


def _nonempty(value: list[object]) -> bool:
    return len(value) > 0


# ruff: disable[E501]
# fmt: off
PARAM_RULES = (
    Depends("", "catalog", physics="Review catalog source is explicit."),
    Depends("", "max_papers", physics="Full-text literature access is bounded."),
    Depends("", "report_language", physics="Review language is explicit."),
    Depends("", "checks", physics="Requested consistency checks are explicit."),
    Value("catalog", str, physics="Catalog is builtin or a root-relative path."),
    Value("max_papers", int, physics="Maximum selected papers is positive.", validator=_positive),
    Value("report_language", Literal["en", "ch"], physics="Review prose is generated directly in English or Chinese."),
    List("checks", "check", physics="At least one review check is requested.", validator=_nonempty),
    Value("checks.check", Literal["identity", "units", "kinematics", "schemes", "grids", "resampling", "extrapolation"], physics="Review check ids are controlled."),
)

INPUT_RULES = (
    Depends("", "results", physics="Review scope is an explicit ordered list of prior results."),
    List("results", "result", physics="Review results are nonempty and preserve authored order.", validator=_nonempty),
    Source("results.result", physics="Each review result is a prior job or external file source."),
)
# fmt: on
# ruff: enable[E501]


JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)

CHECKS = ()
