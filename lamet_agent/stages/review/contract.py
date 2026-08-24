"""Manifest contract for final review."""

from typing import Literal

from lamet_agent.contract import Depends, List, Value


def _positive(value: int) -> bool:
    return value > 0


def _nonempty(value: list[object]) -> bool:
    return len(value) > 0


PARAM_RULES = (
    Depends("", "catalog", physics="Review catalog source is explicit."),
    Depends("", "max_papers", physics="Full-text literature access is bounded."),
    Depends("", "report_language", physics="Review language is explicit."),
    Depends("", "checks", physics="Requested consistency checks are explicit."),
    Value("catalog", str, physics="Catalog is builtin or a root-relative path."),
    Value("max_papers", int, physics="Maximum selected papers is positive.", validator=_positive),
    Value("report_language", str, physics="Report language is a string."),
    List("checks", "check", physics="At least one review check is requested.", validator=_nonempty),
    Value("checks.check", Literal["identity", "units", "kinematics", "schemes", "grids", "resampling", "extrapolation"], physics="Review check ids are controlled."),
)

INPUT_RULES = (
    Depends("", "results", physics="Review scope is an explicit ordered list of prior results."),
    List("results", "result", physics="Review results are nonempty and preserve authored order.", validator=_nonempty),
    Value("results.result", dict, physics="Each review result is a job source."),
)

CHECKS = ()
