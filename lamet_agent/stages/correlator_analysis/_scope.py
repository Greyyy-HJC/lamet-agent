"""Parsing and validation for composable correlator fit scopes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


FIT_SCOPE_ORDER = ("2pt", "3pt", "qda", "FH", "3pt_ratio", "qda_ratio")
FIT_SCOPE_ATOMS = frozenset(FIT_SCOPE_ORDER)
QDA_ATOMS = frozenset({"qda", "qda_ratio"})
ORDINARY_ATOMS = frozenset({"3pt", "3pt_ratio", "FH"})


@dataclass(frozen=True)
class FitScope:
    """One validated ordered pipeline of joint fit stages."""

    stages: tuple[tuple[str, ...], ...]

    @property
    def atoms(self) -> tuple[str, ...]:
        return tuple(atom for stage in self.stages for atom in stage)

    @property
    def atom_set(self) -> frozenset[str]:
        return frozenset(self.atoms)

    @property
    def is_qda(self) -> bool:
        return bool(self.atom_set & QDA_ATOMS)

    @property
    def is_spectrum(self) -> bool:
        return self.stages == (("2pt",),)

    @property
    def needs_pt2_data(self) -> bool:
        return bool(self.atom_set & {"2pt", "3pt_ratio", "qda_ratio", "FH"})

    @property
    def needs_pt3_data(self) -> bool:
        return bool(self.atom_set & ORDINARY_ATOMS)

    @property
    def final_stage(self) -> tuple[str, ...]:
        return self.stages[-1]

    def as_list(self) -> list[str]:
        return ["+".join(stage) for stage in self.stages]

    def key(self) -> tuple[str, ...]:
        atom_order = {atom: index for index, atom in enumerate(FIT_SCOPE_ORDER)}
        return tuple("+".join(sorted(stage, key=atom_order.__getitem__)) for stage in self.stages)


def split_scope_stage(value: str) -> tuple[str, ...]:
    """Split one joint stage and reject malformed or duplicate atoms."""
    if not isinstance(value, str) or not value:
        raise ValueError("fit_scope stages must be nonempty strings")
    atoms = tuple(value.split("+"))
    if any(not atom for atom in atoms) or any(atom not in FIT_SCOPE_ATOMS for atom in atoms):
        allowed = ", ".join(sorted(FIT_SCOPE_ATOMS))
        raise ValueError(f"fit_scope atoms must be selected from {allowed}")
    if len(set(atoms)) != len(atoms):
        raise ValueError("a joint fit_scope stage cannot repeat an atom")
    return atoms


def valid_scope_stage(value: str) -> bool:
    try:
        split_scope_stage(value)
    except (TypeError, ValueError):
        return False
    return True


def parse_fit_scope(values: Sequence[str] | Iterable[str]) -> FitScope:
    """Return a validated ordered fit pipeline.

    List order denotes chained fits and ``+`` denotes a joint likelihood.
    """
    scopes = list(values)
    if not scopes:
        raise ValueError("fit_scope must contain at least one stage")
    stages = tuple(split_scope_stage(value) for value in scopes)
    atoms = tuple(atom for stage in stages for atom in stage)
    if len(set(atoms)) != len(atoms):
        raise ValueError("a fit_scope pipeline cannot repeat an atom")
    if atoms.count("2pt") > 1 or ("2pt" in atoms and "2pt" not in stages[0]):
        raise ValueError("2pt may appear once and only in the first chained stage")
    atom_set = set(atoms)
    if atom_set & QDA_ATOMS and atom_set & ORDINARY_ATOMS:
        raise ValueError("qDA and three-point/FH fit scopes cannot be mixed")
    if {"qda", "qda_ratio"}.issubset(atom_set):
        raise ValueError("raw qda and qda_ratio cannot be fitted in the same pipeline")
    if {"3pt", "3pt_ratio"}.issubset(atom_set):
        raise ValueError("raw 3pt and 3pt_ratio cannot be fitted in the same pipeline")
    if atom_set == {"2pt"} and stages != (("2pt",),):
        raise ValueError("a 2pt-only fit_scope must be exactly ['2pt']")
    return FitScope(stages)


__all__ = [
    "FIT_SCOPE_ATOMS",
    "FIT_SCOPE_ORDER",
    "FitScope",
    "ORDINARY_ATOMS",
    "QDA_ATOMS",
    "parse_fit_scope",
    "split_scope_stage",
    "valid_scope_stage",
]
