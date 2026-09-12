"""Parsing and validation for composable correlator fit scopes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product


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


def resolve_atom_n_states(n_states: Mapping[str, int] | int, pipeline: FitScope) -> dict[str, int]:
    """Normalize a shared or per-atom state count onto the parsed pipeline atoms."""
    atoms = pipeline.atoms
    if isinstance(n_states, Mapping):
        if any("+" in str(key) for key in n_states):
            raise ValueError("n_states keys must be individual correlator atoms, not joint stage strings")
        missing = [atom for atom in atoms if atom not in n_states]
        extra = [atom for atom in n_states if atom not in atoms]
        if missing or extra:
            raise ValueError("n_states keys must match the correlator atoms in fit_scope")
        resolved = {atom: int(n_states[atom]) for atom in atoms}
    elif isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1:
        raise ValueError("n_states must be a positive integer or a per-atom mapping")
    else:
        resolved = {atom: n_states for atom in atoms}
    if any(isinstance(count, bool) or count < 1 for count in resolved.values()):
        raise ValueError("each atom n_states must be a positive integer")
    return resolved


def nstate_key(nstate: Mapping[str, int] | int) -> tuple[tuple[str, int], ...]:
    """Stable identity for one concrete per-atom or scalar state-count choice."""
    if isinstance(nstate, Mapping):
        return tuple(sorted((str(atom), int(count)) for atom, count in nstate.items()))
    return (("__scalar__", int(nstate)),)


def nstate_combinations(
    nstate: Mapping[str, Sequence[int]], fit_scope: Sequence[str]
) -> list[dict[str, int]]:
    """Expand one authored per-atom nstate grid into concrete count mappings."""
    atoms = parse_fit_scope(fit_scope).atoms
    if any("+" in str(key) for key in nstate):
        raise ValueError("nstate keys must be individual correlator atoms, not joint stage strings")
    missing = [atom for atom in atoms if atom not in nstate]
    if missing:
        raise ValueError(f"nstate is missing fit_scope atom {missing[0]!r}")
    extra = [atom for atom in nstate if atom not in atoms]
    if extra:
        raise ValueError(f"nstate has unexpected atom {extra[0]!r}")
    return [
        {atom: int(count) for atom, count in zip(atoms, combo, strict=True)}
        for combo in product(*(list(nstate[atom]) for atom in atoms))
    ]


def atom_state_count(n_states: Mapping[str, int], atom: str) -> int:
    """Return the authored state count for one correlator atom."""
    if atom not in n_states:
        raise ValueError(f"n_states is missing correlator atom {atom!r}")
    count = n_states[atom]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"n_states[{atom!r}] must be a positive integer")
    return count


__all__ = [
    "FIT_SCOPE_ATOMS",
    "FIT_SCOPE_ORDER",
    "FitScope",
    "ORDINARY_ATOMS",
    "QDA_ATOMS",
    "atom_state_count",
    "nstate_combinations",
    "nstate_key",
    "parse_fit_scope",
    "resolve_atom_n_states",
    "split_scope_stage",
    "valid_scope_stage",
]
