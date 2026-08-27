"""Reference-compatible correlator candidate selection."""

from __future__ import annotations

import numpy as np


def select_data_window(
    candidates: list[dict[str, object]], *, q_min: float, chi2_dof_tolerance: float
) -> tuple[dict[str, object], bool]:
    """Apply the original information-preserving primary-z rule."""
    eligible = [
        candidate
        for candidate in candidates
        if not candidate.get("numerical_failure", False)
        and int(candidate.get("n_data", 0)) > int(candidate.get("n_params", 0))
        and np.isfinite(float(candidate.get("chi2_dof", np.inf)))
    ]
    if not eligible:
        raise ValueError("no overdetermined matrix-fit candidate is available")
    passing = [candidate for candidate in eligible if float(candidate.get("Q", 0.0)) >= q_min]
    pool = passing or eligible
    best_chi2_dof = min(float(candidate["chi2_dof"]) for candidate in pool)
    comparable = [candidate for candidate in pool if float(candidate["chi2_dof"]) <= best_chi2_dof + chi2_dof_tolerance]
    selected = max(
        comparable,
        key=lambda candidate: (
            int(candidate["n_data"]),
            -float(candidate["chi2_dof"]),
            float(candidate.get("Q", 0.0)),
        ),
    )
    return selected, not bool(passing)


def select_tuned_candidate(
    candidates: list[dict[str, object]], *, q_min: float, chi2_dof_tolerance: float, qda: bool
) -> tuple[dict[str, object], bool]:
    """Select among candidates usable at every authored tuning separation."""
    feasible = [
        candidate
        for candidate in candidates
        if candidate.get("feasible_at_all_tune_z", True) and not candidate.get("numerical_failure", False)
    ]
    if not feasible:
        raise ValueError("no candidate is feasible at every tune_z value")
    if not qda:
        return select_data_window(
            feasible,
            q_min=q_min,
            chi2_dof_tolerance=chi2_dof_tolerance,
        )
    usable = [
        candidate
        for candidate in feasible
        if int(candidate.get("n_data", 0)) > int(candidate.get("n_params", 0))
        and np.isfinite(float(candidate.get("min_Q", np.nan)))
        and np.isfinite(float(candidate.get("worst_chi2_dof", np.nan)))
    ]
    if not usable:
        raise ValueError("no overdetermined qDA candidate is feasible at every tune_z value")
    selected = min(
        usable,
        key=lambda candidate: (
            -float(candidate["min_Q"]),
            float(candidate["worst_chi2_dof"]),
        ),
    )
    return selected, False
