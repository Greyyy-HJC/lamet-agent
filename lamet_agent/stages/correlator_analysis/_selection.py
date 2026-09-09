"""Reference-compatible correlator candidate selection."""

from __future__ import annotations

from typing import Any

import numpy as np

from lamet_agent.stages.correlator_analysis._scope import parse_fit_scope


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def select_spectrum_candidate(candidates: list[dict[str, object]], *, q_min: float) -> tuple[dict[str, object], bool]:
    """Select the deterministic best finite-Q spectrum candidate."""
    usable = [
        candidate
        for candidate in candidates
        if not candidate.get("numerical_failure", False)
        and candidate.get("error") is None
        and _is_finite(candidate.get("Q"))
    ]
    if not usable:
        raise ValueError("no numerical spectrum candidate has finite Q")

    def rank(candidate: dict[str, object]) -> tuple[float, float, str]:
        value = candidate.get("chi2_dof")
        chi2_dof = float(value) if _is_finite(value) else np.inf
        return (-float(candidate["Q"]), chi2_dof, str(candidate["id"]))

    selected = min(usable, key=rank)
    return selected, float(selected["Q"]) < q_min


def select_data_window(
    candidates: list[dict[str, object]], *, q_min: float, chi2_dof_tolerance: float
) -> tuple[dict[str, object], bool]:
    """Apply the original information-preserving primary-z rule."""
    eligible = [
        candidate
        for candidate in candidates
        if not candidate.get("numerical_failure", False)
        and candidate.get("error") is None
        and int(candidate.get("n_data", 0)) > int(candidate.get("n_params", 0))
        and _is_finite(candidate.get("Q"))
        and _is_finite(candidate.get("chi2_dof"))
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
        if candidate.get("feasible_at_all_tune_z", True)
        and not candidate.get("numerical_failure", False)
        and candidate.get("error") is None
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
        and _is_finite(candidate.get("min_Q"))
        and _is_finite(candidate.get("worst_chi2_dof"))
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
    return selected, not any(float(candidate["min_Q"]) >= q_min for candidate in usable)


def dataset_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Identity of the observations used by one correlator candidate.

    Window and the complete scope pipeline freeze the likelihood data. nstate and
    prior_width are excluded so model averaging can combine those variants on one
    dataset.
    """
    window = candidate.get("window") if isinstance(candidate.get("window"), dict) else {}
    tseps = candidate.get("tsep_values")
    tsep_key = tuple(int(value) for value in tseps) if isinstance(tseps, (list, tuple)) else ()
    tau_min = window.get("tau_min")
    scope = candidate.get("fit_scope", [])
    scope_values = list(scope) if isinstance(scope, (list, tuple)) else [str(scope)]
    scope_key = parse_fit_scope(scope_values).key()
    return (
        str(candidate.get("method", "")),
        scope_key,
        int(window["tmin"]) if window.get("tmin") is not None else -1,
        int(window["tmax"]) if window.get("tmax") is not None else -1,
        int(tau_min) if tau_min is not None else -1,
        tsep_key,
    )


def models_on_dataset(candidates: list[dict[str, Any]], anchor: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every candidate that shares the anchor's frozen dataset."""
    key = dataset_key(anchor)
    return [candidate for candidate in candidates if dataset_key(candidate) == key]
