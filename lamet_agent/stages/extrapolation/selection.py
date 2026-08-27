"""Deterministic selection for the current single-model extrapolation path."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def select_single_candidate(candidates: Sequence[Mapping[str, Any]]) -> tuple[Any, dict[str, Any]]:
    """Return the sole fitted candidate and its unit-weight comparison record."""
    if len(candidates) != 1:
        raise ValueError("the reference extrapolation requires exactly one fitted candidate")
    candidate = candidates[0]
    spread = np.zeros_like(np.asarray(candidate["data"].mean, dtype=float))
    comparison = {
        "candidate_ids": [candidate["id"]],
        "criterion": "single_authored_model",
        "weights": [1.0],
        "between_model_sdev": spread.tolist(),
        "statistical_source": "reference sample-level fit",
        "stability_sigma": 0.0,
        "candidates": [
            {
                "candidate_id": candidate["id"],
                "terms": candidate["terms"],
                "excluded_ensembles": candidate["excluded_ensembles"],
                "chi2": candidate["chi2"],
                "dof": candidate["dof"],
                "chi2_dof": candidate["chi2_dof"],
                "Q": candidate["Q"],
                "aic": candidate["aic"],
                "parameter_mean": candidate["parameter_mean"],
                "parameter_sdev": candidate["parameter_sdev"],
                "momentum_dependence": candidate["momentum_dependence"],
                "weight": 1.0,
            }
        ],
    }
    return candidate["data"], comparison


__all__ = ["select_single_candidate"]
