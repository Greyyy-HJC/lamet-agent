"""Build one sample-bearing spectral candidate."""

from __future__ import annotations

import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.stages.correlator_analysis.physics import fit_spectrum_samples


def run(
    context: ToolContext,
    *,
    tmin: int,
    tmax: int,
    n_states: int,
    prior_means: dict[str, float],
    prior_widths: dict[str, float],
) -> dict[str, object]:
    """Fit a positive two-point correlator and append a candidate."""
    lsqfit = context.params
    if "spectrum" not in lsqfit["fit_scope"]:
        raise ValueError("spectrum fitting is not allowed for this job")
    if tmin >= tmax:
        raise ValueError("spectrum fit window must be increasing")
    if n_states not in context.params["nstate"]:
        raise ValueError("n_states must be selected from the authored candidate list")
    correlators = context.state.get("correlators")
    if not correlators:
        raise RuntimeError("inspect_correlators must run before fit_spectrum")
    source = next(
        (value for value in correlators.values() if value.attrs.get("correlator_type") == "two_point"),
        next(iter(correlators.values())),
    )
    if "t" not in source.dims:
        raise ValueError("spectrum fitting requires a t coordinate")
    time = np.asarray(source.coords["t"])
    requested = np.arange(tmin, tmax, dtype=float)
    if any(not np.any(np.isclose(time, value, rtol=0.0, atol=1e-12)) for value in requested):
        raise ValueError("spectrum fit window is not covered by input time coordinates")
    selection = (time >= tmin) & (time < tmax)
    if selection.sum() < 2 * n_states:
        raise ValueError("spectrum fit window must contain at least 2*n_states times")
    if source.dims != ["t"]:
        raise ValueError("direct spectrum fitting requires only the t physical dimension")
    energy_samples, fit = fit_spectrum_samples(
        np.asarray(source.values)[:, selection],
        time[selection],
        n_states,
        resample=source.resample,
        prior_means=prior_means,
        prior_widths=prior_widths,
        sample_error_mode=str(context.manifest["metadata"]["sample_error_mode"]),
        workers=context.workers,
        _parallel=context._parallel,
    )
    energy_samples = [energies * HBAR_C_GEV_FM / float(source.ensemble.a_t) for energies in energy_samples]
    attrs = dict(source.attrs)
    attrs.update(
        {
            "observable": "spectrum",
            "method": "direct_fit",
            "fit_energy_unit": "lattice",
            "sample_error_mode": context.manifest["metadata"]["sample_error_mode"],
            "units": '{"values":"GeV","state":"index"}',
        }
    )
    candidate = EnsembleData(
        source.ensemble,
        source.resample,
        energy_samples,
        ["state"],
        {"state": list(range(n_states))},
        attrs=attrs,
        name="spectrum",
    )
    candidates = context.state.setdefault("spectrum_candidates", [])
    candidate_id = f"spectrum_{len(candidates) + 1:03d}"
    candidates.append(
        {
            "id": candidate_id,
            "method": "direct_fit",
            "observable": "spectrum",
            "window": {"tmin": tmin, "tmax": tmax},
            "data": candidate,
            "prior_means": dict(prior_means),
            "prior_widths": dict(prior_widths),
            **fit,
        }
    )
    return {
        "summary": f"stored spectral candidate {candidate_id}",
        "metrics": {"candidate_id": candidate_id, "energy_mean_gev": np.asarray(candidate.mean).tolist(), **fit},
        "state_keys": ["spectrum_candidates"],
        "artifacts": [],
    }
