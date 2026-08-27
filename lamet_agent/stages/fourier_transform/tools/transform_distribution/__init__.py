"""Transform selected tail candidates and publish the x-space result."""

from __future__ import annotations

import json
import numpy as np

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.plotting import (
    X_LABEL,
    configure_plot,
    errorband,
    momentum_label,
    quasi_distribution_label,
    save_figure,
    start_plot,
)
from lamet_agent.stages.fourier_transform.physics import fourier_transform


def run(context: ToolContext, *, candidate_id: str) -> dict[str, object]:
    """Transform one selected candidate and finish the stage."""
    candidates = {candidate["id"]: candidate for candidate in context.state.get("tail_candidates", [])}
    if candidate_id not in candidates:
        raise ValueError("candidate_id must name an existing tail candidate")
    selected = candidates[candidate_id]
    conventions = context.state.get("fourier_conventions")
    if not isinstance(conventions, dict):
        raise RuntimeError("inspect_long_distance did not derive Fourier conventions")
    transform = conventions["transform"]
    data = selected["data"]
    momentum = data.attrs.get("momentum_gev")
    if not isinstance(momentum, (int, float)) or not np.isfinite(float(momentum)) or float(momentum) <= 0:
        raise ValueError("Fourier input requires finite positive momentum_gev")
    grid = context.params["quasi_y_ls"]
    if isinstance(grid, dict):
        grid = np.linspace(float(grid["start"]), float(grid["stop"]), int(grid["num"])).tolist()
    result = fourier_transform(
        data,
        grid,
        momentum_gev=float(momentum),
        phase_sign=int(transform["phase_sign"]),
        x_shift=float(transform["x_shift"]),
        prefactor=str(transform["prefactor"]),
        workers=context.workers,
    )
    attrs = result.attrs
    attrs.update(
        {
            "target_observable": context.manifest["metadata"]["target_observable"],
            "parton": conventions["parton"],
            "gfix": conventions["gfix"],
            "tail_model": selected["model_id"],
        }
    )
    result = EnsembleData(
        result.ensemble,
        result.resample,
        [sample for sample in result.values],
        result.dims,
        result.coords,
        attrs=attrs,
        name=result.name,
    )
    context.state["fourier_result"] = {"data": result, "candidate_id": candidate_id}
    result.to_netcdf(context.artifact_directory / "output.nc")
    diagnostics = {
        "candidate_id": candidate_id,
        "tail_model": attrs["tail_model"],
        "sample_count": result.n_sample,
        **{key: selected[key] for key in ("chi2", "dof", "chi2_dof", "Q", "aic")},
    }
    (context.artifact_directory / "diagnostics").mkdir(exist_ok=True)
    (context.artifact_directory / "diagnostics" / "fourier.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    start_plot()
    sample_error_mode = str(context.manifest.get("metadata", {}).get("sample_error_mode", "covariance"))
    errorband(
        result.coords["x"],
        result.real.average(sample_error_mode),
        label=momentum_label(momentum),
    )
    configure_plot(xlabel=X_LABEL, ylabel=quasi_distribution_label("real"), legend=True)
    save_figure(context.artifact_directory / "plots" / "distribution.pdf")
    (context.artifact_directory / "report.md").write_text(
        f"# Fourier transform\n\nTail model: `{attrs['tail_model']}`.\n", encoding="utf-8"
    )
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": "quasi_distribution",
        "decisions": {"candidate_id": candidate_id},
        "diagnostics": diagnostics,
        "artifacts": ["output.nc", "diagnostics/fourier.json", "plots/distribution.pdf", "report.md"],
    }
    context.finish(result, summary)
    return {
        "summary": "published quasi distribution",
        "metrics": diagnostics,
        "state_keys": [],
        "artifacts": summary["artifacts"],
    }
