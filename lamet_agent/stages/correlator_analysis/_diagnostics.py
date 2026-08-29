"""Deterministic per-job LSQFit logs and sample-0 diagnostic figures."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import gvar as gv
import numpy as np

from lamet_agent.plotting import configure_plot, errorband, errorline, save_figure, series_color, start_plot


TSEP_LABEL = r"${t_{\mathrm{sep}}~/~a}$"
TAU_CENTER_LABEL = r"$(\tau - t_{\mathrm{sep}}/2)~/~a$"
RATIO_REAL_LABEL = r"$\Re\left[\mathcal{R}(t_{\mathrm{sep}},\tau)\right]$"
RATIO_IMAG_LABEL = r"$\Im\left[\mathcal{R}(t_{\mathrm{sep}},\tau)\right]$"
FH_REAL_LABEL = r"$\Re\left[\mathrm{FH}(t_{\mathrm{sep}})\right]$"
FH_IMAG_LABEL = r"$\Im\left[\mathrm{FH}(t_{\mathrm{sep}})\right]$"
QDA_TIME_LABEL = r"$t/a$"
QDA_RATIO_REAL_LABEL = r"$\Re\left[R_{\mathrm{qDA}}(t)\right]$"
QDA_RATIO_IMAG_LABEL = r"$\Im\left[R_{\mathrm{qDA}}(t)\right]$"

# Leave asymmetric room above the data for the legend, without emptying the panel.
_FIT_LOG_BOTTOM_MARGIN_FACTOR = 0.35
_FIT_LOG_TOP_MARGIN_FACTOR = 0.70


@dataclass(frozen=True)
class FitArtifactResult:
    artifacts: tuple[str, ...]
    sample_fit_quality: dict[str, Any]
    dispersion_energy: dict[str, Any]
    application_fit: dict[str, Any]


def _slug(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number)).replace("-", "m")
    return f"{number:g}".replace("-", "m").replace(".", "p")


def _metric_text(metrics: Mapping[str, Any]) -> str:
    return (
        f"Q={float(metrics['Q']):.6g} "
        f"chi2/dof={float(metrics['chi2_dof']):.6g} "
        f"chi2={float(metrics['chi2']):.6g} "
        f"dof={float(metrics['dof']):.6g} "
        f"logGBF={float(metrics['logGBF']):.6g}"
    )


def _sample0_plot_labels(kind: str, component: str) -> tuple[str, str]:
    labels = {
        ("pt3_ratio", "re"): (TAU_CENTER_LABEL, RATIO_REAL_LABEL),
        ("pt3_ratio", "im"): (TAU_CENTER_LABEL, RATIO_IMAG_LABEL),
        ("fh", "re"): (TSEP_LABEL, FH_REAL_LABEL),
        ("fh", "im"): (TSEP_LABEL, FH_IMAG_LABEL),
        ("qda_ratio", "re"): (QDA_TIME_LABEL, QDA_RATIO_REAL_LABEL),
        ("qda_ratio", "im"): (QDA_TIME_LABEL, QDA_RATIO_IMAG_LABEL),
    }
    try:
        return labels[(kind, component)]
    except KeyError as exc:
        raise ValueError(f"unsupported sample-0 plot kind/component: {kind}/{component}") from exc


def _sample0_ylim(payload: Mapping[str, Any]) -> tuple[float, float] | None:
    kind = str(payload["kind"])
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for item in payload["series"]:
        pairs = [(item["y"], item["yerr"])]
        if kind == "fh":
            pairs.append((item["fit_mean"], item["fit_sdev"]))
        for values, errors in pairs:
            value_array = np.asarray(values, dtype=float)
            error_array = np.asarray(errors, dtype=float)
            finite = np.isfinite(value_array) & np.isfinite(error_array)
            if np.any(finite):
                lows.append(value_array[finite] - error_array[finite])
                highs.append(value_array[finite] + error_array[finite])
    if not lows:
        return None
    data_min = float(np.min(np.concatenate(lows)))
    data_max = float(np.max(np.concatenate(highs)))
    span = data_max - data_min
    if span <= 0.0:
        scale = max(abs(data_min), 1.0) * 1.0e-6
        span = 2.0 * scale
    return (
        data_min - _FIT_LOG_BOTTOM_MARGIN_FACTOR * span,
        data_max + _FIT_LOG_TOP_MARGIN_FACTOR * span,
    )


def _write_sample0_plot(path: Path, payload: Mapping[str, Any], *, job_id: str) -> None:
    series = payload.get("series", [])
    if not isinstance(series, list) or not series:
        raise ValueError("sample-0 plot payload requires at least one series")
    start_plot()
    fit_x_values: list[float] = []
    for index, item in enumerate(series):
        color = series_color(index)
        x = np.asarray(item["x"], dtype=float)
        fit_x = np.asarray(item["fit_x"], dtype=float)
        fit_x_values.extend(fit_x.tolist())
        errorline(
            x,
            gv.gvar(np.asarray(item["y"], dtype=float), np.asarray(item["yerr"], dtype=float)),
            color=color,
            label=str(item["label"]),
        )
        errorband(
            fit_x,
            gv.gvar(np.asarray(item["fit_mean"], dtype=float), np.asarray(item["fit_sdev"], dtype=float)),
            color=color,
        )
    if fit_x_values:
        plateau_x = np.asarray([min(fit_x_values), max(fit_x_values)], dtype=float)
        errorband(
            plateau_x,
            gv.gvar(
                np.full(2, float(payload["plateau_mean"])),
                np.full(2, float(payload["plateau_sdev"])),
            ),
            color="0.2",
            label="Sample-0 fit matrix element",
        )
    component = str(payload["component"])
    kind = str(payload["kind"])
    xlabel, ylabel = _sample0_plot_labels(kind, component)
    configure_plot(
        xlabel=xlabel,
        ylabel=ylabel,
        ylim=_sample0_ylim(payload),
        legend=True,
        title=f"{job_id}: sample 0, z={payload['z']:g}",
    )
    save_figure(path)


def _candidate_log_lines(candidates: list[Mapping[str, Any]]) -> list[str]:
    lines = ["Candidate scan"]
    for candidate in sorted(candidates, key=lambda item: str(item.get("id", item.get("candidate_id", "")))):
        candidate_id = candidate.get("id", candidate.get("candidate_id"))
        metrics = ""
        if all(key in candidate for key in ("Q", "chi2", "dof", "chi2_dof", "logGBF")):
            metrics = " " + _metric_text(candidate)
        lines.append(
            f"candidate={candidate_id} method={candidate.get('method')} window={candidate.get('window')} "
            f"nstate={candidate.get('nstate')} prior_width={candidate.get('prior_width')} "
            f"accepted={candidate.get('quality_passed')} "
            f"numerical_failure={candidate.get('numerical_failure')}{metrics}"
        )
        tuning = candidate.get("tune_z_diagnostics", {})
        if isinstance(tuning, Mapping):
            for z_value, fit in sorted(tuning.items(), key=lambda item: float(item[0])):
                if isinstance(fit, Mapping) and all(key in fit for key in ("Q", "chi2", "dof", "chi2_dof", "logGBF")):
                    lines.append(f"  tune_z={z_value} {_metric_text(fit)}")
    return lines


def write_fit_artifacts(
    *,
    job_id: str,
    selected: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    preflight_fit: Mapping[str, Any] | None,
    application_fit: Mapping[str, Any],
    application_rejections: list[Mapping[str, Any]],
    artifact_directory: Path,
    component: str,
    q_min: float,
) -> FitArtifactResult:
    """Write logs/PDFs and return compact diagnostics for persisted summaries."""
    strategy = str(selected.get("method", selected.get("fit_strategy", "fit")))
    scope = str(selected.get("fit_scope", "qda_ratio" if strategy == "qda" else "fit"))
    stem = f"{job_id}_{strategy}_{scope}"
    log_directory = artifact_directory / "fit_logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    tuning_path = log_directory / f"{stem}_tuning.log"
    sample_path = log_directory / f"{stem}_samples.log"

    tuning_lines = [
        f"job={job_id} selected_candidate={selected.get('id')} strategy={strategy} scope={scope}",
        f"window={selected.get('window')} nstate={selected.get('nstate')} prior_width={selected.get('prior_width')}",
        *_candidate_log_lines(candidates),
        "Full-grid center preflight",
    ]
    if isinstance(preflight_fit, Mapping):
        for fit in preflight_fit.get("fits", []):
            if isinstance(fit, Mapping):
                tuning_lines.append(f"z={fit.get('z')} {_metric_text(fit)}")
    if application_rejections:
        tuning_lines.append(f"application_rejections={application_rejections}")
    tuning_path.write_text("\n".join(tuning_lines).rstrip() + "\n", encoding="utf-8")

    sample_lines = [f"job={job_id} selected_candidate={selected.get('id')} strategy={strategy} scope={scope}"]
    q_values: list[float] = []
    chi2_values: list[float] = []
    failed_count = 0
    allowed_components = {"re": {"re"}, "im": {"im"}, "both": {"re", "im"}}[component]
    plot_artifacts: list[str] = []
    fits = application_fit.get("fits", [])
    for fit in fits:
        if not isinstance(fit, Mapping):
            continue
        z_value = fit.get("z")
        sample_lines.append(f"=== z={z_value} ===")
        sample_lines.append(f"center {_metric_text(fit)}")
        for record in fit.get("sample_diagnostics", []):
            if not isinstance(record, Mapping):
                continue
            status = "Good" if float(record["Q"]) >= q_min else "Bad"
            sample_lines.append(f"{status} sample={record['sample']} {_metric_text(record)}")
            q = float(record["Q"])
            chi2_dof = float(record["chi2_dof"])
            if np.isfinite(q):
                q_values.append(q)
            if np.isfinite(chi2_dof):
                chi2_values.append(chi2_dof)
        failed_here = int(fit.get("n_failed_samples", 0))
        failed_count += failed_here
        sample_lines.append(f"summary z={z_value} failed={failed_here}")
        plot_payload = fit.get("sample0_plot")
        if isinstance(plot_payload, Mapping):
            for plot in plot_payload.get("plots", []):
                if not isinstance(plot, Mapping) or str(plot.get("component")) not in allowed_components:
                    continue
                relative = (
                    Path("fit_logs")
                    / "plots"
                    / f"{stem}_z{_slug(z_value)}_sample0_{plot['kind']}_{plot['component']}.pdf"
                )
                _write_sample0_plot(artifact_directory / relative, {**plot, "z": float(z_value)}, job_id=job_id)
                plot_artifacts.append(relative.as_posix())
    for failure in application_fit.get("sample_failures", []):
        if isinstance(failure, Mapping):
            sample_lines.append(
                f"Failed z={failure.get('z')} sample={failure.get('sample')} error={failure.get('error')}"
            )
    sample_path.write_text("\n".join(sample_lines).rstrip() + "\n", encoding="utf-8")

    dispersion_energy: dict[str, Any] = {}
    if strategy != "qda":
        energy_fits = [
            fit
            for fit in fits
            if isinstance(fit, Mapping)
            and isinstance(fit.get("E0_samples"), list)
            and fit.get("E0_samples")
            and all(value is not None for value in fit["E0_samples"])
        ]
        if energy_fits:
            reference = min(energy_fits, key=lambda fit: (abs(float(fit["z"])), float(fit["z"])))
            dispersion_energy = {
                "z": float(reference["z"]),
                "energy_unit": "lattice",
                "E0_samples": [float(value) for value in reference["E0_samples"]],
            }

    compact_application = copy.deepcopy(dict(application_fit))
    for fit in compact_application.get("fits", []):
        if not isinstance(fit, dict):
            continue
        for key in (
            "sample_diagnostics",
            "sample0_plot",
            "E0_samples",
            "E0_i_samples",
            "E0_f_samples",
        ):
            fit.pop(key, None)
    return FitArtifactResult(
        artifacts=(
            tuning_path.relative_to(artifact_directory).as_posix(),
            sample_path.relative_to(artifact_directory).as_posix(),
            *plot_artifacts,
        ),
        sample_fit_quality={
            "Q": q_values,
            "chi2_dof": chi2_values,
            "n_successful": len(q_values),
            "n_failed": failed_count,
        },
        dispersion_energy=dispersion_energy,
        application_fit=compact_application,
    )


__all__ = ["FitArtifactResult", "write_fit_artifacts"]
