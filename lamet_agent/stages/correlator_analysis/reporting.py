"""Stage-level correlator-analysis reporting."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from lamet_agent.stages._reporting import (
    StageReportRecord,
    artifact_rows,
    describe_grid,
    format_value,
    output_attrs,
    stage_overlay_lines,
    write_report,
)


_BREIT_METHOD = r"""
The least-squares branch fixes its candidate grid on sample-average data and
then applies the selected model to every resample.  For an ordinary forward
matrix element the two-point and ratio models are

$$
C_2(t)=\sum_n \frac{z_n^2}{2E_n}\left(e^{-E_nt}+e^{-E_n(L_t-t)}\right),
\qquad
R(t,\tau,z)=\frac{1}{C_2(t)}\sum_{m,n}
\frac{O_{mn}(z)z_mz_n}{(2E_m)(2E_n)}
e^{-E_m(t-\tau)}e^{-E_n\tau}.
$$

The published Breit matrix element is $O_{00}(z)/(2E_0)$.
Candidate quality is evaluated at the tool-selected tuning z values; the
chosen window and model are held fixed for the full-z sample fits.
""".strip()


_NONBREIT_METHOD = r"""
For non-forward kinematics, separate initial and final spectra enter the
symmetrized ratio,

$$
R_{\rm NB}(t,\tau,z)=\frac{C_3^{f\leftarrow i}(t,\tau,z)}{C_2^f(t)}
\left[\frac{C_2^i(t-\tau)C_2^f(\tau)C_2^f(t)}
{C_2^f(t-\tau)C_2^i(\tau)C_2^i(t)}\right]^{1/2},
\qquad
h_{\rm NB}(z)=\operatorname{sign}(z_{0,i}z_{0,f})
\frac{O_{00}(z)}{E_{0,i}+E_{0,f}}.
""".strip()


_QDA_METHOD = r"""
The qDA fit uses the nonlocal/local two-point ratio at each spatial separation
and extracts the corresponding matrix element from its constant one-state
ratio.  The selected time window is applied to every production sample.
""".strip()


_FH_METHOD = r"""
For a Feynman--Hellmann scope, the summed ratio and its finite-difference slope
are

$$S(t)=\sum_{\tau=\tau_c}^{t-\tau_c}R(t,\tau),\qquad
R_{\rm FH}(t)=\frac{S(t+\Delta t)-S(t)}{\Delta t}.$$
""".strip()


_CHI2_DOF_XLIM_MAX = 4.0

_LANCZOS_METHOD = r"""
The Lanczos branch constructs the transfer-matrix Krylov problem directly from
resampled correlator moments. Its iteration count is determined by the usable
time grid, while the requested state count controls the exported Ritz states or
diagnostic state matrix. Nested resampling and Cullum--Willoughby filtering are
performed before the median Lanczos result is published.
""".strip()


def _scope_name(scope: object) -> str:
    return {
        "spectrum": "2pt spectrum",
        "3pt_ratio": "3pt ratio",
        "FH": "Feynman--Hellmann",
        "3pt_ratio+FH": "3pt ratio + Feynman--Hellmann",
        "qda_ratio": "qDA nonlocal/local ratio",
        "2pt_spectrum": "2pt spectrum",
        "3pt_matrix": "3pt matrix element",
    }.get(str(scope), str(scope))


def _method_name(method: object, scope: object) -> str:
    method_text = str(method)
    scope_text = str(scope)
    if method_text == "qda" or scope_text == "qda_ratio":
        return "qDA nonlocal/local ratio fit"
    if method_text == "lanczos":
        return f"Lanczos {_scope_name(scope)} extraction"
    if scope_text == "spectrum":
        return "2pt spectrum fit"
    if method_text == "joint":
        return f"2pt + {_scope_name(scope)} joint fit"
    if method_text == "chained":
        return f"2pt + {_scope_name(scope)} chained fit"
    if method_text == "independent":
        return f"{_scope_name(scope)} independent fit"
    return f"{_scope_name(scope)} fit"


def _candidate_for_record(record: StageReportRecord) -> Mapping[str, object] | None:
    diagnostics = record.summary.get("diagnostics", {})
    decisions = record.summary.get("decisions", {})
    candidates = diagnostics.get("candidates", []) if isinstance(diagnostics, Mapping) else []
    candidate_id = decisions.get("candidate_id") if isinstance(decisions, Mapping) else None
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") == candidate_id:
            return candidate
    return None


def _candidate_scope(candidate: Mapping[str, object] | None, record: StageReportRecord) -> str:
    if candidate is not None and candidate.get("fit_scope") is not None:
        return str(candidate["fit_scope"])
    scopes = record.params.get("fit_scope", record.params.get("scope", []))
    if isinstance(scopes, (list, tuple)):
        return str(scopes[0]) if scopes else "n/a"
    return str(scopes)


def _window_text(candidate: Mapping[str, object] | None) -> str:
    if candidate is None:
        return "window not recorded"
    window = candidate.get("window", {})
    if not isinstance(window, Mapping):
        return "window not recorded"
    details: list[str] = []
    if window.get("tmin") is not None and window.get("tmax") is not None:
        details.append(f"2pt window [{window['tmin']}, {window['tmax']})")
    tseps = candidate.get("tsep_values")
    if isinstance(tseps, (list, tuple)) and tseps:
        details.append("t_sep=" + ", ".join(str(value) for value in tseps))
    if window.get("tau_min") is not None:
        details.append(f"tau cut={window['tau_min']}")
    if candidate.get("nstate") is not None:
        state_count = candidate["nstate"]
        details.append(f"{state_count} state" + ("s" if str(state_count) != "1" else ""))
    return "; ".join(details) or "window not recorded"


def _selected_fit_text(record: StageReportRecord) -> str:
    candidate = _candidate_for_record(record)
    decisions = record.summary.get("decisions", {})
    method = (
        candidate.get("method")
        if candidate is not None
        else (decisions.get("method") if isinstance(decisions, Mapping) else record.params.get("analysis_method"))
    )
    scope = _candidate_scope(candidate, record)
    description = _method_name(method, scope)
    if candidate is not None:
        return f"{description}; {_window_text(candidate)}"
    if str(method) == "lanczos":
        diagnostics = record.summary.get("diagnostics", {})
        inspection = diagnostics.get("inspection", {}) if isinstance(diagnostics, Mapping) else {}
        iterations = inspection.get("iterations") if isinstance(inspection, Mapping) else None
        states = record.params.get("nstate")
        if isinstance(states, (list, tuple)) and states:
            states = states[0]
        details = []
        if iterations is not None:
            details.append(f"{iterations} Krylov iterations")
        if states is not None:
            details.append(f"{states} exported state" + ("s" if str(states) != "1" else ""))
        return f"{description}; " + "; ".join(details) if details else description
    return description


def _method_lines(records: tuple[StageReportRecord, ...]) -> list[str]:
    lines = ["## Method", ""]
    lsq_records = [record for record in records if record.params.get("analysis_method") == "lsqfit"]
    forms = {str(record.params.get("fitting_form")) for record in lsq_records if record.params.get("fitting_form")}
    scopes = {str(scope) for record in lsq_records for scope in record.params.get("fit_scope", [])}
    strategies = {str(strategy) for record in lsq_records for strategy in record.params.get("fit_strategy", [])}
    if "Breit" in forms:
        lines.extend([_BREIT_METHOD, ""])
    if "NonBreit" in forms:
        lines.extend([_NONBREIT_METHOD, ""])
    if "qda_ratio" in scopes:
        lines.extend([_QDA_METHOD, ""])
    if scopes & {"FH", "3pt_ratio+FH"}:
        lines.extend([_FH_METHOD, ""])
    if strategies:
        strategy_text = {
            "joint": "2pt and matrix-element data are fit jointly with shared parameters",
            "chained": "the matrix-element fit uses the preceding 2pt posterior as propagated input",
            "independent": "the selected ratio or spectrum is fit without a shared 2pt likelihood",
        }
        lines.append(
            "Fit strategy: " + "; ".join(strategy_text.get(strategy, strategy) for strategy in sorted(strategies)) + "."
        )
        lines.append("")
    if any(record.params.get("analysis_method") == "lanczos" for record in records):
        lines.extend([_LANCZOS_METHOD, ""])
    return lines


def _sample_quality_lines(records: tuple[StageReportRecord, ...], artifact_directory: Path) -> list[str]:
    """Plot selected-production per-sample chi2/dof distributions."""
    import numpy as np

    from lamet_agent.plotting import COLOR_CYCLE, configure_plot, histogram, save_figure, series_color, start_plot

    chi2_series: list[tuple[str, np.ndarray]] = []
    for record in records:
        quality = record.summary.get("diagnostics", {}).get("sample_fit_quality", {})
        values = np.asarray(quality.get("chi2_dof", []), dtype=float) if isinstance(quality, dict) else np.asarray([])
        values = values[np.isfinite(values)]
        if values.size:
            chi2_series.append((record.job_id, values))
    if not chi2_series:
        if any(record.params.get("analysis_method") == "lanczos" for record in records):
            return [
                "Lanczos production uses nested resampling and median aggregation rather than sample-wise "
                "nonlinear fit-quality diagnostics."
            ]
        return ["No successful production sample-fit quality diagnostics were available."]
    lines = [
        "Distributions include every finite result from the selected-window production sample fits, including fits "
        "with Q below the acceptance threshold. Numerical failures are omitted from the distributions and counted "
        "separately.",
        "",
    ]
    pooled_values = np.concatenate([values for _label, values in chi2_series])
    visible = pooled_values[pooled_values <= _CHI2_DOF_XLIM_MAX]
    if visible.size:
        low = float(np.min(visible))
        high = float(np.max(visible))
        bin_source = visible
    else:
        low, high = 0.0, _CHI2_DOF_XLIM_MAX
        bin_source = pooled_values
    if high <= low:
        padding = 0.05 if low == 0.0 else abs(low) * 0.05
        low, high = low - padding, min(high + padding, _CHI2_DOF_XLIM_MAX)
    automatic = max(1, int(np.histogram_bin_edges(bin_source, bins="auto").size - 1))
    bins = np.linspace(low, high, max(1, int(np.round(automatic * 1.5))) + 1)
    start_plot()
    for index, (label, values) in enumerate(chi2_series):
        histogram(
            values, bins, color=series_color(index), label=label, histtype="stepfilled", alpha=0.45, linewidth=0.8
        )
    histogram(pooled_values, bins, color=COLOR_CYCLE[3], label="All", histtype="step", linewidth=2.2)
    span = float(bins[-1] - bins[0])
    padding = 0.02 * span if span > 0 else 0.05
    configure_plot(
        xlabel=r"$\chi^2/\mathrm{d.o.f.}$",
        ylabel="Counts",
        xlim=(float(bins[0]) - padding, min(float(bins[-1]) + padding, _CHI2_DOF_XLIM_MAX)),
        legend=True,
    )
    chi2_pdf = artifact_directory / "plots" / "sample_fit_quality_chi2.pdf"
    chi2_svg = artifact_directory / "plots" / "sample_fit_quality_chi2.svg"
    save_figure(chi2_pdf, chi2_svg)
    lines.extend(
        [
            r"![Histogram of per-sample chi2/dof](plots/sample_fit_quality_chi2.svg)",
            "",
            "[Histogram of per-sample chi2/dof (PDF)](plots/sample_fit_quality_chi2.pdf)",
        ]
    )
    return lines


_Z_PLOT_PATTERN = re.compile(r"_z(?P<z>m?\d+(?:p\d+)?)_sample0_")


def _z_token_value(token: str) -> float:
    return float(token.replace("m", "-").replace("p", "."))


def _representative_figure_lines(record: StageReportRecord, stage_directory: Path) -> list[str]:
    """Render the result plot and plots for the first, middle, and last z values."""
    raw = record.summary.get("artifacts", [])
    if not isinstance(raw, list):
        raise TypeError(f"job '{record.job_id}' summary.artifacts must be a list")
    plot_paths: list[tuple[str, Path]] = []
    by_z: dict[float, list[tuple[str, Path]]] = {}
    for relative in raw:
        if not isinstance(relative, str) or not (relative.startswith("plots/") or "/plots/" in f"/{relative}"):
            continue
        path = (record.artifact_directory / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"job '{record.job_id}' declared missing plot: {path}")
        match = _Z_PLOT_PATTERN.search(Path(relative).name)
        if match is None:
            plot_paths.append((relative, path))
        else:
            by_z.setdefault(_z_token_value(match.group("z")), []).append((relative, path))
    selected_z: list[float] = []
    z_values = sorted(by_z)
    for index in (0, len(z_values) // 2, len(z_values) - 1):
        if z_values and z_values[index] not in selected_z:
            selected_z.append(z_values[index])
    selected_paths = plot_paths + [item for z_value in selected_z for item in by_z[z_value]]
    if not selected_paths:
        return ["No plot artifacts were declared."]
    lines: list[str] = []
    for relative, path in selected_paths:
        link = path.relative_to(stage_directory.resolve()).as_posix()
        label = f"{record.job_id}: {Path(relative).stem}"
        lines.append(f"![{label}]({link})" if path.suffix.lower() == ".svg" else f"[{label}]({link})")
    if z_values and len(z_values) > len(selected_z):
        shown = ", ".join(format_value(value) for value in selected_z)
        lines.insert(0, f"Representative full-z fit plots are shown for z = {shown}.")
    return lines


def _ensemble_text(record: StageReportRecord) -> str:
    ensemble = getattr(record.output, "ensemble", None)
    if ensemble is not None and all(hasattr(ensemble, name) for name in ("series", "id", "a_s", "L_s", "L_t")):
        return (
            f"{ensemble.series} / {ensemble.id}; a_s={format_value(ensemble.a_s)} fm; "
            f"L_s={ensemble.L_s}, L_t={ensemble.L_t}"
        )
    return format_value(output_attrs(record).get("ensemble", ensemble))


def _lanczos_configuration_lines(record: StageReportRecord) -> list[str]:
    diagnostics = record.summary.get("diagnostics", {})
    inspection = diagnostics.get("inspection", {}) if isinstance(diagnostics, Mapping) else {}
    if not isinstance(inspection, Mapping):
        inspection = {}
    plan = inspection.get("sampling_plan", {})
    if not isinstance(plan, Mapping):
        plan = {}
    usage = inspection.get("point_usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    selected_tseps = plan.get("selected_tseps", [])
    tsep_text = ", ".join(format_value(value) for value in selected_tseps) if selected_tseps else "not recorded"
    used = usage.get("used_per_z", plan.get("used_point_count"))
    total = plan.get("total_point_count")
    discarded = usage.get("discarded_per_z", plan.get("discarded_point_count"))
    if used is not None and total is not None:
        point_text = f"{format_value(used)} of {format_value(total)} available three-point (t_sep, tau) points per z"
        if discarded is not None:
            point_text += f" retained; {format_value(discarded)} omitted"
    else:
        point_text = "point usage was not recorded"
    states = record.params.get("nstate")
    if isinstance(states, (list, tuple)) and states:
        states = states[0]
    return [
        f"- Scope: {_scope_name(record.params.get('scope'))}",
        f"- Krylov iterations: {format_value(inspection.get('iterations'))}",
        f"- Exported states: {format_value(states)}",
        f"- Source/sink separations: $t_{{\\mathrm{{sep}}}} = {tsep_text}$",
        f"- Starting time and step: $t_0 = {format_value(inspection.get('lanczos_t0'))}$, "
        f"$\\Delta t = {format_value(inspection.get('lanczos_time_step'))}$",
        f"- Point usage: {point_text}; the retained points form the complete Krylov square.",
    ]


def _selection_policy_lines(records: tuple[StageReportRecord, ...]) -> list[str]:
    has_lsqfit = any(record.params.get("analysis_method") == "lsqfit" for record in records)
    has_lanczos = any(record.params.get("analysis_method") == "lanczos" for record in records)
    lines = ["## Selection Policy", ""]
    if has_lsqfit:
        lines.append(
            "Candidate selection is performed on sample-average fits over the authored strategies, scopes, state "
            "counts, prior widths, and time windows. The selected window and fit method are then held fixed for "
            "the full-z production fits. If recommendation retries are exhausted, the same deterministic selector "
            "is applied once across every retained numerical candidate; numerical failures remain counted in "
            "diagnostics."
        )
    if has_lanczos:
        if has_lsqfit:
            lines.append("")
        lines.append(
            "Lanczos does not scan nonlinear fit candidates. Its effective moment grid is fixed by the usable "
            "two-point and three-point time coordinates; the requested state count controls the published output."
        )
    lines.append("")
    return lines


def _application_summary_lines(record: StageReportRecord) -> list[str]:
    diagnostics = record.summary.get("diagnostics", {})
    application = diagnostics.get("selected_application_fit", {}) if isinstance(diagnostics, Mapping) else {}
    fits = application.get("fits", []) if isinstance(application, Mapping) else []
    quality = diagnostics.get("sample_fit_quality", {}) if isinstance(diagnostics, Mapping) else {}
    by_z = quality.get("by_z", {}) if isinstance(quality, Mapping) else {}
    q_threshold = float(record.params.get("q_min", 0.05))
    rows: list[str] = []
    for fit in fits:
        if not isinstance(fit, Mapping):
            continue
        z_value = fit.get("z")
        stats: object = {}
        if isinstance(by_z, Mapping):
            for key in (str(z_value), f"{float(z_value):g}" if z_value is not None else ""):
                if key in by_z:
                    stats = by_z[key]
                    break
        if not isinstance(stats, Mapping):
            stats = {}
        attempted = stats.get("attempted_samples")
        q_bad = stats.get("q_below_threshold")
        low_q = f"{q_bad}/{attempted}" if q_bad is not None and attempted is not None else "n/a"
        rows.append(
            f"| {format_value(z_value)} | {format_value(fit.get('Q'))} | "
            f"{format_value(fit.get('chi2_dof'))} | {format_value(stats.get('median_chi2_dof'))} | "
            f"{low_q} | {format_value(stats.get('successful_samples'))} | "
            f"{format_value(stats.get('numerical_failures'))} | {format_value(fit.get('E0'))} | "
            f"{format_value(fit.get('E0_sdev'))} |"
        )
    if not rows:
        rows = [
            "| not recorded | not recorded | not recorded | not recorded | "
            "not recorded | not recorded | not recorded | not recorded | not recorded |"
        ]
    sample_note = (
        f"Sample statistics use all production fits for the selected window; "
        f"Q values below {q_threshold:g} are retained and reported as low-quality samples."
    )
    return [
        "### Full-z Application Fit Summary",
        "",
        sample_note,
        "",
        "| z | center Q | center chi2/dof | median sample chi2/dof | "
        "Q < threshold | successful samples | numerical failures | E0 | E0 sdev |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
    ]


def _artifact_summary_lines(record: StageReportRecord, stage_directory: Path) -> list[str]:
    """Validate declared artifacts while exposing only the key user-facing links."""
    artifact_rows(record, stage_directory)
    raw = record.summary.get("artifacts", [])
    links: list[str] = []
    for relative in raw:
        if relative not in {"output.nc", "diagnostics/candidates.json", "diagnostics/lanczos.json"}:
            continue
        path = (record.artifact_directory / relative).resolve()
        link = path.relative_to(stage_directory.resolve()).as_posix()
        links.append(f"[{relative}]({link})")
    link_text = ", ".join(links) if links else "the declared output and diagnostic files"
    return [
        f"This job exported {link_text}, together with the result and representative diagnostic figures.",
        "The complete artifact list remains in the job summary metadata; the links above identify the primary "
        "files for inspection.",
    ]


def _dispersion_model(x_design, parameters):
    """Evaluate E^2 = m^2 + k2 p^2 + k3 p^4 a^2 on the two-column design matrix."""
    return parameters["m2"] + parameters["k2"] * x_design[:, 0] + parameters["k3"] * x_design[:, 1]


_DISPERSION_N_PARAMS = 3


def _dispersion_lines(records: tuple[StageReportRecord, ...], artifact_directory: Path) -> list[str]:
    """Restore the physical-unit ensemble dispersion plot from aligned E0 samples."""
    import gvar as gv
    import lsqfit
    import numpy as np

    from lamet_agent.data import EnsembleData
    from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
    from lamet_agent.plotting import configure_plot, errorband, errorline, line, save_figure, series_color, start_plot

    points = []
    for record in records:
        diagnostics = record.summary.get("diagnostics", {})
        energy = diagnostics.get("dispersion_energy", {}) if isinstance(diagnostics, dict) else {}
        attrs = output_attrs(record)
        momentum = attrs.get("momentum_gev")
        ensemble = getattr(record.output, "ensemble", None)
        samples = energy.get("E0_samples", []) if isinstance(energy, dict) else []
        if momentum is None or not samples:
            continue
        lattice_samples = np.asarray(samples, dtype=float)
        if lattice_samples.ndim != 1 or np.any(~np.isfinite(lattice_samples)):
            continue
        gev2_samples = (lattice_samples * HBAR_C_GEV_FM / float(ensemble.a_t)) ** 2
        mode = str(attrs.get("sample_error_mode", "covariance"))
        point_data = EnsembleData(
            ensemble,
            str(record.output.resample),
            [[value] for value in gev2_samples],
            ["point"],
            {"point": [0]},
        )
        points.append(
            {
                "job_id": record.job_id,
                "ensemble": str(ensemble.id),
                "ensemble_info": ensemble,
                "momentum2": float(momentum) ** 2,
                "samples": gev2_samples,
                "value": point_data.average(mode)[0],
                "mode": mode,
                "resample": str(record.output.resample),
                "resample_id": attrs.get("resample_id"),
            }
        )
    if len(points) < 2:
        return [
            "Fewer than two jobs carried compatible ground-state energy resamples and momentum, so no "
            "dispersion plot was generated."
        ]
    groups: dict[str, list[dict[str, object]]] = {}
    for point in points:
        groups.setdefault(str(point["ensemble"]), []).append(point)
    start_plot()
    maximum = max(float(point["momentum2"]) for point in points)
    p2_line = np.linspace(0.0, maximum * 1.05 if maximum > 0.0 else 1.0, 200)
    notes: list[str] = []
    for group_index, (label, group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda point: (float(point["momentum2"]), str(point["job_id"])))
        color = series_color(group_index)
        p2 = np.asarray([float(point["momentum2"]) for point in group], dtype=float)
        signatures = {(point["resample"], point["resample_id"], len(point["samples"])) for point in group}
        compatible = len(group) >= 2 and len(signatures) == 1 and next(iter(signatures))[1] is not None
        if compatible:
            sample_matrix = np.column_stack([point["samples"] for point in group])
            ensemble = group[0]["ensemble_info"]
            combined = EnsembleData(
                ensemble,
                str(group[0]["resample"]),
                list(sample_matrix),
                ["point"],
                {"point": list(range(len(group)))},
            )
            values = combined.average(str(group[0]["mode"]))
        else:
            values = np.asarray([point["value"] for point in group], dtype=object)
        errorline(p2, values, color=color, label=label or "ensemble")
        if not compatible:
            notes.append(f"`{label}` did not have aligned resamples, so its fit band was omitted.")
            continue
        if len(group) <= _DISPERSION_N_PARAMS:
            notes.append(
                f"`{label}` has {len(group)} momenta and the dispersion model has "
                f"{_DISPERSION_N_PARAMS} parameters, so the fit band was omitted."
            )
            continue
        ensemble = group[0]["ensemble_info"]
        a2 = np.full_like(p2, (float(ensemble.a_s) / HBAR_C_GEV_FM) ** 2)
        design = np.column_stack([p2, p2**2 * a2])
        prior = gv.BufferDict(
            {
                "m2": gv.gvar(float(np.min(gv.mean(values))), 10.0),
                "k2": gv.gvar(1.0, 10.0),
                "k3": gv.gvar(0.0, 10.0),
            }
        )
        try:
            fit = lsqfit.nonlinear_fit(data=(design, values), fcn=_dispersion_model, prior=prior, maxit=2000)
        except (FloatingPointError, OverflowError, RuntimeError, ValueError) as exc:
            notes.append(f"`{label}` dispersion fit failed: {type(exc).__name__}: {exc}")
            continue
        line_design = np.column_stack([p2_line, p2_line**2 * np.full_like(p2_line, float(np.mean(a2)))])
        errorband(p2_line, _dispersion_model(line_design, fit.p), color=color)
    line(p2_line, p2_line, color="0.65", label=r"$E^2=p^2$", linestyle="dashed")
    configure_plot(
        xlabel=r"$p^2\,[\mathrm{GeV}^2]$",
        ylabel=r"$E_0^2\,[\mathrm{GeV}^2]$",
        legend=True,
        legend_loc="upper left",
        title="Dispersion relation",
    )
    pdf = artifact_directory / "plots" / "dispersion_relation.pdf"
    svg = artifact_directory / "plots" / "dispersion_relation.svg"
    save_figure(pdf, svg)
    return [
        "![Ground-state dispersion relation](plots/dispersion_relation.svg)",
        "",
        "[Dispersion relation (PDF)](plots/dispersion_relation.pdf)",
        *(["", *notes] if notes else []),
    ]


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    lines = [
        "# Correlator Analysis Stage Report",
        "",
        "This report summarizes the selected correlator fits and their full-z production-sample results.",
        "",
    ]
    lines.extend(_method_lines(records))
    has_lsqfit = any(record.params.get("analysis_method") == "lsqfit" for record in records)
    summary_header = (
        "| job | fit method | selected fit/configuration | center Q | center chi2/dof | samples |"
        if has_lsqfit
        else "| job | fit method | configuration | samples |"
    )
    summary_separator = "|---|---|---|---:|---:|---:|" if has_lsqfit else "|---|---|---|---:|"
    lines.extend(
        [
            "## Job Summary",
            "",
            summary_header,
            summary_separator,
        ]
    )
    for record in records:
        summary = record.summary
        diagnostics = summary.get("diagnostics", {})
        output = record.output
        decisions = summary.get("decisions", {})
        candidate = _candidate_for_record(record)
        method = (
            decisions.get("method", record.params.get("analysis_method"))
            if isinstance(decisions, Mapping)
            else record.params.get("analysis_method")
        )
        scope = _candidate_scope(candidate, record)
        if has_lsqfit:
            lines.append(
                f"| `{record.job_id}` | {_method_name(method, scope)} | "
                f"{_selected_fit_text(record)} | {format_value(diagnostics.get('Q'))} | "
                f"{format_value(diagnostics.get('chi2_dof'))} | {format_value(getattr(output, 'n_sample', None))} |"
            )
        else:
            lines.append(
                f"| `{record.job_id}` | {_method_name(method, scope)} | "
                f"{_selected_fit_text(record)} | {format_value(getattr(output, 'n_sample', None))} |"
            )
    lines.extend(
        [
            "",
            "## Stage Overview",
            "",
            *stage_overlay_lines(
                records,
                artifact_directory,
                coordinate="z",
                stem="correlator_overview",
                xlabel=r"$z~/~a$",
                ylabel="bare matrix element",
            ),
            "",
            "## Sample Fit Quality",
            "",
            *_sample_quality_lines(records, artifact_directory),
            "",
            "## Dispersion Relation",
            "",
            "The stage compares fitted ground-state energies against momentum when at least two compatible jobs "
            "provide the required posterior provenance.",
            "",
            *_dispersion_lines(records, artifact_directory),
        ]
    )
    lines.extend(
        [
            "",
            *_selection_policy_lines(records),
        ]
    )
    for record in records:
        params = record.params
        summary = record.summary
        diagnostics = summary.get("diagnostics", {})
        attrs = output_attrs(record)
        candidate = _candidate_for_record(record)
        candidates = diagnostics.get("candidates", []) if isinstance(diagnostics, Mapping) else []
        decisions = summary.get("decisions", {})
        output_coordinate = next(iter(record.output.coords))
        grid_symbol = "z/a" if params.get("analysis_method") == "lanczos" else output_coordinate
        output_grid = describe_grid(record.output.coords[output_coordinate], symbol=grid_symbol)
        selected_method = (
            candidate.get("method")
            if candidate
            else decisions.get("method", params.get("analysis_method"))
            if isinstance(decisions, Mapping)
            else params.get("analysis_method")
        )
        lines.extend(
            [
                "",
                f"## `{record.job_id}`",
                "",
                "### Selected Configuration" if params.get("analysis_method") == "lanczos" else "### Selected Fit",
                "",
                f"- Fit method: {_method_name(selected_method, _candidate_scope(candidate, record))}",
            ]
        )
        if params.get("analysis_method") == "lsqfit":
            lines.extend(
                [
                    "",
                    "### Candidate Diagnostics",
                    "",
                    "| selected | fit method | 2pt window | t_sep | tau cut | states | Q | "
                    "chi2/dof | accepted | numerical failure |",
                    "|---|---|---|---|---:|---:|---:|---:|---|---|",
                ]
            )
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                window = candidate.get("window", {})
                if not isinstance(window, Mapping):
                    window = {}
                tseps = candidate.get("tsep_values")
                tsep_text = ", ".join(str(value) for value in tseps) if isinstance(tseps, (list, tuple)) else "n/a"
                selected_marker = candidate.get("candidate_id") == summary.get("decisions", {}).get("candidate_id")
                candidate_method = _method_name(
                    candidate.get("method"), candidate.get("fit_scope", _candidate_scope(None, record))
                )
                lines.append(
                    f"| {'yes' if selected_marker else ''} | {candidate_method} | "
                    f"[{format_value(window.get('tmin'))}, {format_value(window.get('tmax'))}) | {tsep_text} | "
                    f"{format_value(window.get('tau_min'))} | {format_value(candidate.get('nstate'))} | "
                    f"{format_value(candidate.get('Q', candidate.get('min_Q')))} | "
                    f"{format_value(candidate.get('chi2_dof', candidate.get('worst_chi2_dof')))} | "
                    f"{format_value(candidate.get('quality_passed'))} | "
                    f"{format_value(candidate.get('numerical_failure'))} |"
                )
            tune_rows = []
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                tuning = candidate.get("tune_z_diagnostics", {})
                if not isinstance(tuning, dict):
                    continue
                for z_value, fit in tuning.items():
                    if not isinstance(fit, dict):
                        continue
                    candidate_method = _method_name(
                        candidate.get("method"), candidate.get("fit_scope", _candidate_scope(None, record))
                    )
                    tune_rows.append(
                        f"| {candidate_method} | "
                        f"{_window_text(candidate)} | {z_value} | {format_value(fit.get('Q'))} | "
                        f"{format_value(fit.get('chi2_dof'))} | {format_value(fit.get('logGBF'))} |"
                    )
            lines.extend(
                [
                    "",
                    "### Per-tuning-z Fit Summary",
                    "",
                    "| fit method | selected window | z | Q | chi2/dof | logGBF |",
                    "|---|---|---:|---:|---:|---:|",
                    *(tune_rows or ["| n/a | n/a | n/a | n/a | n/a | n/a |"]),
                ]
            )
            lines.extend(["", *_application_summary_lines(record)])
            if diagnostics.get("fallback_no_q_passing"):
                lines.extend(
                    [
                        "",
                        "ATTENTION: no candidate passed `q_min` after the allowed attempts; "
                        "the deterministic best candidate was published anyway.",
                    ]
                )
        else:
            lines.extend(
                [
                    "",
                    *_lanczos_configuration_lines(record),
                ]
            )
        lines.extend(
            [
                "",
                "### Result Context",
                "",
                f"- Ensemble: {_ensemble_text(record)}",
                f"- Momentum: {format_value(attrs.get('momentum_gev'))} GeV",
                f"- Lattice spacing: {format_value(record.output.ensemble.a_s)} fm",
                f"- Output grid: {output_grid}",
                "",
                "### Figures",
                "",
                *_representative_figure_lines(record, artifact_directory),
                "",
                "### Artifacts",
                "",
                *_artifact_summary_lines(record, artifact_directory),
            ]
        )
    return write_report(artifact_directory, lines)
