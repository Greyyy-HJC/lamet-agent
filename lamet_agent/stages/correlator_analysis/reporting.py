"""Stage-level correlator-analysis reporting."""

from __future__ import annotations

from pathlib import Path

from lamet_agent.stages._reporting import (
    StageReportRecord,
    artifact_rows,
    describe_grid,
    figure_lines,
    format_value,
    output_attrs,
    stage_overlay_lines,
    write_report,
)


_LSQFIT_METHOD = r"""
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

The published Breit matrix element is $O_{00}(z)/(2E_0)$.  A `qda_ratio`
job instead fits the nonlocal/local two-point ratio and publishes
$O_{00}(z)/z'_0$.  Candidate quality is evaluated at the tool-selected
`tune_z_values`; the chosen window and model are held fixed for the full-z
sample fits.

For non-forward kinematics, separate initial and final spectra enter the
symmetrized ratio,

$$
R_{\rm NB}(t,\tau,z)=\frac{C_3^{f\leftarrow i}(t,\tau,z)}{C_2^f(t)}
\left[\frac{C_2^i(t-\tau)C_2^f(\tau)C_2^f(t)}
{C_2^f(t-\tau)C_2^i(\tau)C_2^i(t)}\right]^{1/2},
\qquad
h_{\rm NB}(z)=\operatorname{sign}(z_{0,i}z_{0,f})
\frac{O_{00}(z)}{E_{0,i}+E_{0,f}}.
$$

An FH scope additionally uses the finite difference of the summed ratio,

$$S(t)=\sum_{\tau=\tau_c}^{t-\tau_c}R(t,\tau),\qquad
R_{\rm FH}(t)=\frac{S(t+\Delta t)-S(t)}{\Delta t}.$$

`joint` fits all selected channels with shared parameters, `chained` anchors
the matrix-element fit to the two-point posterior, and `independent` fits the
ratio channel without a separate two-point likelihood.
""".strip()


_LANCZOS_METHOD = r"""
The Lanczos branch constructs the transfer-matrix Krylov problem directly from
resampled correlator moments.  Its iteration count is determined by the usable
time grid, while `nstate` limits only the exported Ritz states or diagnostic
state matrix.  Nested resampling and the Cullum--Willoughby filtering are
performed before the median Lanczos result is published.
""".strip()


def _sample_quality_lines(records: tuple[StageReportRecord, ...], artifact_directory: Path) -> list[str]:
    """Plot selected-production per-sample Q and chi2/dof distributions."""
    import numpy as np

    from lamet_agent.plotting import configure_plot, histogram, line, save_figure, series_color, start_plot

    def series(key: str) -> list[tuple[str, np.ndarray]]:
        result = []
        for record in records:
            quality = record.summary.get("diagnostics", {}).get("sample_fit_quality", {})
            values = np.asarray(quality.get(key, []), dtype=float) if isinstance(quality, dict) else np.asarray([])
            values = values[np.isfinite(values)]
            if values.size:
                result.append((record.job_id, values))
        return result

    q_series = series("Q")
    chi2_series = series("chi2_dof")
    if not q_series and not chi2_series:
        return ["No successful production sample-fit quality diagnostics were available."]
    lines = [
        "The LSQFit $Q$ value is the goodness-of-fit p-value. Distributions include successful production "
        "sample fits only; numerical failures remain counted in the job diagnostics.",
        "",
    ]
    if q_series:
        start_plot()
        pooled = []
        for index, (label, values) in enumerate(q_series):
            pooled.append(values)
            ordered = np.sort(values)
            cdf = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
            line(
                np.r_[ordered[0], ordered],
                np.r_[0.0, cdf],
                color=series_color(index),
                label=label,
                linewidth=1.4,
                drawstyle="steps-post",
            )
        all_values = np.sort(np.concatenate(pooled))
        all_cdf = np.arange(1, all_values.size + 1, dtype=float) / all_values.size
        line(
            np.r_[all_values[0], all_values],
            np.r_[0.0, all_cdf],
            color="0.15",
            label="All",
            linewidth=2.0,
            drawstyle="steps-post",
        )
        configure_plot(
            xlabel=r"$Q$",
            ylabel=r"CDF of $Q$",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            legend=True,
            title=r"Per-sample fit $Q$",
        )
        q_pdf = artifact_directory / "plots" / "sample_fit_quality_Q.pdf"
        q_svg = artifact_directory / "plots" / "sample_fit_quality_Q.svg"
        save_figure(q_pdf, q_svg)
        lines.extend(
            [
                "![CDF of per-sample Q](plots/sample_fit_quality_Q.svg)",
                "",
                "[CDF of per-sample Q (PDF)](plots/sample_fit_quality_Q.pdf)",
                "",
            ]
        )
    if chi2_series:
        pooled_values = np.concatenate([values for _label, values in chi2_series])
        low = float(np.min(pooled_values))
        high = float(np.max(pooled_values))
        if high <= low:
            padding = 0.05 if low == 0.0 else abs(low) * 0.05
            low, high = low - padding, high + padding
        automatic = max(1, int(np.histogram_bin_edges(pooled_values, bins="auto").size - 1))
        bins = np.linspace(low, high, max(1, int(np.round(automatic * 1.5))) + 1)
        start_plot()
        for index, (label, values) in enumerate(chi2_series):
            histogram(values, bins, color=series_color(index), label=label)
        histogram(pooled_values, bins, color="0.15", label="All", linewidth=2.0)
        span = float(bins[-1] - bins[0])
        padding = 0.02 * span if span > 0 else 0.05
        configure_plot(
            xlabel=r"$\chi^2/\mathrm{dof}$",
            ylabel="Counts",
            xlim=(float(bins[0]) - padding, float(bins[-1]) + padding),
            legend=True,
            title=r"Per-sample fit $\chi^2/\mathrm{dof}$",
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
    from lamet_agent.plotting import configure_plot, errorband, errorbar, line, save_figure, series_color, start_plot

    points = []
    for record in records:
        diagnostics = record.summary.get("diagnostics", {})
        energy = diagnostics.get("dispersion_energy", {}) if isinstance(diagnostics, dict) else {}
        attrs = output_attrs(record)
        momentum = attrs.get("momentum_gev")
        ensemble = getattr(record.output, "ensemble", None)
        samples = energy.get("E0_samples", []) if isinstance(energy, dict) else []
        if ensemble is None or momentum is None or not samples:
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
        errorbar(p2, values, color=color, label=label or "ensemble")
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
    from lamet_agent.plotting import BARE_MATRIX_ELEMENT_LABEL, Z_OVER_A_LABEL

    methods = {str(record.params["analysis_method"]) for record in records}
    lines = [
        "# Correlator Analysis Stage Report",
        "",
        "This stage extracts bare matrix elements or spectra while preserving the authored resampling axis.",
        "",
        "## Method",
        "",
    ]
    if "lsqfit" in methods:
        lines.extend([_LSQFIT_METHOD, ""])
    if "lanczos" in methods:
        lines.extend([_LANCZOS_METHOD, ""])
    lines.extend(
        [
            "## Selection Policy",
            "",
            "LSQFit jobs enumerate the complete authored Cartesian grid of strategies, scopes, state counts, prior widths, and windows. Numerical failures remain recorded rather than disappearing from the candidate set. Ordinary matrix elements use the reference information/window rule; qDA jobs require feasibility at every selected tuning separation and rank by minimum Q followed by worst chi2/dof. Full-z application may reject a tuned candidate, in which case the next deterministic candidate is tried and the rejection is recorded.",
            "",
        ]
    )
    lines.extend(
        [
            "## Job Summary",
            "",
            "| job | method | result | selected candidate/scope | Q | chi2/dof | samples |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for record in records:
        summary = record.summary
        diagnostics = summary.get("diagnostics", {})
        decisions = summary.get("decisions", {})
        output = record.output
        selected = decisions.get("candidate_id", decisions.get("scope"))
        lines.append(
            f"| `{record.job_id}` | `{decisions.get('method', record.params['analysis_method'])}` | "
            f"`{summary.get('result')}` | `{selected}` | {format_value(diagnostics.get('Q'))} | "
            f"{format_value(diagnostics.get('chi2_dof'))} | {format_value(getattr(output, 'n_sample', None))} |"
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
                xlabel=Z_OVER_A_LABEL,
                ylabel=BARE_MATRIX_ELEMENT_LABEL,
            ),
            "",
            "## Sample Fit Quality",
            "",
            *_sample_quality_lines(records, artifact_directory),
            "",
            "## Dispersion Relation",
            "",
            "The stage compares the fitted ground-state energies against momentum whenever at least two compatible jobs provide the required posterior provenance.",
            "",
            *_dispersion_lines(records, artifact_directory),
        ]
    )
    for record in records:
        params = record.params
        summary = record.summary
        diagnostics = summary.get("diagnostics", {})
        attrs = output_attrs(record)
        lines.extend(
            [
                "",
                f"## `{record.job_id}`",
                "",
                "### Analysis Settings",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| analysis method | `{params['analysis_method']}` |",
                f"| component | `{params['component']}` |",
                f"| nstate | {format_value(params['nstate'])} |",
                f"| output dimensions | {format_value(getattr(record.output, 'dims', None))} |",
                f"| resampling | `{getattr(record.output, 'resample', 'n/a')}` |",
            ]
        )
        if params["analysis_method"] == "lsqfit":
            settings = params["lsqfit"]
            lines.extend(
                [
                    f"| fit scope | {format_value(settings['fit_scope'])} |",
                    f"| fit strategy | {format_value(settings['fit_strategy'])} |",
                    f"| fitting form | `{settings['fitting_form']}` |",
                    f"| pt2 windows | {format_value(settings['pt2_windows'])} |",
                    f"| pt3 windows | {format_value(settings.get('pt3_windows'))} |",
                    f"| SVD cutoff | {format_value(settings['svdcut'])} |",
                    f"| Q threshold | {format_value(settings['q_min'])} |",
                ]
            )
            candidates = diagnostics.get("candidates", [])
            lines.extend(
                [
                    "",
                    "### Candidate Diagnostics",
                    "",
                    "| candidate | method | window | nstate | Q | chi2/dof | accepted | numerical failure |",
                    "|---|---|---|---:|---:|---:|---|---|",
                ]
            )
            for candidate in candidates:
                lines.append(
                    f"| `{candidate.get('candidate_id')}` | `{candidate.get('method')}` | "
                    f"{format_value(candidate.get('window'))} | {format_value(candidate.get('nstate'))} | "
                    f"{format_value(candidate.get('Q', candidate.get('min_Q')))} | "
                    f"{format_value(candidate.get('chi2_dof', candidate.get('worst_chi2_dof')))} | "
                    f"{format_value(candidate.get('quality_passed'))} | {format_value(candidate.get('numerical_failure'))} |"
                )
            tune_rows = []
            for candidate in candidates:
                tuning = candidate.get("tune_z_diagnostics", {})
                if not isinstance(tuning, dict):
                    continue
                for z_value, fit in tuning.items():
                    if not isinstance(fit, dict):
                        continue
                    tune_rows.append(
                        f"| `{candidate.get('candidate_id')}` | {z_value} | {format_value(fit.get('Q'))} | "
                        f"{format_value(fit.get('chi2_dof'))} | {format_value(fit.get('logGBF'))} |"
                    )
            lines.extend(
                [
                    "",
                    "### Per-tuning-z Fit Summary",
                    "",
                    "| candidate | z | Q | chi2/dof | logGBF |",
                    "|---|---:|---:|---:|---:|",
                    *(tune_rows or ["| n/a | n/a | n/a | n/a | n/a |"]),
                    "",
                    "### Full-grid Application Rejections",
                    "",
                    format_value(diagnostics.get("application_rejections", [])),
                ]
            )
            application = diagnostics.get("selected_application_fit")
            application_fits = application.get("fits", []) if isinstance(application, dict) else []
            application_rows = [
                f"| {format_value(fit.get('z'))} | {format_value(fit.get('Q'))} | {format_value(fit.get('chi2_dof'))} | {format_value(fit.get('logGBF'))} | {format_value(fit.get('E0'))} | {format_value(fit.get('E0_sdev'))} |"
                for fit in application_fits
                if isinstance(fit, dict)
            ]
            lines.extend(
                [
                    "",
                    "### Full-z Application Fit Summary",
                    "",
                    "| z | Q | chi2/dof | logGBF | E0 | E0 sdev |",
                    "|---:|---:|---:|---:|---:|---:|",
                    *(
                        application_rows
                        or [
                            "| not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |"
                        ]
                    ),
                    "",
                    "### Runtime-resolved Defaults and Scale Inspection",
                    "",
                    f"- Recommended defaults: {format_value(diagnostics.get('recommended_defaults', {}))}",
                    f"- Correlator scale inspection: {format_value(diagnostics.get('correlator_scale_inspection', {}))}",
                    f"- Center preflight: {format_value(diagnostics.get('selected_preflight_fit'))}",
                ]
            )
        else:
            inspection = diagnostics.get("inspection", {})
            lines.extend(
                [
                    f"| Lanczos scope | `{params['lanczos']['scope']}` |",
                    f"| iterations | {format_value(inspection.get('iterations'))} |",
                    f"| inner samples | {format_value(params['lanczos']['inner_samples'])} |",
                    f"| precision | {format_value(params['lanczos']['precision'])} |",
                    f"| point-usage warning | {format_value(inspection.get('point_usage_warning'))} |",
                ]
            )
        lines.extend(
            [
                "",
                "### Output Provenance",
                "",
                f"- Ensemble: `{attrs.get('ensemble', getattr(record.output, 'ensemble', None))}`",
                f"- Momentum: {format_value(attrs.get('momentum_gev'))} GeV",
                f"- Lattice spacing: {format_value(attrs.get('lattice_spacing_fm'))} fm",
                f"- Output grid: {describe_grid(next(iter(record.output.coords.values())), symbol=next(iter(record.output.coords)))}",
                "",
                "### Field Definitions",
                "",
                "| field | meaning |",
                "|---|---|",
                "| `candidate_id` | Deterministic id of one complete authored fit candidate. |",
                "| `Q`, `chi2_dof`, `logGBF` | Sample-average goodness-of-fit and evidence diagnostics used by the selection rule. |",
                "| `tune_z_diagnostics` | Fits used only to select a common model/window before full-z resample application. |",
                "| `application_rejections` | Candidates that tuned successfully but failed the mandatory full-grid/sample application. |",
                "| `sample_fit_quality` | Successful production-resample Q and chi2/dof values used by the stage "
                "statistics. |",
                "| `dispersion_energy` | Aligned ground-state energy resamples in lattice units used only for the stage "
                "dispersion figure. |",
                "",
                "### Figures",
                "",
                *figure_lines(record, artifact_directory),
                "",
                "### Artifacts",
                "",
                "| job | artifact |",
                "|---|---|",
                *artifact_rows(record, artifact_directory),
            ]
        )
    return write_report(artifact_directory, lines)
