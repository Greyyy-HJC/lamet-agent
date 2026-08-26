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


def _dispersion_lines(records: tuple[StageReportRecord, ...], artifact_directory: Path) -> list[str]:
    points = []
    for record in records:
        diagnostics = record.summary.get("diagnostics", {})
        application = diagnostics.get("selected_application_fit")
        fits = application.get("fits", []) if isinstance(application, dict) else []
        attrs = output_attrs(record)
        momentum = attrs.get("momentum_gev")
        fit = fits[0] if fits and isinstance(fits[0], dict) else None
        if fit is None or momentum is None or fit.get("E0") is None:
            continue
        points.append((record.job_id, float(momentum), float(fit["E0"]), fit.get("E0_sdev")))
    if len(points) < 2:
        return [
            "Fewer than two jobs carried a common ground-state energy and momentum, so no dispersion plot was generated."
        ]
    from lamet_agent.plotting import configure_plot, errorbar, save_figure, start_plot
    import gvar

    start_plot()
    for label, momentum, energy, energy_sdev in points:
        sdev = 0.0 if energy_sdev is None else 2.0 * abs(energy) * float(energy_sdev)
        errorbar([momentum**2], [gvar.gvar(energy**2, sdev)], label=label)
    configure_plot(xlabel=r"$P_z^2$ [GeV$^2$]", ylabel=r"$E_0^2$", legend=True)
    pdf = artifact_directory / "plots" / "dispersion_relation.pdf"
    svg = artifact_directory / "plots" / "dispersion_relation.svg"
    save_figure(pdf, svg)
    return [
        "![Ground-state dispersion relation](plots/dispersion_relation.svg)",
        "",
        "[Dispersion relation (PDF)](plots/dispersion_relation.pdf)",
    ]


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
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
                records, artifact_directory, coordinate="z", stem="correlator_overview", ylabel="bare matrix element"
            ),
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
