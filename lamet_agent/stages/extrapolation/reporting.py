"""Stage-level continuum/infinite-momentum and systematics reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
from lamet_agent.stages.extrapolation.physics import load_data


_TERM_FORMULAS = {
    "a": "a",
    "a2": "a^2",
    "a4": "a^4",
    "ap2": "(aP_z)^2",
    "ap4": "(aP_z)^4",
    "inv_p2": "1/P_z^2",
    "inv_p4": "1/P_z^4",
    "mpi2": "m_\\pi^2-m_{\\pi,\\rm phys}^2",
    "mpi4_log_mpi2": "m_\\pi^4\\log m_\\pi^2-m_{\\pi,\\rm phys}^4\\log m_{\\pi,\\rm phys}^2",
    "exp_mpi_L": "e^{-m_\\pi L}",
    "exp_sqrt2_mpi_L": "e^{-\\sqrt2m_\\pi L}",
}


def _formula(terms: list[str], independent_terms: set[str]) -> str:
    pieces = []
    for term in terms:
        coefficient = f"c_{{{term}}}" if term in independent_terms else f"c_{{{term}}}(x)"
        pieces.append(f"{coefficient}{_TERM_FORMULAS[term]}")
    correction = " + ".join(pieces)
    return f"$$h(x,a,P_z)=h(x,0,\\infty)+{correction}.$$"


def _term_list(attrs: dict[str, object], params: dict[str, object], name: str) -> list[str]:
    value = attrs.get(name, params.get(name, []))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or any(not isinstance(term, str) for term in value):
        raise ValueError(f"extrapolation report requires {name} provenance")
    return list(value)


def _input_rows(record: StageReportRecord) -> list[str]:
    values = record.inputs.get("distributions")
    if not isinstance(values, list) or not values:
        raise ValueError(f"job '{record.job_id}' requires distribution inputs in its report")
    rows = []
    for index, source in enumerate(values):
        data = load_data(source)
        attrs = data.attrs
        ensemble = data.ensemble
        rows.append(
            f"| {index} | `{ensemble.id}` | "
            f"{format_value(ensemble.a_s)} | "
            f"{format_value(attrs.get('momentum_gev'))} | "
            f"{format_value(ensemble.m_pi)} | `{attrs.get('kernel_id', 'n/a')}` |"
        )
    return rows


def _budget_rows(record: StageReportRecord) -> list[str]:
    path = record.artifact_directory / "diagnostics" / "systematics_budget.json"
    if not path.is_file():
        raise FileNotFoundError(f"systematics budget diagnostics are missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    sources = list(record.summary["diagnostics"]["sources"])
    rows = []
    for source in [*sources, "total_systematic_error", "total_error"]:
        values = np.asarray(document.get(source, []), dtype=float)
        if values.ndim != 1 or not values.size:
            raise ValueError(f"systematics budget diagnostics have no one-dimensional '{source}'")
        rows.append(f"| `{source}` | {format_value(float(np.max(np.abs(values))))} |")
    return rows


def _parameter_rows(record: StageReportRecord, candidate: dict[str, object]) -> list[str]:
    means = dict(candidate.get("parameter_mean", {}))
    sdevs = dict(candidate.get("parameter_sdev", {}))
    x = list(record.output.coords["x"])
    indices = list(dict.fromkeys([0, len(x) // 2, len(x) - 1]))
    rows = []
    for name, raw_mean in means.items():
        raw_sdev = sdevs.get(name)
        if isinstance(raw_mean, list):
            if not isinstance(raw_sdev, list) or len(raw_mean) != len(x) or len(raw_sdev) != len(x):
                raise ValueError(f"extrapolation parameter '{name}' does not match the x grid")
            rows.extend(
                f"| `{name}` | {format_value(x[index])} | {format_value(raw_mean[index])} | {format_value(raw_sdev[index])} |"
                for index in indices
            )
        else:
            rows.append(f"| `{name}` | all x | {format_value(raw_mean)} | {format_value(raw_sdev)} |")
    return rows


def _momentum_rows(record: StageReportRecord, candidate: dict[str, object]) -> list[str]:
    diagnostics = candidate.get("momentum_dependence")
    if not isinstance(diagnostics, dict):
        raise ValueError("extrapolation candidate has no momentum-dependence diagnostics")
    x = list(record.output.coords["x"])
    indices = list(dict.fromkeys([0, len(x) // 2, len(x) - 1]))
    rows = []
    for value in diagnostics.values():
        mean = value["mean"]
        sdev = value["sdev"]
        if len(mean) != len(x) or len(sdev) != len(x):
            raise ValueError("momentum-dependence diagnostics do not match the x grid")
        rows.extend(
            f"| {format_value(value['momentum_gev'])} | {format_value(x[index])} | {format_value(mean[index])} | {format_value(sdev[index])} |"
            for index in indices
        )
    return rows


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    fit_records = [record for record in records if record.summary.get("result") == "physical_distribution"]
    budget_records = [record for record in records if record.summary.get("result") == "systematics_budget"]
    lines = [
        "# Extrapolation Stage Report",
        "",
        "This stage fits the lattice-spacing and finite-momentum dependence of matched distributions and publishes the continuum, infinite-momentum result.",
        "",
        "## Job Summary",
        "",
        "| job | result | terms / sources | Q | chi2/dof | samples |",
        "|---|---|---|---:|---:|---:|",
    ]
    for record in records:
        summary = record.summary
        diagnostics = summary.get("diagnostics", {})
        attrs = dict(output_attrs(record))
        terms = attrs.get("extrapolation_terms", diagnostics.get("sources"))
        candidate = (
            (diagnostics.get("candidates") or [{}])[0] if summary.get("result") == "physical_distribution" else {}
        )
        lines.append(
            f"| `{record.job_id}` | `{summary.get('result')}` | {format_value(terms)} | "
            f"{format_value(candidate.get('Q'))} | {format_value(candidate.get('chi2_dof'))} | "
            f"{format_value(getattr(record.output, 'n_sample', None))} |"
        )
    lines.extend(
        [
            "",
            "## Stage Overview",
            "",
            *stage_overlay_lines(
                tuple(fit_records),
                artifact_directory,
                coordinate="x",
                stem="extrapolation_overview",
                ylabel="physical distribution",
                xlabel=r"$x$",
                band=True,
            ),
        ]
    )
    for record in fit_records:
        attrs = dict(output_attrs(record))
        raw_terms = attrs.get("extrapolation_terms", "")
        terms = [term for term in str(raw_terms).split(",") if term]
        fit = record.params
        independent_terms = _term_list(attrs, dict(record.params), "x_independent_terms")
        dependent_terms = _term_list(attrs, dict(record.params), "x_dependent_terms")
        if set(terms) != set(independent_terms) | set(dependent_terms):
            raise ValueError(f"job '{record.job_id}' term provenance differs")
        diagnostics = record.summary["diagnostics"]
        candidate = (diagnostics.get("candidates") or [{}])[0]
        lines.extend(
            [
                "",
                f"## `{record.job_id}`",
                "",
                "### Extrapolation Form",
                "",
                _formula(terms, set(independent_terms)),
                "",
                "The intercept is the published continuum/infinite-momentum distribution.  Coefficients marked as x-dependent are fitted independently across the common x grid; constant coefficients are shared across x.",
                "",
                "### Input Coverage",
                "",
                "| index | ensemble | a [fm] | momentum [GeV] | pion mass [GeV] | matching kernel |",
                "|---:|---|---:|---:|---:|---|",
                *_input_rows(record),
                "",
                "### Fit Settings and Quality",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| x-independent terms | {format_value(independent_terms)} |",
                f"| x-dependent terms | {format_value(dependent_terms)} |",
                f"| cross-x covariance | {format_value(fit.get('x_covariance', False))} |",
                f"| initial prior | {format_value(fit['priors'])} |",
                f"| sample-prior widening | {format_value(fit['posterior_prior_error_scale'])} |",
                f"| momentum diagnostic points [GeV] | {format_value(fit['pdep_gev'])} |",
                f"| inputs | {format_value(candidate.get('n_inputs', candidate.get('input_count')))} |",
                f"| parameters | {format_value(candidate.get('n_params'))} |",
                f"| Q | {format_value(candidate.get('Q'))} |",
                f"| chi2/dof | {format_value(candidate.get('chi2_dof'))} |",
                f"| failed resamples | {format_value(candidate.get('n_failed_samples'))} |",
                f"| output grid | {describe_grid(record.output.coords['x'], symbol='x')} |",
                "",
                "### Fit-model Parameter Table",
                "",
                "Vector coefficients are shown at the first, central, and last x coordinates; scalar coefficients are shared across the complete grid.",
                "",
                "| parameter | x | mean | sdev across resamples |",
                "|---|---:|---:|---:|",
                *(_parameter_rows(record, candidate) or ["| n/a | n/a | n/a | n/a |"]),
                "",
                "### Momentum Dependence",
                "",
                "At each requested momentum the fitted model is evaluated at a=0, retaining the finite-momentum inverse-power terms; the infinite-momentum curve is the published intercept.",
                "",
                "| Pz [GeV] | x | mean | sdev across resamples |",
                "|---:|---:|---:|---:|",
                *_momentum_rows(record, candidate),
                "",
                "### Field Definitions",
                "",
                "| field | meaning |",
                "|---|---|",
                "| `x_independent_terms` | Correction coefficients shared across the complete x grid. |",
                "| `x_dependent_terms` | Correction coefficients fitted independently at every x. |",
                "| `x_covariance` | Whether cross-x covariance is retained within each ensemble source. |",
                "| `priors` | Shared initial Gaussian prior for the intercept and correction coefficients. |",
                "| `posterior_prior_error_scale` | Widening applied when the sample-average posterior seeds resample fits. |",
                "| `pdep_gev` | Requested momenta for the post-fit diagnostic only; it does not select inputs or alter the infinite-momentum result. |",
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
    if budget_records:
        lines.extend(
            [
                "",
                "## Systematics Budget",
                "",
                "For every declared source, the pointwise systematic is the larger absolute displacement from the central result.  Independent source groups are combined in quadrature, followed by the statistical uncertainty:",
                "",
                r"$$\sigma_{\rm sys}(x)=\sqrt{\sum_k\Delta_k(x)^2},\qquad \sigma_{\rm total}(x)=\sqrt{\sigma_{\rm stat}(x)^2+\sigma_{\rm sys}(x)^2}.$$",
            ]
        )
        for record in budget_records:
            lines.extend(
                [
                    "",
                    f"### `{record.job_id}`",
                    "",
                    f"- Source groups: {format_value(record.summary['decisions']['systematics_groups'])}",
                    f"- Published components: {format_value(record.summary['diagnostics']['sources'])}",
                    f"- x points: {format_value(record.summary['diagnostics']['point_count'])}",
                    "",
                    "| uncertainty component | maximum absolute size |",
                    "|---|---:|",
                    *_budget_rows(record),
                    "",
                    "### Figures",
                    "",
                    *figure_lines(record, artifact_directory),
                    "",
                    "| job | artifact |",
                    "|---|---|",
                    *artifact_rows(record, artifact_directory),
                ]
            )
    return write_report(artifact_directory, lines)
