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
    "a": "a_s [fm]",
    "a2": "a_s^2 [fm^2]",
    "a4": "a_s^4 [fm^4]",
    "ap2": "(a_s P_z)^2 [raw]",
    "ap4": "(a_s P_z)^4 [raw]",
    "inv_p2": "P_z^{-2} [GeV^{-2}]",
    "inv_p4": "P_z^{-4} [GeV^{-4}]",
    "mpi2": "m_\\pi^2-m_{\\pi,\\rm phys}^2 [GeV^2]",
    "mpi4_log_mpi2": "m_\\pi^4\\log(m_\\pi^2)-m_{\\pi,\\rm phys}^4\\log(m_{\\pi,\\rm phys}^2) [numerical GeV units]",
    "exp_mpi_L": "e^{-m_\\pi L/(\\hbar c)}",
    "exp_sqrt2_mpi_L": "e^{-\\sqrt2m_\\pi L/(\\hbar c)}",
}


def _formula(terms: list[str], independent_terms: set[str]) -> str:
    pieces = []
    for term in terms:
        coefficient = f"c_{{{term}}}" if term in independent_terms else f"c_{{{term}}}(x)"
        pieces.append(f"{coefficient}{_TERM_FORMULAS[term]}")
    correction = " + ".join(pieces)
    return f"$$h(x,a,P_z)=h_0(x)+{correction}.$$"


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
            f"{format_value(ensemble.L_s * ensemble.a_s)} | "
            f"{format_value(attrs.get('momentum_gev'))} | "
            f"{format_value(ensemble.m_pi)} | `{attrs.get('kernel_id', 'n/a')}` |"
            f" `{data.resample}` / {data.n_sample} | "
            f"[{format_value(data.coords['x'][0])}, {format_value(data.coords['x'][-1])}] |"
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
                f"| `{name}` | {format_value(x[index])} | "
                f"{format_value(raw_mean[index])} | {format_value(raw_sdev[index])} |"
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
                f"| {format_value(value['momentum_gev'])} | {format_value(x[index])} | "
                f"{format_value(mean[index])} | {format_value(sdev[index])} |"
            for index in indices
        )
    return rows


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    fit_records = [record for record in records if record.summary.get("result") == "physical_distribution"]
    budget_records = [record for record in records if record.summary.get("result") == "systematics_budget"]
    lines = [
        "# Extrapolation Stage Report",
        "",
        ("This stage fits authored lattice-spacing, finite-momentum, chiral, and finite-volume "
         "bases and publishes the fitted intercept h_0(x). The report calls this the "
         "continuum/infinite-momentum result; the actual limit is exactly the intercept of "
         "the selected ansatz, not an additional post-fit physics correction."),
        "",
        "## Method",
        "",
        ("The fitted form is h_0(x) plus the authored correction bases. Coefficients marked "
         "x-dependent are fitted independently at each common x point; constant coefficients "
         "are shared across the grid. Basis values use a_s and P_z from input metadata, with "
         "L=L_s a_s and m_pi in GeV."),
        ("The implementation uses a_s in fm for a, a^2, and a^4; raw numerical (a_s P_z)^2 "
         "and (a_s P_z)^4 without an hbar-c conversion; P_z^{-2} and P_z^{-4}; "
         "physical-mass-subtracted pion-mass terms; and finite-volume exponentials with "
         "m_pi L/(hbar c). Coefficients carry the inverse units of their basis when h "
         "is dimensionless."),
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
                "### Selected Fit",
                "",
                "#### Extrapolation Form",
                "",
                _formula(terms, set(independent_terms)),
                "",
                "### Input Coverage",
                "",
                ("| index | ensemble | a [fm] | L [fm] | momentum [GeV] | pion mass [GeV] | "
                 "matching kernel | resampling / N | x coverage |"),
                "|---:|---|---:|---:|---:|---:|---|---|---|",
                *_input_rows(record),
                "",
                "### Candidate Diagnostics",
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
                "",
                "### Fit-model Parameter Table",
                "",
                ("Vector coefficients are shown at the first, central, and last x coordinates; scalar "
                 "coefficients are shared across the complete grid. The joint fit uses the declared "
                 "covariance mode, including cross-x covariance only when `x_covariance=true`."),
                "",
                "| parameter | x | mean | sdev across resamples |",
                "|---|---:|---:|---:|",
                *(_parameter_rows(record, candidate) or ["| n/a | n/a | n/a | n/a |"]),
                "",
                "### Momentum Dependence",
                "",
                ("For each requested diagnostic momentum, the implementation evaluates h_0(x) and "
                 "adds only the fitted `inv_p2` and `inv_p4` terms when present. Other authored bases "
                 "are not re-evaluated in this diagnostic; the P_z -> infinity curve is the "
                 "published intercept h_0(x)."),
                "",
                "| Pz [GeV] | x | mean | sdev across resamples |",
                "|---:|---:|---:|---:|",
                *_momentum_rows(record, candidate),
                "",
                "### Result Context",
                "",
                "- Published output: the fitted intercept $h_0(x)$ at the continuum/infinite-momentum limit.",
                f"- Output grid: {describe_grid(record.output.coords['x'], symbol='x')}.",
                "",
                "### Field Definitions",
                "",
                "| field | meaning |",
                "|---|---|",
                "| `x_independent_terms` | Correction coefficients shared across the complete x grid. |",
                "| `x_dependent_terms` | Correction coefficients fitted independently at every x. |",
                "| `x_covariance` | Whether cross-x covariance is retained within each ensemble source. |",
                ("| `priors` | One common numerical Gaussian mean and width used for h_0 and all "
                 "correction coefficients in their native units. |"),
                ("| `posterior_prior_error_scale` | Multiplier applied to center-fit posterior widths "
                 "when seeding resample fits. |"),
                ("| `pdep_gev` | Requested momenta for the post-fit diagnostic only; it does not select "
                 "inputs or alter the infinite-momentum result. |"),
                ("| `physical_pion_mass_gev` | Reference mass that zeros the two pion-mass correction "
                 "bases when they are selected. |"),
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
                ("For each declared source, only variant central values are used. One variant gives "
                 "|variant - main|; several variants give max(variant) - min(variant), after "
                 "interpolation to the main x grid. Source components are then combined in "
                 "quadrature, followed by the main statistical uncertainty:"),
                "",
                (r"$$\sigma_{\rm sys}(x)=\sqrt{\sum_k\Delta_k(x)^2},\qquad "
                 r"\sigma_{\rm total}(x)=\sqrt{\sigma_{\rm stat}(x)^2+\sigma_{\rm sys}(x)^2}.$$"),
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
