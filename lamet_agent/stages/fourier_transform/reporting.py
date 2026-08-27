"""Stage-level large-distance and Fourier reporting."""

from __future__ import annotations

import json
import math
from numbers import Real
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


def _momentum_series_label(record: StageReportRecord) -> str:
    value = output_attrs(record).get("momentum_gev")
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
        return rf"$P_z={round(float(value), 2):g}\,\mathrm{{GeV}}$"
    return record.job_id


_TAIL_FORMULA = r"""
For a PDF, the GI LA/NLA tail family used away from the origin is

$$
h(z)=\left[A_2e^{i\phi_2\operatorname{sgn}z}
+\frac{A'_2e^{i\phi'_2\operatorname{sgn}z}}{|z|}\right]
e^{-\Lambda |z|/(\hbar c)},
$$

where the primed term is omitted at LA.  The CG family divides this expression
by the fitted power $|z|^n$.  Meson-DA jobs use the corresponding two-endpoint
phase structure fixed by `psi1_flavor_class`, `psi2_flavor_class`, and the
hadron momentum.  The measured short-distance data are never replaced by this
asymptotic ansatz.

For a DA the two endpoint contributions retain the momentum phase explicitly,

$$
h_{\rm DA}(z)=\left[A_1e^{i(\phi_1-P_z|z|)}+A_2e^{i\phi_2}
+\frac{A'_1e^{i(\phi'_1-P_z|z|)}+A'_2e^{i\phi'_2}}{|z|}\right]
e^{-\Lambda|z|/(\hbar c)},
$$

with flavor-class constraints removing or identifying endpoint amplitudes.  If
`phase_transfer_da=true`, the input is first rotated by
$e^{+izP_z/2}$, projected onto its real symmetry channel, and rotated back.
""".strip()


_TRANSFORM_FORMULA = r"""
After the selected tail is connected to the measured matrix element, each
resample is transformed with the reference convention

$$
\widetilde q(x,P_z)=\frac{P_z}{2\pi}
\int dz\,e^{+ixP_zz}\,h(z).
$$

Negative-$z$ completion, component selection, and the final projection factor
are derived from upstream observable, polarization, sector, and gauge-link
provenance.  Range selection is performed on sample-average fits; the selected
range is then fixed for all resamples and LA/NLA prior-width candidates.

For `smooth=linear`, measured data and the fitted tail are linearly blended
over the selected interval.  For `smooth=none`, the measured branch switches
directly to the extrapolated branch at the selected boundary.
""".strip()


def _json_attr(attrs: dict[str, object], name: str) -> object:
    value = attrs.get(name)
    if not isinstance(value, str):
        return value


def _candidate_records(record: StageReportRecord) -> list[dict[str, object]]:
    artifacts = set(record.summary.get("artifacts", []))
    relative = (
        "diagnostics/candidates.json"
        if "diagnostics/candidates.json" in artifacts
        else "diagnostics/fourier.json"
        if "diagnostics/fourier.json" in artifacts
        else None
    )
    if relative is None:
        raise ValueError(f"job '{record.job_id}' declares no Fourier candidate diagnostics")
    path = record.artifact_directory / relative
    if not path.is_file():
        raise FileNotFoundError(f"Fourier candidate diagnostics are missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if relative == "diagnostics/fourier.json":
        if not isinstance(value, dict):
            raise TypeError("Fourier diagnostics must contain one object")
        return [
            {
                "label": value.get("candidate_id"),
                "model_id": value.get("tail_model"),
                **value,
                "model_weight": 1.0,
                "selected": True,
            }
        ]
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError("Fourier candidates.json must contain a list of objects")
    return value


def _range_records(record: StageReportRecord) -> list[dict[str, object]]:
    path = record.artifact_directory / "diagnostics" / "ranges.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError("Fourier ranges.json must contain a list of objects")
    return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    lines = [
        "# Fourier Transform Stage Report",
        "",
        "This stage fits the finite-distance tail and transforms every resample to a quasi-distribution.",
        "",
        "## Large-Distance Extrapolation",
        "",
        _TAIL_FORMULA,
        "",
        "## Fourier Transform and Projection",
        "",
        _TRANSFORM_FORMULA,
        "",
        "## Job Summary",
        "",
        "| job | momentum [GeV] | construction | sector | selected range | selected models | Q | chi2/dof | samples |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for record in records:
        attrs = dict(output_attrs(record))
        diagnostics = record.summary.get("diagnostics", {})
        lines.append(
            f"| `{record.job_id}` | {format_value(attrs.get('momentum_gev'))} | `{attrs.get('gfix', 'n/a')}` | "
            f"`{attrs.get('sector', record.params['scheme_scan']['sector'])}` | "
            f"{format_value(_json_attr(attrs, 'selected_range'))} | "
            f"{format_value(diagnostics.get('selected_fit_model_labels'))} | "
            f"{format_value(diagnostics.get('selected_Q'))} | {format_value(diagnostics.get('selected_chi2_dof'))} | "
            f"{format_value(getattr(record.output, 'n_sample', None))} |"
        )
    lines.extend(
        [
            "",
            "## Stage Overview",
            "",
            *stage_overlay_lines(
                records,
                artifact_directory,
                coordinate="x",
                stem="fourier_overview",
                xlabel=r"$x$",
                ylabel={
                    "real": r"$\mathrm{Re}\,\tilde q(x)$",
                    "imag": r"$\mathrm{Im}\,\tilde q(x)$",
                },
                band=True,
                series_label=_momentum_series_label,
            ),
        ]
    )
    for record in records:
        params = record.params
        scan = params["scheme_scan"]
        attrs = dict(output_attrs(record))
        diagnostics = record.summary.get("diagnostics", {})
        candidates = _candidate_records(record)
        ranges = _range_records(record)
        candidate_rows = [
            f"| `{candidate.get('label')}` | `{candidate.get('model_id')}` | "
            f"{format_value(candidate.get('z_min_fm'))} | {format_value(candidate.get('z_max_fm'))} | "
            f"`{candidate.get('order')}` | {format_value(candidate.get('prior_width'))} | "
            f"{format_value(candidate.get('Q'))} | {format_value(candidate.get('chi2_dof'))} | "
            f"{format_value(candidate.get('model_weight'))} | {format_value(candidate.get('selected'))} |"
            for candidate in candidates
        ]
        range_rows = [
            f"| `{item.get('model_id')}` | {format_value(item.get('z_min_fm'))} | {format_value(item.get('z_max_fm'))} | "
            f"{format_value(item.get('fit_success'))} | {format_value(item.get('Q'))} | {format_value(item.get('chi2_dof'))} | "
            f"{format_value(item.get('logGBF'))} | {format_value(item.get('selected'))} | {format_value(item.get('error'))} |"
            for item in ranges
        ]
        parameter_names = sorted(
            {name for candidate in candidates for name in dict(candidate.get("parameter_mean", {}))}
        )
        parameter_rows = []
        for candidate in candidates:
            means = dict(candidate.get("parameter_mean", {}))
            sdevs = dict(candidate.get("parameter_sdev", {}))
            for name in parameter_names:
                parameter_rows.append(
                    f"| `{candidate.get('label')}` | `{name}` | {format_value(means.get(name))} | {format_value(sdevs.get(name))} |"
                )
        lines.extend(
            [
                "",
                f"## `{record.job_id}`",
                "",
                "### Analysis Settings",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| target observable | `{attrs.get('target_observable', 'n/a')}` |",
                f"| parton / construction | `{attrs.get('parton', 'n/a')}` / `{attrs.get('gfix', 'n/a')}` |",
                f"| momentum | {format_value(attrs.get('momentum_gev'))} GeV |",
                f"| x grid | {describe_grid(record.output.coords['x'], symbol='x')} |",
                f"| candidate $z_{{\\min}}$ [fm] | {format_value(params['zmin_fm'])} |",
                f"| candidate $z_{{\\max}}$ [fm] | {format_value(params['zmax_fm'])} |",
                f"| extension [fm] | {format_value(params['zmax_ext_fm'])} |",
                f"| smoothing | `{params['smooth']}` |",
                f"| orders | {format_value(scan['order'])} |",
                f"| tail-prior scales | {format_value(scan['posterior_prior_error_scale'])} |",
                f"| model average | {format_value(scan['model_average'])} |",
                f"| component / output scale | `{attrs.get('component', 'n/a')}` / {format_value(attrs.get('output_scale'))} |",
                f"| range candidates | {format_value(diagnostics.get('range_candidate_count'))} |",
                f"| model candidates | {format_value(diagnostics.get('model_candidate_count'))} |",
                "",
                "### Selected Model Diagnostics",
                "",
                f"- Selected range: `{diagnostics.get('selected_range_label', 'n/a')}`",
                f"- Selected models: {format_value(diagnostics.get('selected_fit_model_labels'))}",
                f"- Model weights: {format_value(diagnostics.get('fit_model_weights'))}",
                f"- DA phase transfer: {format_value(attrs.get('phase_transfer_da'))}",
                "",
                "### Range and Fit-model Candidates",
                "",
                "The range scan uses the first authored order and prior width to select one physical interval. With that interval fixed, every authored LA/NLA and prior-width model is refitted. `model_average=false` selects the best successful model; `model_average=true` combines successful candidates per resample using their recorded weights.",
                "",
                "#### Range-selection fits",
                "",
                "| tail | zmin [fm] | zmax [fm] | success | Q | chi2/dof | logGBF | selected | failure |",
                "|---|---:|---:|---|---:|---:|---:|---|---|",
                *(range_rows or ["| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"]),
                "",
                "#### Fixed-range model fits",
                "",
                "| label | tail | zmin [fm] | zmax [fm] | order | prior scale | Q | chi2/dof | weight | selected |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
                *(candidate_rows or ["| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"]),
                "",
                "#### Tail posterior parameters",
                "",
                "| candidate | parameter | mean | sdev across resamples |",
                "|---|---|---:|---:|",
                *(parameter_rows or ["| n/a | n/a | n/a | n/a |"]),
                "",
                "### Projection and Field Definitions",
                "",
                f"The output records sector `{attrs.get('sector', 'n/a')}`, component `{attrs.get('component', 'n/a')}`, and multiplicative projection scale {format_value(attrs.get('output_scale'))}. These values are derived from upstream observable and polarization provenance rather than independently authored controls.",
                "",
                "| field | meaning |",
                "|---|---|",
                "| `selected_range` | Sample-average tail interval held fixed during all resample fits. |",
                "| `selected_models`, `model_weights` | LA/NLA/prior candidates retained by selection or model averaging. |",
                "| `component`, `output_scale` | Sector-derived complex channel and normalization of the stored quasi-distribution. |",
                "| `phase_transfer_da` | Whether the midpoint DA phase/symmetry projection was applied before tail fitting. |",
                "| `zmax_ext_fm` | Maximum physical separation of the finite transform, distinct from the fitted data interval. |",
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
