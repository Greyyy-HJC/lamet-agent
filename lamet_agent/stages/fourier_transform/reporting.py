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


def _momentum_series_label(record: StageReportRecord) -> str | None:
    value = output_attrs(record).get("momentum_gev")
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
        return rf"$P_z={round(float(value), 2):g}\,\mathrm{{GeV}}$"
    return None


def _spacing_series_label(record: StageReportRecord) -> str | None:
    ensemble = getattr(record.output, "ensemble", None)
    spacing = getattr(ensemble, "a_s", None)
    if (
        isinstance(spacing, Real)
        and not isinstance(spacing, bool)
        and math.isfinite(float(spacing))
        and float(spacing) > 0
    ):
        return rf"$a={float(spacing):.4g}\,\mathrm{{fm}}$"
    return None


def _variation_series_label(record: StageReportRecord) -> str | None:
    offset = record.params.get("tail_window_step_offset", 0)
    try:
        steps = int(offset)
    except (TypeError, ValueError):
        return None
    if steps == 0:
        return None
    return rf"$\Delta n_z={steps:+d}$"


def _combined_series_label(record: StageReportRecord) -> str:
    parts = [
        label
        for label in (
            _momentum_series_label(record),
            _spacing_series_label(record),
            _variation_series_label(record),
        )
        if label
    ]
    return ", ".join(parts) if parts else record.job_id


_TAIL_FORMULA = r"""
The fitted long-distance branch is used only away from the origin and is
connected to the measured/interpolated branch before the transform.  For the
generic PDF family, the GI/CG LA/NLA ansatz is

$$
h(z)=\left[A_2e^{i\phi_2\operatorname{sgn}z}
+\frac{A'_2e^{i\phi'_2\operatorname{sgn}z}}{|z|}\right]
e^{-\Lambda |z|/(\hbar c)},
$$

where the primed term is omitted at LA.  The CG family divides the result by
the fitted power $|z|^n$.  A pion valence-PDF fit instead uses its dedicated
two-endpoint form

$$
h(z)=\left[A_2+2A_1\cos\left(\phi_1-\frac{P_z|z|}{\hbar c}\right)
 +\frac{\hbar c}{|z|}\left(A'_2+2A'_1\cos\left(\phi'_1-\frac{P_z|z|}{\hbar c}\right)\right)\right]
e^{-\Lambda |z|/(\hbar c)},
$$

with the CG power applied afterward.  GPD tails use the endpoint structures
selected by the hadron and the paired-flow convention.  Thus the parameter
names are model- and observable-dependent; short-distance data are never
replaced by the asymptotic ansatz.

For a DA the two endpoint contributions retain the momentum phase explicitly,

$$
h_{\rm DA}(z)=\left[A_1e^{i(\phi_1-P_z|z|)}+A_2e^{i\phi_2}
+\frac{A'_1e^{i(\phi'_1-P_z|z|)}+A'_2e^{i\phi'_2}}{|z|}\right]
e^{-\Lambda|z|/(\hbar c)},
$$

with flavor-class constraints removing or identifying endpoint amplitudes.  If
`phase_transfer_da=true`, the input is first multiplied by
$e^{+izP_z/(2\hbar c)}$, projected onto the real midpoint-symmetric channel,
and rotated back; if false, the complex input is retained.
""".strip()


_TRANSFORM_FORMULA = r"""
After the selected tail is connected to the measured matrix element, each
resample is transformed with

$$
\widetilde q(x,P_z)=N\sum_z w_z\,
e^{+i(x-x_{\rm shift})P_z z/(\hbar c)}h(z),
\qquad
N=\begin{cases}P_z/(2\pi\hbar c)&\texttt{pz\_over\_2pi},\\
1/(2\pi)&\texttt{one\_over\_2pi},\\1&\texttt{none}.
\end{cases}
$$

The default has phase sign $+1$ and $x_{\rm shift}=0$.  Uniform grids use the
full endpoint rectangle weights; nonuniform grids use trapezoidal weights.
Negative-$z$ completion and the final sector projection follow the target
observable, polarization, and GPD flow provenance.  Range selection is done
on center fits; its selected range is then fixed for all resamples and
LA/NLA-prior candidates.

For `smooth=linear`, the measured weight falls linearly from one at the
selected $z_{\min}$ to zero at $z_{\max}$.  For `smooth=none`, measured data
retain unit weight through the requested extent and the fitted tail is used
only beyond it (including a possible grid-rounded outer point).
""".strip()


def _json_attr(attrs: dict[str, object], name: str) -> object:
    value = attrs.get(name)
    if not isinstance(value, str):
        return value
    return json.loads(value)


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


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    lines = [
        "# Fourier Transform Stage Report",
        "",
        "This stage fits the finite-distance tail and transforms every resample to a quasi-distribution.",
        "",
        "## Method",
        "",
        "### Large-Distance Extrapolation",
        "",
        _TAIL_FORMULA,
        "",
        "### Fourier Transform and Projection",
        "",
        _TRANSFORM_FORMULA,
        "",
        "## Job Summary",
        "",
        ("| job | target / polarization | momentum [GeV] | sector / component | "
         "selected range [fm] | selected models | Q | chi2/dof | samples |"),
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for record in records:
        attrs = dict(output_attrs(record))
        diagnostics = record.summary.get("diagnostics", {})
        lines.append(
            f"| `{record.job_id}` | `{attrs.get('target_observable', 'n/a')}` / "
            f"`{attrs.get('polarization', 'n/a')}` | {format_value(attrs.get('momentum_gev'))} | "
            f"`{attrs.get('sector', record.params['scheme_scan']['sector'])}` / "
            f"`{attrs.get('component', 'n/a')}` | "
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
                series_label=_combined_series_label,
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Selection Policy",
            "",
            ("The range scan uses the first authored order and prior width. Runtime enumerates the authored "
             "model x zmin x zmax prefix up to `max_schemes`, keeps feasible center fits, and selects the "
             "largest-logGBF fit with Q >= `q_min`, falling back to the largest Q. If no center model reaches "
             "`q_min`, range recommendations continue until the job budget is exhausted; a numerically valid "
             "maximum-Q result is then published with an explicit fallback warning. With that interval fixed, "
             "every feasible authored LA/NLA and prior-width model is refitted. `model_average=false` chooses "
             "per-resample models with the same Q/logGBF rule and maximum-Q fallback; `model_average=true` uses "
             "normalized exp(logGBF) weights over all finite-logGBF candidates and adds no separate between-model "
             "variance. The candidate diagnostics below preserve both the selected result and the alternatives."),
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
            f"| `{item.get('model_id')}` | {format_value(item.get('z_min_fm'))} | "
            f"{format_value(item.get('z_max_fm'))} | {format_value(item.get('fit_success'))} | "
            f"{format_value(item.get('Q'))} | {format_value(item.get('chi2_dof'))} | "
            f"{format_value(item.get('logGBF'))} | {format_value(item.get('selected'))} | "
            f"{format_value(item.get('error'))} |"
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
                    f"| `{candidate.get('label')}` | `{name}` | "
                    f"{format_value(means.get(name))} | {format_value(sdevs.get(name))} |"
                )
        fallback_notice = (
            [
                "",
                "ATTENTION: no center model passed `q_min` after the allowed recommendation attempts; "
                "the numerically valid maximum-Q result was published anyway.",
            ]
            if diagnostics.get("fallback_no_q_passing")
            else []
        )
        lines.extend(
            [
                "",
                f"## `{record.job_id}`",
                "",
                "### Selected Fit",
                "",
                f"- Selected range: `{diagnostics.get('selected_range_label', 'n/a')}`",
                f"- Selected models: {format_value(diagnostics.get('selected_fit_model_labels'))}",
                f"- Model weights: {format_value(diagnostics.get('fit_model_weights'))}",
                f"- DA phase transfer: {format_value(attrs.get('phase_transfer_da'))}",
                *fallback_notice,
                "",
                "### Result Context",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| target observable | `{attrs.get('target_observable', 'n/a')}` |",
                f"| hadron / parton | `{attrs.get('hadron', 'n/a')}` / `{attrs.get('parton', 'n/a')}` |",
                f"| polarization / construction | `{attrs.get('polarization', 'n/a')}` / "
                f"`{attrs.get('gfix', 'n/a')}` |",
                f"| momentum | {format_value(attrs.get('momentum_gev'))} GeV |",
                *(
                    [
                        "| GPD initial/final Pz | "
                        f"{format_value(attrs.get('initial_momentum'))} / "
                        f"{format_value(attrs.get('final_momentum'))} |",
                        "| GPD xi / t | "
                        f"{format_value(attrs.get('xi'))} / "
                        f"{format_value(attrs.get('t_gev2'))} GeV^2 |",
                        "| GPD phase transfer / completion | "
                        f"`{attrs.get('phase_transfer_gpd', 'n/a')}` / "
                        f"`{attrs.get('gpd_completion_mode', 'n/a')}` |",
                    ]
                    if str(attrs.get("target_observable", "")).lower() == "gpd"
                    else []
                ),
                f"| x grid | {describe_grid(record.output.coords['x'], symbol='x')} |",
                f"| candidate $z_{{\\min}}$ [fm] | {format_value(params['zmin_fm'])} |",
                f"| candidate $z_{{\\max}}$ [fm] | {format_value(params['zmax_fm'])} |",
                f"| extension [fm] | {format_value(params['zmax_ext_fm'])} |",
                f"| smoothing | `{params['smooth']}` |",
                f"| orders | {format_value(scan['order'])} |",
                f"| Lambda0 [GeV] | {format_value(scan.get('Lambda0_gev', 0.0))} |",
                f"| tail-prior scales | {format_value(scan['posterior_prior_error_scale'])} |",
                f"| model average | {format_value(scan['model_average'])} |",
                f"| component / output scale | `{attrs.get('component', 'n/a')}` / "
                f"{format_value(attrs.get('output_scale'))} |",
                f"| transform | `{attrs.get('fourier_convention', 'n/a')}`; "
                f"prefactor `{attrs.get('prefactor', 'n/a')}` |",
                f"| quadrature | `{attrs.get('quadrature', 'n/a')}` |",
                f"| range candidates | {format_value(diagnostics.get('range_candidate_count'))} |",
                f"| model candidates | {format_value(diagnostics.get('model_candidate_count'))} |",
                "",
                "### Candidate Diagnostics",
                "",
                "#### Range and Fit-model Candidates",
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
                (f"The output records sector `{attrs.get('sector', 'n/a')}`, component "
                 f"`{attrs.get('component', 'n/a')}`, and multiplicative scale "
                 f"{format_value(attrs.get('output_scale'))}. Sector is authored in "
                 "`scheme_scan`; component and scale are derived from the target, polarization, "
                 "and sector. For a non-full GPD, the signed-y transform is projected afterward "
                 "using the polarization relation; a full GPD leaves the complex Fourier result "
                 "unprojected."),
                "",
                "| field | meaning |",
                "|---|---|",
                "| `selected_range` | Sample-average tail interval held fixed during all resample fits. |",
                ("| `selected_models`, `model_weights` | LA/NLA/prior candidates retained by "
                 "selection or model averaging. |"),
                ("| `component`, `output_scale` | Fourier channel and normalization selected "
                 "from target, polarization, and sector. |"),
                ("| `phase_transfer_da` | Whether the midpoint DA phase/symmetry projection "
                 "was applied before tail fitting. |"),
                ("| `phase_transfer_gpd`, `gpd_completion_mode` | GPD endpoint convention and "
                 "whether an exchanged-flow Hermitian partner completed signed z. |"),
                ("| `zmax_ext_fm` | Maximum physical separation of the finite transform, "
                 "distinct from the fitted data interval. |"),
                ("| `Q`, `chi2/dof` | Fit p-value and normalized chi-square diagnostic; neither "
                 "is a Fourier-distribution uncertainty. |"),
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
