"""Stage-level renormalization reporting."""

from __future__ import annotations

from pathlib import Path

from lamet_agent.stages._reporting import StageReportRecord, artifact_rows, describe_grid, figure_lines, format_value, output_attrs, stage_overlay_lines, write_report
from lamet_agent.stages.renormalization.parameters import effective_params


_EXTERNAL_RATIO = r"""
For an external denominator, ratio/MSbar application is pointwise on every
resample,

$$ h_s^R(z)=h_s^{\rm tar}(z)/h_s^{\rm den}(z). $$

A finite nonzero scalar denominator is applied identically at every coordinate.
When `normalization=true`, matrix-element inputs are normalized independently at
$z=0$ before this division.
""".strip()


_EXTERNAL_HYBRID = r"""
The external hybrid scheme uses the pointwise ratio through $z_s$ and freezes
the denominator at the switch for larger separations,

$$
h_s^R(z)=\begin{cases}
h_s^{\rm tar}(z)/h_s^{\rm den}(z),& |z|\le z_s,\\
e^{(\delta m+m_0)(|z|-z_s)/(\hbar c)}
h_s^{\rm tar}(z)/h_s^{\rm den}(z_s),& |z|>z_s.
\end{cases}
$$

The same authored map is applied to all resamples; this stage does not refit the
bare matrix element.
""".strip()


_SELF_RATIO = r"""
Self-renormalization first fits the reusable factor $z_R(z,a)$.  The short
distance PDF conversion fixes the finite slope $m_0$, while target application
uses the ZMSbar finite term selected by `metadata.target_observable`:

$$
g(z)-\ln Z_{\overline{\rm MS}}^{\rm PDF}(z;\mu)\simeq m_0z+b,
\qquad
h_s^R(z)=\frac{h_s^{\rm tar}(z)}{z_R(z,a)
Z_{\overline{\rm MS}}(z;\mu)}.
$$

The origin is passed through unchanged.  If the target extends beyond the
reference grid, the fitted long-distance finite term is completed before the
factor is evaluated; endpoint freezing is not used.

The reference ensemble grid is fitted simultaneously with

$$
\ln M(z,a)=\frac{kz}{a\ln(a\Lambda_{\rm QCD})}+g(z)+f_1(z)a+m_0z.
$$

Only the fitted finite term $f_1(z)$ is extended when longer-distance target
coordinates require completion.
""".strip()


_SELF_HYBRID = r"""
The reusable factor controls the long-distance branch and an external
denominator controls the short-distance branch,

$$
h_s^R(z)=\begin{cases}
h_s^{\rm tar}(z)/h_s^{\rm den}(z),&|z|\le z_s,\\
h_s^{\rm tar}(z)/(z_R(z,a)Z_{T,s}),&|z|>z_s,
\end{cases}\qquad
Z_{T,s}=h_s^{\rm den}(z_s)/z_R(z_s,a).
$$

$Z_{T,s}$ is constructed per resample, preserving continuity and denominator
uncertainty across the switch.
""".strip()


_SELF_MSBAR = r"""
The MSbar self-renormalization application uses
$h_s^R(z)=h_s^{\rm tar}(z)/z_R(z,a)$ while preserving the origin.  Scale,
lattice spacing, and coordinate coverage must agree with the fitted factor.
""".strip()


def _method_text(records: tuple[StageReportRecord, ...]) -> list[str]:
    combinations = {
        (params["scheme"], params["strategy"])
        for params in (effective_params(record.params) for record in records)
    }
    blocks: list[str] = []
    for scheme, strategy in sorted(combinations):
        blocks.append(f"### `{scheme}` / `{strategy}`")
        if strategy == "self_renormalization":
            blocks.append(_SELF_HYBRID if scheme == "hybrid" else _SELF_MSBAR if scheme == "msbar" else _SELF_RATIO)
        elif scheme == "hybrid":
            blocks.append(_EXTERNAL_HYBRID)
        else:
            blocks.append(_EXTERNAL_RATIO)
        blocks.append("")
    return blocks


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    lines = [
        "# Renormalization Stage Report",
        "",
        "This stage maps bare coordinate-space matrix elements to renormalized matrix elements while preserving every resample.",
        "",
        "## Method",
        "",
        *_method_text(records),
        "## Job Summary",
        "",
        "| job | kind | scheme | strategy | normalization | samples | output |",
        "|---|---|---|---|---|---:|---|",
    ]
    for record in records:
        params = effective_params(record.params)
        summary = record.summary
        kind = "fit factor" if summary.get("result") == "renormalization_factor" else "apply"
        lines.append(
            f"| `{record.job_id}` | {kind} | `{params['scheme']}` | `{params['strategy']}` | "
            f"{format_value(params['normalization'])} | {format_value(getattr(record.output, 'n_sample', None))} | "
            f"`{summary.get('result')}` |"
        )
    lines.extend(["", "## Stage Overview", "", *stage_overlay_lines(records, artifact_directory, coordinate="z", stem="renormalization_overview", ylabel="renormalized matrix element")])
    for record in records:
        params = effective_params(record.params)
        attrs = output_attrs(record)
        diagnostics = record.summary.get("diagnostics", {})
        lines.extend([
            "",
            f"## `{record.job_id}`",
            "",
            "### Parameters and Provenance",
            "",
            "| quantity | value |",
            "|---|---|",
            f"| scheme | `{params['scheme']}` |",
            f"| strategy | `{params['strategy']}` |",
            f"| normalization | {format_value(params['normalization'])} |",
            f"| $z_s$ [fm] | {format_value(params.get('zs_fm'))} |",
            f"| $m_0$ [GeV] | {format_value(params.get('m0_gev', attrs.get('m0_gev')))} |",
            f"| $\\delta m$ [GeV] | {format_value(params.get('delta_m_gev'))} |",
            f"| $d$ | {format_value(params.get('d', attrs.get('d')))} |",
            f"| $\\mu$ [GeV] | {format_value(params.get('mu', attrs.get('scale_gev')))} |",
            f"| $\\Lambda_{{\\rm QCD}}$ [GeV] | {format_value(params.get('LambdaQCD_gev'))} |",
            f"| dimensions | {format_value(getattr(record.output, 'dims', diagnostics.get('dims')))} |",
            f"| coordinate unit | `{attrs.get('coord_unit', 'n/a')}` |",
            f"| fitted formula | {format_value(diagnostics.get('formula'))} |",
            f"| short-distance fit range [fm] | {format_value([diagnostics.get('short_distance_min_fm'), diagnostics.get('short_distance_max_fm')])} |",
            f"| lattice-spacing fit range [fm] | {format_value(diagnostics.get('lattice_spacing_range_fm'))} |",
            f"| output z range [fm] | {format_value(diagnostics.get('z_range_fm'))} |",
            f"| input z ranges [fm] | {format_value(diagnostics.get('input_z_ranges_fm'))} |",
            f"| denominator kind | `{diagnostics.get('denominator_kind', 'n/a')}` |",
            f"| ZMSbar model | `{diagnostics.get('zms_model', attrs.get('zms_model', 'n/a'))}` |",
            "",
            "### Coverage and Statistical Semantics",
            "",
            f"- Output grid: {describe_grid(record.output.coords.get('z', record.output.coords.get('a', [])), symbol='z')}",
            "- Every operation acts sample by sample. Matrix denominators are aligned by coordinate value before division; numeric constants carry no artificial uncertainty.",
            "- A reusable factor is selected at the target lattice spacing and must cover every nonzero target coordinate. The origin is preserved exactly.",
            "",
            "### Field Definitions",
            "",
            "| field | meaning |",
            "|---|---|",
            "| `scheme` | Finite prescription: ratio, hybrid, or MSbar. |",
            "| `strategy` | External denominator or a reusable factor fitted from reference ensembles. |",
            "| `d`, `m0_gev` | Finite logarithmic and linear operator corrections used by self-renormalization. |",
            "| `delta_m_gev` | Long-distance exponential correction in the external hybrid prescription. |",
            "| `LambdaQCD_gev`, `mu` | Scales entering the continuum logarithm and perturbative finite conversion. |",
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
        ])
    return write_report(artifact_directory, lines)
