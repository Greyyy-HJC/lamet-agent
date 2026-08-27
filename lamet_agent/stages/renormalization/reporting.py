"""Stage-level renormalization reporting."""

from __future__ import annotations

import json
import math
import re
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
from lamet_agent.stages.renormalization._plotting import (
    RENORMALIZED_MATRIX_ELEMENT_LABELS,
    Z_FM_LABEL,
    lattice_spacing_label,
    momentum_label,
    render_fit_diagnostics,
    render_result,
    render_zmsbar_comparison,
)
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
Self-renormalization first fits the reusable factor $z_R(z,a)$.  The fit job's
explicit conversion kernel fixes the finite slope $m_0$, while target
application uses the finite term selected by the apply job's `kernel_id`:

$$
g(z)-\ln Z_{\overline{\rm MS}}^{\rm fit}(z;\mu)\simeq m_0z+b,
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
        (params["scheme"], params["strategy"]) for params in (effective_params(record.params) for record in records)
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


def _load_plot_data(record: StageReportRecord) -> dict[str, object] | None:
    params = effective_params(record.params)
    name = "self_renormalization.json" if params["type"] == "fit" else "renormalization.json"
    path = record.artifact_directory / "diagnostics" / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    plot_data = payload.get("plot_data")
    return plot_data if isinstance(plot_data, dict) else None


def _record_momentum_label(record: StageReportRecord) -> str:
    attrs = output_attrs(record)
    return momentum_label(
        attrs.get("momentum_gev"),
        momentum=attrs.get("momentum"),
        default=record.job_id,
    )


def _record_spacing_label(record: StageReportRecord) -> str:
    return lattice_spacing_label(output_attrs(record).get("lattice_spacing_fm"), default=record.job_id)


def _combined_series_label(record: StageReportRecord) -> str:
    attrs = output_attrs(record)
    momentum = _record_momentum_label(record)
    spacing = attrs.get("lattice_spacing_fm")
    if isinstance(spacing, (int, float)) and not isinstance(spacing, bool) and math.isfinite(float(spacing)):
        return f"{momentum}, {_record_spacing_label(record)}"
    return momentum


def _slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return text or "unknown"


def _float_slug(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _momentum_key(record: StageReportRecord) -> tuple[str, object] | None:
    attrs = output_attrs(record)
    authored = attrs.get("momentum")
    if isinstance(authored, str) and authored:
        return "authored", authored.upper()
    physical = attrs.get("momentum_gev")
    if isinstance(physical, (int, float)) and not isinstance(physical, bool) and math.isfinite(float(physical)):
        return "physical", round(float(physical), 12)
    return None


def _momentum_stem(record: StageReportRecord) -> str:
    key = _momentum_key(record)
    if key is None:
        return _slug(record.job_id)
    kind, value = key
    return _slug(value) if kind == "authored" else f"pz{_float_slug(float(value))}gev"


def _apply_records(records: tuple[StageReportRecord, ...]) -> list[StageReportRecord]:
    return [
        record
        for record in records
        if effective_params(record.params)["type"] == "apply" and getattr(record.output, "dims", None) == ["z"]
    ]


def _render_formal_figures(
    records: tuple[StageReportRecord, ...], artifact_directory: Path
) -> dict[str, list[tuple[str, str]]]:
    figures: dict[str, list[tuple[str, str]]] = {}
    directory = artifact_directory / "plots"
    for record in records:
        params = effective_params(record.params)
        plot_data = _load_plot_data(record)
        entries: list[tuple[str, str]] = []
        if params["type"] == "fit":
            if plot_data is not None and plot_data.get("kind") == "fit":
                entries.extend(
                    render_fit_diagnostics(
                        plot_data,
                        directory=directory,
                        prefix=f"{record.job_id}_",
                        formats=("pdf", "svg"),
                    )
                )
        elif getattr(record.output, "dims", None) == ["z"]:
            entries.append(
                render_result(
                    record.output,
                    directory=directory,
                    stem=f"{record.job_id}_result",
                    formats=("pdf", "svg"),
                )
            )
            if plot_data is not None and plot_data.get("kind") == "apply":
                entries.append(
                    render_zmsbar_comparison(
                        plot_data,
                        directory=directory,
                        stem=f"{record.job_id}_zmsbar_compare",
                        formats=("pdf", "svg"),
                    )
                )
        figures[record.job_id] = entries
    return figures


def _formal_figure_lines(entries: list[tuple[str, str]]) -> list[str]:
    if not entries:
        return ["No stage-level figures were available."]
    lines: list[str] = []
    for stem, caption in entries:
        lines.extend(
            [
                f"![{caption}](plots/{stem}.svg)",
                "",
                f"[{caption} (PDF)](plots/{stem}.pdf)",
                "",
            ]
        )
    return lines


def _grouped_overlay_lines(records: tuple[StageReportRecord, ...], artifact_directory: Path) -> list[str]:
    apply_records = _apply_records(records)
    lines: list[str] = []
    by_spacing: dict[float, list[StageReportRecord]] = {}
    for record in apply_records:
        spacing = output_attrs(record).get("lattice_spacing_fm")
        if isinstance(spacing, (int, float)) and not isinstance(spacing, bool) and math.isfinite(float(spacing)):
            by_spacing.setdefault(round(float(spacing), 12), []).append(record)
    spacing_groups = [(spacing, group) for spacing, group in sorted(by_spacing.items()) if len(group) >= 2]
    if spacing_groups:
        lines.extend(["### Fixed lattice spacing", ""])
        for spacing, group in spacing_groups:
            lines.extend(
                stage_overlay_lines(
                    tuple(group),
                    artifact_directory,
                    coordinate="z",
                    stem=f"renormalized_a{_float_slug(spacing)}fm",
                    ylabel=RENORMALIZED_MATRIX_ELEMENT_LABELS,
                    xlabel=Z_FM_LABEL,
                    series_label=_record_momentum_label,
                )
            )

    by_momentum: dict[tuple[str, object], list[StageReportRecord]] = {}
    for record in apply_records:
        key = _momentum_key(record)
        if key is not None:
            by_momentum.setdefault(key, []).append(record)
    momentum_groups = [
        group for _key, group in sorted(by_momentum.items(), key=lambda item: str(item[0])) if len(group) >= 2
    ]
    if momentum_groups:
        lines.extend(["### Fixed momentum: lattice-spacing dependence", ""])
        for group in momentum_groups:
            lines.extend(
                stage_overlay_lines(
                    tuple(group),
                    artifact_directory,
                    coordinate="z",
                    stem=f"discrete_effect_{_momentum_stem(group[0])}",
                    ylabel=RENORMALIZED_MATRIX_ELEMENT_LABELS,
                    xlabel=Z_FM_LABEL,
                    series_label=_record_spacing_label,
                )
            )
    return lines


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    formal_figures = _render_formal_figures(records, artifact_directory)
    apply_records = tuple(_apply_records(records))
    overview_lines = (
        stage_overlay_lines(
            apply_records,
            artifact_directory,
            coordinate="z",
            stem="renormalization_overview",
            ylabel=RENORMALIZED_MATRIX_ELEMENT_LABELS,
            xlabel=Z_FM_LABEL,
            series_label=_combined_series_label,
        )
        if apply_records
        else ["No compatible one-dimensional outputs were available for a stage overlay."]
    )
    grouped_lines = _grouped_overlay_lines(records, artifact_directory)
    lines = [
        "# Renormalization Stage Report",
        "",
        "This stage maps bare coordinate-space matrix elements to renormalized matrix elements "
        "while preserving every resample.",
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
        kind = params["type"]
        lines.append(
            f"| `{record.job_id}` | {kind} | `{params['scheme']}` | `{params['strategy']}` | "
            f"{format_value(params['normalization'])} | {format_value(getattr(record.output, 'n_sample', None))} | "
            f"`{summary.get('result')}` |"
        )
    lines.extend(["", "## Stage Overview", "", *overview_lines, *grouped_lines])
    for record in records:
        params = effective_params(record.params)
        attrs = output_attrs(record)
        diagnostics = record.summary.get("diagnostics", {})
        short_distance_range = [
            diagnostics.get("short_distance_min_fm"),
            diagnostics.get("short_distance_max_fm"),
        ]
        output_grid = describe_grid(
            record.output.coords.get("z", record.output.coords.get("a", [])), symbol="z"
        )
        lines.extend(
            [
                "",
                f"## `{record.job_id}`",
                "",
                "### Parameters and Provenance",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| scheme | `{params['scheme']}` |",
                f"| strategy | `{params['strategy']}` |",
                f"| type | `{params['type']}` |",
                f"| kernel id | `{params.get('kernel_id', 'n/a')}` |",
                f"| kernel parameters | {format_value(params.get('kernel_parameters', {}))} |",
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
                f"| short-distance fit range [fm] | {format_value(short_distance_range)} |",
                f"| lattice-spacing fit range [fm] | {format_value(diagnostics.get('lattice_spacing_range_fm'))} |",
                f"| output z range [fm] | {format_value(diagnostics.get('z_range_fm'))} |",
                f"| input z ranges [fm] | {format_value(diagnostics.get('input_z_ranges_fm'))} |",
                f"| denominator kind | `{diagnostics.get('denominator_kind', 'n/a')}` |",
                f"| ZMSbar kernel | `{diagnostics.get('kernel_id', attrs.get('kernel_id', 'n/a'))}` |",
                "",
                "### Coverage and Statistical Semantics",
                "",
                f"- Output grid: {output_grid}",
                (
                    "- Every operation acts sample by sample. Matrix denominators are aligned by coordinate "
                    "value before division; numeric constants carry no artificial uncertainty."
                ),
                (
                    f"- z coverage policy: `{params.get('z_coverage_policy', 'n/a')}`. "
                    "The origin is preserved exactly."
                ),
                "",
                "### Field Definitions",
                "",
                "| field | meaning |",
                "|---|---|",
                "| `scheme` | Finite prescription: ratio, hybrid, or MSbar. |",
                "| `strategy` | External denominator or a reusable factor fitted from reference ensembles. |",
                "| `type` | Fit a reusable factor or apply a renormalization prescription. |",
                "| `kernel_id` | Explicit coordinate-space conversion formula selected for self-renormalization. |",
                "| `kernel_parameters` | Explicit overrides of signature parameters not fixed by input coordinates. |",
                "| `d`, `m0_gev` | Finite logarithmic and linear operator corrections used by self-renormalization. |",
                "| `delta_m_gev` | Long-distance exponential correction in the external hybrid prescription. |",
                (
                    "| `LambdaQCD_gev`, `mu` | Scales entering the continuum logarithm and perturbative "
                    "finite conversion. |"
                ),
                "",
                "### Figures",
                "",
                *_formal_figure_lines(formal_figures.get(record.job_id, [])),
                "",
                "### Artifacts",
                "",
                "| job | artifact |",
                "|---|---|",
                *artifact_rows(record, artifact_directory),
            ]
        )
    return write_report(artifact_directory, lines)
