"""Stage-level renormalization reporting."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from lamet_agent.stages._reporting import (
    StageReportRecord,
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


_EXTERNAL_RATIO = r"""
For an external ratio or MSbar denominator, the renormalized matrix element is
formed pointwise on every resample,

$$ h_s^R(z)=h_s^{\rm tar}(z)/h_s^{\rm den}(z). $$

A finite nonzero scalar denominator is applied at every coordinate. If origin
normalization is enabled, each input sample is first divided by its own value at
$z=0$.
""".strip()


_EXTERNAL_HYBRID = r"""
The external hybrid prescription uses the pointwise ratio up to the switching
distance $z_s$. At larger separations it anchors the denominator at the switch
and applies the long-distance mass correction,

$$
h_s^R(z)=\begin{cases}
h_s^{\rm tar}(z)/h_s^{\rm den}(z),& |z|\le z_s,\\
e^{(\delta m+m_0)(|z|-z_s)/(\hbar c)}
h_s^{\rm tar}(z)/h_s^{\rm den}(z_s),& |z|>z_s.
\end{cases}
$$

The authored prescription is applied sample by sample; the bare matrix element
is not refitted in this path.
""".strip()


_SELF_RATIO = r"""
Self-renormalization extracts a reusable factor from reference matrix elements
at several lattice spacings. First, the reference data are fitted in $(a,z)$,

$$
\ln M(z,a)=\frac{kz}{a\ln(a\Lambda_{\rm QCD})}+g(z)+f_1(z)a.
$$

The short-distance values of $g(z)$ are then matched to the coordinate-space
$Z_{\overline{\rm MS}}$ kernel,

$$
g(z)-\ln Z_{\overline{\rm MS}}^{\rm fit}(z;\mu)\simeq m_0z+b,
$$

which determines the finite linear term and produces the reusable $z_R(z,a)$.
For a target matrix element,

$$
h_s^R(z)=\frac{h_s^{\rm tar}(z)}{z_R(z,a)Z_{\overline{\rm MS}}(z;\mu)}.
$$

The origin is passed through unchanged. If the target reaches beyond the
reference grid, only the fitted long-distance finite term is extended; the
endpoint is not frozen.
""".strip()


_SELF_HYBRID = r"""
Self-hybrid renormalization uses the external ratio at short distance and the
reusable self-renormalization factor at long distance,

$$
h_s^R(z)=\begin{cases}
h_s^{\rm tar}(z)/h_s^{\rm den}(z),&|z|\le z_s,\\
h_s^{\rm tar}(z)/(z_R(z,a)Z_{T,s}),&|z|>z_s,
\end{cases}
\qquad
Z_{T,s}=h_s^{\rm den}(z_s)/z_R(z_s,a).
$$

The transfer factor is constructed for each resample, preserving continuity
and denominator uncertainty at the switch.
""".strip()


_SELF_MSBAR = r"""
For self-renormalization in the MSbar prescription, the fitted reusable factor
is applied as

$$ h_s^R(z)=h_s^{\rm tar}(z)/z_R(z,a). $$

The origin is preserved, while the matching scale, lattice-spacing range and
coordinate coverage are inherited from the fitted factor.
""".strip()


_SCHEME_LABELS = {"ratio": "ratio", "hybrid": "hybrid", "msbar": "MSbar"}
_POLICY_LABELS = {
    "strict": "require full target/factor overlap",
    "intersection": "keep only the common z range",
    "extrapolate": "extend only toward larger z using the fitted long-distance term",
}
_FIT_REPORT_STEMS = (
    ("factor", r"Extracted reusable $Z_R(z,a)$"),
    ("fit_lnM_vs_inv_a", r"Reference $\ln|M|$ fit versus $a^{-1}$"),
    ("fit_m_over_zR", r"$M_{\rm bare}/Z_R$ consistency check"),
)


def _method_text(records: tuple[StageReportRecord, ...]) -> list[str]:
    combinations = {(record.params["scheme"], record.params["strategy"]) for record in records}
    blocks: list[str] = []
    for scheme, strategy in sorted(combinations):
        scheme_label = _SCHEME_LABELS.get(str(scheme), str(scheme))
        if strategy == "self_renormalization":
            title = f"Self-renormalization ({scheme_label} prescription)"
            formula = _SELF_HYBRID if scheme == "hybrid" else _SELF_MSBAR if scheme == "msbar" else _SELF_RATIO
        else:
            title = f"External denominator ({scheme_label} prescription)"
            formula = _EXTERNAL_HYBRID if scheme == "hybrid" else _EXTERNAL_RATIO
        blocks.extend([f"### {title}", "", formula, ""])
    if any(record.params["strategy"] == "self_renormalization" for record in records):
        blocks.extend(
            [
                "The fit job produces one sample-bearing factor that can be reused by the application jobs. "
                "All divisions and remappings preserve the input resample ensemble.",
                "",
            ]
        )
    return blocks


def _load_diagnostic_payload(record: StageReportRecord) -> dict[str, Any]:
    name = "self_renormalization.json" if record.params["type"] == "fit" else "renormalization.json"
    path = record.artifact_directory / "diagnostics" / name
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_plot_data(record: StageReportRecord) -> dict[str, object] | None:
    plot_data = _load_diagnostic_payload(record).get("plot_data")
    return plot_data if isinstance(plot_data, dict) else None


def _record_momentum_label(record: StageReportRecord) -> str:
    attrs = output_attrs(record)
    return momentum_label(attrs.get("momentum_gev"), momentum=attrs.get("momentum"), default=record.job_id)


def _record_spacing_label(record: StageReportRecord) -> str:
    return lattice_spacing_label(record.output.ensemble.a_s)


def _combined_series_label(record: StageReportRecord) -> str:
    momentum = _record_momentum_label(record)
    spacing = record.output.ensemble.a_s
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
        if record.params["type"] == "apply" and getattr(record.output, "dims", None) == ["z"]
    ]


def _relative_link(path: Path, stage_directory: Path) -> str:
    return path.resolve().relative_to(stage_directory.resolve()).as_posix()


def _declared_artifacts(record: StageReportRecord, stage_directory: Path) -> list[tuple[str, str]]:
    raw = record.summary.get("artifacts", [])
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise TypeError(f"job '{record.job_id}' summary.artifacts must be a string list")
    checked: list[tuple[str, str]] = []
    for relative in raw:
        path = (record.artifact_directory / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"job '{record.job_id}' declared missing artifact: {path}")
        checked.append((relative, _relative_link(path, stage_directory)))
    return checked


def _fit_job_figures(record: StageReportRecord, artifact_directory: Path) -> list[tuple[str, str, str]]:
    """Prefer job-local fit figures and fall back to stage rendering for old runs."""
    entries: list[tuple[str, str, str]] = []
    complete = True
    for stem, caption in _FIT_REPORT_STEMS:
        svg = record.artifact_directory / "plots" / f"{stem}.svg"
        pdf = record.artifact_directory / "plots" / f"{stem}.pdf"
        if not svg.is_file() or not pdf.is_file():
            complete = False
            break
        entries.append((
            _relative_link(svg, artifact_directory),
            _relative_link(pdf, artifact_directory),
            caption,
        ))
    if complete:
        return entries

    plot_data = _load_plot_data(record)
    if plot_data is None or plot_data.get("kind") != "fit":
        return []
    rendered = render_fit_diagnostics(
        plot_data,
        directory=artifact_directory / "plots",
        prefix=f"{record.job_id}_",
        formats=("pdf", "svg"),
    )
    rendered_by_stem = {stem: caption for stem, caption in rendered}
    for stem, caption in _FIT_REPORT_STEMS:
        if stem not in rendered_by_stem:
            continue
        entries.append((f"plots/{record.job_id}_{stem}.svg", f"plots/{record.job_id}_{stem}.pdf", caption))
    return entries


def _render_formal_figures(
    records: tuple[StageReportRecord, ...], artifact_directory: Path
) -> dict[str, list[tuple[str, str, str]]]:
    figures: dict[str, list[tuple[str, str, str]]] = {}
    directory = artifact_directory / "plots"
    for record in records:
        plot_data = _load_plot_data(record)
        if record.params["type"] == "fit":
            figures[record.job_id] = _fit_job_figures(record, artifact_directory)
            continue
        entries: list[tuple[str, str, str]] = []
        if getattr(record.output, "dims", None) == ["z"]:
            stem, caption = render_result(
                record.output,
                directory=directory,
                stem=f"{record.job_id}_result",
                formats=("pdf", "svg"),
                sample_error_mode=str(output_attrs(record).get("sample_error_mode", "covariance")),
            )
            entries.append((f"plots/{stem}.svg", f"plots/{stem}.pdf", caption))
            if plot_data is not None and plot_data.get("kind") == "apply":
                stem, caption = render_zmsbar_comparison(
                    plot_data,
                    directory=directory,
                    stem=f"{record.job_id}_zmsbar_compare",
                    formats=("pdf", "svg"),
                )
                entries.append((f"plots/{stem}.svg", f"plots/{stem}.pdf", caption))
        figures[record.job_id] = entries
    return figures


def _formal_figure_lines(entries: list[tuple[str, str, str]]) -> list[str]:
    if not entries:
        return ["No figures were available."]
    lines: list[str] = []
    for svg, pdf, caption in entries:
        lines.extend([f"![{caption}]({svg})", "", f"[{caption} (PDF)]({pdf})", ""])
    return lines


def _range_text(values: Any, symbol: str) -> str:
    if isinstance(values, (list, tuple)) and len(values) == 2:
        return f"{symbol} = {format_value(values[0])}–{format_value(values[1])} fm"
    return f"{symbol} range unavailable"


def _coverage_text(record: StageReportRecord, diagnostics: dict[str, Any]) -> str:
    z_range = diagnostics.get("z_range_fm")
    z_count = diagnostics.get("z_count")
    if z_range is None and hasattr(record.output, "coords"):
        z_values = record.output.coords.get("z", [])
        if len(z_values):
            z_range = [float(min(z_values)), float(max(z_values))]
            z_count = len(z_values)
    text = _range_text(z_range, "z")
    if z_count is not None:
        text += f", {format_value(z_count)} points"
    if record.params["type"] == "fit":
        text += "; " + _range_text(diagnostics.get("lattice_spacing_range_fm"), "a")
    return text


def _operation_label(record: StageReportRecord) -> str:
    if record.params["type"] == "fit":
        return "Extract reusable self-renormalization factor"
    return "Renormalize target matrix element"


def _physical_result_label(record: StageReportRecord) -> str:
    return r"$Z_R(z,a)$" if record.params["type"] == "fit" else r"$h^R(z)$"


def _sample_count(record: StageReportRecord, diagnostics: dict[str, Any]) -> Any:
    if record.params["type"] == "fit" and diagnostics.get("reference_sample_count") is not None:
        return diagnostics["reference_sample_count"]
    return diagnostics.get("target_samples", getattr(record.output, "n_sample", None))


def _prescription_label(record: StageReportRecord) -> str:
    scheme = _SCHEME_LABELS.get(str(record.params.get("scheme")), str(record.params.get("scheme")))
    if record.params.get("strategy") == "self_renormalization":
        return f"Self-renormalization, {scheme}"
    return f"External denominator, {scheme}"


def _quality_text(value: Any) -> str:
    if not isinstance(value, dict):
        return "not recorded"
    chi2_dof = value.get("chi2_dof")
    q = value.get("Q")
    if chi2_dof is None and value.get("chi2") is not None and value.get("dof"):
        chi2_dof = float(value["chi2"]) / float(value["dof"])
    if chi2_dof is None and q is None:
        return "not recorded"
    return f"chi2/dof={format_value(chi2_dof)}, Q={format_value(q)}"


def _grouped_overlay_lines(records: tuple[StageReportRecord, ...], artifact_directory: Path) -> list[str]:
    apply_records = _apply_records(records)
    lines: list[str] = []
    by_spacing: dict[float, list[StageReportRecord]] = {}
    for record in apply_records:
        spacing = record.output.ensemble.a_s
        if isinstance(spacing, (int, float)) and not isinstance(spacing, bool) and math.isfinite(float(spacing)):
            by_spacing.setdefault(round(float(spacing), 12), []).append(record)
    spacing_groups = [(spacing, group) for spacing, group in sorted(by_spacing.items()) if len(group) >= 2]
    if spacing_groups:
        lines.extend(["### Fixed lattice spacing", ""])
        for spacing, group in spacing_groups:
            lines.extend(
                stage_overlay_lines(
                    tuple(group), artifact_directory, coordinate="z",
                    stem=f"renormalized_a{_float_slug(spacing)}fm",
                    ylabel=RENORMALIZED_MATRIX_ELEMENT_LABELS, xlabel=Z_FM_LABEL,
                    series_label=_record_momentum_label,
                )
            )

    by_momentum: dict[tuple[str, object], list[StageReportRecord]] = {}
    for record in apply_records:
        key = _momentum_key(record)
        if key is not None:
            by_momentum.setdefault(key, []).append(record)
    momentum_groups = [
        group
        for _key, group in sorted(by_momentum.items(), key=lambda item: str(item[0]))
        if len(group) >= 2
    ]
    if momentum_groups:
        lines.extend(["### Fixed momentum: lattice-spacing dependence", ""])
        for group in momentum_groups:
            lines.extend(
                stage_overlay_lines(
                    tuple(group), artifact_directory, coordinate="z",
                    stem=f"discrete_effect_{_momentum_stem(group[0])}",
                    ylabel=RENORMALIZED_MATRIX_ELEMENT_LABELS, xlabel=Z_FM_LABEL,
                    series_label=_record_spacing_label,
                )
            )
    return lines


def _artifact_summary(record: StageReportRecord, artifact_directory: Path) -> list[str]:
    checked = _declared_artifacts(record, artifact_directory)
    names = [relative for relative, _link in checked]
    plot_count = sum(relative.startswith("plots/") for relative in names)
    log_count = sum(relative.startswith("fit_logs/") for relative in names)
    lines = [
        "This job exports a sample-bearing NetCDF result and machine-readable diagnostic JSON. "
        f"It also retains {plot_count} diagnostic figure(s) and {log_count} fit log(s) where available.",
    ]
    key_links = [
        (relative, link)
        for relative, link in checked
        if relative == "output.nc" or relative.startswith("diagnostics/")
    ]
    if key_links:
        lines.append("Key files: " + "; ".join(f"[{relative}]({link})" for relative, link in key_links) + ".")
    return lines


def _fit_validation_lines(
    records: tuple[StageReportRecord, ...], figures: dict[str, list[tuple[str, str, str]]]
) -> list[str]:
    fit_records = [record for record in records if record.params["type"] == "fit"]
    if not fit_records:
        return []
    lines = [
        "## Self-renormalization Factor Validation",
        "",
        "The reusable factor is considered available when the reference fit and short-distance matching complete "
        "and a sample-bearing $Z_R(z,a)$ output is published.",
        "",
        "| job | reference fit | short-distance matching | samples | coverage |",
        "|---|---|---|---:|---|",
    ]
    for record in fit_records:
        diagnostics = record.summary.get("diagnostics", {})
        quality = diagnostics.get("fit_quality", {})
        reference_quality = _quality_text(quality.get("reference")) if isinstance(quality, dict) else "not recorded"
        matching_quality = _quality_text(quality.get("m0_matching")) if isinstance(quality, dict) else "not recorded"
        lines.append(
            f"| `{record.job_id}` | {reference_quality} | {matching_quality} | "
            f"{format_value(_sample_count(record, diagnostics))} | {_coverage_text(record, diagnostics)} |"
        )
    lines.extend([""])
    for record in fit_records:
        lines.extend(
            [
                f"### `{record.job_id}` representative diagnostics",
                "",
                *_formal_figure_lines(figures.get(record.job_id, [])),
            ]
        )
    return lines


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    formal_figures = _render_formal_figures(records, artifact_directory)
    apply_records = tuple(_apply_records(records))
    overview_lines = (
        stage_overlay_lines(
            apply_records, artifact_directory, coordinate="z", stem="renormalization_overview",
            ylabel=RENORMALIZED_MATRIX_ELEMENT_LABELS, xlabel=Z_FM_LABEL,
            series_label=_combined_series_label,
        )
        if apply_records else ["No compatible one-dimensional outputs were available for a stage overview."]
    )
    grouped_lines = _grouped_overlay_lines(records, artifact_directory)
    lines = [
        "# Renormalization Stage Report", "",
        "This stage converts bare coordinate-space matrix elements into physical renormalized matrix elements "
        "while preserving the full resample ensemble.", "",
        "## Method and Workflow", "", *_method_text(records),
        "## Job Summary", "",
        "| job | operation | prescription | physical result | coverage | samples |",
        "|---|---|---|---|---|---:|",
    ]
    for record in records:
        diagnostics = record.summary.get("diagnostics", {})
        lines.append(
            f"| `{record.job_id}` | {_operation_label(record)} | {_prescription_label(record)} | "
            f"{_physical_result_label(record)} | {_coverage_text(record, diagnostics)} | "
            f"{format_value(_sample_count(record, diagnostics))} |"
        )
    lines.extend(["", "## Stage Overview", "", *overview_lines, *grouped_lines])
    lines.extend(["", *_fit_validation_lines(records, formal_figures)])

    for record in records:
        params = record.params
        diagnostics = record.summary.get("diagnostics", {})
        output_grid = describe_grid(record.output.coords.get("z", record.output.coords.get("a", [])), symbol="z")
        lines.extend(["", f"## `{record.job_id}`", "", "### Result", ""])
        if params["type"] == "fit":
            quality = diagnostics.get("fit_quality", {})
            reference_quality = _quality_text(quality.get("reference")) if isinstance(quality, dict) else "not recorded"
            matching_quality = (
                _quality_text(quality.get("m0_matching")) if isinstance(quality, dict) else "not recorded"
            )
            lines.extend([
                "The reusable self-renormalization factor was extracted successfully and is available for target "
                "application jobs.",
                f"- Physical output: {_physical_result_label(record)}.",
                f"- Reference fit quality: {reference_quality}.",
                f"- Short-distance matching quality: {matching_quality}.",
            ])
        else:
            lines.extend([
                f"The target matrix element was converted to {_physical_result_label(record)}.",
                f"- Output grid: {output_grid}.",
                "- The transformation is sample by sample; matrix denominators are aligned by physical z value.",
            ])
            policy = params.get("z_coverage_policy")
            if policy:
                lines.append(
                    f"- Target coverage: {_POLICY_LABELS.get(str(policy), str(policy))}; "
                    "the origin is preserved."
                )
        lines.extend(["", "### Figures", ""])
        if params["type"] == "apply":
            lines.extend(_formal_figure_lines(formal_figures.get(record.job_id, [])))
        else:
            lines.append(
                "Representative fit diagnostics are shown in the factor validation section above; "
                "the remaining diagnostics are retained as artifacts."
            )
        lines.extend(["", "### Artifacts", "", *_artifact_summary(record, artifact_directory)])
        if params["type"] == "apply":
            diagnostics = record.summary.get("diagnostics", {})
            lines.extend([
                "", "### Coverage and Technical Details", "",
                f"- Output grid: {output_grid}.",
                f"- Input z ranges: {format_value(diagnostics.get('input_z_ranges_fm'))}.",
                f"- z points dropped for coverage: {format_value(diagnostics.get('n_z_coverage_dropped', 0))}; "
                f"long-distance extrapolated points: {format_value(diagnostics.get('n_z_extrapolated', 0))}.",
            ])
    return write_report(artifact_directory, lines)
