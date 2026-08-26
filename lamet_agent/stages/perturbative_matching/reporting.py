"""Stage-level perturbative-matching reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels import load_kernel_document
from lamet_agent.plotting import X_LABEL
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


def _kernel_document(kernel_id: str) -> str:
    return load_kernel_document(kernel_id)


def _integral(data: EnsembleData, *, lo: float, hi: float) -> float:
    if data.dims != ["x"]:
        raise ValueError("matching report requires one-dimensional x distributions")
    x = np.asarray(data.coords["x"], dtype=float)
    selected = data
    if np.iscomplexobj(data.values):
        component = str(data.attrs.get("matching_component", data.attrs.get("component", ""))).lower()
        if component in {"im", "imag", "imaginary"}:
            selected = data.imag
        elif component in {"re", "real", "both"}:
            selected = data.real
        else:
            raise ValueError("complex matching-report input requires explicit real/imag component provenance")
    values = np.asarray(selected.mean, dtype=float)
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError("matching report integration window has fewer than two points")
    return float(np.trapezoid(values[mask], x[mask]))


def _diagnostics(record: StageReportRecord) -> tuple[float, float, float]:
    quasi = record.inputs.get("quasi")
    if not isinstance(quasi, EnsembleData) or not isinstance(record.output, EnsembleData):
        raise TypeError("matching report requires numerical quasi and matched distributions")
    x_out = np.asarray(record.output.coords["x"], dtype=float)
    lo, hi = float(np.min(x_out)), float(np.max(x_out))
    quasi_integral = _integral(quasi, lo=lo, hi=hi)
    matched_integral = _integral(record.output, lo=lo, hi=hi)
    relative = abs(matched_integral - quasi_integral) / abs(quasi_integral) if quasi_integral else float("nan")
    return quasi_integral, matched_integral, relative


def _scheme_text(scheme: str) -> str:
    return {
        "ratio": "The ratio kernel uses the regular coefficient without an additional finite conversion.",
        "msbar": "The MSbar kernel includes the finite MSbar conversion at the declared scale.",
        "hybrid": "The hybrid kernel adds the Wilson-line sine-integral correction and depends on the dimensionless product $z_sP_z$.",
    }[scheme]


def _kernel_structure(kernel_id: str) -> dict[str, object]:
    tokens = kernel_id.split("_")
    if len(tokens) < 5 or tokens[0] not in {"GI", "CG"}:
        raise ValueError(f"kernel id has no recognized gauge/operator structure: {kernel_id}")
    scheme_tokens = [token for token in tokens if token in {"ratio", "hybrid", "msbar"}]
    if len(scheme_tokens) != 1:
        raise ValueError(f"kernel id must contain exactly one scheme: {kernel_id}")
    distribution = "DA" if "DA" in tokens else "PDF" if "PDF" in tokens else None
    if distribution is None:
        raise ValueError(f"kernel id has no PDF/DA target: {kernel_id}")
    distribution_index = tokens.index(distribution)
    parton_index = tokens.index("quark") if "quark" in tokens else distribution_index
    return {
        "gauge": tokens[0],
        "operator": "_".join(tokens[1:parton_index]),
        "distribution": distribution,
        "scheme": scheme_tokens[0],
        "order": next((token for token in reversed(tokens) if token in {"LO", "NLO", "NNLO"}), "not encoded"),
        "component": next((token for token in tokens if token in {"re", "im"}), "full"),
        "resummation": "RGR" if "RGR" in tokens else "none",
    }


def _is_even_about_zero(data: EnsembleData) -> bool:
    x = np.asarray(data.coords["x"], dtype=float)
    values = np.real(np.asarray(data.mean))
    if x.size < 3 or np.min(x) >= 0 or np.max(x) <= 0:
        return False
    order = np.argsort(x)
    x, values = x[order], values[order]
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale == 0:
        return False
    return bool(np.max(np.abs(values - np.interp(-x, x, values))) <= 1e-6 * scale)


def _has_interior_gap(data: EnsembleData) -> bool:
    x = np.sort(np.asarray(data.coords["x"], dtype=float))
    if x.size < 4:
        return False
    spacing = np.diff(x)
    median = float(np.median(spacing))
    return bool(median > 0 and np.max(spacing) > 2.0 * median)


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    kernel_ids = list(dict.fromkeys(str(record.params["kernel_id"]) for record in records))
    lines = [
        "# Perturbative Matching Stage Report",
        "",
        "This stage applies the selected NLO matching kernel sample by sample to convert quasi-distributions into light-cone distributions.",
        "",
        "## Job Summary",
        "",
        r"| job | kernel | scheme | momentum [GeV] | $\mu$ [GeV] | quasi integral | matched integral | relative change |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    cached_diagnostics: dict[str, tuple[float, float, float]] = {}
    for record in records:
        attrs = output_attrs(record)
        quasi_integral, matched_integral, relative = _diagnostics(record)
        cached_diagnostics[record.job_id] = (quasi_integral, matched_integral, relative)
        lines.append(
            f"| `{record.job_id}` | `{record.params['kernel_id']}` | `{record.params['scheme']}` | "
            f"{format_value(attrs.get('momentum_gev'))} | {format_value(record.params['mu'])} | "
            f"{format_value(quasi_integral)} | {format_value(matched_integral)} | {format_value(100.0 * relative)}% |"
        )
    lines.extend(
        [
            "",
            "The integrals use the light-cone output range for both arrays.  They are diagnostics, not a normalization verdict: the expected normalization is fixed upstream by the coordinate-space matrix element and its projection convention.",
            "",
            "## Kernel-id and Field Definitions",
            "",
            "| field | meaning |",
            "|---|---|",
            "| `kernel_id` | Public kernel filename stem encoding gauge construction, Dirac operator, target distribution, renormalization scheme, resummation options, component, and perturbative order. |",
            "| `mu` | MSbar renormalization/matching scale in GeV. |",
            "| `zs_fm` | Hybrid Wilson-line switching distance; absent for ratio/MSbar kernels. |",
            "| `kernel_parameters` | Parameters owned by a particular kernel, such as RGR kappa and its minimum running scale. |",
            "| matching matrix | Discretized convolution from the quasi input grid to the requested light-cone output grid. |",
            "",
            "## Stage Overview",
            "",
            *stage_overlay_lines(
                records,
                artifact_directory,
                coordinate="x",
                stem="matching_overview",
                xlabel=X_LABEL,
                ylabel="matched distribution",
                band=True,
            ),
        ]
    )
    for kernel_id in kernel_ids:
        structure = _kernel_structure(kernel_id)
        lines.extend(
            [
                "",
                f"## Kernel `{kernel_id}`",
                "",
                "| property | value |",
                "|---|---|",
                *[f"| {name} | `{format_value(value)}` |" for name, value in structure.items()],
                "",
                "### Matching Formula and Literature Consistency Check",
                "",
                _kernel_document(kernel_id),
            ]
        )
    for record in records:
        attrs = output_attrs(record)
        diagnostics = record.summary.get("diagnostics", {})
        quasi_integral, matched_integral, relative = cached_diagnostics[record.job_id]
        quasi = record.inputs["quasi"]
        scale = float(attrs.get("output_scale", 1.0))
        mirrored = abs(scale - 1.0) > 1e-12 and _is_even_about_zero(record.output)
        gap = _has_interior_gap(record.output)
        lines.extend(
            [
                "",
                f"## `{record.job_id}`",
                "",
                "### Analysis Settings",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| kernel | `{record.params['kernel_id']}` |",
                f"| scheme | `{record.params['scheme']}` |",
                f"| momentum | {format_value(attrs.get('momentum_gev'))} GeV |",
                f"| renormalization scale | {format_value(record.params['mu'])} GeV |",
                f"| hybrid switch | {format_value(record.params.get('hybrid', {}).get('zs_fm'))} fm |",
                f"| quasi grid | {describe_grid(quasi.coords['x'], symbol='x')} |",
                f"| light-cone grid | {describe_grid(record.output.coords['x'], symbol='x')} |",
                f"| kernel parameters | {format_value(record.params.get('kernel_parameters', {}))} |",
                f"| matching matrix shape | {format_value(diagnostics.get('matrix_shape'))} |",
                f"| resampling | `{getattr(record.output, 'resample', 'n/a')}` with {format_value(getattr(record.output, 'n_sample', None))} samples |",
                "",
                "### Integral Diagnostic",
                "",
                f"- Quasi input: {format_value(quasi_integral)}",
                f"- Matched output: {format_value(matched_integral)}",
                f"- Relative change: {format_value(100.0 * relative)}%",
                f"- Fourier projection scale: {format_value(scale)}",
                *(
                    [
                        f"- The stored matched distribution is symmetric about x=0; one-sided quasi/matched integrals after removing the projection scale are {format_value(quasi_integral / scale)} / {format_value(matched_integral / scale)}."
                    ]
                    if mirrored
                    else []
                ),
                *(
                    [
                        "- The matched grid contains an interior gap. The trapezoid diagnostic bridges that interval linearly, so part of the integral is interpolation."
                    ]
                    if gap
                    else []
                ),
                "- Compare these values with the normalization convention fixed upstream (`normalization=true` gives unity only for the corresponding operator/projection convention).",
                "",
                "### Matching Scheme",
                "",
                _scheme_text(str(record.params["scheme"])),
                "",
                "The LO contribution is the identity. The shipped kernel document above is the source of truth for the implemented NLO coefficient, plus prescription, support regions, and any RGR or hybrid correction.",
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
