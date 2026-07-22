"""Reporting helpers for extrapolation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gvar
import numpy as np
import xarray as xr

from lamet_agent.core.reporting import markdown_artifact_paths, resolve_report_target, translate_markdown_report


def write_extrapolation_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
    report_language: str = "en",
    backend: str = "",
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Path]:
    """Write a compact stage-level extrapolation report."""
    output = Path(path)
    target, language = resolve_report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Extrapolation Report", ""]
    for record in jobs:
        result = record.get("result", {})
        raw_artifacts = record.get("artifacts", {})
        artifacts = markdown_artifact_paths(
            raw_artifacts,
            base_dir=target.parent,
            path_keys=(
                "extrapolated_artifact",
                "fit_info_artifact",
                "extrapolated_plot",
                "extrapolated_plot_image",
                "chi2_xdep_plot",
                "chi2_xdep_plot_image",
                "adep_plot",
                "adep_plot_image",
                "pdep_plot",
                "pdep_plot_image",
            ),
        )
        a_orders = result.get("allow_order_a", [2])
        p_orders = result.get("allow_order_1overp", [2])
        ap_orders = result.get("allow_order_ap", [])
        xdep = result.get("fitting_param_xdep", [False, True, False])
        a_xdep = bool(xdep[0]) if xdep else True
        p_xdep = bool(xdep[1]) if len(xdep) > 1 else True
        include_ap = bool(xdep[2]) if len(xdep) > 2 else False
        a_text = ",".join(str(int(value)) for value in a_orders)
        p_text = ",".join(str(int(value)) for value in p_orders)
        ap_text = ",".join(str(int(value)) for value in ap_orders)
        ca_label = "c_{a,i}(x)" if a_xdep else "c_{a,i}"
        cp_label = "c_{p,j}(x)" if p_xdep else "c_{p,j}"
        cap_term = rf"+\sum_{{k\in\{{{ap_text}\}}}} c_{{ap,k}}(x)a^k p_z^k" if include_ap and ap_orders else ""
        formula = (
            rf"$h(x,p_z,a)=h(x,\infty,0)+\sum_{{i\in\{{{a_text}\}}}} {ca_label}a^i"
            rf"+\sum_{{j\in\{{{p_text}\}}}}\frac{{{cp_label}}}{{p_z^j}}{cap_term}$"
        )
        pdep_text = ", ".join(f"{float(value):.2f}" for value in result.get("pdep_gev", [])) or "not set"
        xdep_text = f"[{str(a_xdep).lower()}, {str(p_xdep).lower()}, {str(include_ap).lower()}]"
        chi_text = f"{float(result.get('chi2_dof', 0.0)):.3g}"
        fit_columns: list[tuple[str, np.ndarray]] = []
        fit_x = np.asarray([], dtype=float)
        fit_indices: list[int] = []
        artifact_path = raw_artifacts.get("extrapolated_artifact")
        if artifact_path:
            with xr.open_dataset(artifact_path) as dataset:
                fit_x = np.asarray(dataset.coords.get("x", []), dtype=float)
                fit_indices = [index for index in (0, len(fit_x) // 2, len(fit_x) - 1) if 0 <= index < len(fit_x)]
                fit_indices = list(dict.fromkeys(fit_indices))
                for name in dataset.data_vars:
                    if not (name.startswith("c_a_") or name.startswith("c_p_") or name.startswith("c_ap_")):
                        continue
                    if name.startswith("c_a_"):
                        label = rf"$c_{{a,{name.removeprefix('c_a_')}}}$"
                    elif name.startswith("c_p_"):
                        label = rf"$c_{{p,{name.removeprefix('c_p_')}}}$"
                    else:
                        label = rf"$c_{{ap,{name.removeprefix('c_ap_')}}}$"
                    fit_columns.append((label, np.asarray(dataset[name].values, dtype=float)))
        header = "| $x$ | " + " | ".join(label for label, _values in fit_columns) + " |" if fit_columns else "| $x$ |"
        divider = "|---" * (len(fit_columns) + 1) + "|"
        fit_rows = []
        for index in fit_indices:
            cells = [f"{float(fit_x[index]):.4g}"]
            for _label, values in fit_columns:
                samples = values[:, index]
                sdev = 0.0 if samples.size < 2 else float(np.std(samples, ddof=1))
                cells.append(str(gvar.gvar(float(np.mean(samples)), sdev)))
            fit_rows.append("| " + " | ".join(cells) + " |")
        lines.extend(
            [
                f"## {record.get('job_id')}",
                "",
                "This report summarizes the light-cone distributions from perturbative matching and extrapolates their lattice-spacing and momentum dependence.",
                "",
                "## Extrapolation Form",
                "",
                formula,
                "",
                "## Job Summary",
                "| job | mode | inputs | parameters | $\\chi^2/\\mathrm{dof}$ | output |",
                "|---|---|---:|---:|---:|---|",
                f"| `{record.get('job_id')}` | {result.get('mode')} | {result.get('n_inputs')} | {result.get('n_parameters')} | {chi_text} | {Path(str(artifacts.get('extrapolated_artifact'))).name if artifacts.get('extrapolated_artifact') else 'n/a'} |",
                "",
                "## Analysis Settings",
                "| Item | Value or setting | Explanation |",
                "|---|---|---|",
                f"| `allow_order_a` | {a_orders} | Allowed lattice-spacing correction powers; the code fits $a^i$ terms. |",
                f"| `allow_order_1overp` | {p_orders} | Allowed finite-momentum correction powers; the code fits $1/p_z^j$ terms. |",
                f"| `allow_order_ap` | {ap_orders} | Allowed lattice-spacing-momentum cross-term powers; when enabled, the code fits $a^k p_z^k$ terms. |",
                f"| `fitting_param_xdep` | {xdep_text} | The first two values control whether $c_{{a,i}}$ and $c_{{p,j}}$ depend on $x$; the third controls whether $c_{{ap,k}}(x)a^k p_z^k$ is included. |",
                f"| `pdep_gev` | {pdep_text} | Requested $p_z$ values in GeV for the extra momentum-dependence figure; if unset, `extrapolate_pdep` is not generated. |",
                "",
                "## Fit Model Parameter Table",
                header,
                divider,
                *fit_rows,
                "",
            ]
        )
        if result.get("warning"):
            lines.append("Warning: " + str(result["warning"]))
            lines.append("")
        if result.get("use_lattice_spacing_dependence") and not result.get("use_momentum_dependence"):
            lines.append("The inputs contain multiple ensembles at a single momentum, so only the lattice-spacing-dependence figure can be generated.")
            lines.append("")
        if result.get("use_momentum_dependence") and not result.get("use_lattice_spacing_dependence"):
            lines.append("The inputs contain one ensemble at multiple momenta, so only the momentum-dependence figure can be generated; it is omitted unless `pdep_gev` is set.")
            lines.append("")
        title = "Extrapolated Result"
        if artifacts.get("extrapolated_plot_image"):
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['extrapolated_plot_image']})", ""])
        if artifacts.get("chi2_xdep_plot_image"):
            title = "Fit Quality"
            lines.extend([f"## {title}", "", f"![chi2_xdep]({artifacts['chi2_xdep_plot_image']})", ""])
        if artifacts.get("adep_plot_image"):
            title = "Lattice-Spacing Dependence"
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['adep_plot_image']})", ""])
        if artifacts.get("pdep_plot_image"):
            title = "Momentum Dependence"
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['pdep_plot_image']})", ""])
    markdown = "\n".join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    if language == "ch":
        translated = translate_markdown_report(
            markdown,
            backend=backend,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        target.write_text(translated, encoding="utf-8")
    return {"report": target}
