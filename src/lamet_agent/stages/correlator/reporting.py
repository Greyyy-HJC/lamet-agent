"""Markdown reporting helpers for the correlator-analysis stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lamet_agent.core.reporting import (
    format_report_list,
    format_report_value,
    markdown_artifact_paths,
    resolve_report_target,
    translate_markdown_report,
)


CORRELATOR_ARTIFACT_DESCRIPTIONS = {
    "bare_artifact": "Bare matrix element samples (EnsembleData NetCDF)",
    "summary_plot": "PDF plot of the bare matrix element versus Wilson-line length",
    "summary_plot_image": "SVG companion for Markdown embedding",
    "tuning_log": "Window tuning and sample-average fit-quality log",
    "sample_log": "Per-sample and per-z fit-quality log",
    "E0_artifact": "Stage-level dispersion-relation table (NetCDF)",
    "dispersion_relation_plot": "Stage-level dispersion-relation PDF",
    "dispersion_relation_image": "Stage-level dispersion-relation SVG",
}

CORRELATOR_ARTIFACT_ORDER = (
    "bare_artifact",
    "summary_plot",
    "summary_plot_image",
    "tuning_log",
    "sample_log",
    "E0_artifact",
    "dispersion_relation_plot",
    "dispersion_relation_image",
)


def _job_settings_table(result: dict[str, Any]) -> list[str]:
    z_grid = result.get("bz", result.get("z_values", []))
    rows = [
        ("Fitting form", f"`{result.get('fitting_form', 'not recorded')}`"),
        ("Fit scope", f"`{result.get('fit_scope', 'not recorded')}`"),
        ("Fit strategy", f"`{result.get('fit_strategy', 'not recorded')}`"),
        ("Fit mode", f"`{result.get('fit_mode', 'not recorded')}`"),
        ("Model average", f"`{result.get('model_average', 'not recorded')}`"),
        ("Selection rule", f"`{result.get('selection_rule', 'not recorded')}`"),
        ("Resampling", f"`{result.get('resample_mode', 'not recorded')}` with {result.get('n_samples', 'n/a')} samples"),
        ("z grid", format_report_list(z_grid)),
        ("Tuning z values", format_report_list(result.get("tune_z_values", [result.get("tune_z")] if result.get("tune_z") is not None else []))),
        ("correlator_rescale", format_report_value(result.get("correlator_rescale"))),
    ]
    lines = ["| Quantity | Value |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _window_text(result: dict[str, Any]) -> list[str]:
    specs = result.get("shared_window_specs")
    if not specs:
        return ["No shared window metadata was recorded."]
    if not isinstance(specs, list):
        specs = [specs]
    lines = [
        "| scope | strategy | nstate | pt2 window | pt3 window | n_data | n_params |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        pt2_window = spec.get("pt2_window", f"[{spec.get('tmin', 'n/a')},{spec.get('tmax', 'n/a')})")
        pt3_window = spec.get(
            "pt3_window",
            f"tsep={format_report_list(spec.get('tsep_ls', []))}, tau_cut={spec.get('tau_cut', 'n/a')}",
        )
        lines.append(
            f"| `{spec.get('fit_scope', spec.get('scope', 'n/a'))}` | "
            f"`{spec.get('fit_strategy', spec.get('strategy', 'n/a'))}` | "
            f"{spec.get('nstate', 'n/a')} | "
            f"{pt2_window} | "
            f"{pt3_window} | "
            f"{spec.get('n_data', 'n/a')} | "
            f"{spec.get('n_params', 'n/a')} |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _auto_window_scan_text(result: dict[str, Any]) -> list[str]:
    scan = result.get("auto_window_scan")
    if not isinstance(scan, dict):
        return ["No automatic-window diagnostics were recorded."]
    lines = [
        "| channel | source | candidates | stable tmax | fallback |",
        "|---|---|---:|---:|---|",
    ]
    candidate_lines: list[str] = []
    for scan_key, channel, window_key in (
        ("pt2", "2pt", "pt2_windows"),
        ("pt3", "3pt", "pt3_windows"),
    ):
        details = scan.get(scan_key)
        if not isinstance(details, dict):
            continue
        windows = details.get(window_key, [])
        fallback = details.get("fallback_reason") or "none"
        lines.append(
            f"| `{channel}` | `{details.get('source', 'not recorded')}` | "
            f"{len(windows) if isinstance(windows, list) else 'n/a'} | "
            f"{format_report_value(details.get('stable_tmax'))} | {fallback} |"
        )
        if isinstance(windows, list) and windows:
            payload = json.dumps(windows, separators=(",", ":"))
            candidate_lines.append(f"- Generated candidates `{channel}`: `{payload}`")
    if candidate_lines:
        lines.extend(["", *candidate_lines])
    return lines


def _z_fit_table(result: dict[str, Any]) -> list[str]:
    z_fits = result.get("z_fits") or []
    lines = [
        "| z | Q | chi2/dof | logGBF | failed samples | Re sys | Im sys |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fit in z_fits[:20]:
        if not isinstance(fit, dict):
            continue
        window = fit.get("window") if isinstance(fit.get("window"), dict) else {}
        lines.append(
            f"| {format_report_value(fit.get('z'))} | {format_report_value(fit.get('Q', window.get('Q')))} | "
            f"{format_report_value(fit.get('chi2_dof', fit.get('chi2/DOF', window.get('chi2_dof'))))} | "
            f"{format_report_value(fit.get('logGBF', window.get('logGBF')))} | {fit.get('n_failed_samples', 0)} | "
            f"{format_report_value(fit.get('real_sys_sdev'))} | {format_report_value(fit.get('imag_sys_sdev'))} |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _outputs_table(artifacts: dict[str, Any]) -> list[str]:
    lines = ["| Artifact | Description |", "|---|---|"]
    for key in CORRELATOR_ARTIFACT_ORDER:
        if key in {"E0_artifact", "dispersion_relation_plot", "dispersion_relation_image"}:
            continue
        value = artifacts.get(key)
        if value:
            lines.append(f"| `{value}` | {CORRELATOR_ARTIFACT_DESCRIPTIONS[key]} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def _fit_form_text(*, has_qda_ratio: bool = False, has_3pt_ratio: bool = True) -> str:
    if has_qda_ratio:
        text = r"""
`fit_scope="qda_ratio"` builds the ratio of a nonlocal qDA two-point correlator to the selected two-point denominator from shared resamples,

$$
R_{\rm qDA}^{(a)}(z,P,t)=\frac{C_{\rm qDA}^{(a)}(z,P,t)}{C_2(P,t)}.
$$

The numerator and denominator use a shared multi-state spectrum,

$$
C_{\rm qDA}^{(a)}(t)=\sum_n\frac{z_n O_{0n}^{(a)}}{2E_n}
\left[e^{-E_nt}+e^{-E_n(L_t-t)}\right],\qquad
C_2(t)=\sum_n\frac{z_n z'_n}{2E_n}\left[e^{-E_nt}+e^{-E_n(L_t-t)}\right].
$$

With an ordinary local-local 2pt input, $z'_n=z_n$ and the output is $O_{00}^{(a)}/z_0$. When that input is omitted, the same qDA operator at $b_z=0$ supplies the denominator, $z'_n$ is the nonlocal sink overlap, and the output is $O_{00}^{(a)}/z'_0$; the $z=0$ ratio is then automatically normalized by identical numerator and denominator data. This scope has no 3pt correlator, $t_{\rm sep}$, $\tau$, or current insertion.
""".strip()
        return text + ("\n\n" + _fit_form_text() if has_3pt_ratio else "")
    return r"""
lamet-agent builds 2pt/3pt data from the same resampled ensemble and extracts bare matrix elements in the selected time windows. The 2pt input is $C_2(t)$ for the chosen momentum and interpolator; the 3pt input is $C_3(t_{\rm sep},\tau,z)$ for each source-sink separation, insertion time, and Wilson-line length. For each job, tuning first fixes the window, state count, `fit_scope`, and `fit_strategy` on sample-average data; those choices are then held fixed for all $z$ and all resampled samples.

The 2pt spectral form is

$$
C_2^\alpha(t)=\sum_{n=0}^{N_{\rm st}-1}
\frac{z_{n,\alpha}^2}{2E_{n,\alpha}}
\left(e^{-E_{n,\alpha}t}+e^{-E_{n,\alpha}(L_t-t)}\right),
\qquad
E_{n,\alpha}=E_{0,\alpha}+\sum_{k=1}^{n}e^{\log\Delta E_{k,\alpha}} .
$$

Here $\alpha$ labels the initial or final momentum channel. Breit/forward fits use one set of $\{E_n,z_n\}$, while NonBreit fits use separate initial and final sets. The 2pt parameters are $E_0$, the gaps $\log\Delta E_k$, and overlaps $z_n$.

For Breit kinematics the ratio model is

$$
R_{\rm B}(t,\tau,z)=\frac{C_3(t,\tau,z)}{C_2(t)}
=\frac{1}{C_2(t)}
\sum_{m,n}\frac{O^\Gamma_{mn}(z)z_mz_n}{(2E_m)(2E_n)}
e^{-E_m(t-\tau)}e^{-E_n\tau},
\qquad h_{\rm B}(z)=\frac{O_{00}(z)}{2E_0}.
$$

The inputs are the same-momentum 2pt and 3pt ratio. In addition to the 2pt spectral parameters, the fit determines the matrix elements $O_{mn}(z)$ for each Wilson-line length. The reported Breit bare matrix element is the ground-state normalized combination $h_{\rm B}(z)$.

For NonBreit kinematics the symmetrized non-forward ratio is

$$
R_{\rm NB}(t,\tau,z)=
\frac{C_3^{f\leftarrow i}(t,\tau,z)}{C_2^f(t)}
\left[
\frac{C_2^i(t-\tau)C_2^f(\tau)C_2^f(t)}
{C_2^f(t-\tau)C_2^i(\tau)C_2^i(t)}
\right]^{1/2},
\qquad
h_{\rm NB}(z)={\rm sign}(z_{0,i}z_{0,f})\frac{O_{00}(z)}{E_{0,i}+E_{0,f}} .
$$

The inputs are initial 2pt, final 2pt, and non-forward 3pt data. The fit parameters include both 2pt spectra and the transition matrix elements $O_{mn}(z)$. The reported NonBreit summary uses $O_{00}/(E_{0,i}+E_{0,f})$ with the ground-state overlap sign convention.

When `fit_scope` is `FH` or `3pt_ratio+FH`, a summed-ratio/Feynman-Hellmann constraint is also formed:

$$
S(t)=\sum_{\tau=\tau_c}^{t-\tau_c}R(t,\tau),
\qquad
R_{\rm FH}(t)=\frac{S(t+\Delta t)-S(t)}{\Delta t}.
$$

`fit_strategy="joint"` fits 2pt and 3pt/FH constraints in one nonlinear fit with shared floating parameters. `fit_strategy="chained"` fits the 2pt data first and uses the resulting energies and overlaps as anchored priors for the following 3pt/FH fit. `fit_strategy="independent"` fits the ratio/FH/`qda_ratio` alone with no 2pt channel and no prior 2pt fit. `fit_scope="3pt_ratio"` uses only 3pt-ratio data, `fit_scope="FH"` uses only summed-ratio/FH data, and `fit_scope="3pt_ratio+FH"` uses both.
""".strip()


def _diagnostic_plots(artifacts: dict[str, Any]) -> list[str]:
    plots = list(artifacts.get("sample0_fit_plots", [])) + list(artifacts.get("sample0_pt2_plots", []))
    plots = [plot for plot in plots if plot and str(plot).endswith(".svg")]
    if not plots:
        return ["No sample-0 diagnostic SVGs were recorded."]
    lines = ["Sample-0 diagnostic SVGs:"]
    for start in range(0, len(plots), 4):
        row = plots[start : start + 4]
        lines.append("<table><tr>")
        for plot in row:
            lines.append(f'<td><img src="{plot}" alt="{Path(str(plot)).stem}" width="230"><br><code>{Path(str(plot)).stem}</code></td>')
        lines.append("</tr></table>")
    return lines


def build_correlator_stage_report_markdown(
    *,
    jobs: list[dict[str, Any]],
    base_dir: Path,
) -> str:
    """Build one English Markdown report for all correlator-analysis jobs."""
    all_qda_ratio = bool(jobs) and all((item.get("result", {}).get("fit_scope") == "qda_ratio") for item in jobs)
    has_qda_ratio = any((item.get("result", {}).get("fit_scope") == "qda_ratio") for item in jobs)
    has_3pt_ratio = any((item.get("result", {}).get("fit_scope") != "qda_ratio") for item in jobs)
    intro = (
        "This report summarizes correlator fits that extract bare matrix elements from 2pt correlators."
        if all_qda_ratio
        else "This report summarizes correlator fits that extract bare matrix elements from 2pt/3pt data."
    )
    lines = [
        "# Correlator Analysis Stage Report",
        "",
        intro,
        "",
        "## Fitting Form",
        _fit_form_text(has_qda_ratio=has_qda_ratio, has_3pt_ratio=has_3pt_ratio),
        "",
        "## Job Summary",
        "| job | fit scope | strategy | output | plot |",
        "|---|---|---|---|---|",
    ]
    markdown_jobs = []
    for item in jobs:
        result = item.get("result", {})
        artifacts = markdown_artifact_paths(
            item.get("artifacts", {}),
            base_dir=base_dir,
            path_keys=(
                *CORRELATOR_ARTIFACT_ORDER,
                *(key for key in item.get("artifacts", {}) if key.startswith("matrix_overlay_")),
            ),
            list_path_keys=("sample0_pt2_plots", "sample0_fit_plots"),
        )
        markdown_jobs.append((item, result, artifacts))
        lines.append(
            f"| `{item['job_id']}` | `{result.get('fit_scope', 'n/a')}` | "
            f"`{result.get('fit_strategy', 'n/a')}` | "
            f"{artifacts.get('bare_artifact', 'n/a')} | {artifacts.get('summary_plot', 'n/a')} |"
        )

    stage_artifacts = markdown_jobs[0][2] if markdown_jobs else {}
    if stage_artifacts.get("dispersion_relation_image"):
        lines.extend(
            [
                "",
                "## Dispersion Relation",
                "",
                "The dispersion-relation plot is designed to check the dependence of $E_0^2$ on $p^2$ and shows the ground-state energy posterior obtained from 2pt correlator fits at different ensembles and momenta. "
                r"For each ensemble, the fit form is $E_0^2=m^2+k_2P^2+k_3P^4a^2$.",
                "",
                f"![Dispersion relation]({stage_artifacts['dispersion_relation_image']})",
            ]
        )
    overlay_images = [
        value for key, value in sorted(stage_artifacts.items()) if key.startswith("matrix_overlay_") and "_image_" in key
    ]
    if overlay_images:
        overlay_groups: dict[str, list[str]] = {}
        for image in overlay_images:
            stem = Path(image).stem
            label = stem[3:] if stem.startswith("ca_") else stem
            label = label[:-3] if label.endswith(("_re", "_im")) else label
            overlay_groups.setdefault(label, []).append(image)
        for label, images in overlay_groups.items():
            title = f"{label} ensemble overview"
            lines.extend(["", f"## {title}", ""])
            images.sort(key=lambda image: 0 if Path(image).stem.endswith("_re") else 1)
            lines.extend(f"![{title}]({image})" for image in images)

    for item, result, artifacts in markdown_jobs:
        lines.extend(
            [
                "",
                f"## `{item['job_id']}`",
                "",
                "### Fit Setup",
                *_job_settings_table(result),
                "",
                "### Shared Windows",
                *_window_text(result),
                "",
                "### Automatic Window Scan",
                *_auto_window_scan_text(result),
                "",
                "### Per-z Fit Summary",
                *_z_fit_table(result),
                "",
                "### fit_logs",
                "The `fit_logs` directory contains split logs: the tuning log records window selection and sample-average fit quality, while the sample log records per-sample and per-z fit quality, including failures.",
                "",
                *_outputs_table(artifacts),
                "",
                "### Diagnostic SVGs",
                *_diagnostic_plots(artifacts),
                "",
                "### Summary Figure",
                (
                    f"![Bare matrix element summary]({artifacts.get('summary_plot_image')})"
                    if artifacts.get("summary_plot_image")
                    else "Not available."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def write_correlator_stage_report(
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
    """Write one report summarizing all correlator-analysis jobs."""
    output = Path(path)
    target, language = resolve_report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_correlator_stage_report_markdown(jobs=jobs, base_dir=output.parent)
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
