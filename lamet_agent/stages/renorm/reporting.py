"""Markdown reporting helpers for the renormalization stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lamet_agent.core.reporting import (
    format_report_list,
    format_report_value,
    markdown_artifact_paths,
    resolve_report_target,
    translate_markdown_report,
)


RENORM_ARTIFACT_DESCRIPTIONS = {
    "renormalized_artifact": "Renormalized matrix element samples (EnsembleData NetCDF)",
    "zR_artifact": "Fitted self-renormalization factor zR (EnsembleData NetCDF)",
    "renormalized_plot": "PDF plot of the renormalized matrix element",
    "renormalized_plot_image": "SVG companion for Markdown embedding",
    "diag_fit_lnM_vs_inv_a": "Self-renorm fit: ln|M| vs 1/a",
    "diag_fit_lnM_vs_inv_a_image": "SVG for ln|M| vs 1/a",
    "diag_fit_mR_zmsbar": "mR vs ZMSbar",
    "diag_fit_mR_zmsbar_image": "SVG for mR vs ZMSbar",
    "diag_fit_m_over_zR": "M_bare/zR by a",
    "diag_fit_m_over_zR_image": "SVG for M_bare/zR",
    "diag_fit_f1": "Discretization coefficient f1(z)",
    "diag_fit_f1_image": "SVG for f1(z)",
    "diag_zmsbar_compare": "H/zR compared with ZMSbar",
    "diag_zmsbar_compare_image": "SVG for ZMSbar compare",
    "diag_discrete_effect_re": "Multi-a discrete-effect overlay (Re)",
    "diag_discrete_effect_re_image": "SVG for discrete-effect Re",
    "diag_discrete_effect_im": "Multi-a discrete-effect overlay (Im)",
    "diag_discrete_effect_im_image": "SVG for discrete-effect Im",
}

RENORM_ARTIFACT_ORDER = (
    "zR_artifact",
    "renormalized_artifact",
    "renormalized_plot",
    "renormalized_plot_image",
    "diag_fit_lnM_vs_inv_a",
    "diag_fit_lnM_vs_inv_a_image",
    "diag_fit_mR_zmsbar",
    "diag_fit_mR_zmsbar_image",
    "diag_fit_m_over_zR",
    "diag_fit_m_over_zR_image",
    "diag_fit_f1",
    "diag_fit_f1_image",
    "diag_zmsbar_compare",
    "diag_zmsbar_compare_image",
    "diag_discrete_effect_re",
    "diag_discrete_effect_re_image",
    "diag_discrete_effect_im",
    "diag_discrete_effect_im_image",
)


def _outputs_table(artifacts: dict[str, Any]) -> list[str]:
    lines = ["| Artifact | Description |", "|---|---|"]
    keys = [key for key in RENORM_ARTIFACT_ORDER if artifacts.get(key)]
    for key in artifacts:
        if key.startswith("diag_") and key not in keys:
            keys.append(key)
    for key in keys:
        value = artifacts.get(key)
        if value:
            lines.append(f"| `{value}` | {RENORM_ARTIFACT_DESCRIPTIONS.get(key, key)} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def _scheme_table(result: dict[str, Any]) -> list[str]:
    scheme = str(result.get("scheme", "ratio"))
    strategy = str(result.get("strategy", "external_denominator"))
    if strategy == "self_renormalization":
        job_kind = str(
            result.get("job_kind")
            or ("fit" if result.get("d") is not None and "lattice_spacing_fm" not in result else "apply")
        )
        rows = [
            ("Scheme", f"`{scheme}`"),
            ("Strategy", f"`{strategy}`"),
            ("Job kind", f"`{job_kind}`"),
            ("kernel_id", f"`{result.get('kernel_id', 'n/a')}`"),
            ("$\\mu$ [GeV]", format_report_value(result.get("mu"))),
            ("$\\Lambda_{\\mathrm{QCD}}$ [GeV]", format_report_value(result.get("LambdaQCD_gev"))),
            ("Derived $\\alpha_s$", format_report_value(result.get("alpha_s_derived"))),
            ("Running helper", f"`{result.get('alpha_s_source', 'n/a')}`"),
            ("$m_0$ [GeV]", format_report_value(result.get("m0", result.get("m0_gev")))),
            ("$d$", format_report_value(result.get("d"))),
            ("svdcut", format_report_value(result.get("svdcut"))),
            ("$z_s$ [fm]", format_report_value(result.get("zs_fm"))),
            ("Selected $z_s$ grid [fm]", format_report_value(result.get("zs_grid_fm"))),
            ("Mean Re $Z_T$", format_report_value(result.get("ZT_re_mean"))),
            ("z coverage policy", format_report_value(result.get("z_coverage_policy"))),
            ("Dropped z points", format_report_value(result.get("n_z_dropped"))),
            ("Extrapolated z points", format_report_value(result.get("n_z_extrapolated"))),
            ("Extrapolation method", format_report_value(result.get("z_extrapolation_method"))),
            ("Input z range [fm]", format_report_list(result.get("z_input_range_fm", []))),
            ("Output z range [fm]", format_report_list(result.get("z_output_range_fm", []))),
            ("$a$ [fm]", format_report_value(result.get("lattice_spacing_fm"))),
            ("z grid", format_report_list(result.get("z_grid", result.get("z_values", [])))),
            ("Resampling", f"{result.get('n_sample', 'n/a')} samples"),
        ]
    elif scheme in {"ratio", "msbar"}:
        rows = [
            ("Scheme", f"`{scheme}`"),
            ("Strategy", f"`{strategy}`"),
            ("z grid", format_report_list(result.get("z_grid", []))),
            ("Resampling", f"{result.get('n_sample', 'n/a')} samples"),
        ]
        if result.get("denominator_constant") is not None:
            rows.insert(2, ("Denominator constant", format_report_value(result.get("denominator_constant"))))
    else:
        rows = [
            ("Scheme", f"`{scheme}`"),
            ("Strategy", f"`{strategy}`"),
            ("$z_s$ [fm]", format_report_value(result.get("zs_fm"))),
            ("$z_s/a$", format_report_value(result.get("zs_lattice"))),
            ("Selected denominator z grid", format_report_value(result.get("zs_grid"))),
            ("$\\delta m$ [GeV]", format_report_value(result.get("delta_m_gev"))),
            ("$m_0$ [GeV]", format_report_value(result.get("m0_gev"))),
            ("z grid", format_report_list(result.get("z_grid", []))),
            ("Resampling", f"{result.get('n_sample', 'n/a')} samples"),
        ]
    lines = ["| Quantity | Value |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _formula_text(*, scheme: str = "ratio", strategy: str = "external_denominator") -> str:
    if strategy == "self_renormalization" and scheme == "hybrid":
        return r"""
The self-renormalization strategy first fits the full-range factor $z_R(z,a)$.
For the hybrid scheme, the apply job also consumes a denominator:

$$
h^R_s(z)=
\begin{cases}
h^{\rm tar}_s(z)/h^{\rm den}_s(z), & |z|\le z_s,\\
h^{\rm tar}_s(z)/(z_R(z,a)Z_{T,s}), & |z|>z_s,
\end{cases}
\qquad
Z_{T,s}=\frac{h^{\rm den}_s(z_s^{\rm grid})}{z_R(z_s^{\rm grid},a)}.
$$

$Z_{T,s}$ is independent of $z$ and is constructed per resample, so the two
branches are continuous at the switch and denominator uncertainty propagates
through the long-distance branch.
""".strip()
    if strategy == "self_renormalization" and scheme == "msbar":
        return r"""
The self-renormalization strategy fits $z_R(z,a)$ from the reference and the
MSbar-scheme apply job acts sample by sample as

$$
h^R_s(z)=\frac{h^{\rm tar}_s(z)}{z_R(z,a)}.
$$

The $z=0$ point is passed through unchanged. Coverage and optional long-distance
extension follow the declared `z_coverage_policy`.
""".strip()
    if strategy == "self_renormalization" and scheme == "ratio":
        return r"""
The self-renormalization strategy fits the zero-momentum reference over the full $z$ range and uses short-distance
$Z_{\overline{\mathrm{MS}}}^{\mathrm{PDF}}$ matching to fix the finite renormalization $m_0$:

$$
g(z)-\ln Z_{\overline{\mathrm{MS}}}^{\mathrm{PDF}}(z;\mu)\simeq m_0 z+b,
\qquad
z_R(z,a)=\exp[\ln M_{\mathrm{fit}}(z,a)-g(z)+m_0z].
$$

It then acts sample by sample on the coordinate grid covered by $z_R$ as

$$
h^R_s(z)=\frac{h^{\rm tar}_s(z)}{z_R(z,a)\,Z_{\overline{\mathrm{MS}}}(z;\mu)}.
$$

$Z_{\overline{\mathrm{MS}}}$ comes from the `inputs.kernels` entry with `stage='renormalization'` (`ZMSbar_pdf` or `ZMSbar_da`). Lattice-unit targets are converted inside the scheme as $z_{\rm fm}=|z/a|a_{\rm fm}$. The $z=0$ samples are excluded from $z_R$ and $Z_{\overline{\mathrm{MS}}}$ evaluation but passed through unchanged into the complete output, preserving $h^R(0)=1$. `LambdaQCD_gev` is the required $\Lambda_{\mathrm{QCD}}$ scale in GeV for the self-renormalization ansatz and is recorded in artifact provenance. The coupling is still derived independently by `alphas_nloop(mu)` and recorded as provenance; a numerical coupling cannot be supplied. The `strict` coverage policy requires the nonzero target to lie within the $z_R$ grid, `intersection` explicitly clips to their overlap, and `extrapolate` automatically extends the long-distance $f_1(z)$ quadratically and rebuilds only the missing $z_R$ points without endpoint freezing. There is no explicit $z_s$ switch; the hybrid character is the combination of full-range self-renormalization and short-distance MSbar finite matching.
""".strip()
    if strategy == "external_denominator" and scheme in {"ratio", "msbar"}:
        constant_note = (
            " The denominator may be a z-dependent reference matrix element or a single finite nonzero constant $C$ applied to every sample and coordinate, $h^R_s(z)=h^{\\rm tar}_s(z)/C$."
        )
        scheme_label = "ratio" if scheme == "ratio" else r"$\overline{\mathrm{MS}}$"
        return rf"""
The {scheme_label} scheme with the external-denominator strategy acts pointwise on every resampled sample $s$ across the complete coordinate grid:

$$
h^R_s(z)=\frac{{h^{{\rm tar}}_s(z)}}{{h^{{\rm den}}_s(z)}}.
$$

Here $h^{{\rm tar}}_s(z)$ is the bare target matrix element and $h^{{\rm den}}_s(z)$ is the reference/denominator matrix element.{constant_note} This scheme has no switching distance, frozen denominator, or long-distance exponential correction. With `normalization=true`, a matrix-element denominator is normalized sample by sample by its own $z=0$ value before the tool runs, while a numeric constant is used as-is; with `normalization=false`, the raw inputs are divided directly.
""".strip()
    return r"""
The hybrid-ratio scheme acts sample by sample on the target and denominator matrix elements. The normalization factor for resampled sample $s$ is

$$
N_s=\frac{h^{\rm den}_s(0)}{h^{\rm tar}_s(0)} .
$$

The renormalized matrix element is

$$
h^R_s(z)=
\begin{cases}
N_s h^{\rm tar}_s(z)/h^{\rm den}_s(z), & |z|_{\rm fm}\le z_s,\\
N_s e^{(\delta m+m_0)(|z|_{\rm fm}-z_s)/(\hbar c)}
h^{\rm tar}_s(z)/h^{\rm den}_s(z_s^{\rm grid}), & |z|_{\rm fm}>z_s .
\end{cases}
$$

Here $h^{\rm tar}_s(z)$ is the bare target matrix element, $h^{\rm den}_s(z)$ is the reference denominator matrix element, $z_s$ is the hybrid-ratio switching distance, and $z_s^{\rm grid}$ is the denominator point on the available coordinate grid nearest to $z_s$. The short-distance region uses a pointwise ratio; the long-distance region freezes the denominator at $z_s^{\rm grid}$ and applies the exponential correction governed by $\delta m+m_0$. When `normalization=true`, $N_s$ is not multiplied again inside the renormalization tool; it is implemented equivalently by the per-sample $z=0$ normalization of target and denominator before the renormalization job starts. When `normalization=false`, this $N_s$ factor is not applied. This stage does not refit matrix elements; it applies one renormalization map to all resampled samples.
""".strip()


def build_renorm_stage_report_markdown(
    *,
    jobs: list[dict[str, Any]],
    systematics_jobs: list[dict[str, Any]] | None = None,
    base_dir: Path,
) -> str:
    """Build one English Markdown report for all renormalization jobs."""
    combinations = {
        (
            str(item.get("result", {}).get("scheme", "ratio")),
            str(item.get("result", {}).get("strategy", "external_denominator")),
        )
        for item in jobs
    }
    primary_scheme, primary_strategy = (
        next(iter(combinations)) if len(combinations) == 1 else ("mixed", "mixed")
    )
    intro = {
        "ratio": (
            "This report summarizes ratio-scheme jobs that convert bare matrix elements into "
            "renormalized coordinate-space matrix elements."
        ),
        "hybrid": (
            "This report summarizes hybrid-scheme renormalization jobs that convert bare matrix elements into "
            "renormalized coordinate-space matrix elements."
        ),
        "msbar": (
            "This report summarizes MSbar-scheme jobs that convert bare matrix elements into "
            "renormalized coordinate-space matrix elements."
        ),
    }.get(
        primary_scheme,
        "This report summarizes renormalization jobs that convert bare matrix elements into "
        "renormalized coordinate-space matrix elements.",
    )
    lines = [
        "# Renormalization Stage Report",
        "",
        intro,
        "",
        "## Job Summary",
        "| job | scheme | strategy | key | output | plot |",
        "|---|---|---|---:|---|---|",
    ]
    markdown_jobs = []
    for item in jobs + list(systematics_jobs or []):
        result = item.get("result", {})
        raw_artifacts = item.get("artifacts", {})
        artifacts = markdown_artifact_paths(
            raw_artifacts,
            base_dir=base_dir,
            path_keys=(
                key
                for key in raw_artifacts
                if key in RENORM_ARTIFACT_ORDER or key.startswith("diag_") or key.startswith("matrix_overlay_")
            ),
        )
        if item in jobs:
            markdown_jobs.append((item, result, artifacts))
        if result.get("strategy") == "self_renormalization":
            key = result.get("kernel_id")
        elif result.get("denominator_constant") is not None:
            key = result.get("denominator_constant")
        elif result.get("scheme") in {"ratio", "msbar"}:
            key = "pointwise"
        else:
            key = result.get("zs_fm")
        output_path = artifacts.get("renormalized_artifact") or artifacts.get("zR_artifact") or "n/a"
        plot_path = artifacts.get("renormalized_plot") or "n/a"
        lines.append(
            f"| `{item['job_id']}` | `{result.get('scheme', 'ratio')}` | "
            f"`{result.get('strategy', 'external_denominator')}` | "
            f"{key if key is not None else format_report_value(result.get('zs_fm'))} | "
            f"{output_path} | "
            f"{plot_path} |"
        )

    stage_artifacts = markdown_jobs[0][2] if markdown_jobs else {}
    overlay_images = [
        value for key, value in sorted(stage_artifacts.items()) if key.startswith("matrix_overlay_") and "_image_" in key
    ]
    method_text = (
        "\n\n".join(
            _formula_text(scheme=scheme, strategy=strategy)
            for scheme, strategy in sorted(combinations)
        )
        if primary_scheme == "mixed"
        else _formula_text(scheme=primary_scheme, strategy=primary_strategy)
    )
    lines.extend(["", "## Method", method_text])
    if overlay_images:
        overlay_groups: dict[str, list[str]] = {}
        for image in overlay_images:
            stem = Path(image).stem
            label = stem[3:] if stem.startswith("rn_") else stem
            label = label[:-3] if label.endswith(("_re", "_im")) else label
            overlay_groups.setdefault(label, []).append(image)
        for label, images in overlay_groups.items():
            title = f"{label} ensemble overview"
            lines.extend(["", f"## {title}", ""])
            images.sort(key=lambda image: 0 if Path(image).stem.endswith("_re") else 1)
            lines.extend(f"![{title}]({image})" for image in images)
    for item, result, artifacts in markdown_jobs:
        is_fit_job = result.get("job_kind") == "fit" or (
            result.get("strategy") == "self_renormalization"
            and artifacts.get("zR_artifact")
            and not artifacts.get("renormalized_artifact")
        )
        lines.extend(["", f"## `{item['job_id']}`", "", "### Scheme Parameters", *_scheme_table(result)])
        if not is_fit_job:
            lines.extend(
                [
                    "",
                    "### Renormalized Matrix Element",
                    (
                        f"![Renormalized matrix element]({artifacts.get('renormalized_plot_image')})"
                        if artifacts.get("renormalized_plot_image")
                        else "Not available."
                    ),
                    f"[PDF artifact]({artifacts.get('renormalized_plot')})" if artifacts.get("renormalized_plot") else "",
                ]
            )
        lines.extend(["", "### Diagnostic Plots"])
        diag_images = [
            key for key in artifacts
            if key.startswith("diag_") and key.endswith("_image") and artifacts.get(key)
        ]
        if not diag_images:
            lines.append("Not available.")
        else:
            for key in diag_images:
                label = key.removeprefix("diag_").removesuffix("_image")
                lines.append(f"![{label}]({artifacts[key]})")
                pdf_key = key.removesuffix("_image")
                if artifacts.get(pdf_key):
                    lines.append(f"[PDF]({artifacts[pdf_key]})")
        lines.extend(["", "### Output Artifacts", *_outputs_table(artifacts)])
        for systematics in item.get("systematics", []):
            systematics_artifacts = markdown_artifact_paths(
                systematics.get("artifacts", {}),
                base_dir=base_dir,
                path_keys=tuple(systematics.get("artifacts", {})),
            )
            lines.append(f"| `{systematics['job_id']}` | Systematics variation artifacts |")
            for value in systematics_artifacts.values():
                if value:
                    lines.append(f"| `{value}` | `{systematics['job_id']}` output |")
    return "\n".join(lines) + "\n"


def write_renorm_stage_report(
    *,
    jobs: list[dict[str, Any]],
    systematics_jobs: list[dict[str, Any]] | None = None,
    path: str | Path,
    report_language: str = "en",
    backend: str = "",
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Path]:
    """Write one report summarizing all renormalization jobs."""
    output = Path(path)
    target, language = resolve_report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_renorm_stage_report_markdown(jobs=jobs, systematics_jobs=systematics_jobs, base_dir=output.parent)
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
