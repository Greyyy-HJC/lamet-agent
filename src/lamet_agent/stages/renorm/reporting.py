"""Markdown reporting helpers for the renormalization stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lamet_agent.core.reporting import (
    format_report_list,
    format_report_value,
    markdown_artifact_paths,
    resolve_report_target,
)


RENORM_ARTIFACT_DESCRIPTIONS = {
    "renormalized_artifact": ("Renormalized matrix element samples (EnsembleData NetCDF)", "重整化矩阵元样本（EnsembleData NetCDF）"),
    "zR_artifact": ("Fitted self-renormalization factor zR (EnsembleData NetCDF)", "拟合得到的 self-renorm 因子 zR（EnsembleData NetCDF）"),
    "renormalized_plot": ("PDF plot of the renormalized matrix element", "重整化矩阵元 PDF 图"),
    "renormalized_plot_image": ("SVG companion for Markdown embedding", "供 Markdown 嵌入的重整化矩阵元 SVG 图"),
    "diag_fit_lnM_vs_inv_a": ("Self-renorm fit: ln|M| vs 1/a", "Self-renorm 拟合：ln|M| vs 1/a"),
    "diag_fit_lnM_vs_inv_a_image": ("SVG for ln|M| vs 1/a", "ln|M| vs 1/a 的 SVG"),
    "diag_fit_mR_zmsbar": ("mR vs ZMSbar", "mR 与 ZMSbar 对比"),
    "diag_fit_mR_zmsbar_image": ("SVG for mR vs ZMSbar", "mR 与 ZMSbar 对比的 SVG"),
    "diag_fit_m_over_zR": ("M_bare/zR by a", "各 a 的 M_bare/zR"),
    "diag_fit_m_over_zR_image": ("SVG for M_bare/zR", "M_bare/zR 的 SVG"),
    "diag_fit_f1": ("Discretization coefficient f1(z)", "离散化系数 f1(z)"),
    "diag_fit_f1_image": ("SVG for f1(z)", "f1(z) 的 SVG"),
    "diag_zmsbar_compare": ("H/zR compared with ZMSbar", "H/zR 与 ZMSbar 对比"),
    "diag_zmsbar_compare_image": ("SVG for ZMSbar compare", "ZMSbar 对比的 SVG"),
    "diag_discrete_effect_re": ("Multi-a discrete-effect overlay (Re)", "多 a 离散效应叠图（实部）"),
    "diag_discrete_effect_re_image": ("SVG for discrete-effect Re", "离散效应实部 SVG"),
    "diag_discrete_effect_im": ("Multi-a discrete-effect overlay (Im)", "多 a 离散效应叠图（虚部）"),
    "diag_discrete_effect_im_image": ("SVG for discrete-effect Im", "离散效应虚部 SVG"),
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


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    header = "| Artifact | Description |" if language == "en" else "| 文件 | 说明 |"
    lines = [header, "|---|---|"]
    keys = [key for key in RENORM_ARTIFACT_ORDER if artifacts.get(key)]
    for key in artifacts:
        if key.startswith("diag_") and key not in keys:
            keys.append(key)
    for key in keys:
        value = artifacts.get(key)
        if not value:
            continue
        desc_pair = RENORM_ARTIFACT_DESCRIPTIONS.get(key)
        if desc_pair is None:
            desc = key
        else:
            desc = desc_pair[0 if language == "en" else 1]
        lines.append(f"| `{value}` | {desc} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def _scheme_table(result: dict[str, Any], *, language: str) -> list[str]:
    scheme = str(result.get("scheme", "hybrid_ratio"))
    if scheme == "hybrid_self_renormalization":
        job_kind = str(
            result.get("job_kind")
            or ("fit" if result.get("d") is not None and "lattice_spacing_fm" not in result else "apply")
        )
        if language == "zh":
            rows = [
                ("方案", f"`{scheme}`"),
                ("job 类型", f"`{job_kind}`"),
                ("kernel_id", f"`{result.get('kernel_id', 'n/a')}`"),
                ("$\\mu$ [GeV]", format_report_value(result.get("mu"))),
                ("$\\Lambda_{\\mathrm{QCD}}$ [GeV]", format_report_value(result.get("LambdaQCD_gev"))),
                ("派生 $\\alpha_s$", format_report_value(result.get("alpha_s_derived"))),
                ("running helper", f"`{result.get('alpha_s_source', 'n/a')}`"),
                ("$m_0$ [GeV]", format_report_value(result.get("m0", result.get("m0_gev")))),
                ("$d$", format_report_value(result.get("d"))),
                ("svdcut", format_report_value(result.get("svdcut"))),
                ("z 覆盖策略", format_report_value(result.get("z_coverage_policy"))),
                ("丢弃 z 点数", format_report_value(result.get("n_z_dropped"))),
                ("延拓 z 点数", format_report_value(result.get("n_z_extrapolated"))),
                ("延拓方法", format_report_value(result.get("z_extrapolation_method"))),
                ("输入 z 范围 [fm]", format_report_list(result.get("z_input_range_fm", []))),
                ("输出 z 范围 [fm]", format_report_list(result.get("z_output_range_fm", []))),
                ("$a$ [fm]", format_report_value(result.get("lattice_spacing_fm"))),
                ("z 网格", format_report_list(result.get("z_grid", result.get("z_values", [])))),
                ("重采样", f"{result.get('n_sample', 'n/a')} 个样本"),
            ]
            header = "| 条目 | 数值或设置 |"
        else:
            rows = [
                ("Scheme", f"`{scheme}`"),
                ("Job kind", f"`{job_kind}`"),
                ("kernel_id", f"`{result.get('kernel_id', 'n/a')}`"),
                ("$\\mu$ [GeV]", format_report_value(result.get("mu"))),
                ("$\\Lambda_{\\mathrm{QCD}}$ [GeV]", format_report_value(result.get("LambdaQCD_gev"))),
                ("Derived $\\alpha_s$", format_report_value(result.get("alpha_s_derived"))),
                ("Running helper", f"`{result.get('alpha_s_source', 'n/a')}`"),
                ("$m_0$ [GeV]", format_report_value(result.get("m0", result.get("m0_gev")))),
                ("$d$", format_report_value(result.get("d"))),
                ("svdcut", format_report_value(result.get("svdcut"))),
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
            header = "| Quantity | Value |"
    elif scheme == "ratio" and language == "zh":
        rows = [
            ("方案", f"`{scheme}`"),
            ("z 网格", format_report_list(result.get("z_grid", []))),
            ("重采样", f"{result.get('n_sample', 'n/a')} 个样本"),
        ]
        header = "| 条目 | 数值或设置 |"
    elif scheme == "ratio":
        rows = [
            ("Scheme", f"`{scheme}`"),
            ("z grid", format_report_list(result.get("z_grid", []))),
            ("Resampling", f"{result.get('n_sample', 'n/a')} samples"),
        ]
        header = "| Quantity | Value |"
    elif language == "zh":
        rows = [
            ("方案", f"`{scheme}`"),
            ("$z_s$ [fm]", format_report_value(result.get("zs_fm"))),
            ("$z_s/a$", format_report_value(result.get("zs_lattice"))),
            ("选中的 denominator z grid", format_report_value(result.get("zs_grid"))),
            ("$\\delta m$ [GeV]", format_report_value(result.get("delta_m_gev"))),
            ("$m_0$ [GeV]", format_report_value(result.get("m0_gev"))),
            ("z 网格", format_report_list(result.get("z_grid", []))),
            ("重采样", f"{result.get('n_sample', 'n/a')} 个样本"),
        ]
        header = "| 条目 | 数值或设置 |"
    else:
        rows = [
            ("Scheme", f"`{scheme}`"),
            ("$z_s$ [fm]", format_report_value(result.get("zs_fm"))),
            ("$z_s/a$", format_report_value(result.get("zs_lattice"))),
            ("Selected denominator z grid", format_report_value(result.get("zs_grid"))),
            ("$\\delta m$ [GeV]", format_report_value(result.get("delta_m_gev"))),
            ("$m_0$ [GeV]", format_report_value(result.get("m0_gev"))),
            ("z grid", format_report_list(result.get("z_grid", []))),
            ("Resampling", f"{result.get('n_sample', 'n/a')} samples"),
        ]
        header = "| Quantity | Value |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _formula_text(*, language: str, scheme: str = "hybrid_ratio") -> str:
    if scheme == "hybrid_self_renormalization":
        if language == "zh":
            return r"""
Hybrid self-renormalization 在完整 $z$ 范围拟合零动量 reference，并以短程
$Z_{\overline{\mathrm{MS}}}^{\mathrm{PDF}}$ matching 固定有限重整化 $m_0$：

$$
g(z)-\ln Z_{\overline{\mathrm{MS}}}^{\mathrm{PDF}}(z;\mu)\simeq m_0 z+b,
\qquad
z_R(z,a)=\exp[\ln M_{\mathrm{fit}}(z,a)-g(z)+m_0z].
$$

然后在 $z_R$ 覆盖的坐标网格上对每个重采样样本作用

$$
h^R_s(z)=\frac{h^{\rm tar}_s(z)}{z_R(z,a)\,Z_{\overline{\mathrm{MS}}}(z;\mu)}.
$$

$Z_{\overline{\mathrm{MS}}}$ 由 `inputs.kernels` 中 `stage='renormalization'` 的 kernel（`ZMSbar_pdf` 或 `ZMSbar_da`）给出。lattice 单位的 target 在 scheme 内转换为 $z_{\rm fm}=|z/a|a_{\rm fm}$。$z=0$ 不参与 $z_R$ 与 $Z_{\overline{\mathrm{MS}}}$ 的计算，但其已归一样本会原样合并回输出，因此重整化矩阵元保留 $h^R(0)=1$。`scheme_parameters.LambdaQCD_gev` 是 self-renormalization ansatz 中必填的 $\Lambda_{\mathrm{QCD}}$（GeV），并记录在产物 provenance 中。$\alpha_s$ 仍由 `alphas_nloop(mu)` 独立派生并记录，不接受手动数值。`strict` 覆盖策略要求非零 target 完全位于 $z_R$ 网格内；`intersection` 显式裁剪到两者交集；`extrapolate` 在 target 超出拟合范围时自动对长程 $f_1(z)$ 作二次延拓并重建缺少的 $z_R$，不冻结端点。该 scheme 没有显式 $z_s$ 切换点；hybrid 性来自全程 self-renormalization 与短程 MSbar 有限 matching 的结合。
""".strip()
        return r"""
Hybrid self-renormalization fits the zero-momentum reference over the full $z$ range and uses short-distance
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

$Z_{\overline{\mathrm{MS}}}$ comes from the `inputs.kernels` entry with `stage='renormalization'` (`ZMSbar_pdf` or `ZMSbar_da`). Lattice-unit targets are converted inside the scheme as $z_{\rm fm}=|z/a|a_{\rm fm}$. The $z=0$ samples are excluded from $z_R$ and $Z_{\overline{\mathrm{MS}}}$ evaluation but passed through unchanged into the complete output, preserving $h^R(0)=1$. `scheme_parameters.LambdaQCD_gev` is the required $\Lambda_{\mathrm{QCD}}$ scale in GeV for the self-renormalization ansatz and is recorded in artifact provenance. The coupling is still derived independently by `alphas_nloop(mu)` and recorded as provenance; a numerical coupling cannot be supplied. The `strict` coverage policy requires the nonzero target to lie within the $z_R$ grid, `intersection` explicitly clips to their overlap, and `extrapolate` automatically extends the long-distance $f_1(z)$ quadratically and rebuilds only the missing $z_R$ points without endpoint freezing. There is no explicit $z_s$ switch; the hybrid character is the combination of full-range self-renormalization and short-distance MSbar finite matching.
""".strip()
    if scheme == "ratio":
        if language == "zh":
            return r"""
Ratio 方案对每个重采样样本 $s$ 在完整坐标网格上逐点计算

$$
h^R_s(z)=\frac{h^{\rm tar}_s(z)}{h^{\rm den}_s(z)}.
$$

这里 $h^{\rm tar}_s(z)$ 是待重整化的裸矩阵元，$h^{\rm den}_s(z)$ 是 reference/denominator 裸矩阵元。该方案不使用切换距离、固定 denominator 或长距离指数修正。当 `normalization=true` 时，target 和 denominator 在进入工具前分别按各自的 $z=0$ 值逐样本归一化；当 `normalization=false` 时直接使用原始输入。
""".strip()
        return r"""
The ratio scheme acts pointwise on every resampled sample $s$ across the complete coordinate grid:

$$
h^R_s(z)=\frac{h^{\rm tar}_s(z)}{h^{\rm den}_s(z)}.
$$

Here $h^{\rm tar}_s(z)$ is the bare target matrix element and $h^{\rm den}_s(z)$ is the reference/denominator matrix element. This scheme has no switching distance, frozen denominator, or long-distance exponential correction. With `normalization=true`, target and denominator are normalized sample by sample by their own $z=0$ values before the tool runs; with `normalization=false`, the raw inputs are divided directly.
""".strip()
    if language == "zh":
        return r"""
Hybrid-ratio 方案对每个重采样样本 $s$ 单独作用在目标矩阵元和 denominator 矩阵元上。逐样本归一化因子为

$$
N_s=\frac{h^{\rm den}_s(0)}{h^{\rm tar}_s(0)} .
$$

重整化矩阵元定义为

$$
h^R_s(z)=
\begin{cases}
N_s h^{\rm tar}_s(z)/h^{\rm den}_s(z), & |z|_{\rm fm}\le z_s,\\
N_s e^{(\delta m+m_0)(|z|_{\rm fm}-z_s)/(\hbar c)}
h^{\rm tar}_s(z)/h^{\rm den}_s(z_s^{\rm grid}), & |z|_{\rm fm}>z_s .
\end{cases}
$$

这里 $h^{\rm tar}_s(z)$ 是待重整化的裸矩阵元，$h^{\rm den}_s(z)$ 是 reference/denominator 裸矩阵元，$z_s$ 是 hybrid-ratio 的切换距离，$z_s^{\rm grid}$ 是实际数据网格上最接近 $z_s$ 的 denominator 点。短距离区域使用逐点 ratio；长距离区域固定 denominator 到 $z_s^{\rm grid}$，并用 $\delta m+m_0$ 的指数因子延拓 Wilson 线线性发散修正。当 `normalization=true` 时，$N_s$ 不在 renormalization tool 内再次显式相乘，而是在进入 renormalization job 前通过 target/denominator 的 $z=0$ 逐样本归一化等价实现；当 `normalization=false` 时不施加这个 $N_s$ 因子。该步骤不重新拟合矩阵元，而是对所有样本施加同一个重整化 map。
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
    base_dir: Path,
    language: str = "en",
) -> str:
    """Build one Markdown report for all renormalization jobs."""
    schemes = {str(item.get("result", {}).get("scheme", "hybrid_ratio")) for item in jobs}
    primary_scheme = next(iter(schemes)) if len(schemes) == 1 else "mixed"
    title = "# Renormalization Stage Report" if language == "en" else "# Renormalization 阶段报告"
    if language == "en":
        intro = {
            "hybrid_self_renormalization": (
                "This report summarizes hybrid-self-renormalization jobs that convert bare matrix elements into "
                "renormalized coordinate-space matrix elements."
            ),
            "ratio": (
                "This report summarizes ratio-scheme jobs that convert bare matrix elements into "
                "renormalized coordinate-space matrix elements."
            ),
            "hybrid_ratio": (
                "This report summarizes hybrid-ratio renormalization jobs that convert bare matrix elements into "
                "renormalized coordinate-space matrix elements."
            ),
        }.get(
            primary_scheme,
            "This report summarizes renormalization jobs that convert bare matrix elements into "
            "renormalized coordinate-space matrix elements.",
        )
        summary_header = "| job | scheme | key | output | plot |"
    else:
        intro = {
            "hybrid_self_renormalization": "本报告汇总 hybrid-self-renormalization job，将裸矩阵元转换为坐标空间重整化矩阵元。",
            "ratio": "本报告汇总 ratio scheme job，将裸矩阵元转换为坐标空间重整化矩阵元。",
            "hybrid_ratio": "本报告汇总 hybrid-ratio 重整化 job，将裸矩阵元转换为坐标空间重整化矩阵元。",
        }.get(primary_scheme, "本报告汇总重整化 job，将裸矩阵元转换为坐标空间重整化矩阵元。")
        summary_header = "| job | 方案 | 关键参数 | 输出 | 图像 |"
    lines = [
        title,
        "",
        intro,
        "",
        "## Job Summary" if language == "en" else "## Job 汇总",
        summary_header,
        "|---|---|---:|---|---|",
    ]
    markdown_jobs = []
    for item in jobs:
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
        markdown_jobs.append((item, result, artifacts))
        if result.get("scheme") == "hybrid_self_renormalization":
            key = result.get("kernel_id")
        elif result.get("scheme") == "ratio":
            key = "pointwise"
        else:
            key = result.get("zs_fm")
        output_path = artifacts.get("renormalized_artifact") or artifacts.get("zR_artifact") or "n/a"
        plot_path = artifacts.get("renormalized_plot") or "n/a"
        lines.append(
            f"| `{item['job_id']}` | `{result.get('scheme', 'hybrid_ratio')}` | "
            f"{key if key is not None else format_report_value(result.get('zs_fm'))} | "
            f"{output_path} | "
            f"{plot_path} |"
        )

    stage_artifacts = markdown_jobs[0][2] if markdown_jobs else {}
    overlay_images = [
        value for key, value in sorted(stage_artifacts.items()) if key.startswith("matrix_overlay_") and "_image_" in key
    ]
    method_text = (
        "\n\n".join(_formula_text(language=language, scheme=scheme) for scheme in sorted(schemes))
        if primary_scheme == "mixed"
        else _formula_text(language=language, scheme=primary_scheme)
    )
    lines.extend(
        [
            "",
            "## Method" if language == "en" else "## 方法",
            method_text,
        ]
    )
    if overlay_images:
        overlay_groups: dict[str, list[str]] = {}
        for image in overlay_images:
            stem = Path(image).stem
            label = stem[3:] if stem.startswith("rn_") else stem
            label = label[:-3] if label.endswith(("_re", "_im")) else label
            overlay_groups.setdefault(label, []).append(image)
        for label, images in overlay_groups.items():
            title = f"{label} ensemble overview" if language == "en" else f"{label}组态总览图"
            lines.extend(["", f"## {title}", ""])
            images.sort(key=lambda image: 0 if Path(image).stem.endswith("_re") else 1)
            lines.extend(f"![{title}]({image})" for image in images)
    for item, result, artifacts in markdown_jobs:
        is_fit_job = result.get("job_kind") == "fit" or (
            result.get("scheme") == "hybrid_self_renormalization" and artifacts.get("zR_artifact") and not artifacts.get("renormalized_artifact")
        )
        lines.extend(
            [
                "",
                f"## `{item['job_id']}`",
                "",
                "### Scheme Parameters" if language == "en" else "### 方案参数",
                *_scheme_table(result, language=language),
            ]
        )
        if not is_fit_job:
            lines.extend(
                [
                    "",
                    "### Renormalized Matrix Element" if language == "en" else "### 重整化矩阵元图",
                    (
                        f"![Renormalized matrix element]({artifacts.get('renormalized_plot_image')})"
                        if artifacts.get("renormalized_plot_image")
                        else ("Not available." if language == "en" else "未生成。")
                    ),
                    (
                        f"[PDF artifact]({artifacts.get('renormalized_plot')})"
                        if artifacts.get("renormalized_plot")
                        else ""
                    ),
                ]
            )
        lines.extend(
            [
                "",
                "### Diagnostic Plots" if language == "en" else "### 诊断图",
            ]
        )
        diag_images = [
            key for key in artifacts
            if key.startswith("diag_") and key.endswith("_image") and artifacts.get(key)
        ]
        if not diag_images:
            lines.append("Not available." if language == "en" else "未生成。")
        else:
            for key in diag_images:
                label = key.removeprefix("diag_").removesuffix("_image")
                lines.append(f"![{label}]({artifacts[key]})")
                pdf_key = key.removesuffix("_image")
                if artifacts.get(pdf_key):
                    lines.append(f"[PDF]({artifacts[pdf_key]})")
        lines.extend(
            [
                "",
                "### Output Artifacts" if language == "en" else "### 输出文件",
                *_outputs_table(artifacts, language=language),
            ]
        )
    return "\n".join(lines) + "\n"


def write_renorm_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
    report_language: str = "en",
) -> dict[str, Path]:
    """Write one report summarizing all renormalization jobs."""
    output = Path(path)
    target, language = resolve_report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_renorm_stage_report_markdown(jobs=jobs, base_dir=target.parent, language=language),
        encoding="utf-8",
    )
    return {"report": target}
