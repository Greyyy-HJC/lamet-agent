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
    if scheme == "self_renormalization":
        job_kind = str(
            result.get("job_kind")
            or ("fit" if result.get("d") is not None and "a_fm" not in result else "apply")
        )
        if language == "zh":
            rows = [
                ("方案", f"`{scheme}`"),
                ("job 类型", f"`{job_kind}`"),
                ("kernel_id", f"`{result.get('kernel_id', 'n/a')}`"),
                ("$\\mu$ [GeV]", format_report_value(result.get("mu"))),
                ("$m_0$ [GeV]", format_report_value(result.get("m0", result.get("m0_gev")))),
                ("$d$", format_report_value(result.get("d"))),
                ("svdcut", format_report_value(result.get("svdcut"))),
                ("$a$ [fm]", format_report_value(result.get("a_fm"))),
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
                ("$m_0$ [GeV]", format_report_value(result.get("m0", result.get("m0_gev")))),
                ("$d$", format_report_value(result.get("d"))),
                ("svdcut", format_report_value(result.get("svdcut"))),
                ("$a$ [fm]", format_report_value(result.get("a_fm"))),
                ("z grid", format_report_list(result.get("z_grid", result.get("z_values", [])))),
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
    if scheme == "self_renormalization":
        if language == "zh":
            return r"""
Self-renormalization 先从零动量 reference 拟合 $z_R(z,a)$，再对每个重采样样本作用

$$
h^R_s(z)=\frac{h^{\rm tar}_s(z)}{z_R(z,a)\,Z_{\overline{\mathrm{MS}}}(z;\mu)}.
$$

$Z_{\overline{\mathrm{MS}}}$ 由 `inputs.kernels` 中 `stage='renormalization'` 的 kernel（`ZMSbar_pdf` 或 `ZMSbar_da`）给出。该步骤不重新拟合矩阵元，而是对所有样本施加同一个重整化 map。
""".strip()
        return r"""
Self-renormalization first fits $z_R(z,a)$ from a zero-momentum reference, then acts sample by sample as

$$
h^R_s(z)=\frac{h^{\rm tar}_s(z)}{z_R(z,a)\,Z_{\overline{\mathrm{MS}}}(z;\mu)}.
$$

$Z_{\overline{\mathrm{MS}}}$ comes from the `inputs.kernels` entry with `stage='renormalization'` (`ZMSbar_pdf` or `ZMSbar_da`). This stage does not refit matrix elements; it applies one renormalization map to all resampled samples.
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
        intro = (
            "This report summarizes self-renormalization jobs that convert bare matrix elements into renormalized coordinate-space matrix elements."
            if primary_scheme == "self_renormalization"
            else "This report summarizes hybrid-ratio renormalization jobs that convert bare matrix elements into renormalized coordinate-space matrix elements."
        )
        summary_header = "| job | scheme | key | output | plot |"
    else:
        intro = (
            "本报告汇总 self-renormalization job，将裸矩阵元转换为坐标空间重整化矩阵元。"
            if primary_scheme == "self_renormalization"
            else "本报告汇总 hybrid-ratio 重整化 job，将裸矩阵元转换为坐标空间重整化矩阵元。"
        )
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
                if key in RENORM_ARTIFACT_ORDER or key.startswith("diag_")
            ),
        )
        markdown_jobs.append((item, result, artifacts))
        key = result.get("kernel_id") if result.get("scheme") == "self_renormalization" else result.get("zs_fm")
        output_path = artifacts.get("renormalized_artifact") or artifacts.get("zR_artifact") or "n/a"
        plot_path = artifacts.get("renormalized_plot") or "n/a"
        lines.append(
            f"| `{item['job_id']}` | `{result.get('scheme', 'hybrid_ratio')}` | "
            f"{key if key is not None else format_report_value(result.get('zs_fm'))} | "
            f"{output_path} | "
            f"{plot_path} |"
        )

    lines.extend(
        [
            "",
            "## Method" if language == "en" else "## 方法",
            _formula_text(language=language, scheme=primary_scheme if primary_scheme != "mixed" else "hybrid_ratio"),
        ]
    )
    for item, result, artifacts in markdown_jobs:
        is_fit_job = result.get("job_kind") == "fit" or (
            result.get("scheme") == "self_renormalization" and artifacts.get("zR_artifact") and not artifacts.get("renormalized_artifact")
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
