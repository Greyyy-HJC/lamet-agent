"""Markdown reporting helpers for the renormalization stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


RENORM_ARTIFACT_DESCRIPTIONS = {
    "renormalized_artifact": ("Renormalized matrix element samples (EnsembleData NetCDF)", "重整化矩阵元样本（EnsembleData NetCDF）"),
    "renormalized_plot": ("PDF plot of the renormalized matrix element", "重整化矩阵元 PDF 图"),
    "renormalized_plot_image": ("SVG companion for Markdown embedding", "供 Markdown 嵌入的重整化矩阵元 SVG 图"),
}

RENORM_ARTIFACT_ORDER = ("renormalized_artifact", "renormalized_plot", "renormalized_plot_image")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not set"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return str(number)
    return f"{number:.{digits}g}"


def _fmt_list(values: Any, *, max_items: int = 8, digits: int = 4) -> str:
    arr = np.asarray(values)
    if arr.size == 0:
        return "[]"
    flat = arr.reshape(-1)
    items = [_fmt(item, digits=digits) for item in flat[:max_items]]
    suffix = ", ..." if flat.size > max_items else ""
    return "[" + ", ".join(items) + suffix + "]"


def _cn_report_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_CN{path.suffix or '.md'}")


def _report_target(path: Path, report_language: str) -> tuple[Path, str]:
    language = report_language.lower()
    if language == "en":
        return path, "en"
    if language == "ch":
        return _cn_report_path(path), "zh"
    raise ValueError("report_language must be 'en' or 'ch'")


def _md_path(value: Any, *, base_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return os.path.relpath(path, base_dir)
    return str(value)


def _markdown_artifacts(artifacts: dict[str, Any] | None, *, base_dir: Path) -> dict[str, Any]:
    output = dict(artifacts or {})
    for key in RENORM_ARTIFACT_ORDER:
        if key in output:
            output[key] = _md_path(output[key], base_dir=base_dir)
    return output


def _scheme_table(result: dict[str, Any], *, language: str) -> list[str]:
    if language == "zh":
        rows = [
            ("方案", f"`{result.get('scheme', 'hybrid_ratio')}`"),
            ("$z_s$ [fm]", _fmt(result.get("zs_fm"))),
            ("$z_s/a$", _fmt(result.get("zs_lattice"))),
            ("选中的 denominator z grid", _fmt(result.get("zs_grid"))),
            ("$\\delta m$ [GeV]", _fmt(result.get("delta_m_gev"))),
            ("$m_0$ [GeV]", _fmt(result.get("m0_gev"))),
            ("z 网格", _fmt_list(result.get("z_grid", []))),
            ("重采样", f"{result.get('n_sample', 'n/a')} 个样本"),
        ]
        header = "| 条目 | 数值或设置 |"
    else:
        rows = [
            ("Scheme", f"`{result.get('scheme', 'hybrid_ratio')}`"),
            ("$z_s$ [fm]", _fmt(result.get("zs_fm"))),
            ("$z_s/a$", _fmt(result.get("zs_lattice"))),
            ("Selected denominator z grid", _fmt(result.get("zs_grid"))),
            ("$\\delta m$ [GeV]", _fmt(result.get("delta_m_gev"))),
            ("$m_0$ [GeV]", _fmt(result.get("m0_gev"))),
            ("z grid", _fmt_list(result.get("z_grid", []))),
            ("Resampling", f"{result.get('n_sample', 'n/a')} samples"),
        ]
        header = "| Quantity | Value |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _formula_text(*, language: str) -> str:
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


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    header = "| Artifact | Description |" if language == "en" else "| 文件 | 说明 |"
    lines = [header, "|---|---|"]
    for key in RENORM_ARTIFACT_ORDER:
        value = artifacts.get(key)
        if not value:
            continue
        desc = RENORM_ARTIFACT_DESCRIPTIONS[key][0 if language == "en" else 1]
        lines.append(f"| `{value}` | {desc} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def build_renorm_stage_report_markdown(
    *,
    jobs: list[dict[str, Any]],
    base_dir: Path,
    language: str = "en",
) -> str:
    """Build one Markdown report for all renormalization jobs."""
    title = "# Renormalization Stage Report" if language == "en" else "# Renormalization 阶段报告"
    intro = (
        "This report summarizes hybrid-ratio renormalization jobs that convert bare matrix elements into renormalized coordinate-space matrix elements."
        if language == "en"
        else "本报告汇总 hybrid-ratio 重整化 job，将裸矩阵元转换为坐标空间重整化矩阵元。"
    )
    lines = [
        title,
        "",
        intro,
        "",
        "## Job Summary" if language == "en" else "## Job 汇总",
        "| job | scheme | $z_s$ [fm] | output | plot |" if language == "en" else "| job | 方案 | $z_s$ [fm] | 输出 | 图像 |",
        "|---|---|---:|---|---|",
    ]
    markdown_jobs = []
    for item in jobs:
        result = item.get("result", {})
        artifacts = _markdown_artifacts(item.get("artifacts", {}), base_dir=base_dir)
        markdown_jobs.append((item, result, artifacts))
        lines.append(
            f"| `{item['job_id']}` | `{result.get('scheme', 'hybrid_ratio')}` | "
            f"{_fmt(result.get('zs_fm'))} | "
            f"{artifacts.get('renormalized_artifact', 'n/a')} | "
            f"{artifacts.get('renormalized_plot', 'n/a')} |"
        )

    lines.extend(["", "## Method" if language == "en" else "## 方法", _formula_text(language=language)])
    for item, result, artifacts in markdown_jobs:
        lines.extend(
            [
                "",
                f"## `{item['job_id']}`",
                "",
                "### Scheme Parameters" if language == "en" else "### 方案参数",
                *_scheme_table(result, language=language),
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
    target, language = _report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_renorm_stage_report_markdown(jobs=jobs, base_dir=target.parent, language=language),
        encoding="utf-8",
    )
    return {"report": target}
