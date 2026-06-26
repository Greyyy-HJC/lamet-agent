"""Markdown reporting helpers for the renormalization stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


RENORM_ARTIFACT_DESCRIPTIONS = {
    "renormalized_artifact": ("Renormalized matrix element samples (EnsembleData NetCDF)", "重整化矩阵元样本（EnsembleData NetCDF）"),
    "renormalized_plot": ("PDF plot of the renormalized matrix element", "重整化矩阵元 PDF 图"),
}

RENORM_ARTIFACT_ORDER = ("renormalized_artifact", "renormalized_plot")


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
    formula = (
        r"h^R(z)=N\,h(z)/h_{\rm den}(z)\quad (|z|_{\rm fm}\le z_s),\qquad "
        r"h^R(z)=N\,e^{(\delta m+m_0)(|z|_{\rm fm}-z_s)}h(z)/h_{\rm den}(z_s)\quad (|z|_{\rm fm}>z_s)."
    )
    if language == "zh":
        return (
            "Hybrid-ratio 方案在短距离使用逐点 ratio，在长距离固定 denominator 的 $z_s$ 点并乘以指数修正：\n\n"
            f"$$\n{formula}\n$$"
        )
    return (
        "The hybrid-ratio scheme uses a pointwise ratio at short distances and switches to the denominator at $z_s$ with an exponential correction at long distances:\n\n"
        f"$$\n{formula}\n$$"
    )


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
                "### Output Artifacts" if language == "en" else "### 输出文件",
                *_outputs_table(artifacts, language=language),
            ]
        )
    return "\n".join(lines) + "\n"


def write_renorm_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
) -> dict[str, Path]:
    """Write one bilingual report summarizing all renormalization jobs."""
    output = Path(path)
    cn_output = _cn_report_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cn_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_renorm_stage_report_markdown(jobs=jobs, base_dir=output.parent, language="en"),
        encoding="utf-8",
    )
    cn_output.write_text(
        build_renorm_stage_report_markdown(jobs=jobs, base_dir=cn_output.parent, language="zh"),
        encoding="utf-8",
    )
    return {"en": output, "zh": cn_output}
