"""Markdown reporting helpers for the perturbative-matching stage.

Mirrors ``stages/fourier/reporting.py``: it turns the matching-stage result and
artifacts into an English report plus a Chinese companion. The matching stage is
simpler than the Fourier stage (no scheme scan / model averaging), so the report
focuses on the chosen kernel, the matching convolution, and a small set of
"is this a sane perturbative correction" diagnostics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


# Logical operator -> human text, keyed by the ``<operator>`` field of a
# ``CG_<operator>_PDF_<scheme>`` kernel_id.
OPERATOR_TEXT = {
    "gt": ("unpolarized $\\gamma^t$ quark PDF", "非极化 $\\gamma^t$ 夸克 PDF"),
    "gtg5": ("helicity $\\gamma^t\\gamma_5$ quark PDF", "螺旋度 $\\gamma^t\\gamma_5$ 夸克 PDF"),
    "gluon": ("unpolarized gluon PDF", "非极化胶子 PDF"),
}

# Scheme -> (human text, reference equation in arXiv:2602.11283).
SCHEME_TEXT = {
    "msbar": ("MSbar", "arXiv:2602.11283 Eq. (2.14)"),
    "ratio": ("ratio", "arXiv:2602.11283 Eq. (2.16)"),
    "hybrid": ("hybrid", "arXiv:2602.11283 Eqs. (2.19)/(2.20)"),
}


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


def _parse_kernel_id(kernel_id: str) -> tuple[str, str]:
    """Split a ``CG_<operator>_PDF_<scheme>`` id into (operator, scheme).

    Falls back to ('', '') for any id that does not follow the convention so the
    report degrades gracefully instead of raising.
    """
    parts = str(kernel_id).split("_")
    # CG, <op...>, PDF, <scheme>
    if len(parts) >= 4 and parts[0] == "CG" and "PDF" in parts:
        pdf_idx = parts.index("PDF")
        operator = "_".join(parts[1:pdf_idx])
        scheme = "_".join(parts[pdf_idx + 1 :])
        return operator, scheme
    return "", ""


def _format_grid(x_grid: np.ndarray, *, language: str) -> str:
    if x_grid.size == 0:
        return "未记录" if language == "zh" else "not recorded"
    if x_grid.size == 1:
        return f"one point at $x={_fmt(x_grid[0])}$"
    diffs = np.diff(x_grid)
    if np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        if language == "zh":
            return f"从 $x={_fmt(x_grid[0])}$ 到 $x={_fmt(x_grid[-1])}$，每隔 $\\Delta x={_fmt(diffs[0])}$ 取一个点，共 {x_grid.size} 个点"
        return f"from $x={_fmt(x_grid[0])}$ to $x={_fmt(x_grid[-1])}$ with spacing $\\Delta x={_fmt(diffs[0])}$, for {x_grid.size} points"
    if language == "zh":
        return f"非均匀网格，共 {x_grid.size} 个点；预览 `{_fmt_list(x_grid)}`"
    return f"nonuniform grid with {x_grid.size} points; preview `{_fmt_list(x_grid)}`"


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
    for key in ("matched_plot", "lightcone_npz"):
        if key in output:
            output[key] = _md_path(output[key], base_dir=base_dir)
    return output


def _trapz_norm(x_grid: np.ndarray, values: np.ndarray) -> float:
    """Integral of ``values`` over the x grid (the PDF norm sum rule check)."""
    if x_grid.size < 2 or values.size != x_grid.size:
        return float("nan")
    order = np.argsort(x_grid)
    # np.trapezoid is the NumPy 2.x name; fall back to np.trapz on older NumPy.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values[order], x_grid[order]))


def _settings_table(data: dict[str, Any], *, language: str) -> list[str]:
    kernel_id = str(data.get("kernel_id", "not recorded"))
    operator, scheme = _parse_kernel_id(kernel_id)
    op_en, op_zh = OPERATOR_TEXT.get(operator, (operator or "not recorded",) * 2)
    scheme_en, _scheme_ref = SCHEME_TEXT.get(scheme, (scheme or "not recorded", ""))
    x_grid = np.asarray(data.get("x_grid", []), dtype=float)
    zspz = data.get("zspz")

    if language == "zh":
        rows = [
            ("矩阵元/算符", f"`{kernel_id}`（{op_zh}）"),
            ("匹配方案", f"`{scheme}`（{scheme_en}）"),
            ("夸克/胶子分量", f"`{data.get('component', 'not recorded')}`"),
            ("强子动量", f"$P_z={_fmt(data.get('pz_gev'))}$ GeV"),
            ("重整化标度", f"$\\mu={_fmt(data.get('mu'))}$ GeV"),
        ]
        if zspz is not None:
            rows.append(("Wilson 线标度", f"$z_sP_z={_fmt(zspz)}$"))
        rows.extend(
            [
                ("重采样模式", f"`{data.get('resample', 'not recorded')}`，共 {data.get('n_sample', 'n/a')} 个样本"),
                ("x 网格", _format_grid(x_grid, language="zh")),
                ("quasi-PDF 来源", f"`{data.get('source', 'not recorded')}`"),
            ]
        )
        header = "| 条目 | 数值或设置 |"
    else:
        rows = [
            ("Operator / kernel", f"`{kernel_id}` ({op_en})"),
            ("Matching scheme", f"`{scheme}` ({scheme_en})"),
            ("Quark/gluon component", f"`{data.get('component', 'not recorded')}`"),
            ("Hadron momentum", f"$P_z={_fmt(data.get('pz_gev'))}$ GeV"),
            ("Renormalization scale", f"$\\mu={_fmt(data.get('mu'))}$ GeV"),
        ]
        if zspz is not None:
            rows.append(("Wilson-line scale", f"$z_sP_z={_fmt(zspz)}$"))
        rows.extend(
            [
                ("Resampling mode", f"`{data.get('resample', 'not recorded')}` with {data.get('n_sample', 'n/a')} samples"),
                ("x grid", _format_grid(x_grid, language="en")),
                ("Quasi-PDF source", f"`{data.get('source', 'not recorded')}`"),
            ]
        )
        header = "| Quantity | Value |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _field_definitions(*, language: str) -> list[str]:
    if language == "zh":
        return [
            "| 条目 | 含义 |",
            "|---|---|",
            "| Operator / kernel | 选定的匹配核 `CG_<算符>_PDF_<方案>`；算符决定 Dirac 结构（gt、gtg5、gluon），方案决定有限项。 |",
            "| Matching scheme | `msbar` / `ratio` / `hybrid`，由 kernel_id 后缀选定；hybrid 还需要 Wilson 线长度 $z_s$。 |",
            "| Hadron momentum | $P_z$，必须与傅立叶阶段一致，进入核的 $\\log(4y^2P_z^2/\\mu^2)$ 项。 |",
            "| Renormalization scale | MSbar 重整化标度 $\\mu$（GeV），默认 2.0。 |",
            "| Resampling mode | quasi-PDF 携带的重采样轴（bootstrap/jackknife）；匹配逐样本进行以保留关联结构。 |",
        ]
    return [
        "| Entry | Meaning |",
        "|---|---|",
        "| Operator / kernel | The selected matching kernel `CG_<operator>_PDF_<scheme>`; the operator sets the Dirac structure (gt, gtg5, gluon) and the scheme sets the finite terms. |",
        "| Matching scheme | `msbar` / `ratio` / `hybrid`, chosen by the kernel_id suffix; hybrid also needs the Wilson-line length $z_s$. |",
        "| Hadron momentum | $P_z$, which must match the Fourier stage and enters the kernel's $\\log(4y^2P_z^2/\\mu^2)$ terms. |",
        "| Renormalization scale | MSbar renormalization scale $\\mu$ in GeV (default 2.0). |",
        "| Resampling mode | The resampling axis carried by the quasi-PDF (bootstrap/jackknife); matching is done sample by sample to preserve the correlation structure. |",
    ]


def _matching_formula_text(data: dict[str, Any], *, language: str) -> str:
    kernel_id = str(data.get("kernel_id", ""))
    _operator, scheme = _parse_kernel_id(kernel_id)
    _scheme_en, reference = SCHEME_TEXT.get(scheme, ("", "arXiv:2602.11283"))
    formula = (
        r"f(x,\mu)=\int\frac{dy}{|y|}\,C^{-1}\!\left(\frac{x}{y},\frac{\mu}{yP_z}\right)"
        r"\tilde f\!\left(y,P_z\right),"
    )
    discrete = r"f_i=\sum_j K_{ij}\,\tilde f_j,\qquad K=\text{(nx, ny) NLO matrix}."
    if language == "zh":
        return (
            f"{reference}。光锥 PDF 由 quasi-PDF 经 NLO 匹配核反卷积得到：\n\n"
            f"$$\n{formula}\n$$\n\n"
            "离散化后即矩阵乘法（本阶段对每个重采样样本独立施加，再重建统计量）：\n\n"
            f"$$\n{discrete}\n$$"
        )
    return (
        f"{reference}. The light-cone PDF is obtained from the quasi-PDF by inverting the NLO matching kernel:\n\n"
        f"$$\n{formula}\n$$\n\n"
        "After discretization this is a matrix product (applied to every resampling sample independently, then the statistics are rebuilt):\n\n"
        f"$$\n{discrete}\n$$"
    )


def _scheme_explanation(data: dict[str, Any], *, language: str) -> list[str]:
    kernel_id = str(data.get("kernel_id", ""))
    _operator, scheme = _parse_kernel_id(kernel_id)
    if language == "zh":
        notes = {
            "msbar": "MSbar 方案在裸 ratio 系数上加上有限的 MSbar 转换项（Eq. 2.14）。",
            "ratio": "ratio 方案直接使用裸的正则系数 $C_r$（Eq. 2.16），不含额外有限项。",
            "hybrid": "hybrid 方案在 ratio 系数上加上 Wilson 线的正弦积分修正，依赖 $z_sP_z$（Eq. 2.19-2.20）。",
        }
        body = notes.get(scheme, "未识别的匹配方案，仅记录所选 kernel_id。")
        return ["## 匹配方案", body]
    notes = {
        "msbar": "The MSbar scheme adds a finite MSbar conversion on top of the bare ratio coefficient (Eq. 2.14).",
        "ratio": "The ratio scheme uses the bare regular coefficient $C_r$ directly (Eq. 2.16) with no extra finite terms.",
        "hybrid": "The hybrid scheme adds a Wilson-line sine-integral correction to the ratio coefficient and depends on $z_sP_z$ (Eqs. 2.19-2.20).",
    }
    body = notes.get(scheme, "Unrecognized matching scheme; only the selected kernel_id is recorded.")
    return ["## Matching Scheme", body]


def _diagnostics(data: dict[str, Any], *, language: str) -> list[str]:
    x_grid = np.asarray(data.get("x_grid", []), dtype=float)
    quasi_mean = np.asarray(data.get("quasi_mean", []), dtype=float)
    lc_mean = np.asarray(data.get("lightcone_mean", []), dtype=float)
    lines: list[str] = []

    if x_grid.size >= 2 and quasi_mean.size == x_grid.size and lc_mean.size == x_grid.size:
        quasi_norm = _trapz_norm(x_grid, quasi_mean)
        lc_norm = _trapz_norm(x_grid, lc_mean)
        rel = abs(lc_norm - quasi_norm) / abs(quasi_norm) if quasi_norm not in (0.0, float("nan")) else float("nan")
        denom = np.where(np.abs(quasi_mean) > 1e-12, np.abs(quasi_mean), np.nan)
        max_dev = float(np.nanmax(np.abs(lc_mean - quasi_mean) / denom))
        if language == "zh":
            lines.extend(
                [
                    f"- quasi-PDF 归一 $\\int f\\,dx={_fmt(quasi_norm)}$；光锥 PDF 归一 $\\int f\\,dx={_fmt(lc_norm)}$。",
                    f"- 归一相对变化 {_fmt(100 * rel)}%。NLO 匹配是微扰修正，应当接近守恒。",
                    f"- quasi 与光锥逐点最大相对偏差 {_fmt(100 * max_dev)}%。偏差过大或剧烈振荡通常意味着 x 网格触及 0 或 $P_z$ 设置有误。",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Quasi-PDF norm $\\int f\\,dx={_fmt(quasi_norm)}$; light-cone norm $\\int f\\,dx={_fmt(lc_norm)}$.",
                    f"- Relative norm change {_fmt(100 * rel)}%. NLO matching is a perturbative correction and should nearly preserve the norm.",
                    f"- Maximum pointwise relative quasi-vs-light-cone deviation {_fmt(100 * max_dev)}%. A large deviation or wild oscillation usually signals an x grid hitting 0 or a wrong $P_z$.",
                ]
            )
    else:
        lines.append("- 无可用的匹配诊断。" if language == "zh" else "- Matching diagnostics were not available.")
    return lines


def _figure_block(artifacts: dict[str, Any], *, language: str) -> list[str]:
    heading = "## 图像与可视化评估" if language == "zh" else "## Figures and Visual Assessment"
    label = "quasi 与光锥 PDF 对比图" if language == "zh" else "Quasi vs light-cone comparison"
    pdf_value = artifacts.get("matched_plot")
    lines = [heading, "", f"### {label}"]
    if pdf_value:
        # The plot is a single PDF artifact; Markdown cannot embed PDFs inline, so link it.
        lines.append(f"[{label} (PDF)]({pdf_value})" if language == "en" else f"[{label}（PDF）]({pdf_value})")
    else:
        lines.append("未生成。" if language == "zh" else "Not available.")
    return lines


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    descriptions = {
        "matched_plot": ("PDF plot comparing quasi and light-cone PDFs", "quasi 与光锥 PDF 对比 PDF 图"),
        "lightcone_npz": ("Matched light-cone PDF samples (EnsembleData npz)", "匹配后的光锥 PDF 样本（EnsembleData npz）"),
    }
    order = ("lightcone_npz", "matched_plot")
    header = "| File | Description |" if language == "en" else "| 文件名 | 文件描述 |"
    lines = [header, "|---|---|"]
    for key in order:
        value = artifacts.get(key)
        if not value:
            continue
        desc = descriptions[key][1 if language == "zh" else 0]
        lines.append(f"| `{value}` | {desc} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def build_matching_report_markdown(
    *,
    result: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    language: str = "en",
) -> str:
    artifacts = artifacts or {}
    kernel_id = str(result.get("kernel_id", "not recorded"))
    operator, scheme = _parse_kernel_id(kernel_id)
    op_en, op_zh = OPERATOR_TEXT.get(operator, (operator or "not recorded",) * 2)
    scheme_en, _ref = SCHEME_TEXT.get(scheme, (scheme or "not recorded", ""))

    if language == "zh":
        lines = [
            "# 微扰匹配分析报告",
            "",
            "## 摘要",
            f"本报告总结将 `{kernel_id}`（{op_zh}）quasi-PDF 经 `{scheme_en}` 方案 NLO 匹配核转换为光锥 PDF 的过程。",
            "",
            "## 分析设置",
            *_settings_table(result, language="zh"),
            "",
            "### 条目解释",
            *_field_definitions(language="zh"),
            "",
            "## 匹配公式",
            _matching_formula_text(result, language="zh"),
            "",
            *_scheme_explanation(result, language="zh"),
            "",
            "## 诊断与一致性检查",
            *_diagnostics(result, language="zh"),
            "",
            *_figure_block(artifacts, language="zh"),
            "",
            "## 输出文件",
            *_outputs_table(artifacts, language="zh"),
        ]
    else:
        lines = [
            "# Perturbative Matching Analysis Report",
            "",
            "## Abstract",
            f"This report summarizes converting the `{kernel_id}` ({op_en}) quasi-PDF into the light-cone PDF using the `{scheme_en}`-scheme NLO matching kernel.",
            "",
            "## Analysis Setup",
            *_settings_table(result, language="en"),
            "",
            "### Field Definitions",
            *_field_definitions(language="en"),
            "",
            "## Matching Formula",
            _matching_formula_text(result, language="en"),
            "",
            *_scheme_explanation(result, language="en"),
            "",
            "## Diagnostics and Consistency Checks",
            *_diagnostics(result, language="en"),
            "",
            *_figure_block(artifacts, language="en"),
            "",
            "## Output Artifacts",
            *_outputs_table(artifacts, language="en"),
        ]
    return "\n".join(lines) + "\n"


def write_matching_report(
    *,
    result: dict[str, Any],
    artifacts: dict[str, Any] | None,
    path: str | Path,
) -> dict[str, Path]:
    """Write English and Chinese matching reports and return their paths."""
    output = Path(path)
    cn_output = _cn_report_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cn_output.parent.mkdir(parents=True, exist_ok=True)
    report_artifacts = _markdown_artifacts(artifacts, base_dir=output.parent)
    output.write_text(
        build_matching_report_markdown(result=result, artifacts=report_artifacts, language="en"),
        encoding="utf-8",
    )
    cn_output.write_text(
        build_matching_report_markdown(result=result, artifacts=report_artifacts, language="zh"),
        encoding="utf-8",
    )
    return {"en": output, "zh": cn_output}
