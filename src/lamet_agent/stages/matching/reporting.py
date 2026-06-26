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

MATCHING_ARTIFACT_DESCRIPTIONS = {
    "lightcone_artifact": ("Matched light-cone PDF samples (EnsembleData NetCDF)", "匹配后的光锥 PDF 样本（EnsembleData NetCDF）"),
    "matched_plot": ("PDF plot comparing quasi and light-cone PDFs", "quasi 与光锥 PDF 对比 PDF 图"),
}

MATCHING_ARTIFACT_ORDER = ("lightcone_artifact", "matched_plot")


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
    for key in ("matched_plot", "lightcone_artifact"):
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
    # The `CG` prefix of the kernel_id marks the Coulomb-gauge (no Wilson line)
    # construction; anything else is the conventional gauge-invariant one.
    is_coulomb = kernel_id.upper().startswith("CG")
    gauge_en = "Coulomb gauge ($\\partial_i A_i=0$, no Wilson line)" if is_coulomb else "gauge-invariant (straight Wilson line)"
    gauge_zh = "库伦规范（Coulomb gauge，$\\partial_i A_i=0$，无 Wilson 线）" if is_coulomb else "规范不变（gauge-invariant，含直 Wilson 线）"
    x_grid = np.asarray(data.get("x_grid", []), dtype=float)
    zspz = data.get("zspz")
    pz_value = data.get("pz_gev")
    try:
        pz_text = f"$P_z={_fmt(float(pz_value))}$ GeV"
    except (TypeError, ValueError):
        pz_text = str(pz_value or "not recorded")

    if language == "zh":
        rows = [
            ("矩阵元/算符", f"`{kernel_id}`（{op_zh}）"),
            ("规范约定", gauge_zh),
            ("匹配方案", f"`{scheme}`（{scheme_en}）"),
            ("夸克/胶子分量", f"`{data.get('component', 'not recorded')}`"),
            ("强子动量", pz_text),
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
            ("Gauge convention", gauge_en),
            ("Matching scheme", f"`{scheme}` ({scheme_en})"),
            ("Quark/gluon component", f"`{data.get('component', 'not recorded')}`"),
            ("Hadron momentum", pz_text),
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


def _explicit_kernel_text(operator: str, scheme: str, *, language: str) -> str:
    r"""Return the explicit analytic NLO matching coefficient for ``operator``.

    Mirrors the closed forms implemented in :mod:`lamet_agent.kernels`
    (``_ratio_regular_entry``, ``_hybrid_delta_entry`` and ``CG_gluon_PDF_msbar``).
    """
    if operator == "gluon":
        poly = r"P_g(\xi)=\frac{2\,(1-\xi+\xi^2)^2}{1-\xi}"
        gluon = (
            r"C_g^{(1)}(\xi)=P_g(\xi)\Big[L+\ln\!\big(\xi(1-\xi)\big)\Big]"
            r"-\frac{15-56\xi+102\xi^2-96\xi^3+48\xi^4}{6\,(1-\xi)},\qquad 0<\xi<1,"
        )
        if language == "zh":
            return (
                "胶子核为 $C_A$ 正比、纯 MSbar（无 ratio/hybrid 方案）。记 $\\xi=x/y$、$L=\\ln(4y^2P_z^2/\\mu^2)$，"
                "并令\n\n"
                f"$$\n{poly}\n$$\n\n"
                "则物理区 $0<\\xi<1$ 的正则系数为\n\n"
                f"$$\n{gluon}\n$$\n\n"
                "$\\xi=1$ 处的奇异性由加法（plus）规则恢复，即令每个 $y$ 列积分为零。"
            )
        return (
            "The gluon kernel is $C_A$-proportional and pure MSbar (no ratio/hybrid scheme). "
            "With $\\xi=x/y$, $L=\\ln(4y^2P_z^2/\\mu^2)$ and\n\n"
            f"$$\n{poly}\n$$\n\n"
            "the regular coefficient in the physical region $0<\\xi<1$ is\n\n"
            f"$$\n{gluon}\n$$\n\n"
            "The $\\xi=1$ singularity is restored by the plus prescription (each $y$ column integrates to zero)."
        )

    # gt / gtg5 quark kernels share the same gamma^t structure.
    c_ratio = (
        r"C_r^{(1)}(\xi)=\frac{1+\xi^2}{1-\xi}\Big[L+\ln|\xi|+\ln|1-\xi|\Big]"
        r"+(\xi-1)+1+A(\xi)-\frac{3}{2\,|1-\xi|},\qquad 0<\xi<1,"
    )
    arctan = (
        r"A(\xi)=\frac{3\xi-1}{\xi-1}\times"
        r"\begin{cases}\dfrac{\arctan\!\big(\sqrt{1-2\xi}/|\xi|\big)}{\sqrt{1-2\xi}},&\xi<\tfrac12\\[2.2ex]"
        r"\dfrac{\operatorname{artanh}\!\big(\sqrt{2\xi-1}/|\xi|\big)}{\sqrt{2\xi-1}},&\xi>\tfrac12\end{cases}"
    )
    corrections = (
        r"\Delta C_{\overline{\rm MS}}=+\frac{1}{2|1-\xi|},\qquad"
        r"\Delta C_{\rm hy}=\frac{1}{2}\Big[\frac{1}{|1-\xi|}"
        r"-\frac{2}{\pi}\,\frac{\mathrm{Si}\!\big((1-\xi)\,z_sP_z\big)}{1-\xi}\Big]."
    )
    diag = r"+\frac{1}{2}\big(1+L\big)\quad(\text{MSbar diagonal, plus-prescription row})."
    if language == "zh":
        return (
            "夸克 $\\gamma^t$（及 $\\gamma^t\\gamma_5$）核的骨架是 ratio 方案正则系数。记 $\\xi=x/y$、"
            "$L=\\ln(4y^2P_z^2/\\mu^2)$，物理区 $0<\\xi<1$：\n\n"
            f"$$\n{c_ratio}\n$$\n\n"
            "其中 arctan/arctanh 项按 $\\xi$ 相对 $1/2$ 取分支：\n\n"
            f"$$\n{arctan}\n$$\n\n"
            "三种方案仅相差一个加在 $C_r^{(1)}$ 上的有限修正（off-diagonal）：\n\n"
            f"$$\n{corrections}\n$$\n\n"
            "ratio 方案修正为零；$\\xi=1$ 由加法规则恢复，MSbar 另在对角元加上 "
            f"${diag}$"
        )
    return (
        "The quark $\\gamma^t$ (and $\\gamma^t\\gamma_5$) kernels share the ratio-scheme regular "
        "coefficient as their backbone. With $\\xi=x/y$ and $L=\\ln(4y^2P_z^2/\\mu^2)$, in the "
        "physical region $0<\\xi<1$:\n\n"
        f"$$\n{c_ratio}\n$$\n\n"
        "where the arctan/arctanh term picks its branch by where $\\xi$ sits relative to $1/2$:\n\n"
        f"$$\n{arctan}\n$$\n\n"
        "The three schemes differ only by a finite off-diagonal correction added on top of $C_r^{(1)}$:\n\n"
        f"$$\n{corrections}\n$$\n\n"
        "the ratio scheme adds zero; the $\\xi=1$ singularity is restored by the plus prescription, "
        f"and MSbar additionally adds ${diag}$"
    )


def _matching_formula_text(data: dict[str, Any], *, language: str) -> str:
    kernel_id = str(data.get("kernel_id", ""))
    operator, scheme = _parse_kernel_id(kernel_id)
    _scheme_en, reference = SCHEME_TEXT.get(scheme, ("", "arXiv:2602.11283"))
    formula = (
        r"f(x,\mu)=\int\frac{dy}{|y|}\,C^{-1}\!\left(\frac{x}{y},\frac{\mu}{yP_z}\right)"
        r"\tilde f\!\left(y,P_z\right),"
    )
    discrete = r"f_i=\sum_j K_{ij}\,\tilde f_j,\qquad K=\text{(nx, ny) NLO matrix}."
    explicit = _explicit_kernel_text(operator, scheme, language=language)
    if language == "zh":
        return (
            f"{reference}。光锥 PDF 由 quasi-PDF 经 NLO 匹配核反卷积得到：\n\n"
            f"$$\n{formula}\n$$\n\n"
            "离散化后即矩阵乘法（本阶段对每个重采样样本独立施加，再重建统计量）：\n\n"
            f"$$\n{discrete}\n$$\n\n"
            "其中 LO 为单位阵，NLO 修正为 "
            "$K=\\mathbb{1}-\\dfrac{\\alpha_s\\,C_{F/A}}{2\\pi}\\,C^{(1)}(\\xi)\\,\\dfrac{dy}{|y|}$，"
            "解析形式为：\n\n"
            f"{explicit}"
        )
    return (
        f"{reference}. The light-cone PDF is obtained from the quasi-PDF by inverting the NLO matching kernel:\n\n"
        f"$$\n{formula}\n$$\n\n"
        "After discretization this is a matrix product (applied to every resampling sample independently, then the statistics are rebuilt):\n\n"
        f"$$\n{discrete}\n$$\n\n"
        "Here the LO part is the identity and the NLO correction is "
        "$K=\\mathbb{1}-\\dfrac{\\alpha_s\\,C_{F/A}}{2\\pi}\\,C^{(1)}(\\xi)\\,\\dfrac{dy}{|y|}$, "
        "with the explicit coefficient:\n\n"
        f"{explicit}"
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
        if language == "zh":
            lines.extend(
                [
                    f"- quasi-PDF 归一 $\\int f\\,dx={_fmt(quasi_norm)}$；光锥 PDF 归一 $\\int f\\,dx={_fmt(lc_norm)}$。",
                    f"- 归一相对变化 {_fmt(100 * rel)}%。NLO 匹配是微扰修正，应当接近守恒。",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Quasi-PDF norm $\\int f\\,dx={_fmt(quasi_norm)}$; light-cone norm $\\int f\\,dx={_fmt(lc_norm)}$.",
                    f"- Relative norm change {_fmt(100 * rel)}%. NLO matching is a perturbative correction and should nearly preserve the norm.",
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
    header = "| File | Description |" if language == "en" else "| 文件名 | 文件描述 |"
    lines = [header, "|---|---|"]
    for key in MATCHING_ARTIFACT_ORDER:
        value = artifacts.get(key)
        if not value:
            continue
        desc = MATCHING_ARTIFACT_DESCRIPTIONS[key][1 if language == "zh" else 0]
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


def write_matching_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
) -> dict[str, Path]:
    """Write one bilingual report summarizing all matching jobs in a stage."""
    output = Path(path)
    cn_output = _cn_report_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cn_output.parent.mkdir(parents=True, exist_ok=True)
    first = jobs[0]["result"]
    for language, target in (("en", output), ("zh", cn_output)):
        kernel_id = str(first.get("kernel_id", "not recorded"))
        operator, scheme = _parse_kernel_id(kernel_id)
        op_en, op_zh = OPERATOR_TEXT.get(operator, (operator or "not recorded",) * 2)
        scheme_en, _ref = SCHEME_TEXT.get(scheme, (scheme or "not recorded", ""))
        lines = [
            "# Perturbative Matching Stage Report" if language == "en" else "# 微扰匹配阶段报告",
            "",
            f"This report summarizes all perturbative-matching jobs for `{kernel_id}` ({op_en}) using the `{scheme_en}` scheme."
            if language == "en"
            else f"本报告汇总 `{kernel_id}`（{op_zh}）在 `{scheme_en}` 方案下的所有动量匹配。",
            "",
            "## Job Summary" if language == "en" else "## Job 汇总",
            "| job | kernel | $P_z$ | output | plot |"
            if language == "en"
            else "| job | kernel | $P_z$ | 输出 | 图像 |",
            "|---|---|---:|---|---|",
        ]
        for item in jobs:
            result = item["result"]
            artifacts = _markdown_artifacts(item.get("artifacts", {}), base_dir=target.parent)
            lines.append(
                f"| `{item['job_id']}` | {result.get('kernel_id', 'n/a')} | "
                f"{_fmt(result.get('pz_gev'))} | "
                f"{artifacts.get('lightcone_artifact', 'n/a')} | "
                f"{artifacts.get('matched_plot', 'n/a')} |"
            )
        setting_data = {**first, "pz_gev": "see per-momentum table" if language == "en" else "见下方动量表"}
        lines.extend(
            [
                "",
                "## Analysis Setup" if language == "en" else "## 分析设置",
                *_settings_table(setting_data, language=language),
                "",
                "### Field Definitions" if language == "en" else "### 条目解释",
                *_field_definitions(language=language),
                "",
                "## Matching Formula" if language == "en" else "## 匹配公式",
                _matching_formula_text(first, language=language),
                "",
                *_scheme_explanation(first, language=language),
                "",
                "## Diagnostics and Consistency Checks" if language == "en" else "## 诊断与一致性检查",
                "| job | $P_z$ | quasi norm | matched norm | norm change |"
                if language == "en"
                else "| job | $P_z$ | quasi 归一 | 匹配后归一 | 归一变化 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in jobs:
            result = item["result"]
            x_grid = np.asarray(result.get("x_grid", []), dtype=float)
            quasi_mean = np.asarray(result.get("quasi_mean", []), dtype=float)
            lc_mean = np.asarray(result.get("lightcone_mean", []), dtype=float)
            if x_grid.size >= 2 and quasi_mean.size == x_grid.size and lc_mean.size == x_grid.size:
                quasi_norm = _trapz_norm(x_grid, quasi_mean)
                lc_norm = _trapz_norm(x_grid, lc_mean)
                rel = abs(lc_norm - quasi_norm) / abs(quasi_norm) if quasi_norm != 0.0 else float("nan")
                lines.append(
                    f"| `{item['job_id']}` | {_fmt(result.get('pz_gev'))} | {_fmt(quasi_norm)} | "
                    f"{_fmt(lc_norm)} | {_fmt(100 * rel)}% |"
                )
            else:
                lines.append(f"| `{item['job_id']}` | {_fmt(result.get('pz_gev'))} | n/a | n/a | n/a |")
        lines.extend(
            [
                "",
                "The table compares the quasi-PDF and matched light-cone PDF norm for each momentum. Moderate norm changes are expected from the NLO kernel; a very large norm change usually indicates an x-grid or momentum-convention issue."
                if language == "en"
                else "上表逐动量比较 quasi-PDF 与匹配后光锥 PDF 的归一。NLO 匹配会带来有限修正；若归一变化很大，通常需要检查 x 网格或动量约定。",
                "",
                "## Figures and Visual Assessment" if language == "en" else "## 图像与可视化评估",
            ]
        )
        for item in jobs:
            result = item["result"]
            artifacts = _markdown_artifacts(item.get("artifacts", {}), base_dir=target.parent)
            plot = artifacts.get("matched_plot")
            lines.extend(["", f"### `{item['job_id']}`: $P_z={_fmt(result.get('pz_gev'))}$ GeV"])
            if plot:
                lines.append(
                    f"[Quasi vs light-cone comparison (PDF)]({plot})"
                    if language == "en"
                    else f"[quasi 与光锥 PDF 对比图（PDF）]({plot})"
                )
            else:
                lines.append("未生成。" if language == "zh" else "Not available.")
        lines.extend(
            [
                "",
                "## Output Artifacts" if language == "en" else "## 输出文件",
                "| File | Description |" if language == "en" else "| 文件名 | 文件描述 |",
                "|---|---|",
            ]
        )
        for item in jobs:
            artifacts = _markdown_artifacts(item.get("artifacts", {}), base_dir=target.parent)
            for key in MATCHING_ARTIFACT_ORDER:
                value = artifacts.get(key)
                if value:
                    desc = MATCHING_ARTIFACT_DESCRIPTIONS[key]
                    lines.append(f"| [{Path(value).name}]({value}) | `{item['job_id']}`: {desc[1 if language == 'zh' else 0]} |")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"en": output, "zh": cn_output}
