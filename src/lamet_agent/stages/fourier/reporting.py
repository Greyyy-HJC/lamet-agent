"""Markdown reporting helpers for the Fourier-transform stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


OBSERVABLE_TEXT = {
    "pion_quark_quasi_pdf": "pion quark quasi-PDF",
    "nucleon_quark_unpolarized_quasi_pdf": "nucleon quark unpolarized quasi-PDF",
    "nucleon_quark_transversity_quasi_pdf": "nucleon quark transversity quasi-PDF",
    "pion_gluon_quasi_pdf": "pion gluon quasi-PDF",
    "nucleon_gluon_quasi_pdf": "nucleon gluon quasi-PDF",
    "meson_quasi_da": "meson quasi-DA",
    "pion_quark_quasi_gpd": "pion quark quasi-GPD",
    "nucleon_quark_quasi_gpd": "nucleon quark quasi-GPD",
}

FORMULA_REFERENCES = {
    "pion_quark_quasi_pdf": ("arXiv:2601.12189 Eqs. (2.1)/(2.2)", "quark-like oscillatory tail"),
    "nucleon_quark_unpolarized_quasi_pdf": ("arXiv:2601.12189 Eqs. (2.3)/(2.4)", "quark-like oscillatory tail"),
    "nucleon_quark_transversity_quasi_pdf": ("arXiv:2601.12189 Eqs. (2.5)/(2.6)", "quark-like oscillatory tail"),
    "meson_quasi_da": ("arXiv:2601.12189 Eqs. (2.7)/(2.8)", "quark-like oscillatory tail"),
    "pion_quark_quasi_gpd": ("arXiv:2601.12189 Eqs. (2.9)/(2.10)", "quark-like oscillatory tail"),
    "nucleon_quark_quasi_gpd": ("arXiv:2601.12189 Eqs. (2.11)/(2.12)", "quark-like oscillatory tail"),
    "nucleon_gluon_quasi_pdf": ("arXiv:2601.12189 Appendix F Eqs. (F.6)/(F.7)", "nucleon gluon real tail"),
    "pion_gluon_quasi_pdf": ("arXiv:2601.12189 Appendix F Eqs. (F.8)/(F.9)", "pion gluon real tail"),
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


def _display_unit(unit: Any) -> str:
    text = str(unit or "not recorded").lower()
    if text == "gev_inv":
        return r"$\mathrm{GeV}^{-1}$"
    if text == "lambda":
        return r"$\lambda$"
    if text == "fm":
        return r"$\mathrm{fm}$"
    if text == "lattice":
        return r"$z/a$"
    return f"`{unit}`"


def _format_fit_range(fit_range: Any, *, language: str) -> str:
    if fit_range is None:
        return "不可用" if language == "zh" else "not available"
    return rf"$z^{{\rm min}}={_fmt(fit_range[0])}$ to $z^{{\rm max}}={_fmt(fit_range[1])}$"


def _format_grid(k_grid: np.ndarray, *, language: str) -> str:
    if k_grid.size == 0:
        return "未记录" if language == "zh" else "not recorded"
    if k_grid.size == 1:
        return f"one point at $x={_fmt(k_grid[0])}$"
    diffs = np.diff(k_grid)
    if np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        if language == "zh":
            return f"从 $x={_fmt(k_grid[0])}$ 到 $x={_fmt(k_grid[-1])}$，每隔 $\\Delta x={_fmt(diffs[0])}$ 取一个点，共 {k_grid.size} 个点"
        return f"from $x={_fmt(k_grid[0])}$ to $x={_fmt(k_grid[-1])}$ with spacing $\\Delta x={_fmt(diffs[0])}$, for {k_grid.size} points"
    if language == "zh":
        return f"非均匀网格，共 {k_grid.size} 个点；预览 `{_fmt_list(k_grid)}`"
    return f"nonuniform grid with {k_grid.size} points; preview `{_fmt_list(k_grid)}`"


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
    for key in (
        "fourier_npz",
        "fit_info_npz",
        "fourier_plot",
        "fourier_plot_image",
        "extension_plot_re",
        "extension_plot_re_image",
        "extension_plot_im",
        "extension_plot_im_image",
    ):
        if key in output:
            output[key] = _md_path(output[key], base_dir=base_dir)
    return output


def _tail_formula_text(result: dict[str, Any], *, language: str) -> str:
    method = str(result.get("method", "")).upper()
    order = str(result.get("order", "")).upper()
    observable = str(result.get("observable", ""))
    reference, family = FORMULA_REFERENCES.get(observable, ("code-selected LA/NLA formula", "tail formula"))
    common_tail = r"\exp(-\Lambda z)"
    method_note = ""
    if method == "CG":
        common_tail += r"z^{-n}"
        method_note = (
            "本地 CG 实现会在该尾部上额外乘以 $z^{-n}$。"
            if language == "zh"
            else "The local CG implementation additionally multiplies this tail by $z^{-n}$."
        )

    if observable == "nucleon_gluon_quasi_pdf":
        body = r"A z" if order == "LA" else r"A z + A'"
        formula = rf"\mathrm{{Re}}\,h(z)=({body}){common_tail},\qquad \mathrm{{Im}}\,h(z)=0."
    elif observable == "pion_gluon_quasi_pdf":
        body = r"A_2 z" if order == "LA" else r"A_2 z + A_2' + 2 A_1\cos(\phi-P_z z)"
        formula = rf"\mathrm{{Re}}\,h(z)=({body}){common_tail},\qquad \mathrm{{Im}}\,h(z)=0."
    else:
        nla = r" + \sum_j \frac{A'_j}{z}\exp[i(\phi'_j+\omega_j z)]" if order == "NLA" else ""
        formula = rf"h(z)=\left[\sum_j A_j\exp[i(\phi_j+\omega_j z)]{nla}\right]{common_tail}."

    if language == "zh":
        return f"{reference}。代码实现的 {family} 为\n\n$$\n{formula}\n$$\n\n{method_note}"
    return f"{reference}. The implemented {family} is\n\n$$\n{formula}\n$$\n\n{method_note}"


def _field_definitions(*, language: str) -> list[str]:
    if language == "zh":
        return [
            "| 条目 | 含义 |",
            "|---|---|",
            "| Observable | 本阶段要变换的物理矩阵元类型；它决定长程外推公式中的相位结构和参数结构。 |",
            "| Tail method/order | $\\mathrm{order}$ 选择 LA 或 NLA；$\\mathrm{method}=\\mathrm{CG}$ 表示在基础尾部上额外乘以 $z^{-n}$。 |",
            "| Active fitted component | `both` 表示实部和虚部同时参与拟合；`re` 只拟合实部；`im` 只拟合虚部。 |",
            "| Coordinate unit | `lattice` 表示 $z/a$。代码先转为 $z_{\\rm GeV^{-1}}=(z/a)a_{\\rm fm}\\,5.067731237$，再计算 $\\lambda=P_z z_{\\rm GeV^{-1}}$。 |",
            "| Posterior-prior error scale | 样本平均拟合得到 $\\bar p_i\\pm\\sigma_{p_i}$ 后，重采样拟合使用弱先验 $p_i=\\bar p_i\\pm s\\sigma_{p_i}$。 |",
        ]
    return [
        "| Entry | Meaning |",
        "|---|---|",
        "| Observable | Physical matrix element transformed by this stage. |",
        "| Tail method/order | $\\mathrm{order}$ selects LA or NLA; $\\mathrm{method}=\\mathrm{CG}$ adds $z^{-n}$ to the base tail. |",
        "| Active fitted component | `both` fits $\\mathrm{Re}\\,\\tilde h^R$ and $\\mathrm{Im}\\,\\tilde h^R$ together; `re` or `im` fits only one component. |",
        "| Coordinate unit | `lattice` means $z/a$. The code converts it to $z_{\\rm GeV^{-1}}=(z/a)a_{\\rm fm}\\,5.067731237$ and then $\\lambda=P_z z_{\\rm GeV^{-1}}$. |",
        "| Posterior-prior error scale | The mean fit gives $\\bar p_i\\pm\\sigma_{p_i}$; resampled fits use $p_i=\\bar p_i\\pm s\\sigma_{p_i}$. |",
    ]


def _fit_assessment(result: dict[str, Any], *, language: str) -> list[str]:
    failures = np.asarray(result.get("fit_failures", []), dtype=float)
    chi2 = np.asarray(result.get("scheme_fit_chi2_dof", []), dtype=float)
    weights = np.asarray(result.get("scheme_weights", []), dtype=float)
    roughness = np.asarray(result.get("scheme_roughness", []), dtype=float)
    labels = list(result.get("scheme_labels", []))
    lines: list[str] = []

    if failures.size:
        lines.append(
            f"- 拟合失败次数：共 {int(np.sum(failures))} 次，分布在 {failures.size} 个 scheme 中。"
            if language == "zh"
            else f"- Fit failures: {int(np.sum(failures))} total across {failures.size} scheme(s)."
        )
    if chi2.size and np.any(np.isfinite(chi2)):
        finite = chi2[np.isfinite(chi2)]
        lines.append(
            f"- $\\chi^2/{{\\rm dof}}$ 范围：{_fmt(np.min(finite))} 到 {_fmt(np.max(finite))}。"
            if language == "zh"
            else f"- $\\chi^2/{{\\rm dof}}$ range: {_fmt(np.min(finite))} to {_fmt(np.max(finite))}."
        )
    if weights.size:
        best = int(np.argmax(weights))
        label = labels[best] if best < len(labels) else str(best)
        if language == "zh":
            lines.append(
                f"- Model averaging 使用 $S_s=\\chi_s^2/{{\\rm dof}}+w_{{\\rm rough}}R_s+100N_s^{{\\rm fail}}$ "
                f"与 $W_s=\\exp[-(S_s-S_{{\\rm min}})/2]/\\sum_t\\exp[-(S_t-S_{{\\rm min}})/2]$。"
                f"最高权重 scheme 为 {label}，权重 {_fmt(weights[best])}。"
            )
        else:
            lines.append(
                f"- Model averaging uses $S_s=\\chi_s^2/{{\\rm dof}}+w_{{\\rm rough}}R_s+100N_s^{{\\rm fail}}$ "
                f"and $W_s=\\exp[-(S_s-S_{{\\rm min}})/2]/\\sum_t\\exp[-(S_t-S_{{\\rm min}})/2]$. "
                f"The highest-weight scheme is {label} with weight {_fmt(weights[best])}."
            )
    if roughness.size and np.any(np.isfinite(roughness)):
        lines.append(
            f"- 傅立叶结果粗糙度分数 $R_s$ 范围：{_fmt(np.nanmin(roughness))} 到 {_fmt(np.nanmax(roughness))}。"
            if language == "zh"
            else f"- Fourier roughness score $R_s$ range: {_fmt(np.nanmin(roughness))} to {_fmt(np.nanmax(roughness))}."
        )
    return lines or (["- 无可用拟合诊断。"] if language == "zh" else ["- Fit diagnostics were not available."])


def _scheme_table(result: dict[str, Any], *, language: str) -> list[str]:
    schemes = list(result.get("scheme_results", []))
    labels = list(result.get("scheme_labels", []))
    weights = np.asarray(result.get("scheme_weights", []), dtype=float)
    chi2 = np.asarray(result.get("scheme_fit_chi2_dof", []), dtype=float)
    roughness = np.asarray(result.get("scheme_roughness", []), dtype=float)
    failures = np.asarray(result.get("fit_failures", []), dtype=float)
    if not schemes:
        return ["没有可用的 scheme 诊断。" if language == "zh" else "Scheme diagnostics were not available."]

    header = (
        "| # | label | weight | $\\chi^2/{\\rm dof}$ | roughness | failures | fit range | $z_{\\rm ext}^{\\rm max}$ | smooth |"
        if language == "en"
        else "| # | 标签 | 权重 | $\\chi^2/{\\rm dof}$ | 粗糙度 | 失败次数 | 拟合区间 | $z_{\\rm ext}^{\\rm max}$ | 平滑方式 |"
    )
    lines = [header, "|---:|---|---:|---:|---:|---:|---|---:|---|"]
    for idx, scheme in enumerate(schemes):
        label = labels[idx] if idx < len(labels) else scheme.get("label", str(idx))
        fit_range = scheme.get("fit_range")
        fit_range_text = "n/a" if fit_range is None else _format_fit_range(fit_range, language=language)
        lines.append(
            "| "
            f"{idx} | {label} | "
            f"{_fmt(weights[idx]) if idx < weights.size else 'n/a'} | "
            f"{_fmt(chi2[idx]) if idx < chi2.size else 'n/a'} | "
            f"{_fmt(roughness[idx]) if idx < roughness.size else 'n/a'} | "
            f"{int(failures[idx]) if idx < failures.size else 'n/a'} | "
            f"{fit_range_text} | "
            f"{_fmt(scheme.get('z_ext_max'))} | "
            f"`{scheme.get('smooth', 'n/a')}` |"
        )
    return lines


def _smooth_explanation(result: dict[str, Any], *, language: str) -> list[str]:
    schemes = list(result.get("scheme_results", []))
    best_idx = int(result.get("best_scheme_index", 0) or 0)
    smooth = "linear"
    if schemes and 0 <= best_idx < len(schemes):
        smooth = str(schemes[best_idx].get("smooth", smooth)).lower()
    if smooth == "linear":
        if language == "zh":
            return [
                "平滑方式 `linear` 表示",
                "$$",
                "h_{\\rm ext}(z)=[1-w(z)]h_{\\rm data}(z)+w(z)h_{\\rm fit}(z),\\quad "
                "w(z)=\\frac{z-z_{\\rm min}}{z_{\\rm max}-z_{\\rm min}}.",
                "$$",
            ]
        return [
            "`linear` smoothing means",
            "$$",
            "h_{\\rm ext}(z)=[1-w(z)]h_{\\rm data}(z)+w(z)h_{\\rm fit}(z),\\quad "
            "w(z)=\\frac{z-z_{\\rm min}}{z_{\\rm max}-z_{\\rm min}}.",
            "$$",
        ]
    return ["平滑方式 `none` 表示直接切换到拟合尾部。"] if language == "zh" else ["`none` smoothing switches directly to the fitted tail."]


def _figure_block(artifacts: dict[str, Any], *, language: str) -> list[str]:
    labels = {
        "fourier_plot": ("Fourier result", "傅立叶变换结果图"),
        "extension_plot_re": ("Real-part extension", "实部长程外推图"),
        "extension_plot_im": ("Imaginary-part extension", "虚部长程外推图"),
    }
    lines = ["## Figures and Visual Assessment" if language == "en" else "## 图像与可视化评估"]
    for key, (en_title, zh_title) in labels.items():
        title = zh_title if language == "zh" else en_title
        pdf_value = artifacts.get(key)
        image_value = artifacts.get(f"{key}_image") or pdf_value
        lines.extend(["", f"### {title}"])
        if image_value:
            lines.extend(["", f"![{title}]({image_value})"])
            if pdf_value:
                lines.append(f"[PDF artifact]({pdf_value})")
        else:
            lines.append("未生成。" if language == "zh" else "Not available.")
    return lines


def _settings_table(
    *,
    result: dict[str, Any],
    observable: str,
    observable_text: str,
    method: str,
    order: str,
    fit_range_text: str,
    z_ext_max: Any,
    k_grid: np.ndarray,
    language: str,
) -> list[str]:
    rows = [
        ("Observable", f"`{observable}` ({observable_text})"),
        ("Tail method/order", f"`{method}` / `{order}`"),
        ("Active fitted component", f"`{result.get('part', 'both')}`"),
        ("Resampling mode", f"`{result.get('resample_mode', 'not recorded')}`"),
        ("Coordinate unit", f"{_display_unit(result.get('coord_unit', 'not recorded'))}; fit unit {_display_unit(result.get('fit_coord_unit', 'not recorded'))}"),
        ("Fit lower bound", f"$\\Lambda_0\\ge {_fmt(result.get('Lambda0'))}$"),
        ("Weak-prior scale", f"$p_i=\\bar p_i\\pm s\\sigma_{{p_i}}$ with $s={_fmt(result.get('posterior_prior_error_scale'))}$"),
        ("Best fit range", fit_range_text),
        ("Extension endpoint", f"$z_{{\\rm ext}}^{{\\rm max}}={_fmt(z_ext_max)}$"),
        ("Fourier grid", _format_grid(k_grid, language=language)),
    ]
    if language == "zh":
        rows = [
            ("物理量", f"`{observable}`（{observable_text}）"),
            ("长程外推 method/order", f"`{method}` / `{order}`"),
            ("参与拟合的分量", f"`{result.get('part', 'both')}`"),
            ("重采样模式", f"`{result.get('resample_mode', 'not recorded')}`"),
            ("坐标单位", f"{_display_unit(result.get('coord_unit', 'not recorded'))}；拟合单位 {_display_unit(result.get('fit_coord_unit', 'not recorded'))}"),
            ("拟合下界", f"$\\Lambda_0\\ge {_fmt(result.get('Lambda0'))}$"),
            ("弱先验尺度", f"$p_i=\\bar p_i\\pm s\\sigma_{{p_i}}$，其中 $s={_fmt(result.get('posterior_prior_error_scale'))}$"),
            ("最优拟合区间", fit_range_text),
            ("外推终点", f"$z_{{\\rm ext}}^{{\\rm max}}={_fmt(z_ext_max)}$"),
            ("傅立叶网格", _format_grid(k_grid, language=language)),
        ]
    header = "| Quantity | Value |" if language == "en" else "| 条目 | 数值或设置 |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _npz_help(*, language: str) -> list[str]:
    if language == "zh":
        return [
            "## 如何读取 NPZ 输出",
            "`fourier_result.npz` 保存傅立叶变换后的样本、均值、统计误差、系统误差和 scheme 权重；`fourier_fit_info.npz` 保存长程拟合参数与拟合质量。",
            "```python",
            "import numpy as np",
            "from lamet_agent.core.data import EnsembleData",
            "data, extra = EnsembleData.load_npz('fourier_result.npz')",
            "raw = np.load('fourier_result.npz', allow_pickle=False)",
            "print(data.values.shape, raw.files)",
            "```",
        ]
    return [
        "## Reading the NPZ Outputs",
        "`fourier_result.npz` stores transformed samples, means, statistical/systematic errors, and scheme weights; `fourier_fit_info.npz` stores tail-fit parameters and quality diagnostics.",
        "```python",
        "import numpy as np",
        "from lamet_agent.core.data import EnsembleData",
        "data, extra = EnsembleData.load_npz('fourier_result.npz')",
        "raw = np.load('fourier_result.npz', allow_pickle=False)",
        "print(data.values.shape, raw.files)",
        "```",
    ]


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    rows = [
        ("Fourier NPZ", artifacts.get("fourier_npz")),
        ("Fit-info NPZ", artifacts.get("fit_info_npz")),
        ("Fourier PDF", artifacts.get("fourier_plot")),
        ("Fourier PNG", artifacts.get("fourier_plot_image")),
        ("Real extension PDF", artifacts.get("extension_plot_re")),
        ("Real extension PNG", artifacts.get("extension_plot_re_image")),
        ("Imaginary extension PDF", artifacts.get("extension_plot_im")),
        ("Imaginary extension PNG", artifacts.get("extension_plot_im_image")),
    ]
    header = "| Artifact | Path |" if language == "en" else "| 文件 | 路径 |"
    lines = [header, "|---|---|"]
    for name, value in rows:
        lines.append(f"| {name} | `{value}` |" if value else f"| {name} | not available |")
    return lines


def build_fourier_report_markdown(
    *,
    result: dict[str, Any],
    summary: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    language: str = "en",
) -> str:
    summary = summary or {}
    artifacts = artifacts or {}
    observable = str(result.get("observable", ""))
    observable_text = OBSERVABLE_TEXT.get(observable, observable or "not recorded")
    method = str(result.get("method", "not recorded"))
    order = str(result.get("order", "not recorded"))
    k_grid = np.asarray(result.get("k_grid", []), dtype=float)
    schemes = list(result.get("scheme_results", []))
    best_idx = int(result.get("best_scheme_index", 0) or 0)
    best_scheme = schemes[best_idx] if schemes and 0 <= best_idx < len(schemes) else {}
    fit_range_text = _format_fit_range(best_scheme.get("fit_range"), language=language)
    z_ext_max = best_scheme.get("z_ext_max", "not available")

    if language == "zh":
        title = "# 傅立叶变换分析报告"
        abstract = f"本报告总结 `{observable}`（{observable_text}）的傅立叶变换分析，长程外推采用 `{method}` / `{order}`。"
        transform_text = (
            "$$\n"
            "q(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}e^{ix\\lambda}h(\\lambda),\\qquad \\lambda=P_z z.\n"
            "$$"
        )
        lines = [
            title,
            "",
            "## 摘要",
            abstract,
            "",
            "## 分析设置",
            *_settings_table(result=result, observable=observable, observable_text=observable_text, method=method, order=order, fit_range_text=fit_range_text, z_ext_max=z_ext_max, k_grid=k_grid, language="zh"),
            "",
            "### 条目解释",
            *_field_definitions(language="zh"),
            "",
            "## 长程外推形式",
            _tail_formula_text(result, language="zh"),
            "",
            "## 傅立叶变换方法",
            transform_text,
            "",
            "## 拟合质量与 Scheme 诊断",
            *_fit_assessment(result, language="zh"),
            "",
            *_smooth_explanation(result, language="zh"),
            "",
            *_scheme_table(result, language="zh"),
            "",
            *_figure_block(artifacts, language="zh"),
            "",
            "## 数值结果预览",
            f"实部中心值预览：`{_fmt_list(summary.get('ft_re_mean', result.get('ft_re_mean', [])))}`；虚部中心值预览：`{_fmt_list(summary.get('ft_im_mean', result.get('ft_im_mean', [])))}`。",
            "",
            "## 输出文件",
            *_outputs_table(artifacts, language="zh"),
            "",
            *_npz_help(language="zh"),
        ]
    else:
        title = "# Fourier Transform Analysis Report"
        abstract = f"This report summarizes the Fourier transform for `{observable}` ({observable_text}) using `{method}` / `{order}` large-distance extrapolation."
        transform_text = (
            "$$\n"
            "q(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}e^{ix\\lambda}h(\\lambda),\\qquad \\lambda=P_z z.\n"
            "$$"
        )
        lines = [
            title,
            "",
            "## Abstract",
            abstract,
            "",
            "## Analysis Setup",
            *_settings_table(result=result, observable=observable, observable_text=observable_text, method=method, order=order, fit_range_text=fit_range_text, z_ext_max=z_ext_max, k_grid=k_grid, language="en"),
            "",
            "### Field Definitions",
            *_field_definitions(language="en"),
            "",
            "## Large-Distance Extrapolation",
            _tail_formula_text(result, language="en"),
            "",
            "## Fourier Transform Method",
            transform_text,
            "",
            "## Fit Quality and Scheme Diagnostics",
            *_fit_assessment(result, language="en"),
            "",
            *_smooth_explanation(result, language="en"),
            "",
            *_scheme_table(result, language="en"),
            "",
            *_figure_block(artifacts, language="en"),
            "",
            "## Numerical Result Preview",
            f"Real mean preview: `{_fmt_list(summary.get('ft_re_mean', result.get('ft_re_mean', [])))}`; imaginary mean preview: `{_fmt_list(summary.get('ft_im_mean', result.get('ft_im_mean', [])))}`.",
            "",
            "## Output Artifacts",
            *_outputs_table(artifacts, language="en"),
            "",
            *_npz_help(language="en"),
        ]
    return "\n".join(lines) + "\n"


def write_fourier_report(
    *,
    result: dict[str, Any],
    summary: dict[str, Any] | None,
    artifacts: dict[str, Any] | None,
    path: str | Path,
) -> dict[str, Path]:
    """Write English and Chinese Fourier reports and return their paths."""
    output = Path(path)
    cn_output = _cn_report_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cn_output.parent.mkdir(parents=True, exist_ok=True)
    report_artifacts = _markdown_artifacts(artifacts, base_dir=output.parent)
    output.write_text(
        build_fourier_report_markdown(result=result, summary=summary, artifacts=report_artifacts, language="en"),
        encoding="utf-8",
    )
    cn_output.write_text(
        build_fourier_report_markdown(result=result, summary=summary, artifacts=report_artifacts, language="zh"),
        encoding="utf-8",
    )
    return {"en": output, "zh": cn_output}
