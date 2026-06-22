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

FOURIER_ARTIFACT_DESCRIPTIONS = {
    "fourier_artifact": ("Fourier result samples and diagnostics", "傅立叶变换后的样本、均值、误差和 scheme 权重"),
    "fit_info_artifact": ("Tail-fit parameters and fit-quality diagnostics", "长程外推拟合参数和拟合质量诊断"),
    "fourier_plot": ("PDF plot of the Fourier-space result", "傅立叶变换结果 PDF 图"),
    "fourier_plot_image": ("PNG companion for Markdown embedding", "供 Markdown 嵌入的傅立叶结果 PNG 图"),
    "extension_plot_re": ("PDF plot of real-part extension quality", "实部长程外推质量 PDF 图"),
    "extension_plot_re_image": ("PNG companion for real-part extension quality", "供 Markdown 嵌入的实部长程外推 PNG 图"),
    "extension_plot_im": ("PDF plot of imaginary-part extension quality", "虚部长程外推质量 PDF 图"),
    "extension_plot_im_image": ("PNG companion for imaginary-part extension quality", "供 Markdown 嵌入的虚部长程外推 PNG 图"),
}

FOURIER_ARTIFACT_ORDER = (
    "fourier_artifact",
    "fit_info_artifact",
    "fourier_plot",
    "fourier_plot_image",
    "extension_plot_re",
    "extension_plot_re_image",
    "extension_plot_im",
    "extension_plot_im_image",
)


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


def _format_grid(y_grid: np.ndarray, *, language: str) -> str:
    if y_grid.size == 0:
        return "未记录" if language == "zh" else "not recorded"
    if y_grid.size == 1:
        return f"one point at $x={_fmt(y_grid[0])}$"
    diffs = np.diff(y_grid)
    if np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        if language == "zh":
            return f"从 $x={_fmt(y_grid[0])}$ 到 $x={_fmt(y_grid[-1])}$，每隔 $\\Delta x={_fmt(diffs[0])}$ 取一个点，共 {y_grid.size} 个点"
        return f"from $x={_fmt(y_grid[0])}$ to $x={_fmt(y_grid[-1])}$ with spacing $\\Delta x={_fmt(diffs[0])}$, for {y_grid.size} points"
    if language == "zh":
        return f"非均匀网格，共 {y_grid.size} 个点；预览 `{_fmt_list(y_grid)}`"
    return f"nonuniform grid with {y_grid.size} points; preview `{_fmt_list(y_grid)}`"


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
        "fourier_artifact",
        "fit_info_artifact",
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


def _projection_text(result: dict[str, Any], *, language: str) -> list[str]:
    part = str(result.get("part", "both")).lower()
    scale = float(result.get("output_scale", 1.0))
    if language == "zh":
        intro = (
            f"本次设置为 `part={part}` 且 `output_scale={_fmt(scale)}`。"
            "在常用的扩展分布约定"
            "$q_{\\rm ext}(x)=q(x)$（$x>0$）和 $q_{\\rm ext}(-x)=-\\bar q(x)$ 下，"
            "坐标空间矩阵元满足"
            "$h(\\lambda)=\\int dx\\,e^{ix\\lambda}q_{\\rm ext}(x)$。"
        )
        if part == "both":
            meaning = (
                "因此 `both` 使用完整复矩阵元，目标是重建完整的扩展 quasi 分布 "
                "$q_{\\rm ext}(x)$。若 `output_scale=1`，这就是未额外归一化的完整 Fourier 结果；"
                "若使用其它缩放因子，则只是整体改变输出归一化，并不自动变成新的投影物理量。"
            )
        elif part == "re":
            meaning = (
                "实部对应余弦投影，即"
                "$\\mathrm{FT}[\\mathrm{Re}\\,h]=[q_{\\rm ext}(x)+q_{\\rm ext}(-x)]/2$。"
                "对 $x>0$，这等于 $[q(x)-\\bar q(x)]/2$。"
            )
            if np.isclose(scale, 2.0):
                meaning += " 因此 `part=re, output_scale=2` 给出 valence 组合 $q(x)-\\bar q(x)$。"
            elif np.isclose(scale, 1.0):
                meaning += " 因此 `output_scale=1` 给出 valence 组合的一半。"
            else:
                meaning += f" 当前缩放因子 {_fmt(scale)} 给出按该因子归一化后的实部投影。"
        elif part == "im":
            meaning = (
                "虚部对应正弦/反对称投影，即"
                "$\\mathrm{FT}[\\mathrm{Im}\\,h]$ 隔离与 $[q_{\\rm ext}(x)-q_{\\rm ext}(-x)]/2$ 对应的组合；"
                "对 $x>0$，在符号约定一致时对应 $[q(x)+\\bar q(x)]/2$。"
            )
            if np.isclose(scale, 2.0):
                meaning += " 因此 `part=im, output_scale=2` 对应 $q(x)+\\bar q(x)$ 型组合，整体符号仍取决于虚部和 `im_flip_for_ft` 约定。"
            elif np.isclose(scale, 1.0):
                meaning += " 因此 `output_scale=1` 给出该组合的一半。"
            else:
                meaning += f" 当前缩放因子 {_fmt(scale)} 给出按该因子归一化后的虚部投影。"
        else:
            meaning = "该 `part` 设置未识别，报告只记录数值缩放，不赋予额外物理投影解释。"
        return ["## Part/Scale 物理解释", intro, "", meaning]

    intro = (
        f"This run uses `part={part}` and `output_scale={_fmt(scale)}`. "
        "With the common extended-distribution convention "
        "$q_{\\rm ext}(x)=q(x)$ for $x>0$ and $q_{\\rm ext}(-x)=-\\bar q(x)$, "
        "the coordinate-space matrix element obeys "
        "$h(\\lambda)=\\int dx\\,e^{ix\\lambda}q_{\\rm ext}(x)$."
    )
    if part == "both":
        meaning = (
            "`both` uses the full complex matrix element and reconstructs the full extended quasi-distribution "
            "$q_{\\rm ext}(x)$. With `output_scale=1`, this is the unscaled full Fourier result; other scale values "
            "change only the overall normalization and do not define a new projection by themselves."
        )
    elif part == "re":
        meaning = (
            "The real part gives the cosine projection "
            "$\\mathrm{FT}[\\mathrm{Re}\\,h]=[q_{\\rm ext}(x)+q_{\\rm ext}(-x)]/2$, "
            "which equals $[q(x)-\\bar q(x)]/2$ for $x>0$."
        )
        if np.isclose(scale, 2.0):
            meaning += " Therefore `part=re, output_scale=2` gives the valence combination $q(x)-\\bar q(x)$."
        elif np.isclose(scale, 1.0):
            meaning += " Therefore `output_scale=1` gives one half of the valence combination."
        else:
            meaning += f" The current scale {_fmt(scale)} returns this real-part projection with that overall normalization."
    elif part == "im":
        meaning = (
            "The imaginary part gives the sine/antisymmetric projection associated with "
            "$[q_{\\rm ext}(x)-q_{\\rm ext}(-x)]/2$, which corresponds to $[q(x)+\\bar q(x)]/2$ for $x>0$ "
            "when the sign convention is aligned."
        )
        if np.isclose(scale, 2.0):
            meaning += " Therefore `part=im, output_scale=2` corresponds to a $q(x)+\\bar q(x)$-type combination, with the overall sign set by the imaginary-part and `im_flip_for_ft` convention."
        elif np.isclose(scale, 1.0):
            meaning += " Therefore `output_scale=1` gives one half of that combination."
        else:
            meaning += f" The current scale {_fmt(scale)} returns this imaginary-part projection with that overall normalization."
    else:
        meaning = "This `part` setting is not recognized, so only the numerical output scale is reported."
    return ["## Part/Scale Physical Interpretation", intro, "", meaning]


def _fit_assessment(result: dict[str, Any], *, language: str) -> list[str]:
    failures = np.asarray(result.get("fit_failures", []), dtype=float)
    chi2 = np.asarray(result.get("scheme_fit_chi2_dof", []), dtype=float)
    weights = np.asarray(result.get("scheme_weights", []), dtype=float)
    roughness = np.asarray(result.get("scheme_roughness", []), dtype=float)
    labels = list(result.get("scheme_labels", []))
    selection_mode = str(result.get("selection_mode", "model_average"))
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
    if selection_mode == "sample_average_best_scheme":
        candidate_labels = list(result.get("candidate_scheme_labels", []))
        candidate_weights = np.asarray(result.get("candidate_scheme_weights", []), dtype=float)
        candidate_chi2 = np.asarray(result.get("candidate_scheme_fit_chi2_dof", []), dtype=float)
        selected = int(result.get("selected_candidate_index", 0) or 0)
        selected_label = str(result.get("selected_candidate_label", candidate_labels[selected] if selected < len(candidate_labels) else selected))
        if language == "zh":
            lines.append(
                f"- `model_average=false` 时，代码先用 sample-average 矩阵元扫描所有候选 scheme，"
                f"按 $\\chi_s^2/{{\\rm dof}}$ 选择拟合质量最好的窗口；随后只对选中的 scheme 跑全部重采样样本。"
                f"本次选中 {selected_label}。"
            )
        else:
            lines.append(
                f"- With `model_average=false`, the code first scans all candidate schemes using the sample-average matrix element, "
                f"selects the window with the best $\\chi_s^2/{{\\rm dof}}$, and then runs all resampled Fourier transforms only for that scheme. "
                f"The selected scheme is {selected_label}."
            )
        if candidate_weights.size:
            best = int(np.argmax(candidate_weights))
            label = candidate_labels[best] if best < len(candidate_labels) else str(best)
            lines.append(
                f"- 候选 scheme 权重是一热选择；{label} 的权重为 {_fmt(candidate_weights[best])}，其它候选权重为 0。"
                if language == "zh"
                else f"- Candidate-scheme weights are one-hot; {label} has weight {_fmt(candidate_weights[best])}, and all other candidates have weight 0."
            )
        if candidate_chi2.size and np.any(np.isfinite(candidate_chi2)):
            finite = candidate_chi2[np.isfinite(candidate_chi2)]
            lines.append(
                f"- 候选 scheme 的 $\\chi^2/{{\\rm dof}}$ 范围：{_fmt(np.min(finite))} 到 {_fmt(np.max(finite))}。"
                if language == "zh"
                else f"- Candidate $\\chi^2/{{\\rm dof}}$ range: {_fmt(np.min(finite))} to {_fmt(np.max(finite))}."
            )
    elif weights.size:
        best = int(np.argmax(weights))
        label = labels[best] if best < len(labels) else str(best)
        if language == "zh":
            lines.append(
                f"- Model averaging 对每个 scheme $s$ 定义综合分数 "
                f"$S_s=\\chi_s^2/{{\\rm dof}}+w_{{\\rm rough}}R_s+100N_s^{{\\rm fail}}$，其中 "
                f"$\\chi_s^2/{{\\rm dof}}$ 衡量长程拟合质量，$R_s$ 惩罚傅立叶结果的振荡粗糙度，"
                f"$w_{{\\rm rough}}$ 是粗糙度权重，$N_s^{{\\rm fail}}$ 是该 scheme 下重采样拟合失败次数。"
                f"权重按 $W_s=\\exp[-(S_s-S_{{\\rm min}})/2]/\\sum_t\\exp[-(S_t-S_{{\\rm min}})/2]$ 归一化。"
                f"因此分数越低的 scheme 越优先；拟合更好、曲线更平滑、失败次数更少会得到更大权重。"
                f"本次最高权重 scheme 为 {label}，权重 {_fmt(weights[best])}。"
            )
        else:
            lines.append(
                f"- Model averaging assigns each scheme $s$ the score "
                f"$S_s=\\chi_s^2/{{\\rm dof}}+w_{{\\rm rough}}R_s+100N_s^{{\\rm fail}}$, where "
                f"$\\chi_s^2/{{\\rm dof}}$ measures tail-fit quality, $R_s$ penalizes oscillatory Fourier roughness, "
                f"$w_{{\\rm rough}}$ is the roughness weight, and $N_s^{{\\rm fail}}$ is the number of failed resampled fits. "
                f"Weights are normalized as $W_s=\\exp[-(S_s-S_{{\\rm min}})/2]/\\sum_t\\exp[-(S_t-S_{{\\rm min}})/2]$. "
                f"Lower scores are preferred: better fits, smoother Fourier curves, and fewer failures get larger weights. "
                f"The highest-weight scheme is {label} with weight {_fmt(weights[best])}."
            )
    if selection_mode != "sample_average_best_scheme" and roughness.size and np.any(np.isfinite(roughness)):
        if language == "zh":
            lines.append(
                f"- 傅立叶结果粗糙度分数 $R_s$ 是对输出 $x$ 空间曲线局部二阶差分的归一化惩罚，"
                f"用来压低由外推区间选择造成的高频抖动；$R_s$ 越小表示 Fourier 曲线越平滑。"
                f"本次 $R_s$ 范围为 {_fmt(np.nanmin(roughness))} 到 {_fmt(np.nanmax(roughness))}。"
            )
        else:
            lines.append(
                f"- The Fourier roughness score $R_s$ is a normalized penalty on local second differences of the output curve in $x$ space. "
                f"It suppresses high-frequency oscillations induced by the extrapolation-window choice; smaller $R_s$ means a smoother Fourier curve. "
                f"In this run, $R_s$ ranges from {_fmt(np.nanmin(roughness))} to {_fmt(np.nanmax(roughness))}."
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
    y_grid: np.ndarray,
    language: str,
) -> list[str]:
    try:
        z_ext_text = f"$z_{{\\rm ext}}^{{\\rm max}}={_fmt(float(z_ext_max))}$"
    except (TypeError, ValueError):
        z_ext_text = str(z_ext_max)
    rows = [
        ("Observable", f"`{observable}` ({observable_text})"),
        ("Tail method/order", f"`{method}` / `{order}`"),
        ("Active fitted component", f"`{result.get('part', 'both')}`"),
        ("Resampling mode", f"`{result.get('resample_mode', 'not recorded')}`"),
        ("Coordinate unit", f"{_display_unit(result.get('coord_unit', 'not recorded'))}; fit unit {_display_unit(result.get('fit_coord_unit', 'not recorded'))}"),
        ("Fit lower bound", f"$\\Lambda_0\\ge {_fmt(result.get('Lambda0'))}$"),
        ("Weak-prior scale", f"$p_i=\\bar p_i\\pm s\\sigma_{{p_i}}$ with $s={_fmt(result.get('posterior_prior_error_scale'))}$"),
        ("Output scale", f"$q(x)\\rightarrow {_fmt(result.get('output_scale', 1.0))}\\,q(x)$"),
        ("Best fit range", fit_range_text),
        ("Extension endpoint", z_ext_text),
        ("Fourier grid", _format_grid(y_grid, language=language)),
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
            ("输出缩放", f"$q(x)\\rightarrow {_fmt(result.get('output_scale', 1.0))}\\,q(x)$"),
            ("最优拟合区间", fit_range_text),
            ("外推终点", z_ext_text),
            ("傅立叶网格", _format_grid(y_grid, language=language)),
        ]
    header = "| Quantity | Value |" if language == "en" else "| 条目 | 数值或设置 |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _artifact_field_table(kind: str, *, language: str) -> list[str]:
    if kind == "result":
        rows = [
            ("`values`", "Complex model-averaged Fourier samples with dimensions `(resample, x)`.", "复数模型平均 Fourier 样本，维度为 `(resample, x)`。"),
            ("coordinate `x`", "Fourier momentum-fraction grid.", "傅立叶变换后的动量分数网格。"),
            ("attr `resample`", "Resampling mode recorded by `EnsembleData`.", "`EnsembleData` 记录的重采样模式。"),
            ("attr `ft_re_mean` / `ft_im_mean`", "Model-averaged real/imaginary central values.", "模型平均后的实部/虚部中心值。"),
            ("attr `ft_re_stat_sdev` / `ft_im_stat_sdev`", "Statistical standard deviations from bootstrap/jackknife samples.", "由 bootstrap/jackknife 样本给出的统计误差。"),
            ("attr `ft_re_sys_sdev` / `ft_im_sys_sdev`", "Systematic spread from scheme variation.", "由 scheme 变化估计的系统误差。"),
            ("attr `scheme_labels`", "Text labels for scanned extrapolation schemes.", "所有外推 scheme 的文本标签。"),
            ("attr `fit_failures`", "Number of failed resampled tail fits in each scheme.", "每个 scheme 中重采样长程拟合失败的次数。"),
            ("attr `scheme_weights`", "Model-averaging weights $W_s$ for schemes.", "scheme 的模型平均权重 $W_s$。"),
            ("attr `scheme_fit_chi2_dof`", "Scheme-level fit-quality diagnostic $\\chi_s^2/{\\rm dof}$.", "scheme 级别的拟合质量诊断 $\\chi_s^2/{\\rm dof}$。"),
            ("attr `scheme_roughness`", "Fourier roughness score $R_s$ used in model averaging.", "模型平均中使用的 Fourier 粗糙度分数 $R_s$。"),
            ("attr `scheme_scores`", "Total scheme score $S_s$ used to compute $W_s$.", "用于计算 $W_s$ 的总评分 $S_s$。"),
            ("attr `best_scheme_index`", "Index of the highest-weight scheme.", "最高权重 scheme 的编号。"),
            ("attrs `candidate_scheme_*`", "Full candidate-window diagnostics when `model_average=false` selects one scheme before fitting all samples.", "`model_average=false` 时先选一个 scheme 再跑所有样本；这些字段保存完整候选窗口诊断。"),
            ("attr `selection_mode`", "Scheme-selection mode: model averaging or sample-average best-scheme selection.", "scheme 选择模式：模型平均或 sample-average 最优窗口选择。"),
            ("attrs `pz_gev`, `pz_prime_gev`, `a_fm`", "Momentum and lattice-spacing metadata.", "动量和格距元数据。"),
            ("attrs `method`, `order`, `observable`, `part`, `output_scale`", "Physics/formula choices and final output normalization.", "物理公式选择和最终输出归一化。"),
        ]
    else:
        rows = [
            ("`values`", "Fit-parameter samples with dimensions `(resample, scheme, parameter)`.", "拟合参数样本，维度为 `(resample, scheme, parameter)`。"),
            ("coordinates `scheme`, `parameter`", "Scheme labels and fitted parameter names.", "scheme 标签和拟合参数名。"),
            ("attr `fit_params`", "Tail-fit parameters for every scheme and resample.", "每个 scheme 和每个重采样样本的长程拟合参数。"),
            ("attr `fit_param_center` / `fit_param_sdev`", "Sample mean and statistical standard deviation of fit parameters.", "拟合参数的样本平均值和统计误差。"),
            ("attrs `fit_chi2`, `fit_dof`, `fit_q`, `fit_chi2_dof`", "Per-resample fit quality diagnostics.", "每个重采样样本的拟合质量诊断。"),
            ("attrs `fit_chi2_center`, `fit_chi2_dof_center`, `fit_q_center`", "Sample-averaged fit quality diagnostics for each scheme.", "每个 scheme 的样本平均拟合质量诊断。"),
            ("attrs `mean_fit_params`, `mean_fit_chi2`, `mean_fit_dof`, `mean_fit_q`", "Initial sample-average fit results used to seed resampled fits.", "用于初始化重采样拟合的样本平均拟合结果。"),
            ("attrs `scheme_weights`, `scheme_fit_chi2_dof`, `scheme_roughness`, `scheme_scores`", "Scheme scoring diagnostics used in model averaging.", "模型平均中使用的 scheme 评分诊断。"),
            ("attr `best_scheme_index`", "Index of the highest-weight scheme.", "最高权重 scheme 的编号。"),
            ("attrs `candidate_scheme_*`, `selection_mode`", "Candidate diagnostics and selection mode when `model_average=false`.", "`model_average=false` 时的候选窗口诊断和选择模式。"),
        ]
    header = "| Field | Meaning |" if language == "en" else "| 字段 | 含义 |"
    lines = [header, "|---|---|"]
    for field, en, zh in rows:
        lines.append(f"| {field} | {zh if language == 'zh' else en} |")
    return lines


def _artifact_help(*, language: str) -> list[str]:
    if language == "zh":
        return [
            "## 如何读取 NetCDF 输出",
            "`fourier_result.nc` 保存傅立叶变换后的复数样本；`fourier_fit_info.nc` 保存长程拟合参数样本。"
            "两者都可以用 `EnsembleData.from_netcdf` 读取主数组；诊断量保存在 `data.attrs` 中。",
            "```python",
            "from lamet_agent.core.data import EnsembleData",
            "data = EnsembleData.from_netcdf('fourier_result.nc')",
            "print(data.values.shape, data.coords, data.attrs.keys())",
            "```",
            "",
            "### `fourier_result.nc` 字段说明",
            *_artifact_field_table("result", language="zh"),
            "",
            "### `fourier_fit_info.nc` 字段说明",
            *_artifact_field_table("fit_info", language="zh"),
        ]
    return [
        "## Reading the NetCDF Outputs",
        "`fourier_result.nc` stores complex Fourier-transform samples; `fourier_fit_info.nc` stores large-distance fit-parameter samples. "
        "Both files can be read with `EnsembleData.from_netcdf`; diagnostics are stored in `data.attrs`.",
        "```python",
        "from lamet_agent.core.data import EnsembleData",
        "data = EnsembleData.from_netcdf('fourier_result.nc')",
        "print(data.values.shape, data.coords, data.attrs.keys())",
        "```",
        "",
        "### `fourier_result.nc` Field Reference",
        *_artifact_field_table("result", language="en"),
        "",
        "### `fourier_fit_info.nc` Field Reference",
        *_artifact_field_table("fit_info", language="en"),
    ]


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    header = "| File | Description |" if language == "en" else "| 文件名 | 文件描述 |"
    lines = [header, "|---|---|"]
    for key in FOURIER_ARTIFACT_ORDER:
        value = artifacts.get(key)
        if not value:
            continue
        desc = FOURIER_ARTIFACT_DESCRIPTIONS[key][1 if language == "zh" else 0]
        lines.append(f"| `{value}` | {desc} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
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
    y_grid = np.asarray(result.get("y_grid", []), dtype=float)
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
            *_settings_table(result=result, observable=observable, observable_text=observable_text, method=method, order=order, fit_range_text=fit_range_text, z_ext_max=z_ext_max, y_grid=y_grid, language="zh"),
            "",
            "### 条目解释",
            *_field_definitions(language="zh"),
            "",
            *_projection_text(result, language="zh"),
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
            "## 输出文件",
            *_outputs_table(artifacts, language="zh"),
            "",
            *_artifact_help(language="zh"),
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
            *_settings_table(result=result, observable=observable, observable_text=observable_text, method=method, order=order, fit_range_text=fit_range_text, z_ext_max=z_ext_max, y_grid=y_grid, language="en"),
            "",
            "### Field Definitions",
            *_field_definitions(language="en"),
            "",
            *_projection_text(result, language="en"),
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
            "## Output Artifacts",
            *_outputs_table(artifacts, language="en"),
            "",
            *_artifact_help(language="en"),
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


def write_fourier_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
) -> dict[str, Path]:
    """Write one bilingual report summarizing all Fourier jobs in a stage."""
    output = Path(path)
    cn_output = _cn_report_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cn_output.parent.mkdir(parents=True, exist_ok=True)
    first = jobs[0]["result"]
    for language, target in (("en", output), ("zh", cn_output)):
        observable = str(first.get("observable", ""))
        observable_text = OBSERVABLE_TEXT.get(observable, observable or "not recorded")
        method = str(first.get("method", "not recorded"))
        order = str(first.get("order", "not recorded"))
        y_grid = np.asarray(first.get("y_grid", []), dtype=float)
        z_ext_values = []
        for item in jobs:
            result = item["result"]
            schemes = list(result.get("scheme_results", []))
            best_idx = int(result.get("best_scheme_index", 0) or 0)
            if schemes and 0 <= best_idx < len(schemes):
                z_ext_values.append(schemes[best_idx].get("z_ext_max"))
        finite_z_ext = [float(value) for value in z_ext_values if value is not None]
        same_z_ext = bool(finite_z_ext) and np.allclose(finite_z_ext, finite_z_ext[0])
        fit_range_text = "见下方各动量诊断表" if language == "zh" else "see the per-momentum diagnostics below"
        z_ext_max = finite_z_ext[0] if same_z_ext else ("见下方各动量诊断表" if language == "zh" else "see the per-momentum diagnostics below")
        transform_text = (
            "$$\n"
            "q(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}e^{ix\\lambda}h(\\lambda),\\qquad \\lambda=P_z z.\n"
            "$$"
        )
        lines = [
            "# Fourier Transform Stage Report" if language == "en" else "# 傅立叶变换阶段报告",
            "",
            f"This report summarizes all Fourier-transform jobs in this stage for `{observable}` ({observable_text})."
            if language == "en"
            else f"本报告汇总当前 Fourier transform 阶段中 `{observable}`（{observable_text}）的所有动量。",
            "",
            "## Job Summary" if language == "en" else "## Job 汇总",
            "| job | $P_z$ | best scheme | output | plot |"
            if language == "en"
            else "| job | $P_z$ | 最优 scheme | 输出 | 图像 |",
            "|---|---:|---|---|---|",
        ]
        for item in jobs:
            result = item["result"]
            artifacts = _markdown_artifacts(item.get("artifacts", {}), base_dir=target.parent)
            lines.append(
                f"| `{item['job_id']}` | {_fmt(result.get('pz_gev'))} | "
                f"{result.get('best_scheme_label', 'n/a')} | "
                f"{artifacts.get('fourier_artifact', 'n/a')} | "
                f"{artifacts.get('fourier_plot', 'n/a')} |"
            )
        lines.extend(
            [
                "",
                "## Analysis Setup" if language == "en" else "## 分析设置",
                *_settings_table(
                    result=first,
                    observable=observable,
                    observable_text=observable_text,
                    method=method,
                    order=order,
                    fit_range_text=fit_range_text,
                    z_ext_max=z_ext_max,
                    y_grid=y_grid,
                    language=language,
                ),
                "",
                "### Field Definitions" if language == "en" else "### 条目解释",
                *_field_definitions(language=language),
                "",
                *_projection_text(first, language=language),
                "",
                "## Large-Distance Extrapolation" if language == "en" else "## 长程外推形式",
                _tail_formula_text(first, language=language),
                "",
                "## Fourier Transform Method" if language == "en" else "## 傅立叶变换方法",
                transform_text,
                "",
                "## Fit Quality and Scheme Diagnostics" if language == "en" else "## 拟合质量与 Scheme 诊断",
            ]
        )
        if str(first.get("selection_mode", "model_average")) == "sample_average_best_scheme":
            lines.append(
                "`model_average=false` 时，代码先用 sample-average 矩阵元选择一个最优候选窗口，再把该窗口用于所有重采样样本。"
                if language == "zh"
                else "With `model_average=false`, the sample-average matrix element selects one best candidate window, which is then applied to all resampled samples."
            )
        else:
            lines.append(
                "Model averaging 使用 $S_s=\\chi_s^2/{\\rm dof}+w_{\\rm rough}R_s+100N_s^{\\rm fail}$ 给每个 scheme 评分，并用 $W_s=\\exp[-(S_s-S_{\\rm min})/2]/\\sum_t\\exp[-(S_t-S_{\\rm min})/2]$ 归一化权重。分数越低的 scheme 权重越大。"
                if language == "zh"
                else "Model averaging scores each scheme with $S_s=\\chi_s^2/{\\rm dof}+w_{\\rm rough}R_s+100N_s^{\\rm fail}$ and normalizes weights as $W_s=\\exp[-(S_s-S_{\\rm min})/2]/\\sum_t\\exp[-(S_t-S_{\\rm min})/2]$. Lower-scored schemes receive larger weights."
            )
        lines.extend(
            [
                "",
                "| job | $P_z$ | best scheme | best fit range | $\\chi^2/{\\rm dof}$ range | fit failures |"
                if language == "en"
                else "| job | $P_z$ | 最优 scheme | 最优拟合区间 | $\\chi^2/{\\rm dof}$ 范围 | 拟合失败次数 |",
                "|---|---:|---|---|---:|---:|",
            ]
        )
        for item in jobs:
            result = item["result"]
            schemes = list(result.get("scheme_results", []))
            best_idx = int(result.get("best_scheme_index", 0) or 0)
            best_scheme = schemes[best_idx] if schemes and 0 <= best_idx < len(schemes) else {}
            chi2 = np.asarray(result.get("scheme_fit_chi2_dof", []), dtype=float)
            finite = chi2[np.isfinite(chi2)]
            chi_text = "n/a" if finite.size == 0 else f"{_fmt(np.min(finite))} to {_fmt(np.max(finite))}"
            lines.append(
                f"| `{item['job_id']}` | {_fmt(result.get('pz_gev'))} | "
                f"{result.get('best_scheme_label', 'n/a')} | "
                f"{_format_fit_range(best_scheme.get('fit_range'), language=language)} | "
                f"{chi_text} | "
                f"{int(np.sum(np.asarray(result.get('fit_failures', []), dtype=float)))} |"
            )
        lines.extend(["", *_smooth_explanation(first, language=language)])
        for item in jobs:
            result = item["result"]
            lines.extend(
                [
                    "",
                    f"### `{item['job_id']}`: $P_z={_fmt(result.get('pz_gev'))}$ GeV",
                    "",
                    *_scheme_table(result, language=language),
                ]
            )
        lines.append("")
        lines.append("## Figures and Visual Assessment" if language == "en" else "## 图像与可视化评估")
        for item in jobs:
            result = item["result"]
            artifacts = _markdown_artifacts(item.get("artifacts", {}), base_dir=target.parent)
            lines.extend(["", f"### `{item['job_id']}`: $P_z={_fmt(result.get('pz_gev'))}$ GeV"])
            for key, title in (
                ("fourier_plot", "Fourier result" if language == "en" else "傅立叶变换结果图"),
                ("extension_plot_re", "Real-part extension" if language == "en" else "实部长程外推图"),
                ("extension_plot_im", "Imaginary-part extension" if language == "en" else "虚部长程外推图"),
            ):
                image_value = artifacts.get(f"{key}_image") or artifacts.get(key)
                pdf_value = artifacts.get(key)
                lines.append("")
                lines.append(f"#### {title}")
                if image_value:
                    lines.append(f"![{title}]({image_value})")
                    if pdf_value:
                        lines.append(f"[PDF artifact]({pdf_value})")
                else:
                    lines.append("未生成。" if language == "zh" else "Not available.")
        lines.extend(["", "## Output Artifacts" if language == "en" else "## 输出文件"])
        lines.extend(
            [
                "| File | Description |" if language == "en" else "| 文件名 | 文件描述 |",
                "|---|---|",
            ]
        )
        for item in jobs:
            artifacts = _markdown_artifacts(item.get("artifacts", {}), base_dir=target.parent)
            for key in FOURIER_ARTIFACT_ORDER:
                value = artifacts.get(key)
                if value:
                    desc = FOURIER_ARTIFACT_DESCRIPTIONS[key]
                    lines.append(f"| [{Path(value).name}]({value}) | `{item['job_id']}`: {desc[1 if language == 'zh' else 0]} |")
        lines.extend(["", *_artifact_help(language=language)])
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"en": output, "zh": cn_output}
