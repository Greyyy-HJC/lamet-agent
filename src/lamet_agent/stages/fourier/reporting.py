"""Markdown reporting helpers for the Fourier-transform stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.core.reporting import (
    format_report_list as _fmt_list,
    format_report_value as _fmt,
    markdown_artifact_paths,
    resolve_report_target as _report_target,
)
from lamet_agent.core.resampling import sample_mean_and_sdev


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
    "pion_quark_quasi_pdf": "arXiv:2601.12189 Eqs. (2.1)/(2.2)",
    "nucleon_quark_unpolarized_quasi_pdf": "arXiv:2601.12189 Eqs. (2.3)/(2.4)",
    "nucleon_quark_transversity_quasi_pdf": "arXiv:2601.12189 Eqs. (2.5)/(2.6)",
    "meson_quasi_da": "arXiv:2601.12189 Eqs. (2.7)/(2.8)",
    "pion_quark_quasi_gpd": "arXiv:2601.12189 Eqs. (2.9)/(2.10)",
    "nucleon_quark_quasi_gpd": "arXiv:2601.12189 Eqs. (2.11)/(2.12)",
    "nucleon_gluon_quasi_pdf": "arXiv:2601.12189 Appendix F Eqs. (F.6)/(F.7)",
    "pion_gluon_quasi_pdf": "arXiv:2601.12189 Appendix F Eqs. (F.8)/(F.9)",
}

FOURIER_ARTIFACT_DESCRIPTIONS = {
    "fourier_artifact": ("Fourier result samples and diagnostics", "傅立叶变换后的样本、均值、误差和 scheme 权重"),
    "fit_info_artifact": ("Tail-fit parameters and fit-quality diagnostics", "长程外推拟合参数和拟合质量诊断"),
    "fourier_plot": ("PDF plot of the Fourier-space result", "傅立叶变换结果 PDF 图"),
    "fourier_plot_image": ("SVG companion for Markdown embedding", "供 Markdown 嵌入的傅立叶结果 SVG 图"),
    "extension_plot_re": ("PDF plot of real-part extension quality", "实部长程外推质量 PDF 图"),
    "extension_plot_re_image": ("SVG companion for real-part extension quality", "供 Markdown 嵌入的实部长程外推 SVG 图"),
    "extension_plot_im": ("PDF plot of imaginary-part extension quality", "虚部长程外推质量 PDF 图"),
    "extension_plot_im_image": ("SVG companion for imaginary-part extension quality", "供 Markdown 嵌入的虚部长程外推 SVG 图"),
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


def _tail_formula_text(result: dict[str, Any], *, language: str) -> str:
    method = str(result.get("method", "")).upper()
    order = str(result.get("order", "")).upper()
    observable = str(result.get("observable", ""))
    sector = str(result.get("sector", "")).lower()
    psi1_class = str(result.get("psi1_flavor_class", "heavy") or "heavy").lower()
    psi2_class = str(result.get("psi2_flavor_class", "heavy") or "heavy").lower()
    orders = [item.strip().upper() for item in order.split(",") if item.strip()]
    if len(orders) > 1:
        article_lines = []
        implementation_lines = []
        mapping_scope_text = ""
        for item in orders:
            one = dict(result)
            one["order"] = item
            text = _tail_formula_text(one, language=language)
            if language == "zh":
                article_part = text.split("### lamet-agent 实际拟合公式", 1)[0]
                implementation_part = text.split("### lamet-agent 实际拟合公式", 1)[1].split("### 参数对应关系", 1)[0]
                mapping_scope_text = text.split("### 参数对应关系", 1)[1]
            else:
                article_part = text.split("### lamet-agent Implementation", 1)[0]
                implementation_part = text.split("### lamet-agent Implementation", 1)[1].split("### Parameter Correspondence", 1)[0]
                mapping_scope_text = text.split("### Parameter Correspondence", 1)[1]
            article_formula = article_part.split("$$", 2)[1]
            implementation_formula = implementation_part.split("$$", 2)[1]
            article_lines.append(f"$$\n{article_formula}\n$$")
            implementation_lines.append(f"$$\n{implementation_formula}\n$$")
        if language == "zh":
            return "\n\n".join(
                [
                    "### 文献公式",
                    FORMULA_REFERENCES.get(observable, "code-selected LA/NLA formula") + "。",
                    *article_lines,
                    "### lamet-agent 实际拟合公式",
                    "本次分析中代码实际使用的拟合形式为",
                    *implementation_lines,
                    "### 参数对应关系",
                    mapping_scope_text,
                ]
            )
        return "\n\n".join(
            [
                "### Article Formula",
                FORMULA_REFERENCES.get(observable, "code-selected LA/NLA formula") + ".",
                *article_lines,
                "### lamet-agent Implementation",
                "The fit actually used by lamet-agent in this run is",
                *implementation_lines,
                "### Parameter Correspondence",
                mapping_scope_text,
            ]
        )
    reference = FORMULA_REFERENCES.get(observable, "code-selected LA/NLA formula")
    article_tail = r"\exp[-(m+\Lambda_0)|z|]"
    implementation_tail = r"\exp[-(m+\Lambda_0)z]"
    if method == "CG":
        implementation_tail += r"\,z^{-n}"

    if observable == "pion_quark_quasi_gpd":
        if order == "LA":
            article_formula = (
                r"\tilde{h}^{\rm LA}(z,P^z,P'^z)="
                r"\left["
                r"A_1 e^{i\phi_1\,{\rm sign}(z)} e^{-i z P^z}"
                r"+A_3 e^{i\phi_3\,{\rm sign}(z)} e^{i z P'^z}"
                r"+A_2 e^{i\phi_2\,{\rm sign}(z)}"
                r"+\tilde{A}_2 e^{i\tilde{\phi}_2\,{\rm sign}(z)} e^{-i(P^z-P'^z)z}"
                r"\right]"
                + article_tail
                + r"."
            )
            implementation_formula = (
                r"\tilde{h}^{\rm LA}_{\rm agent}(z>0;P^z,P'^z)="
                r"\left["
                r"A_1 e^{i(\phi_1-P^z z)}"
                r"+A_3 e^{i(\phi_3+P'^z z)}"
                r"+A_2 e^{i\phi_2}"
                r"+\tilde{A}_2 e^{i(\tilde{\phi}_2-(P^z-P'^z)z)}"
                r"\right]"
                + implementation_tail
                + r"."
            )
            mapping_lines = [
                "- $A_1,\\phi_1$ correspond to the incoming-momentum oscillatory term $e^{-i z P^z}$."
                if language == "en"
                else "- $A_1,\\phi_1$ 对应入射动量振荡项 $e^{-i z P^z}$。",
                "- $A_3,\\phi_3$ correspond to the outgoing-momentum oscillatory term $e^{+i z P'^z}$."
                if language == "en"
                else "- $A_3,\\phi_3$ 对应出射动量振荡项 $e^{+i z P'^z}$。",
                "- $A_2,\\phi_2$ correspond to the non-oscillatory central term."
                if language == "en"
                else "- $A_2,\\phi_2$ 对应不带额外动量相位的中心项。",
                "- $\\tilde A_2,\\tilde\\phi_2$ correspond to the momentum-transfer term $e^{-i(P^z-P'^z)z}$."
                if language == "en"
                else "- $\\tilde A_2,\\tilde\\phi_2$ 对应动量转移项 $e^{-i(P^z-P'^z)z}$。",
                "- $m$ is the fitted non-negative offset, so the effective decay rate is $m+\\Lambda_0$."
                if language == "en"
                else "- $m$ 是非负拟合偏移量，因此有效衰减率为 $m+\\Lambda_0$。",
                "- `Lambda0_gev` is the fixed offset $\\Lambda_0$ in the reparameterized decay rate, not a hard bound on a fitted $\\Lambda$."
                if language == "en"
                else "- `Lambda0_gev` 是重参数化衰减率中的固定偏移 $\\Lambda_0$，不是对拟合参数 $\\Lambda$ 的硬边界。",
            ]
        else:
            article_formula = (
                r"\tilde{h}^{\rm NLA}(z,P^z,P'^z)="
                r"\left["
                r"A_1 e^{i\phi_1\,{\rm sign}(z)} e^{-i z P^z}"
                r"+A_3 e^{i\phi_3\,{\rm sign}(z)} e^{i z P'^z}"
                r"+A_2 e^{i\phi_2\,{\rm sign}(z)}"
                r"+\tilde{A}_2 e^{i\tilde{\phi}_2\,{\rm sign}(z)} e^{-i(P^z-P'^z)z}"
                r"+\frac{A'_1}{|z|} e^{i\phi'_1\,{\rm sign}(z)} e^{-i z P^z}"
                r"+\frac{A'_3}{|z|} e^{i\phi'_3\,{\rm sign}(z)} e^{i z P'^z}"
                r"+\frac{A'_2}{|z|} e^{i\phi'_2\,{\rm sign}(z)}"
                r"+\frac{\tilde{A}'_2}{|z|} e^{i\tilde{\phi}'_2\,{\rm sign}(z)} e^{-i(P^z-P'^z)z}"
                r"\right]"
                + article_tail
                + r"."
            )
            implementation_formula = (
                r"\tilde{h}^{\rm NLA}_{\rm agent}(z>0;P^z,P'^z)="
                r"\left["
                r"A_1 e^{i(\phi_1-P^z z)}"
                r"+A_3 e^{i(\phi_3+P'^z z)}"
                r"+A_2 e^{i\phi_2}"
                r"+\tilde{A}_2 e^{i(\tilde{\phi}_2-(P^z-P'^z)z)}"
                r"+\frac{A'_1}{z} e^{i(\phi'_1-P^z z)}"
                r"+\frac{A'_3}{z} e^{i(\phi'_3+P'^z z)}"
                r"+\frac{A'_2}{z} e^{i\phi'_2}"
                r"+\frac{\tilde{A}'_2}{z} e^{i(\tilde{\phi}'_2-(P^z-P'^z)z)}"
                r"\right]"
                + implementation_tail
                + r"."
            )
            mapping_lines = [
                "- $A_1,\\phi_1$ and $A'_1,\\phi'_1$ correspond to the incoming-momentum terms proportional to $e^{-i z P^z}$."
                if language == "en"
                else "- $A_1,\\phi_1$ 以及 $A'_1,\\phi'_1$ 对应与 $e^{-i z P^z}$ 相乘的入射动量项。",
                "- $A_3,\\phi_3$ and $A'_3,\\phi'_3$ correspond to the outgoing-momentum terms proportional to $e^{+i z P'^z}$."
                if language == "en"
                else "- $A_3,\\phi_3$ 以及 $A'_3,\\phi'_3$ 对应与 $e^{+i z P'^z}$ 相乘的出射动量项。",
                "- $A_2,\\phi_2$ and $A'_2,\\phi'_2$ correspond to the central non-oscillatory LA/NLA terms."
                if language == "en"
                else "- $A_2,\\phi_2$ 以及 $A'_2,\\phi'_2$ 对应中心的 LA/NLA 非振荡项。",
                "- $\\tilde A_2,\\tilde\\phi_2$ and $\\tilde A'_2,\\tilde\\phi'_2$ correspond to the momentum-transfer terms proportional to $e^{-i(P^z-P'^z)z}$."
                if language == "en"
                else "- $\\tilde A_2,\\tilde\\phi_2$ 以及 $\\tilde A'_2,\\tilde\\phi'_2$ 对应与 $e^{-i(P^z-P'^z)z}$ 相乘的动量转移项。",
                "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$, while the primed amplitudes are the $1/|z|$ NLA corrections."
                if language == "en"
                else "- $m$ 是共同衰减率 $m+\\Lambda_0$ 中的非负拟合偏移量，所有带撇振幅对应 $1/|z|$ 的 NLA 修正。",
                "- `Lambda0_gev` is the fixed offset $\\Lambda_0$ in the reparameterized decay rate, not a hard bound on a fitted $\\Lambda$."
                if language == "en"
                else "- `Lambda0_gev` 是重参数化衰减率中的固定偏移 $\\Lambda_0$，不是对拟合参数 $\\Lambda$ 的硬边界。",
            ]
        scope_lines = [
            "The article formula is the full $\\pm z$ expression. The lamet-agent fit uses the explicit positive-$z$ branch, so ${\\rm sign}(z)=1$ and $|z|=z$ on the fitted interval."
            if language == "en"
            else "文献公式是完整的 $\\pm z$ 表达式；lamet-agent 实际拟合时只用正 $z$ 分支，因此在拟合区间上有 ${\\rm sign}(z)=1$ 且 $|z|=z$。",
            "When `method=CG`, the implementation multiplies the positive-$z$ branch by the extra factor $z^{-n}$."
            if language == "en"
            else "当 `method=CG` 时，lamet-agent 会在该正 $z$ 分支外再乘一个 $z^{-n}$ 因子。",
        ]
    elif observable == "nucleon_quark_quasi_gpd":
        if order == "LA":
            article_formula = (
                r"\tilde{h}^{\rm LA}(z,P^z,P'^z)="
                r"\left["
                r"A_2 e^{i\phi_2\,{\rm sign}(z)}"
                r"+\tilde{A}_2 e^{i\tilde{\phi}_2\,{\rm sign}(z)} e^{-i(P^z-P'^z)z}"
                r"\right]"
                + article_tail
                + r"."
            )
            implementation_formula = (
                r"\tilde{h}^{\rm LA}_{\rm agent}(z>0;P^z,P'^z)="
                r"\left["
                r"A_2 e^{i\phi_2}"
                r"+\tilde{A}_2 e^{i(\tilde{\phi}_2-(P^z-P'^z)z)}"
                r"\right]"
                + implementation_tail
                + r"."
            )
            mapping_lines = [
                "- $A_2,\\phi_2$ correspond to the forward-like central term."
                if language == "en"
                else "- $A_2,\\phi_2$ 对应 forward-like 的中心项。",
                "- $\\tilde A_2,\\tilde\\phi_2$ correspond to the momentum-transfer term $e^{-i(P^z-P'^z)z}$."
                if language == "en"
                else "- $\\tilde A_2,\\tilde\\phi_2$ 对应动量转移项 $e^{-i(P^z-P'^z)z}$。",
                "- $m$ is the fitted non-negative offset, so the effective decay rate is $m+\\Lambda_0$."
                if language == "en"
                else "- $m$ 是非负拟合偏移量，因此有效衰减率为 $m+\\Lambda_0$。",
                "- `Lambda0_gev` is the fixed offset $\\Lambda_0$ in the reparameterized decay rate, not a hard bound on a fitted $\\Lambda$."
                if language == "en"
                else "- `Lambda0_gev` 是重参数化衰减率中的固定偏移 $\\Lambda_0$，不是对拟合参数 $\\Lambda$ 的硬边界。",
            ]
        else:
            article_formula = (
                r"\tilde{h}^{\rm NLA}(z,P^z,P'^z)="
                r"\left["
                r"A_2 e^{i\phi_2\,{\rm sign}(z)}"
                r"+\tilde{A}_2 e^{i\tilde{\phi}_2\,{\rm sign}(z)} e^{-i(P^z-P'^z)z}"
                r"+\frac{A'_2}{|z|} e^{i\phi'_2\,{\rm sign}(z)}"
                r"+\frac{\tilde{A}'_2}{|z|} e^{i\tilde{\phi}'_2\,{\rm sign}(z)} e^{-i(P^z-P'^z)z}"
                r"\right]"
                + article_tail
                + r"."
            )
            implementation_formula = (
                r"\tilde{h}^{\rm NLA}_{\rm agent}(z>0;P^z,P'^z)="
                r"\left["
                r"A_2 e^{i\phi_2}"
                r"+\tilde{A}_2 e^{i(\tilde{\phi}_2-(P^z-P'^z)z)}"
                r"+\frac{A'_2}{z} e^{i\phi'_2}"
                r"+\frac{\tilde{A}'_2}{z} e^{i(\tilde{\phi}'_2-(P^z-P'^z)z)}"
                r"\right]"
                + implementation_tail
                + r"."
            )
            mapping_lines = [
                "- $A_2,\\phi_2$ and $A'_2,\\phi'_2$ correspond to the forward-like LA/NLA central terms."
                if language == "en"
                else "- $A_2,\\phi_2$ 以及 $A'_2,\\phi'_2$ 对应 forward-like 的 LA/NLA 中心项。",
                "- $\\tilde A_2,\\tilde\\phi_2$ and $\\tilde A'_2,\\tilde\\phi'_2$ correspond to the momentum-transfer terms proportional to $e^{-i(P^z-P'^z)z}$."
                if language == "en"
                else "- $\\tilde A_2,\\tilde\\phi_2$ 以及 $\\tilde A'_2,\\tilde\\phi'_2$ 对应与 $e^{-i(P^z-P'^z)z}$ 相乘的动量转移项。",
                "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$, while the primed amplitudes are the $1/|z|$ NLA corrections."
                if language == "en"
                else "- $m$ 是共同衰减率 $m+\\Lambda_0$ 中的非负拟合偏移量，所有带撇振幅对应 $1/|z|$ 的 NLA 修正。",
                "- `Lambda0_gev` is the fixed offset $\\Lambda_0$ in the reparameterized decay rate, not a hard bound on a fitted $\\Lambda$."
                if language == "en"
                else "- `Lambda0_gev` 是重参数化衰减率中的固定偏移 $\\Lambda_0$，不是对拟合参数 $\\Lambda$ 的硬边界。",
            ]
        scope_lines = [
            "The article formula is the full $\\pm z$ expression. The lamet-agent fit uses the explicit positive-$z$ branch, so ${\\rm sign}(z)=1$ and $|z|=z$ on the fitted interval."
            if language == "en"
            else "文献公式是完整的 $\\pm z$ 表达式；lamet-agent 实际拟合时只用正 $z$ 分支，因此在拟合区间上有 ${\\rm sign}(z)=1$ 且 $|z|=z$。",
            "When `method=CG`, the implementation multiplies the positive-$z$ branch by the extra factor $z^{-n}$."
            if language == "en"
            else "当 `method=CG` 时，lamet-agent 会在该正 $z$ 分支外再乘一个 $z^{-n}$ 因子。",
        ]
    elif observable in {
        "pion_quark_quasi_pdf",
        "nucleon_quark_unpolarized_quasi_pdf",
        "nucleon_quark_transversity_quasi_pdf",
        "meson_quasi_da",
    }:
        if observable == "pion_quark_quasi_pdf":
            phases_text = (
                "- In the implementation rewrite, $\\omega_2=0$, $\\omega_1=-P^z$, and $\\omega_3=+P^z$."
                if language == "en"
                else "- 在实现中的等价重写里，$\\omega_2=0$、$\\omega_1=-P^z$、$\\omega_3=+P^z$。"
            )
        elif observable == "meson_quasi_da":
            phases_text = (
                "- In the implementation rewrite, $\\omega_1=-P^z$ and $\\omega_2=0$."
                if language == "en"
                else "- 在实现中的等价重写里，$\\omega_1=-P^z$、$\\omega_2=0$。"
            )
        else:
            phases_text = (
                "- In the implementation rewrite, the only retained phase is the central frequency $\\omega_2=0$."
                if language == "en"
                else "- 在实现中的等价重写里，只保留中心频率 $\\omega_2=0$。"
            )
        if observable == "pion_quark_quasi_pdf":
            article_core = (
                r"A_2 e^{i\phi_2\,{\rm sign}(z)}"
                r"+A_1 e^{i\phi_1\,{\rm sign}(z)} e^{-i z P^z}"
                r"+A_3 e^{i\phi_3\,{\rm sign}(z)} e^{i z P^z}"
            )
            article_nla_core = (
                r"\frac{A'_2}{|z|} e^{i\phi'_2\,{\rm sign}(z)}"
                r"+\frac{A'_1}{|z|} e^{i\phi'_1\,{\rm sign}(z)} e^{-i z P^z}"
                r"+\frac{A'_3}{|z|} e^{i\phi'_3\,{\rm sign}(z)} e^{i z P^z}"
            )
        elif observable == "meson_quasi_da":
            article_core = (
                r"A_1 e^{i\phi_1\,{\rm sign}(z)} e^{-i z P^z}"
                r"+A_2 e^{i\phi_2\,{\rm sign}(z)}"
            )
            article_nla_core = (
                r"\frac{A'_1}{|z|} e^{i\phi'_1\,{\rm sign}(z)} e^{-i z P^z}"
                r"+\frac{A'_2}{|z|} e^{i\phi'_2\,{\rm sign}(z)}"
            )
        else:
            article_core = r"A_2 e^{i\phi_2\,{\rm sign}(z)}"
            article_nla_core = r"\frac{A'_2}{|z|} e^{i\phi'_2\,{\rm sign}(z)}"
        article_formula = (
            r"h^{\rm " + order + r"}_{\rm art}(z)="
            r"\left["
            + article_core
            + (r"+" + article_nla_core if order == "NLA" else "")
            + r"\right]"
            + article_tail
            + r"."
        )
        implementation_formula = (
            r"h^{\rm " + order + r"}_{\rm agent}(z>0)="
            r"\left[\sum_j A_j e^{i(\phi_j+\omega_j z)}"
            + (r"+\sum_j \frac{A'_j}{z} e^{i(\phi'_j+\omega_j z)}" if order == "NLA" else "")
            + r"\right]"
            + implementation_tail
            + r"."
        )
        mapping_lines = [
            phases_text,
            "- The article form keeps explicit ${\\rm sign}(z)$ and $|z|$, while lamet-agent rewrites the same positive-$z$ branch as a sum over frequencies $\\omega_j$."
            if language == "en"
            else "- 文献公式保留显式的 ${\\rm sign}(z)$ 和 $|z|$；lamet-agent 则把同一正 $z$ 分支等价重写为频率和形式 $\\omega_j$ 的求和。",
            "- The amplitudes $A_j,\\phi_j$ map one-to-one between the two formulas, and the primed amplitudes give the NLA $1/|z|$ corrections when present."
            if language == "en"
            else "- 两种写法中的 $A_j,\\phi_j$ 一一对应；若存在带撇项，它们对应 NLA 的 $1/|z|$ 修正。",
            "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$; `Lambda0_gev` is the fixed offset, not a hard bound on a fitted $\\Lambda$."
            if language == "en"
            else "- $m$ 是共同衰减率 $m+\\Lambda_0$ 中的非负拟合偏移量；`Lambda0_gev` 是固定偏移，不是对拟合参数 $\\Lambda$ 的硬边界。",
        ]
        scope_lines = [
            "For these forward-like quark observables, the report distinguishes the article formula from the lamet-agent parameterized equivalent rewrite."
            if language == "en"
            else "对这些 forward-like 夸克物理量，报告会明确区分文献原式与 lamet-agent 的参数化等价重写。",
            "The implementation fits only positive coordinates, so ${\\rm sign}(z)=1$ and $|z|=z$ on the fitted interval; `method=CG` adds the explicit factor $z^{-n}$ shown in the implementation formula."
            if language == "en"
            else "实现时只拟合正坐标，因此在拟合区间上有 ${\\rm sign}(z)=1$、$|z|=z$；`method=CG` 时再额外乘上实现公式中已经写出的 $z^{-n}$。",
        ]
    elif observable == "nucleon_gluon_quasi_pdf":
        article_formula = (
            (
                r"\mathrm{Re}\,h^{\rm LA}_{\rm art}(z)=\left[A\,|z|\right]"
                if order == "LA"
                else r"\mathrm{Re}\,h^{\rm NLA}_{\rm art}(z)=\left[A\,|z|+A'\right]"
            )
            + article_tail
            + r",\qquad \mathrm{Im}\,h(z)=0."
        )
        implementation_formula = (
            (
                r"\mathrm{Re}\,h^{\rm LA}_{\rm agent}(z>0)=\left[A\,z\right]"
                if order == "LA"
                else r"\mathrm{Re}\,h^{\rm NLA}_{\rm agent}(z>0)=\left[A\,z+A'\right]"
            )
            + implementation_tail
            + r",\qquad \mathrm{Im}\,h(z)=0."
        )
        mapping_lines = [
            "- This is the implementation-oriented real-tail rewrite of the Appendix-F gluon form; the report does not claim a universal term-by-term correspondence beyond this specialized real-part ansatz."
            if language == "en"
            else "- 这里给出的是 Appendix F 胶子实部尾项在代码中的实现版本；除该特定实部参数化外，报告不强行宣称逐项完全对应。",
            "- $A$ controls the linear large-distance growth before exponential damping; $A'$ is the NLA constant correction when present."
            if language == "en"
            else "- $A$ 控制指数衰减前的线性长程增长；若存在，$A'$ 是 NLA 常数修正。",
            "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$; `Lambda0_gev` is the fixed offset, not a hard bound on a fitted $\\Lambda$."
            if language == "en"
            else "- $m$ 是共同衰减率 $m+\\Lambda_0$ 中的非负拟合偏移量；`Lambda0_gev` 是固定偏移，不是对拟合参数 $\\Lambda$ 的硬边界。",
        ]
        scope_lines = [
            "The article form is written with $|z|$ and the lamet-agent form uses the positive-$z$ implementation; `method=CG` adds the explicit factor $z^{-n}$ shown above."
            if language == "en"
            else "文献式写成 $|z|$ 形式，而 lamet-agent 采用正 $z$ 实现；`method=CG` 时再额外乘上上式中写出的 $z^{-n}$。",
        ]
    elif observable == "pion_gluon_quasi_pdf":
        article_formula = (
            (
                r"\mathrm{Re}\,h^{\rm LA}_{\rm art}(z)=\left[A_2\,|z|\right]"
                if order == "LA"
                else r"\mathrm{Re}\,h^{\rm NLA}_{\rm art}(z)=\left[A_2\,|z|+A_2'+2A_1\cos(\phi-P^z z)\right]"
            )
            + article_tail
            + r",\qquad \mathrm{Im}\,h(z)=0."
        )
        implementation_formula = (
            (
                r"\mathrm{Re}\,h^{\rm LA}_{\rm agent}(z>0)=\left[A_2\,z\right]"
                if order == "LA"
                else r"\mathrm{Re}\,h^{\rm NLA}_{\rm agent}(z>0)=\left[A_2\,z+A_2'+2A_1\cos(\phi-P^z z)\right]"
            )
            + implementation_tail
            + r",\qquad \mathrm{Im}\,h(z)=0."
        )
        mapping_lines = [
            "- This is the implementation-oriented real-tail rewrite of the Appendix-F gluon form; the report does not claim a universal term-by-term correspondence beyond this specialized real-part ansatz."
            if language == "en"
            else "- 这里给出的是 Appendix F 胶子实部尾项在代码中的实现版本；除该特定实部参数化外，报告不强行宣称逐项完全对应。",
            "- $A_2$ controls the linear large-distance part, while $A_2'$, $A_1$, and $\\phi$ parameterize the NLA constant and oscillatory corrections."
            if language == "en"
            else "- $A_2$ 控制线性的长程部分，$A_2'$、$A_1$ 和 $\\phi$ 则参数化 NLA 的常数与振荡修正。",
            "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$; `Lambda0_gev` is the fixed offset, not a hard bound on a fitted $\\Lambda$."
            if language == "en"
            else "- $m$ 是共同衰减率 $m+\\Lambda_0$ 中的非负拟合偏移量；`Lambda0_gev` 是固定偏移，不是对拟合参数 $\\Lambda$ 的硬边界。",
        ]
        scope_lines = [
            "The article form is written with $|z|$ and the lamet-agent form uses the positive-$z$ implementation; `method=CG` adds the explicit factor $z^{-n}$ shown above."
            if language == "en"
            else "文献式写成 $|z|$ 形式，而 lamet-agent 采用正 $z$ 实现；`method=CG` 时再额外乘上上式中写出的 $z^{-n}$。",
        ]
    else:
        article_formula = rf"h^{{\rm {order}}}_{{\rm art}}(z)=\left[\sum_j A_j e^{{i\phi_j\,{{\rm sign}}(z)}} e^{{i\omega_j z}}\right]{article_tail}."
        implementation_formula = (
            rf"h^{{\rm {order}}}_{{\rm agent}}(z>0)=\left[\sum_j A_j e^{{i(\phi_j+\omega_j z)}}"
            + (r"+\sum_j \frac{A'_j}{z} e^{i(\phi'_j+\omega_j z)}" if order == "NLA" else "")
            + rf"\right]{implementation_tail}."
        )
        mapping_lines = [
            "- The implementation is the positive-$z$ rewrite of the article-style oscillatory tail."
            if language == "en"
            else "- 实现公式是文献风格振荡尾项在正 $z$ 分支上的重写。",
            "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$; `Lambda0_gev` is the fixed offset, not a hard bound on a fitted $\\Lambda$."
            if language == "en"
            else "- $m$ 是共同衰减率 $m+\\Lambda_0$ 中的非负拟合偏移量；`Lambda0_gev` 是固定偏移，不是对拟合参数 $\\Lambda$ 的硬边界。",
        ]
        scope_lines = [
            "The implementation fits only positive coordinates."
            if language == "en"
            else "实现时只拟合正坐标。",
        ]

    constraint_lines = []
    if observable == "pion_quark_quasi_pdf" and sector == "valence":
        constraint_lines.append(
            "- 由 arXiv:2601.12189 可知，`pion_quark_quasi_pdf` 的 `valence` sector 应在拟合输入中施加约束 $\\phi_2=\\phi'_2=0$、$A_3=A_1$、$\\phi_3=-\\phi_1$、$A'_3=A'_1$、$\\phi'_3=-\\phi'_1$。"
            if language == "zh"
            else "- Following arXiv:2601.12189, the fit input for the `valence` sector of `pion_quark_quasi_pdf` imposes $\\phi_2=\\phi'_2=0$, $A_3=A_1$, $\\phi_3=-\\phi_1$, $A'_3=A'_1$, and $\\phi'_3=-\\phi'_1$."
        )
    if observable == "pion_quark_quasi_pdf" and sector == "sea":
        constraint_lines.append(
            "- 由 arXiv:2601.12189 可知，`pion_quark_quasi_pdf` 的 `sea` sector 应在拟合输入中施加约束 $A_1=A_3=0$；NLA 项同时满足 $A'_1=A'_3=0$。"
            if language == "zh"
            else "- Following arXiv:2601.12189, the fit input for the `sea` sector of `pion_quark_quasi_pdf` imposes $A_1=A_3=0$; for NLA terms it also imposes $A'_1=A'_3=0$."
        )
    if observable == "meson_quasi_da" and psi1_class == "light" and psi2_class == "light":
        constraint_lines.append(
            "- 由 arXiv:2601.12189 可知，`psi1_flavor_class=light, psi2_flavor_class=light` 情形下的 `meson_quasi_da` 应在拟合输入中施加约束 $A_2=A_1$、$\\phi_2=-\\phi_1$、$A'_2=A'_1$、$\\phi'_2=-\\phi'_1$。"
            if language == "zh"
            else "- Following arXiv:2601.12189, the fit input for `meson_quasi_da` with `psi1_flavor_class=light, psi2_flavor_class=light` imposes $A_2=A_1$, $\\phi_2=-\\phi_1$, $A'_2=A'_1$, and $\\phi'_2=-\\phi'_1$."
        )
    if observable == "meson_quasi_da" and psi1_class == "light" and psi2_class == "heavy":
        constraint_lines.append(
            "- 由 arXiv:2601.12189 可知，`psi1_flavor_class=light, psi2_flavor_class=heavy` 情形下的 `meson_quasi_da` 应在拟合输入中施加约束 $A_1=A'_1=0$。"
            if language == "zh"
            else "- Following arXiv:2601.12189, the fit input for `meson_quasi_da` with `psi1_flavor_class=light, psi2_flavor_class=heavy` imposes $A_1=A'_1=0$."
        )
    if observable == "meson_quasi_da" and psi1_class == "heavy" and psi2_class == "light":
        constraint_lines.append(
            "- 由 arXiv:2601.12189 可知，`psi1_flavor_class=heavy, psi2_flavor_class=light` 情形下的 `meson_quasi_da` 应在拟合输入中施加约束 $A_2=A'_2=0$。"
            if language == "zh"
            else "- Following arXiv:2601.12189, the fit input for `meson_quasi_da` with `psi1_flavor_class=heavy, psi2_flavor_class=light` imposes $A_2=A'_2=0$."
        )
    if observable == "pion_quark_quasi_gpd" and sector == "sea":
        constraint_lines.append(
            "- 由 arXiv:2601.12189 可知，`pion_quark_quasi_gpd` 的 `sea` sector 应在拟合输入中施加约束 $A_1=A_3=0$；NLA 项同时满足 $A'_1=A'_3=0$。"
            if language == "zh"
            else "- Following arXiv:2601.12189, the fit input for the `sea` sector of `pion_quark_quasi_gpd` imposes $A_1=A_3=0$; for NLA terms it also imposes $A'_1=A'_3=0$."
        )
    mapping_lines.extend(constraint_lines)

    if language == "zh":
        lines = [
            f"### 文献公式\n{reference}。\n\n$$\n{article_formula}\n$$",
            f"### lamet-agent 实际拟合公式\n本次分析中代码实际使用的拟合形式为\n\n$$\n{implementation_formula}\n$$",
            "### 参数对应关系",
            *mapping_lines,
            "### 适用范围说明",
            *scope_lines,
        ]
        return "\n\n".join(lines)
    lines = [
        f"### Article Formula\n{reference}.\n\n$$\n{article_formula}\n$$",
        f"### lamet-agent Implementation\nThe fit actually used by lamet-agent in this run is\n\n$$\n{implementation_formula}\n$$",
        "### Parameter Correspondence",
        *mapping_lines,
        "### Scope and Equivalence",
        *scope_lines,
    ]
    return "\n\n".join(lines)


def _fourier_transform_text(result: dict[str, Any], *, language: str) -> str:
    part = str(result.get("part", "both")).lower()
    shift = float(result.get("phase_shift", 0.0) or 0.0)
    phase = f"(x-{_fmt(shift)})\\lambda" if shift else "x\\lambda"
    convention = (
        f"本阶段采用 $e^{{+i{phase}}}$ 的 Fourier convention，即 $q(x)=\\frac{{\\Delta\\lambda}}{{2\\pi}}\\sum_\\lambda e^{{+i{phase}}}h(\\lambda)$；下面给出该 convention 对应的实部/虚部分解。"
        if language == "zh"
        else f"This stage uses the $e^{{+i{phase}}}$ Fourier convention, i.e. $q(x)=\\frac{{\\Delta\\lambda}}{{2\\pi}}\\sum_\\lambda e^{{+i{phase}}}h(\\lambda)$; the corresponding real/imaginary decomposition is shown below."
    )
    note = (
        f"\n\n这里使用 `phase_shift={_fmt(shift)}`，即 Fourier 相位为 ${phase}$。"
        if language == "zh" and shift
        else f"\n\nThis run uses `phase_shift={_fmt(shift)}`, i.e. the Fourier phase is ${phase}$."
        if shift
        else ""
    )
    if language == "zh":
        if part == "re":
            return (
                f"{convention}\n\n"
                "$$\n"
                "q_{\\rm re}(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
                f"\\cos({phase})\\,\\mathrm{{Re}}\\,h(\\lambda).\n"
                "$$"
                f"{note}"
            )
        if part == "im":
            return (
                f"{convention}\n\n"
                "$$\n"
                "q_{\\rm im}(x)=-\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
                f"\\sin({phase})\\,\\mathrm{{Im}}\\,h(\\lambda).\n"
                "$$"
                f"{note}"
            )
        return (
            f"{convention}\n\n"
            "$$\n"
            "\\mathrm{Re}\\,q(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
            f"\\left[\\cos({phase})\\,\\mathrm{{Re}}\\,h(\\lambda)-\\sin({phase})\\,\\mathrm{{Im}}\\,h(\\lambda)\\right],\n"
            "$$\n"
            "$$\n"
            "\\mathrm{Im}\\,q(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
            f"\\left[\\sin({phase})\\,\\mathrm{{Re}}\\,h(\\lambda)+\\cos({phase})\\,\\mathrm{{Im}}\\,h(\\lambda)\\right].\n"
            "$$"
            f"{note}"
        )
    if part == "re":
        return (
            f"{convention}\n\n"
            "$$\n"
            "q_{\\rm re}(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
            f"\\cos({phase})\\,\\mathrm{{Re}}\\,h(\\lambda).\n"
            "$$"
            f"{note}"
        )
    if part == "im":
        return (
            f"{convention}\n\n"
            "$$\n"
            "q_{\\rm im}(x)=-\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
            f"\\sin({phase})\\,\\mathrm{{Im}}\\,h(\\lambda).\n"
            "$$"
            f"{note}"
        )
    return (
        f"{convention}\n\n"
        "$$\n"
        "\\mathrm{Re}\\,q(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
        f"\\left[\\cos({phase})\\,\\mathrm{{Re}}\\,h(\\lambda)-\\sin({phase})\\,\\mathrm{{Im}}\\,h(\\lambda)\\right],\n"
        "$$\n"
        "$$\n"
        "\\mathrm{Im}\\,q(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
        f"\\left[\\sin({phase})\\,\\mathrm{{Re}}\\,h(\\lambda)+\\cos({phase})\\,\\mathrm{{Im}}\\,h(\\lambda)\\right].\n"
        "$$"
        f"{note}"
    )


def _field_definitions(*, language: str) -> list[str]:
    if language == "zh":
        return [
            "| 条目 | 含义 |",
            "|---|---|",
            "| Observable | 本阶段要变换的物理矩阵元类型；它决定长程外推公式中的相位结构和参数结构。 |",
            "| Sector | 用户请求的物理投影；PDF/GPD 可取 `valence`、`total`、`full`、`sea`，DA 固定为 `full`。 |",
            "| Tail method/order | $\\mathrm{order}$ 选择 LA 或 NLA；$\\mathrm{method}=\\mathrm{CG}$ 表示在基础尾部上额外乘以 $z^{-n}$。 |",
            "| Active fitted component | `sector` 展开后的执行通道；`both` 表示实部和虚部同时参与拟合，`re`/`im` 只拟合单个分量。 |",
            "| Coordinate unit | `lattice` 表示 $z/a$。代码先转为 $z_{\\rm GeV^{-1}}=(z/a)a_{\\rm fm}\\,5.067731237$，再计算 $\\lambda=P_z z_{\\rm GeV^{-1}}$。 |",
            "| Posterior-prior error scale | 样本平均拟合得到 $\\bar p_i\\pm\\sigma_{p_i}$ 后，重采样拟合使用弱先验 $p_i=\\bar p_i\\pm s\\sigma_{p_i}$。 |",
        ]
    return [
        "| Entry | Meaning |",
        "|---|---|",
        "| Observable | Physical matrix element transformed by this stage. |",
        "| Sector | Requested physics projection; PDF/GPD accept `valence`, `total`, `full`, and `sea`, while DA uses `full`. |",
        "| Tail method/order | $\\mathrm{order}$ selects LA or NLA; $\\mathrm{method}=\\mathrm{CG}$ adds $z^{-n}$ to the base tail. |",
        "| Active fitted component | Execution channel resolved from `sector`; `both` fits $\\mathrm{Re}\\,\\tilde h^R$ and $\\mathrm{Im}\\,\\tilde h^R$ together, while `re` or `im` fits one component. |",
        "| Coordinate unit | `lattice` means $z/a$. The code converts it to $z_{\\rm GeV^{-1}}=(z/a)a_{\\rm fm}\\,5.067731237$ and then $\\lambda=P_z z_{\\rm GeV^{-1}}$. |",
        "| Posterior-prior error scale | The mean fit gives $\\bar p_i\\pm\\sigma_{p_i}$; resampled fits use $p_i=\\bar p_i\\pm s\\sigma_{p_i}$. |",
    ]


def _projection_text(result: dict[str, Any], *, language: str) -> list[str]:
    sector = str(result.get("sector", "full")).lower()
    part = str(result.get("part", "both")).lower()
    scale = float(result.get("output_scale", 1.0))
    truncated = str(result.get("short_distance_policy", "full_from_zero")) == "truncate_missing"
    missing = result.get("missing_short_distance_coord", [])
    if language == "zh":
        intro = (
            f"本次设置为 `sector={sector}`，代码展开为 `part={part}`、"
            f"`output_scale={_fmt(scale)}`、`im_flip_for_ft={result.get('im_flip_for_ft', False)}`。"
            "在常用的扩展分布约定"
            "$q_{\\rm ext}(x)=q(x)$（$x>0$）和 $q_{\\rm ext}(-x)=-\\bar q(x)$ 下，"
            "坐标空间矩阵元满足"
            "$h(\\lambda)=\\int dx\\,e^{ix\\lambda}q_{\\rm ext}(x)$。"
        )
        if truncated:
            intro += f" 本次输入缺少短距离坐标 {missing}，这些点未参与 Fourier 求和，因此输出是缺短距离截断投影。"
        if sector == "sea":
            meaning = "因此 `sea` 由两次投影组合得到：先计算 `total=q(x)+\\bar q(x)` 和 `valence=q(x)-\\bar q(x)`，再取 $(\\mathrm{total}-\\mathrm{valence})/2=\\bar q(x)$。"
        elif part == "both":
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
        if truncated:
            meaning += " 由于近零坐标缺失，上述投影解释只适用于当前截断求和结果，不能按完整从零开始的 Fourier 结果解释其归一化或矩。"
        return ["## Sector 物理解释", intro, "", meaning]

    intro = (
        f"This run uses `sector={sector}`, resolved internally to `part={part}`, "
        f"`output_scale={_fmt(scale)}`, and `im_flip_for_ft={result.get('im_flip_for_ft', False)}`. "
        "With the common extended-distribution convention "
        "$q_{\\rm ext}(x)=q(x)$ for $x>0$ and $q_{\\rm ext}(-x)=-\\bar q(x)$, "
        "the coordinate-space matrix element obeys "
        "$h(\\lambda)=\\int dx\\,e^{ix\\lambda}q_{\\rm ext}(x)$."
    )
    if truncated:
        intro += f" This input misses short-distance coordinates {missing}; these points are omitted from the Fourier sum, so the output is a short-distance-truncated projection."
    if sector == "sea":
        meaning = "`sea` is a derived projection: the code computes `total=q(x)+\\bar q(x)` and `valence=q(x)-\\bar q(x)` and then returns $(\\mathrm{total}-\\mathrm{valence})/2=\\bar q(x)$."
    elif part == "both":
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
    if truncated:
        meaning += " Because near-zero coordinates are missing, this projection statement applies only to the truncated sum and should not be interpreted as a fully normalized Fourier result or moment."
    return ["## Sector Physical Interpretation", intro, "", meaning]


def _range_selection_table(result: dict[str, Any], *, language: str) -> list[str]:
    labels = list(result.get("candidate_scheme_labels", []))
    if not labels:
        return []
    chi2 = np.asarray(result.get("candidate_scheme_fit_chi2_dof", []), dtype=float)
    q_values = np.asarray(result.get("candidate_scheme_q", []), dtype=float)
    log_gbf = np.asarray(result.get("candidate_scheme_logGBF", []), dtype=float)
    selected = int(result.get("selected_candidate_index", -1))
    title = "### Range Selection Candidates" if language == "en" else "### 区间选择候选"
    header = (
        "| # | range label | selected | $Q$ | logGBF | $\\chi^2/{\\rm dof}$ |"
        if language == "en"
        else "| # | 区间标签 | 选中 | $Q$ | logGBF | $\\chi^2/{\\rm dof}$ |"
    )
    lines = [title, "", header, "|---:|---|---:|---:|---:|---:|"]
    for idx, label in enumerate(labels):
        lines.append(
            f"| {idx} | {label} | {'yes' if idx == selected else ''} | "
            f"{_fmt(q_values[idx]) if idx < q_values.size else 'n/a'} | "
            f"{_fmt(log_gbf[idx]) if idx < log_gbf.size else 'n/a'} | "
            f"{_fmt(chi2[idx]) if idx < chi2.size else 'n/a'} |"
        )
    return lines


def _fit_model_table(result: dict[str, Any], *, language: str) -> list[str]:
    labels = list(result.get("fit_model_labels", []))
    if not labels:
        return []
    schemes = list(result.get("scheme_results", []))
    orders = list(result.get("fit_model_orders", []))
    widths = np.asarray(result.get("fit_model_prior_widths", []), dtype=float)
    weights = np.asarray(result.get("fit_model_mean_weights", []), dtype=float)
    q_values = np.asarray(result.get("fit_model_q", []), dtype=float)
    log_gbf = np.asarray(result.get("fit_model_logGBF", []), dtype=float)
    chi2 = np.asarray(result.get("fit_model_chi2_dof", []), dtype=float)
    failures = np.asarray(result.get("fit_failures", []), dtype=float)
    title = "### Fit-Model Average Candidates" if language == "en" else "### 拟合模型平均候选"
    header = (
        "| # | model | order | prior width | mean sample weight | $Q$ | logGBF | $\\chi^2/{\\rm dof}$ | failures | selected range | $z_{\\rm ext}^{\\rm max}$ | smooth |"
        if language == "en"
        else "| # | 模型 | order | 先验宽度 | 样本平均权重 | $Q$ | logGBF | $\\chi^2/{\\rm dof}$ | 失败次数 | 选定区间 | $z_{\\rm ext}^{\\rm max}$ | 平滑方式 |"
    )
    lines = [title, "", header, "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for idx, label in enumerate(labels):
        scheme = schemes[idx] if idx < len(schemes) else {}
        fit_range = scheme.get("fit_range")
        fit_range_text = "n/a" if fit_range is None else _format_fit_range(fit_range, language=language)
        lines.append(
            f"| {idx} | {label} | `{orders[idx] if idx < len(orders) else 'n/a'}` | "
            f"{_fmt(widths[idx]) if idx < widths.size else 'n/a'} | "
            f"{_fmt(weights[idx]) if idx < weights.size else 'n/a'} | "
            f"{_fmt(q_values[idx]) if idx < q_values.size else 'n/a'} | "
            f"{_fmt(log_gbf[idx]) if idx < log_gbf.size else 'n/a'} | "
            f"{_fmt(chi2[idx]) if idx < chi2.size else 'n/a'} | "
            f"{int(failures[idx]) if idx < failures.size else 'n/a'} | "
            f"{fit_range_text} | "
            f"{_fmt(scheme.get('z_ext_max'))} | "
            f"`{scheme.get('smooth', 'n/a')}` |"
        )
    return lines


def _format_fit_value(mean: float, sdev: float) -> str:
    if not np.isfinite(mean):
        return "n/a"
    if not np.isfinite(sdev) or sdev <= 0.0:
        return _fmt(mean)
    exponent = 0 if sdev == 0 else int(np.floor(np.log10(abs(sdev))))
    decimals = max(0, -exponent + 1)
    scale = 10**decimals
    mean_rounded = round(mean, decimals)
    err_digits = int(round(sdev * scale))
    return f"{mean_rounded:.{decimals}f}({err_digits:0d})"


def _fit_model_parameter_table(result: dict[str, Any], *, language: str) -> list[str]:
    schemes = list(result.get("scheme_results", []))
    if not schemes:
        return []
    labels = list(result.get("scheme_labels", []))
    param_labels = []
    for scheme in schemes:
        for label in scheme.get("fit_param_labels", []):
            if label not in param_labels:
                param_labels.append(label)
    if not param_labels:
        return []

    header_title = "### Fit-Model Parameters" if language == "en" else "### 拟合模型参数"
    header = (
        "| # | label | " + " | ".join(f"`{label}`" for label in param_labels) + " |"
        if language == "en"
        else "| # | 标签 | " + " | ".join(f"`{label}`" for label in param_labels) + " |"
    )
    lines = [header_title, "", header, "|" + "---|" * (len(param_labels) + 2)]
    resample_mode = str(result.get("resample_mode", "bootstrap"))
    sample_error_mode = str(result.get("sample_error_mode", "covariance"))
    for idx, scheme in enumerate(schemes):
        label = labels[idx] if idx < len(labels) else scheme.get("label", str(idx))
        fit_params = np.asarray(scheme.get("fit_params", []), dtype=float)
        local_labels = list(scheme.get("fit_param_labels", []))
        values = []
        for param_label in param_labels:
            if fit_params.ndim != 2 or param_label not in local_labels:
                values.append("n/a")
                continue
            local_idx = local_labels.index(param_label)
            samples = fit_params[:, local_idx]
            mean_arr, sdev_arr = sample_mean_and_sdev(samples, mode=resample_mode, sample_error_mode=sample_error_mode)
            mean = float(mean_arr)
            sdev = float(sdev_arr)
            values.append(_format_fit_value(mean, sdev))
        lines.append("| " + f"{idx} | {label} | " + " | ".join(values) + " |")
    return lines


def _smooth_explanation(result: dict[str, Any], *, language: str) -> list[str]:
    schemes = list(result.get("scheme_results", []))
    best_idx = 0
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
    missing = list(result.get("missing_short_distance_coord", []))
    shift = float(result.get("phase_shift", 0.0) or 0.0)
    phase_text = "x\\lambda" if shift == 0.0 else f"(x-{_fmt(shift)})\\lambda"
    if missing:
        short_distance_text = (
            f"`truncate_missing`; omitted short-distance coordinates {missing}; Fourier starts at {_fmt(result.get('fourier_positive_coord_start'))}"
        )
    else:
        short_distance_text = "`full_from_zero`; no omitted short-distance coordinate"
    rows = [
        ("Observable", f"`{observable}` ({observable_text})"),
        ("Sector", f"`{result.get('sector', 'full')}`"),
        ("Tail method/order", f"`{method}` / `{order}`"),
        ("Active fitted component", f"`{result.get('part', 'both')}`"),
        ("Resampling mode", f"`{result.get('resample_mode', 'not recorded')}`"),
        ("Coordinate unit", f"{_display_unit(result.get('coord_unit', 'not recorded'))}; fit unit {_display_unit(result.get('fit_coord_unit', 'not recorded'))}"),
        ("Decay offset", f"$\\Lambda_0={_fmt(result.get('Lambda0_gev'))}$"),
        ("Phase shift", f"`phase_shift={_fmt(shift)}`; phase ${phase_text}$"),
        ("Output scale", f"$q(x)\\rightarrow {_fmt(result.get('output_scale', 1.0))}\\,q(x)$"),
        ("Short-distance treatment", short_distance_text),
        ("Best fit range", fit_range_text),
        ("Extension endpoint", z_ext_text),
        ("Fourier grid", _format_grid(y_grid, language=language)),
    ]
    if observable == "meson_quasi_da":
        rows.insert(2, ("DA flavor classes", f"`psi1={result.get('psi1_flavor_class', 'heavy')}`, `psi2={result.get('psi2_flavor_class', 'heavy')}`"))
    if language == "zh":
        short_distance_text = (
            f"`truncate_missing`；短距离坐标 {missing} 未参与；Fourier 从 {_fmt(result.get('fourier_positive_coord_start'))} 开始"
            if missing
            else "`full_from_zero`；没有省略短距离坐标"
        )
        rows = [
            ("物理量", f"`{observable}`（{observable_text}）"),
            ("Sector", f"`{result.get('sector', 'full')}`"),
            ("长程外推 method/order", f"`{method}` / `{order}`"),
            ("参与拟合的分量", f"`{result.get('part', 'both')}`"),
            ("重采样模式", f"`{result.get('resample_mode', 'not recorded')}`"),
            ("坐标单位", f"{_display_unit(result.get('coord_unit', 'not recorded'))}；拟合单位 {_display_unit(result.get('fit_coord_unit', 'not recorded'))}"),
            ("衰减偏移", f"$\\Lambda_0={_fmt(result.get('Lambda0_gev'))}$"),
            ("相位平移", f"`phase_shift={_fmt(shift)}`；相位 ${phase_text}$"),
            ("输出缩放", f"$q(x)\\rightarrow {_fmt(result.get('output_scale', 1.0))}\\,q(x)$"),
            ("短距离处理", short_distance_text),
            ("最优拟合区间", fit_range_text),
            ("外推终点", z_ext_text),
            ("傅立叶网格", _format_grid(y_grid, language=language)),
        ]
        if observable == "meson_quasi_da":
            rows.insert(2, ("DA flavor class", f"`psi1={result.get('psi1_flavor_class', 'heavy')}`，`psi2={result.get('psi2_flavor_class', 'heavy')}`"))
    header = "| Quantity | Value |" if language == "en" else "| 条目 | 数值或设置 |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _artifact_field_table(kind: str, *, language: str) -> list[str]:
    if kind == "result":
        rows = [
            ("`values`", "Complex final Fourier samples after fit-model averaging or best-model selection, with dimensions `(resample, x)`.", "经过拟合模型平均或最优模型选择后的最终复数 Fourier 样本，维度为 `(resample, x)`。"),
            ("coordinate `x`", "Fourier momentum-fraction grid.", "傅立叶变换后的动量分数网格。"),
            ("attr `resample`", "Resampling mode recorded by `EnsembleData`.", "`EnsembleData` 记录的重采样模式。"),
            ("attr `ft_re_mean` / `ft_im_mean`", "Final real/imaginary central values after fit-model averaging or best-model selection.", "经过拟合模型平均或最优模型选择后的实部/虚部中心值。"),
            ("attr `ft_re_stat_sdev` / `ft_im_stat_sdev`", "Statistical standard deviations from bootstrap/jackknife samples.", "由 bootstrap/jackknife 样本给出的统计误差。"),
            ("attr `ft_re_sys_sdev` / `ft_im_sys_sdev`", "Weighted spread among fit-model candidates at fixed selected range.", "固定选定区间后，不同拟合模型候选的加权离散度。"),
            ("attr `scheme_labels`", "Fit-model labels at the selected range.", "选定区间上的拟合模型标签。"),
            ("attr `fit_failures`", "Number of failed resampled tail fits in each fit model.", "每个拟合模型中重采样长程拟合失败的次数。"),
            ("attrs `fit_model_*`", "Per-sample fit-model weights and diagnostics for `(order, prior width)` candidates.", "`(order, prior width)` 候选的逐样本权重和诊断。"),
            ("attrs `candidate_scheme_*`", "Sample-average range-scan diagnostics used before model averaging.", "进入模型平均前 sample-average 区间扫描的诊断。"),
            ("attr `selection_mode`", "Two-stage selection mode: range selection followed by fit-model averaging or best-model selection.", "两阶段选择模式：先选区间，再做拟合模型平均或最优模型选择。"),
            ("attrs `momentum_gev`, `final_momentum_gev`, `lattice_spacing_fm`", "Momentum and lattice-spacing metadata.", "动量和格距元数据。"),
            ("attrs `sector`, `method`, `order`, `observable`, `part`, `output_scale`, `phase_shift`, `psi1_flavor_class`, `psi2_flavor_class`", "Physics projection, formula choices, execution channel, final output normalization, Fourier phase convention, and DA flavor-class metadata.", "物理投影、公式选择、执行通道、最终输出归一化、Fourier 相位约定和 DA flavor-class 元数据。"),
        ]
    else:
        rows = [
            ("`values`", "Fit-parameter samples with dimensions `(resample, scheme, parameter)`.", "拟合参数样本，维度为 `(resample, scheme, parameter)`。"),
            ("coordinates `scheme`, `parameter`", "Scheme labels and fitted parameter names.", "scheme 标签和拟合参数名。"),
            ("attr `fit_params`", "Tail-fit parameters for every scheme and resample.", "每个 scheme 和每个重采样样本的长程拟合参数。"),
            ("attr `fit_param_center` / `fit_param_sdev`", "Sample mean and statistical standard deviation of fit parameters.", "拟合参数的样本平均值和统计误差。"),
            ("attrs `fit_chi2`, `fit_dof`, `fit_q`, `fit_chi2_dof`", "Per-resample fit quality diagnostics.", "每个重采样样本的拟合质量诊断。"),
            ("attrs `fit_chi2_center`, `fit_chi2_dof_center`, `fit_q_center`", "Sample-averaged fit quality diagnostics for each scheme.", "每个 scheme 的样本平均拟合质量诊断。"),
            ("attrs `mean_fit_params`, `mean_fit_chi2`, `mean_fit_dof`, `mean_fit_q`, `mean_fit_log_gbf`", "Initial sample-average fit results used to seed resampled fits.", "用于初始化重采样拟合的样本平均拟合结果。"),
            ("attrs `fit_model_*`", "Per-sample weights and diagnostics for fixed-range fit-model averaging.", "固定区间后拟合模型平均的逐样本权重和诊断。"),
            ("attrs `candidate_scheme_*`, `selection_mode`", "Range-scan diagnostics and the two-stage selection mode.", "区间扫描诊断和两阶段选择模式。"),
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
    selected_model = schemes[0] if schemes else {}
    fit_range_text = _format_fit_range(selected_model.get("fit_range"), language=language)
    z_ext_max = selected_model.get("z_ext_max", "not available")

    if language == "zh":
        title = "# 傅立叶变换分析报告"
        abstract = f"本报告总结 `{observable}`（{observable_text}）的傅立叶变换分析，长程外推采用 `{method}` / `{order}`。"
        transform_text = _fourier_transform_text(result, language="zh")
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
            "## 拟合质量与模型诊断",
            "本单 job 报告列出 sample-average 选出的区间和固定区间后的拟合模型候选；完整统计口径见 stage 汇总报告。",
            "",
            *_range_selection_table(result, language="zh"),
            "",
            *_fit_model_table(result, language="zh"),
            "",
            *_fit_model_parameter_table(result, language="zh"),
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
        transform_text = _fourier_transform_text(result, language="en")
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
            "## Fit Quality and Model Diagnostics",
            "This single-job report lists the sample-average selected range and the fixed-range fit-model candidates; the full statistical prescription lives in the stage summary report.",
            "",
            *_range_selection_table(result, language="en"),
            "",
            *_fit_model_table(result, language="en"),
            "",
            *_fit_model_parameter_table(result, language="en"),
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
    report_language: str = "en",
) -> dict[str, Path]:
    """Write one Fourier report and return its path."""
    output = Path(path)
    target, language = _report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    report_artifacts = markdown_artifact_paths(
        artifacts,
        base_dir=target.parent,
        path_keys=FOURIER_ARTIFACT_ORDER,
    )
    target.write_text(
        build_fourier_report_markdown(result=result, summary=summary, artifacts=report_artifacts, language=language),
        encoding="utf-8",
    )
    return {"report": target}


def write_fourier_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
    report_language: str = "en",
) -> dict[str, Path]:
    """Write one report summarizing all Fourier jobs in a stage."""
    output = Path(path)
    target, language = _report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    first = jobs[0]["result"]
    for language, target in ((language, target),):
        observable = str(first.get("observable", ""))
        observable_text = OBSERVABLE_TEXT.get(observable, observable or "not recorded")
        method = str(first.get("method", "not recorded"))
        order = str(first.get("order", "not recorded"))
        y_grid = np.asarray(first.get("y_grid", []), dtype=float)
        z_ext_values = []
        for item in jobs:
            result = item["result"]
            schemes = list(result.get("scheme_results", []))
            if schemes:
                z_ext_values.append(schemes[0].get("z_ext_max"))
        finite_z_ext = [float(value) for value in z_ext_values if value is not None]
        same_z_ext = bool(finite_z_ext) and np.allclose(finite_z_ext, finite_z_ext[0])
        fit_range_text = "见下方各动量诊断表" if language == "zh" else "see the per-momentum diagnostics below"
        z_ext_max = finite_z_ext[0] if same_z_ext else ("见下方各动量诊断表" if language == "zh" else "see the per-momentum diagnostics below")
        transform_text = _fourier_transform_text(first, language=language)
        lines = [
            "# Fourier Transform Stage Report" if language == "en" else "# 傅立叶变换阶段报告",
            "",
            f"This report summarizes all Fourier-transform jobs in this stage for `{observable}` ({observable_text})."
            if language == "en"
            else f"本报告汇总当前 Fourier transform 阶段中 `{observable}`（{observable_text}）的所有动量。",
            "",
            "## Job Summary" if language == "en" else "## Job 汇总",
            "| job | $P_z$ | selected range | output | plot |"
            if language == "en"
            else "| job | $P_z$ | 选定区间 | 输出 | 图像 |",
            "|---|---:|---|---|---|",
        ]
        for item in jobs:
            result = item["result"]
            pz_value = result.get("momentum_gev")
            pz_text = "n/a" if pz_value is None else f"{float(pz_value):.2f}"
            artifacts = markdown_artifact_paths(
                item.get("artifacts", {}),
                base_dir=target.parent,
                path_keys=FOURIER_ARTIFACT_ORDER,
            )
            lines.append(
                f"| `{item['job_id']}` | {pz_text} | "
                f"{result.get('selected_range_label', 'n/a')} | "
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
                "## Fit Quality and Model Diagnostics" if language == "en" else "## 拟合质量与模型诊断",
                ]
            )
        lines.append(
            "本步骤先用 sample-average 矩阵元扫描 `zmin_values × zmax_values`，按 $Q\\ge0.05$ 过门后取 `logGBF` 最大的区间；"
            "若没有区间过门，则回退到 $Q$ 最大的成功候选。选定的区间固定后才进入模型层，区间变化不参与 model average。"
            if language == "zh"
            else "This stage first scans `zmin_values × zmax_values` on the sample-average matrix element, selects the largest-`logGBF` range among candidates passing $Q\\ge0.05$, and falls back to the largest-$Q$ successful range if none passes. The selected range is then fixed; range variation is not part of model averaging."
        )
        lines.append(
            "`model_average=true` 时，每个 resample sample 在固定区间和固定 method 下分别拟合 `(order, prior width)` 候选，并用该 sample 自己的 "
            "$w_{s,m}=\\exp(\\log\\mathrm{GBF}_{s,m}-\\max_n\\log\\mathrm{GBF}_{s,n})/\\sum_k\\exp(\\log\\mathrm{GBF}_{s,k}-\\max_n\\log\\mathrm{GBF}_{s,n})$ 加权；"
            "`model_average=false` 时，每个 sample 在过 $Q$ 门的候选中取 `logGBF` 最大者。"
            if language == "zh"
            else "With `model_average=true`, each resample sample refits the `(order, prior width)` candidates at fixed range and fixed method, then uses that sample's normalized evidence weight $w_{s,m}=\\exp(\\log\\mathrm{GBF}_{s,m}-\\max_n\\log\\mathrm{GBF}_{s,n})/\\sum_k\\exp(\\log\\mathrm{GBF}_{s,k}-\\max_n\\log\\mathrm{GBF}_{s,n})$. With `model_average=false`, each sample selects the largest-`logGBF` candidate after the $Q$ gate."
        )
        lines.extend(
            [
                "",
                "| job | $P_z$ | selected range | selected fit range | omitted short z | $\\chi^2/{\\rm dof}$ range | fit failures |"
                if language == "en"
                else "| job | $P_z$ | 选定区间 | 选定拟合区间 | 省略的短距离 z | $\\chi^2/{\\rm dof}$ 范围 | 拟合失败次数 |",
                "|---|---:|---|---|---|---:|---:|",
            ]
        )
        for item in jobs:
            result = item["result"]
            pz_value = result.get("momentum_gev")
            pz_text = "n/a" if pz_value is None else f"{float(pz_value):.2f}"
            schemes = list(result.get("scheme_results", []))
            selected_model = schemes[0] if schemes else {}
            chi2 = np.asarray(result.get("fit_model_chi2_dof", []), dtype=float)
            finite = chi2[np.isfinite(chi2)]
            chi_text = "n/a" if finite.size == 0 else f"{_fmt(np.min(finite))} to {_fmt(np.max(finite))}"
            missing = result.get("missing_short_distance_coord", [])
            lines.append(
                f"| `{item['job_id']}` | {pz_text} | "
                f"{result.get('selected_range_label', 'n/a')} | "
                f"{_format_fit_range(selected_model.get('fit_range'), language=language)} | "
                f"{missing if missing else 'none'} | "
                f"{chi_text} | "
                f"{int(np.sum(np.asarray(result.get('fit_failures', []), dtype=float)))} |"
            )
        lines.extend(["", *_smooth_explanation(first, language=language)])
        if language == "zh":
            lines.extend(
                [
                    "",
                    "- `range grid` 表示区间候选，不是模型平均对象；模型平均对象是固定区间后的 `(order, prior width)`。",
                    "- `method` 是理论输入，保持 manifest 给定值，不参与 model average。",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "- The `range grid` denotes range candidates, not the model-averaging space; the model candidates are `(order, prior width)` at fixed range.",
                    "- `method` is a fixed theory input from the manifest and is not model averaged.",
                ]
            )
        for item in jobs:
            result = item["result"]
            pz_value = result.get("momentum_gev")
            pz_text = "n/a" if pz_value is None else f"{float(pz_value):.2f}"
            lines.extend(
                [
                    "",
                    f"### `{item['job_id']}`: $P_z={pz_text}$ GeV",
                    "",
                    *_range_selection_table(result, language=language),
                    "",
                    *_fit_model_table(result, language=language),
                    "",
                    *_fit_model_parameter_table(result, language=language),
                ]
            )
        lines.append("")
        lines.append("## Figures and Visual Assessment" if language == "en" else "## 图像与可视化评估")
        for item in jobs:
            result = item["result"]
            pz_value = result.get("momentum_gev")
            pz_text = "n/a" if pz_value is None else f"{float(pz_value):.2f}"
            artifacts = markdown_artifact_paths(
                item.get("artifacts", {}),
                base_dir=target.parent,
                path_keys=FOURIER_ARTIFACT_ORDER,
            )
            lines.extend(["", f"### `{item['job_id']}`: $P_z={pz_text}$ GeV"])
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
            artifacts = markdown_artifact_paths(
                item.get("artifacts", {}),
                base_dir=target.parent,
                path_keys=FOURIER_ARTIFACT_ORDER,
            )
            for key in FOURIER_ARTIFACT_ORDER:
                value = artifacts.get(key)
                if value:
                    desc = FOURIER_ARTIFACT_DESCRIPTIONS[key]
                    lines.append(f"| [{Path(value).name}]({value}) | `{item['job_id']}`: {desc[1 if language == 'zh' else 0]} |")
        lines.extend(["", *_artifact_help(language=language)])
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": target}
