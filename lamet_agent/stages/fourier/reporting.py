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
    translate_markdown_report,
)
from lamet_agent.core.resampling import sample_mean_and_sdev


OBSERVABLE_TEXT = {
    "pion_quark_quasi_pdf": "pion quark quasi-PDF",
    "nucleon_quark_quasi_pdf": "nucleon quark quasi-PDF",
    "pion_gluon_quasi_pdf": "pion gluon quasi-PDF",
    "nucleon_gluon_quasi_pdf": "nucleon gluon quasi-PDF",
    "meson_quasi_da": "meson quasi-DA",
    "pion_quark_quasi_gpd": "pion quark quasi-GPD",
    "nucleon_quark_quasi_gpd": "nucleon quark quasi-GPD",
}

FORMULA_REFERENCES = {
    "pion_quark_quasi_pdf": "arXiv:2601.12189 Eqs. (2.1)/(2.2)",
    "pion_quark_unpolarized_quasi_pdf": "arXiv:2601.12189 Eqs. (2.1)/(2.2)",
    "nucleon_quark_unpolarized_quasi_pdf": "arXiv:2601.12189 Eqs. (2.3)/(2.4)",
    "nucleon_quark_transversity_quasi_pdf": "arXiv:2601.12189 Eqs. (2.5)/(2.6)",
    "meson_quasi_da": "arXiv:2601.12189 Eqs. (2.7)/(2.8)",
    "pion_quark_quasi_gpd": "arXiv:2601.12189 Eqs. (2.9)/(2.10)",
    "pion_quark_unpolarized_quasi_gpd": "arXiv:2601.12189 Eqs. (2.9)/(2.10)",
    "nucleon_quark_quasi_gpd": "arXiv:2601.12189 Eqs. (2.11)/(2.12)",
    "nucleon_quark_unpolarized_quasi_gpd": "arXiv:2601.12189 Eqs. (2.11)/(2.12)",
    "nucleon_gluon_quasi_pdf": "arXiv:2601.12189 Appendix F Eqs. (F.6)/(F.7)",
    "pion_gluon_quasi_pdf": "arXiv:2601.12189 Appendix F Eqs. (F.8)/(F.9)",
}

FOURIER_ARTIFACT_DESCRIPTIONS = {
    "fourier_artifact": "Fourier result samples and diagnostics",
    "fit_info_artifact": "Tail-fit parameters and fit-quality diagnostics",
    "fourier_plot": "PDF plot of the Fourier-space result",
    "fourier_plot_image": "SVG companion for Markdown embedding",
    "extension_plot_re": "PDF plot of real-part extension quality",
    "extension_plot_re_image": "SVG companion for real-part extension quality",
    "extension_plot_im": "PDF plot of imaginary-part extension quality",
    "extension_plot_im_image": "SVG companion for imaginary-part extension quality",
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


def _format_fit_range(fit_range: Any, *, language: str) -> str:
    if fit_range is None:
        return "not available"
    return rf"$z^{{\rm min}}={_fmt(fit_range[0])}$ to $z^{{\rm max}}={_fmt(fit_range[1])}$"


def _format_grid(y_grid: np.ndarray, *, language: str) -> str:
    if y_grid.size == 0:
        return "not recorded"
    if y_grid.size == 1:
        return f"one point at $x={_fmt(y_grid[0])}$"
    diffs = np.diff(y_grid)
    if np.allclose(diffs, diffs[0], rtol=1e-7, atol=1e-12):
        return f"from $x={_fmt(y_grid[0])}$ to $x={_fmt(y_grid[-1])}$ with spacing $\\Delta x={_fmt(diffs[0])}$, for {y_grid.size} points"
    return f"nonuniform grid with {y_grid.size} points; preview `{_fmt_list(y_grid)}`"


def _tail_formula_text(result: dict[str, Any], *, language: str) -> str:
    gfix = str(result.get("gfix", "")).upper()
    order = str(result.get("order", "")).upper()
    observable = str(result.get("observable_backend", result.get("observable", "")))
    sector = str(result.get("sector", "")).lower()
    psi1_class = str(result.get("psi1_flavor_class", "heavy") or "heavy").lower()
    psi2_class = str(result.get("psi2_flavor_class", "heavy") or "heavy").lower()
    symmetry_guarantee = bool(result.get("symmetry_guarantee", False))
    orders = [item.strip().upper() for item in order.split(",") if item.strip()]
    if len(orders) > 1:
        article_lines = []
        implementation_lines = []
        mapping_scope_text = ""
        for item in orders:
            one = dict(result)
            one["order"] = item
            text = _tail_formula_text(one, language=language)

            article_part = text.split("### lamet-agent Implementation", 1)[0]
            implementation_part = text.split("### lamet-agent Implementation", 1)[1].split("### Parameter Correspondence", 1)[0]
            mapping_scope_text = text.split("### Parameter Correspondence", 1)[1]
            article_formula = article_part.split("$$", 2)[1]
            implementation_formula = implementation_part.split("$$", 2)[1]
            article_lines.append(f"$$\n{article_formula}\n$$")
            implementation_lines.append(f"$$\n{implementation_formula}\n$$")
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
    if gfix == "CG":
        implementation_tail += r"\,z^{-n}"

    if observable == "pion_quark_quasi_gpd" or observable.startswith("pion_quark_") and observable.endswith("_quasi_gpd"):
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
                "- $A_1,\\phi_1$ correspond to the incoming-momentum oscillatory term $e^{-i z P^z}$.",
                "- $A_3,\\phi_3$ correspond to the outgoing-momentum oscillatory term $e^{+i z P'^z}$.",
                "- $A_2,\\phi_2$ correspond to the non-oscillatory central term.",
                "- $\\tilde A_2,\\tilde\\phi_2$ correspond to the momentum-transfer term $e^{-i(P^z-P'^z)z}$.",
                "- $m$ is the fitted non-negative offset, so the effective decay rate is $m+\\Lambda_0$.",
                "- `Lambda0_gev` is the fixed offset $\\Lambda_0$ in the reparameterized decay rate, not a hard bound on a fitted $\\Lambda$.",
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
                "- $A_1,\\phi_1$ and $A'_1,\\phi'_1$ correspond to the incoming-momentum terms proportional to $e^{-i z P^z}$.",
                "- $A_3,\\phi_3$ and $A'_3,\\phi'_3$ correspond to the outgoing-momentum terms proportional to $e^{+i z P'^z}$.",
                "- $A_2,\\phi_2$ and $A'_2,\\phi'_2$ correspond to the central non-oscillatory LA/NLA terms.",
                "- $\\tilde A_2,\\tilde\\phi_2$ and $\\tilde A'_2,\\tilde\\phi'_2$ correspond to the momentum-transfer terms proportional to $e^{-i(P^z-P'^z)z}$.",
                "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$, while the primed amplitudes are the $1/|z|$ NLA corrections.",
                "- `Lambda0_gev` is the fixed offset $\\Lambda_0$ in the reparameterized decay rate, not a hard bound on a fitted $\\Lambda$.",
            ]
        scope_lines = [
            "The article formula is the full $\\pm z$ expression. The lamet-agent fit uses the explicit positive-$z$ branch, so ${\\rm sign}(z)=1$ and $|z|=z$ on the fitted interval.",
            "When `gfix=CG`, the implementation multiplies the positive-$z$ branch by the extra factor $z^{-n}$.",
        ]
    elif observable == "nucleon_quark_quasi_gpd" or observable.startswith("nucleon_quark_") and observable.endswith("_quasi_gpd"):
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
                "- $A_2,\\phi_2$ correspond to the forward-like central term.",
                "- $\\tilde A_2,\\tilde\\phi_2$ correspond to the momentum-transfer term $e^{-i(P^z-P'^z)z}$.",
                "- $m$ is the fitted non-negative offset, so the effective decay rate is $m+\\Lambda_0$.",
                "- `Lambda0_gev` is the fixed offset $\\Lambda_0$ in the reparameterized decay rate, not a hard bound on a fitted $\\Lambda$.",
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
                "- $A_2,\\phi_2$ and $A'_2,\\phi'_2$ correspond to the forward-like LA/NLA central terms.",
                "- $\\tilde A_2,\\tilde\\phi_2$ and $\\tilde A'_2,\\tilde\\phi'_2$ correspond to the momentum-transfer terms proportional to $e^{-i(P^z-P'^z)z}$.",
                "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$, while the primed amplitudes are the $1/|z|$ NLA corrections.",
                "- `Lambda0_gev` is the fixed offset $\\Lambda_0$ in the reparameterized decay rate, not a hard bound on a fitted $\\Lambda$.",
            ]
        scope_lines = [
            "The article formula is the full $\\pm z$ expression. The lamet-agent fit uses the explicit positive-$z$ branch, so ${\\rm sign}(z)=1$ and $|z|=z$ on the fitted interval.",
            "When `gfix=CG`, the implementation multiplies the positive-$z$ branch by the extra factor $z^{-n}$.",
        ]
    elif observable in {
        "pion_quark_quasi_pdf",
        "nucleon_quark_unpolarized_quasi_pdf",
        "nucleon_quark_transversity_quasi_pdf",
        "meson_quasi_da",
    } or "_quark_" in observable and observable.endswith("_quasi_pdf"):
        if observable == "pion_quark_quasi_pdf" or observable.startswith("pion_quark_"):
            phases_text = (
                "- In the implementation rewrite, $\\omega_2=0$, $\\omega_1=-P^z$, and $\\omega_3=+P^z$."
            )
        elif observable == "meson_quasi_da":
            phases_text = (
                "- In the implementation rewrite, $\\omega_1=-P^z$ and $\\omega_2=0$."
            )
        else:
            phases_text = (
                "- In the implementation rewrite, the only retained phase is the central frequency $\\omega_2=0$."
            )
        if observable == "pion_quark_quasi_pdf" or observable.startswith("pion_quark_"):
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
            "- The article form keeps explicit ${\\rm sign}(z)$ and $|z|$, while lamet-agent rewrites the same positive-$z$ branch as a sum over frequencies $\\omega_j$.",
            "- The amplitudes $A_j,\\phi_j$ map one-to-one between the two formulas, and the primed amplitudes give the NLA $1/|z|$ corrections when present.",
            "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$; `Lambda0_gev` is the fixed offset, not a hard bound on a fitted $\\Lambda$.",
        ]
        scope_lines = [
            "For these forward-like quark observables, the report distinguishes the article formula from the lamet-agent parameterized equivalent rewrite.",
            "The implementation fits only positive coordinates, so ${\\rm sign}(z)=1$ and $|z|=z$ on the fitted interval; `gfix=CG` adds the explicit factor $z^{-n}$ shown in the implementation formula.",
        ]
        if observable == "meson_quasi_da":
            if symmetry_guarantee:
                scope_lines.append(
                    "With `symmetry_guarantee=true`, lamet-agent forms "
                    "$h_{+}(z)=e^{+izP_z/2}h^R(z)$, discards $\\operatorname{Im}h_{+}$, "
                    "and rotates back as $h_{\\rm proj}(z)=e^{-izP_z/2}\\operatorname{Re}h_{+}(z)$. "
                    "Range selection, asymptotic fitting, extension plots, negative-$z$ completion, "
                    "and the ordinary $e^{+ix\\lambda}$ Fourier transform all use $h_{\\rm proj}$."
                )
            else:
                scope_lines.append(
                    "With `symmetry_guarantee=false`, lamet-agent applies no DA phase rotation "
                    "or real-part projection; range selection, asymptotic fitting, extension, and "
                    "the ordinary $e^{+ix\\lambda}$ Fourier transform use the input $h^R$ unchanged."
                )
    elif observable in {"nucleon_gluon_quasi_pdf", "nucleon_gluon_unpolarized_quasi_pdf"}:
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
            "- This is the implementation-oriented real-tail rewrite of the Appendix-F gluon form; the report does not claim a universal term-by-term correspondence beyond this specialized real-part ansatz.",
            "- $A$ controls the linear large-distance growth before exponential damping; $A'$ is the NLA constant correction when present.",
            "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$; `Lambda0_gev` is the fixed offset, not a hard bound on a fitted $\\Lambda$.",
        ]
        scope_lines = [
            "The article form is written with $|z|$ and the lamet-agent form uses the positive-$z$ implementation; `gfix=CG` adds the explicit factor $z^{-n}$ shown above.",
        ]
    elif observable in {"pion_gluon_quasi_pdf", "pion_gluon_unpolarized_quasi_pdf"}:
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
            "- This is the implementation-oriented real-tail rewrite of the Appendix-F gluon form; the report does not claim a universal term-by-term correspondence beyond this specialized real-part ansatz.",
            "- $A_2$ controls the linear large-distance part, while $A_2'$, $A_1$, and $\\phi$ parameterize the NLA constant and oscillatory corrections.",
            "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$; `Lambda0_gev` is the fixed offset, not a hard bound on a fitted $\\Lambda$.",
        ]
        scope_lines = [
            "The article form is written with $|z|$ and the lamet-agent form uses the positive-$z$ implementation; `gfix=CG` adds the explicit factor $z^{-n}$ shown above.",
        ]
    else:
        article_formula = rf"h^{{\rm {order}}}_{{\rm art}}(z)=\left[\sum_j A_j e^{{i\phi_j\,{{\rm sign}}(z)}} e^{{i\omega_j z}}\right]{article_tail}."
        implementation_formula = (
            rf"h^{{\rm {order}}}_{{\rm agent}}(z>0)=\left[\sum_j A_j e^{{i(\phi_j+\omega_j z)}}"
            + (r"+\sum_j \frac{A'_j}{z} e^{i(\phi'_j+\omega_j z)}" if order == "NLA" else "")
            + rf"\right]{implementation_tail}."
        )
        mapping_lines = [
            "- The implementation is the positive-$z$ rewrite of the article-style oscillatory tail.",
            "- $m$ is the fitted non-negative offset in the common decay rate $m+\\Lambda_0$; `Lambda0_gev` is the fixed offset, not a hard bound on a fitted $\\Lambda$.",
        ]
        scope_lines = [
            "The implementation fits only positive coordinates.",
        ]

    constraint_lines = []
    if observable in {"pion_quark_quasi_pdf", "pion_quark_unpolarized_quasi_pdf"} and sector == "valence":
        constraint_lines.append(
            "- Following arXiv:2601.12189, the fit input for the `valence` sector of `pion_quark_quasi_pdf` imposes $\\phi_2=\\phi'_2=0$, $A_3=A_1$, $\\phi_3=-\\phi_1$, $A'_3=A'_1$, and $\\phi'_3=-\\phi'_1$."
        )
    if observable == "meson_quasi_da" and psi1_class == "light" and psi2_class == "light":
        constraint_lines.append(
            "- Following arXiv:2601.12189, the fit input for `meson_quasi_da` with `psi1_flavor_class=light, psi2_flavor_class=light` imposes $A_2=A_1$, $\\phi_2=-\\phi_1$, $A'_2=A'_1$, and $\\phi'_2=-\\phi'_1$."
        )
    if observable == "meson_quasi_da" and psi1_class == "light" and psi2_class == "heavy":
        constraint_lines.append(
            "- Following arXiv:2601.12189, the fit input for `meson_quasi_da` with `psi1_flavor_class=light, psi2_flavor_class=heavy` imposes $A_1=A'_1=0$."
        )
    if observable == "meson_quasi_da" and psi1_class == "heavy" and psi2_class == "light":
        constraint_lines.append(
            "- Following arXiv:2601.12189, the fit input for `meson_quasi_da` with `psi1_flavor_class=heavy, psi2_flavor_class=light` imposes $A_2=A'_2=0$."
        )
    mapping_lines.extend(constraint_lines)

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
    phase = "x\\lambda"
    rotation = ""
    if str(result.get("target_observable", "")).lower() == "da":
        if bool(result.get("symmetry_guarantee", False)):
            rotation = (
                "For `target_observable=da` with `symmetry_guarantee=true`, before range selection "
                "and large-distance fitting, lamet-agent computes $h_{+}=e^{+i\\lambda/2}h^R$, "
                "sets $\\operatorname{Im}h_{+}=0$, and defines "
                "$h_{\\rm proj}=e^{-i\\lambda/2}\\operatorname{Re}h_{+}$. The extrapolation and "
                "Fourier transform use the real and imaginary parts of $h_{\\rm proj}$.\n\n"
            )
        else:
            rotation = (
                "For `target_observable=da` with `symmetry_guarantee=false`, the input matrix "
                "element is not phase-rotated or projected before extrapolation.\n\n"
            )
    convention = rotation + f"This stage uses the $e^{{+i{phase}}}$ Fourier convention, i.e. $q(x)=\\frac{{\\Delta\\lambda}}{{2\\pi}}\\sum_\\lambda e^{{+i{phase}}}h(\\lambda)$; the corresponding real/imaginary decomposition is shown below."
    if part == "re":
        return (
            f"{convention}\n\n"
            "$$\n"
            "q_{\\rm re}(x)=\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
            f"\\cos({phase})\\,\\mathrm{{Re}}\\,h(\\lambda).\n"
            "$$"
        )
    if part == "im":
        return (
            f"{convention}\n\n"
            "$$\n"
            "q_{\\rm im}(x)=-\\frac{\\Delta\\lambda}{2\\pi}\\sum_{\\lambda}"
            f"\\sin({phase})\\,\\mathrm{{Im}}\\,h(\\lambda).\n"
            "$$"
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
    )


def _field_definitions(result: dict[str, Any], *, language: str) -> list[str]:
    lines = [
        "| Entry | Meaning |",
        "|---|---|",
        "| Observable | Physical matrix element transformed by this stage. |",
        "| Sector | Requested physics projection; PDF/GPD accept `sea`, `valence`, `singlet`, and `full`, while DA uses `full`. |",
        "| Gauge-link treatment / tail order | $\\mathrm{order}$ selects LA or NLA; $\\mathrm{gfix}=\\mathrm{CG}$ adds $z^{-n}$ to the base tail. |",
        "| Active fitted part | Execution channel resolved from `sector`, or supplied manually when `sector` is omitted; `both` fits $\\mathrm{Re}\\,\\tilde h^R$ and $\\mathrm{Im}\\,\\tilde h^R$ together, while `re` or `im` fits one channel. |",
        "| Coordinate unit | Input coordinates and `scheme_scan` ranges are fixed physical distances in fm; the fit uses $z_{\\rm GeV^{-1}}=z_{\\rm fm}/(\\hbar c)$ and $\\lambda=\\bar P^z z_{\\rm GeV^{-1}}$, where $\\bar P^z=(P_i^z+P_f^z)/2$ (and $P_i^z=P_f^z$ for forward kinematics). |",
        "| Posterior-prior error scale | The mean fit gives $\\bar p_i\\pm\\sigma_{p_i}$; resampled fits use $p_i=\\bar p_i\\pm s\\sigma_{p_i}$. |",
    ]
    if str(result.get("target_observable", "")).lower() in {"pdf", "gpd"}:
        lines[3] = "| Sector | Quark PDF/GPD projection: `sea`, `valence`, `singlet`, or `full`; gluon uses `full`. |"
        lines.insert(3, "| Polarization | Operator family: `unpolarized`, `helicity`, or `transversity`. |")
    return lines


def _projection_text(result: dict[str, Any], *, language: str) -> list[str]:
    sector = str(result.get("sector", "full")).lower()
    part = str(result.get("part", "both")).lower()
    scale = float(result.get("output_scale", 1.0))
    target = str(result.get("target_observable", "pdf")).lower()
    parton = str(result.get("parton", "quark")).lower()
    polarization = str(result.get("polarization", "unpolarized")).lower()
    truncated = str(result.get("short_distance_policy", "full_from_zero")) == "truncate_missing"
    missing = result.get("missing_short_distance_coord", [])
    if target == "da":
        intro = (
            f"This run uses `sector={sector}`, resolved internally to `part={part}`, "
            f"`output_scale={_fmt(scale)}`, and `im_flip_for_ft={result.get('im_flip_for_ft', False)}`. "
            "With the vector/tensor quark extended-distribution convention "
            "$q_{\\rm ext}(x)=q(x)$ for $x>0$ and $q_{\\rm ext}(-x)=-\\bar q(x)$, "
            "the coordinate-space matrix element obeys "
            "$h(\\lambda)=\\int dx\\,e^{-ix\\lambda}q_{\\rm ext}(x)$."
        )
        if truncated:
            intro += f" This input misses short-distance coordinates {missing}; these points are omitted from the Fourier sum, so the output is a short-distance-truncated projection."
        if sector == "sea":
            meaning = "`sea` is reconstructed from the full vector/tensor quark distribution as $\\bar q(x)=-q_{\\rm ext}(-x)$, using one joint fit of the real and imaginary matrix-element channels; for a nonzero-skewness GPD, the reflected ERBL region remains a quark-antiquark amplitude rather than an antiquark density."
        elif sector == "singlet":
            meaning = "`singlet` returns the per-flavor C-even combination $q(x)+\\bar q(x)$; a strict flavor-singlet distribution additionally sums this combination over quark flavors."
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
    intro = (
        f"This run uses `sector={sector}`, resolved internally to `part={part}`, "
        f"`output_scale={_fmt(scale)}`, and `im_flip_for_ft={result.get('im_flip_for_ft', False)}`."
    )
    if sector == "manual":
        intro = (
            f"This run omits a named sector and directly uses `part={part}`, "
            f"`output_scale={_fmt(scale)}`, and `im_flip_for_ft={result.get('im_flip_for_ft', False)}`."
        )
        meaning = "The reported result is a manual real/imaginary projection; these numerical controls do not assign a named `sea`, `valence`, `singlet`, or `full` interpretation by themselves."
    elif parton == "gluon":
        intro += " Gluon operator families use only the full complex Fourier result; quark/antiquark sector projections are not applied."
        meaning = "`full` preserves the gluon result without assigning quark `sea`, `valence`, or `singlet` semantics."
    elif target == "gpd":
        sign = "+" if polarization == "helicity" else "-"
        negative_y_relation = (
            "$\\Delta q_{\\rm ext}(-x)=+\\Delta\\bar q(x)$"
            if polarization == "helicity"
            else "$q_{\\rm ext}(-x)=-\\bar q(x)$"
        )
        intro += (
            " GPD sectors are constructed after the full complex paired reconstruction and Fourier transform; "
            "no real or imaginary coordinate-space channel is discarded before fitting. "
            f"The signed-extension convention is {negative_y_relation}."
        )
        meaning = {
            "sea": f"`sea` returns $\\bar H(y,\\xi)={sign}H(-y,\\xi)$.",
            "valence": f"`valence` returns $H(y,\\xi){'-' if sign == '+' else '+'}H(-y,\\xi)$.",
            "singlet": f"`singlet` returns $H(y,\\xi){'+' if sign == '+' else '-'}H(-y,\\xi)$ per flavor; a strict flavor singlet also sums over flavors.",
            "full": "`full` preserves the complete complex $H(y,\\xi)$ reconstructed from the paired flows.",
        }.get(sector, "The selected part is reported without an additional named projection.")
        meaning += " The upper sign is the helicity convention; unpolarized and transversity use the lower sign."
    elif polarization == "helicity":
        intro += " The helicity convention is $\\Delta q_{\\rm ext}(-x)=+\\Delta\\bar q(x)$."
        meaning = {
            "sea": "`sea` is reconstructed with one joint fit as $\\Delta\\bar q(x)=+\\Delta q_{\\rm ext}(-x)$.",
            "valence": "`valence` uses the sine/odd projection and returns $\\Delta q(x)-\\Delta\\bar q(x)$.",
            "singlet": "`singlet` uses the cosine/even projection and returns the per-flavor C-even combination $\\Delta q(x)+\\Delta\\bar q(x)$; a strict flavor singlet also sums over flavors.",
            "full": "`full` uses one real/imaginary joint fit and reconstructs the complete extended helicity distribution $\\Delta q_{\\rm ext}(x)$.",
        }.get(sector, "The selected part is reported without an additional named projection.")
    else:
        intro += " The unpolarized/transversity convention is $q_{\\rm ext}(-x)=-\\bar q(x)$."
        meaning = {
            "sea": "`sea` is reconstructed with one joint fit as $\\bar q(x)=-q_{\\rm ext}(-x)$.",
            "valence": "`valence` uses the cosine/even projection and returns $q(x)-\\bar q(x)$.",
            "singlet": "`singlet` uses the sine/odd projection and returns the per-flavor C-even combination $q(x)+\\bar q(x)$; a strict flavor singlet also sums over flavors.",
            "full": "`full` uses one real/imaginary joint fit and reconstructs the complete extended distribution $q_{\\rm ext}(x)$.",
        }.get(sector, "The selected part is reported without an additional named projection.")
    if truncated:
        intro += f" This input misses short-distance coordinates {missing}; these points are omitted from the Fourier sum, so the output is a short-distance-truncated projection."
    if target == "gpd":
        family, decomposition = {
            "unpolarized": ("the vector family $H,E$ for a spin-$1/2$ hadron (only $H$ for a spin-0 hadron)", "$H/E$"),
            "helicity": ("the axial family $\\widetilde H,\\widetilde E$ for a spin-$1/2$ hadron", "$\\widetilde H/\\widetilde E$"),
            "transversity": ("the tensor family $H_T,E_T,\\widetilde H_T,\\widetilde E_T$ for a spin-$1/2$ hadron", "the tensor-GPD"),
        }.get(polarization, ("the recorded operator family", "the corresponding invariant-GPD"))
        meaning += f" This run labels {family}, but the input is a projected quasi-GPD matrix element; `polarization` alone does not perform {decomposition} decomposition."
        anchor = str(result.get("bilocal_anchor", "mid_at_0"))
        completion = str(result.get("gpd_completion_mode", "single_flow"))
        partner = str(result.get("hermitian_partner_id", ""))
        meaning += (
            f" The input uses `bilocal_anchor={anchor}` and is converted to the centered convention with "
            "$h_{fi}^{\\rm mid}(z)=e^{-i(P_f^z-P_i^z)z/2}h_{fi}^{\\bar\\psi@0}(z)$."
        )
        if completion == "paired_flow":
            meaning += (
                f" Its negative-$z$ branch is reconstructed from the exchanged-flow partner `{partner}` through "
                "$h_{fi}^{\\rm mid}(-z)=h_{if}^{\\rm mid}(z)^*$ for the supported Hermitian current phase $\\eta_\\Gamma=+1$."
            )
        if sector == "sea":
            meaning += " The antiquark interpretation applies only in the negative-$x$ DGLAP region; the ERBL region $|x|<|\\xi|$ is a quark-antiquark amplitude, not a pure sea density."
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
    title = "### Range Selection Candidates"
    header = (
        "| # | range label [fm] | selected | $Q$ | logGBF | $\\chi^2/{\\rm dof}$ |"
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
    title = "### Fit-Model Average Candidates"
    header = (
        "| # | model | order | prior width | mean sample weight | $Q$ | logGBF | $\\chi^2/{\\rm dof}$ | failures | selected range [fm] | $z_{\\rm ext}^{\\rm max}$ [fm] | smooth |"
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

    header_title = "### Fit-Model Parameters"
    header = "| # | label | " + " | ".join(f"`{label}`" for label in param_labels) + " |"
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
        return [
            "`linear` smoothing means",
            "$$",
            "h_{\\rm ext}(z)=[1-w(z)]h_{\\rm data}(z)+w(z)h_{\\rm fit}(z),\\quad "
            "w(z)=\\frac{z-z_{\\rm min}}{z_{\\rm max}-z_{\\rm min}}.",
            "$$",
        ]
    return ["`none` smoothing switches directly to the fitted tail."]


def _figure_block(artifacts: dict[str, Any], *, language: str) -> list[str]:
    labels = {
        "fourier_plot": "Fourier result",
        "extension_plot_re": "Real-part extension",
        "extension_plot_im": "Imaginary-part extension",
    }
    lines = ["## Figures and Visual Assessment"]
    for key, title in labels.items():
        pdf_value = artifacts.get(key)
        image_value = artifacts.get(f"{key}_image") or pdf_value
        lines.extend(["", f"### {title}"])
        if image_value:
            lines.extend(["", f"![{title}]({image_value})"])
            if pdf_value:
                lines.append(f"[PDF artifact]({pdf_value})")
        else:
            lines.append("Not available.")
    return lines


def _settings_table(
    *,
    result: dict[str, Any],
    observable: str,
    observable_text: str,
    gfix: str,
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
    if missing:
        short_distance_text = (
            f"`truncate_missing`; omitted short-distance coordinates {missing}; Fourier starts at {_fmt(result.get('fourier_positive_coord_start'))}"
        )
    else:
        short_distance_text = "`full_from_zero`; no omitted short-distance coordinate"
    rows = [
        ("Observable", f"`{observable}` ({observable_text})"),
        ("Sector", f"`{result.get('sector', 'full')}`"),
        ("Gauge-link treatment / tail order", f"`{gfix}` / `{order}`"),
        ("Active fitted part", f"`{result.get('part', 'both')}`"),
        ("Resampling mode", f"`{result.get('resample_mode', 'not recorded')}`"),
        ("Coordinate unit", r"fm input and scan; fit unit $\mathrm{GeV}^{-1}$"),
        ("Decay offset", f"$\\Lambda_0={_fmt(result.get('Lambda0_gev'))}$"),
        ("Output scale", f"$q(x)\\rightarrow {_fmt(result.get('output_scale', 1.0))}\\,q(x)$"),
        ("Short-distance treatment", short_distance_text),
        ("Best fit range [fm]", fit_range_text),
        ("Extension endpoint [fm]", z_ext_text),
        ("Fourier grid", _format_grid(y_grid, language=language)),
    ]
    if str(result.get("target_observable", "")).lower() in {"pdf", "gpd"}:
        rows[1:1] = [
            ("Polarization", f"`{result.get('polarization', 'not recorded')}`"),
            ("Current operator", f"`{result.get('current_operator', 'not recorded')}`"),
            ("Parton", f"`{result.get('parton', 'not recorded')}`"),
            ("Hadron", f"`{result.get('hadron', 'not recorded')}`"),
        ]
    if str(result.get("target_observable", "")).lower() == "gpd":
        partner = str(result.get("hermitian_partner_id", "")) or "not used"
        rows[1:1] = [
            ("Bilocal anchor", f"`{result.get('bilocal_anchor', 'mid_at_0')}`"),
            ("Negative-z completion", f"`{result.get('gpd_completion_mode', 'single_flow')}`; partner `{partner}`"),
            (
                "Signed momentum transfer",
                f"$\\Delta P_z={_fmt(result.get('delta_momentum_gev'))}$ GeV from `{result.get('phase_momentum_source', 'not recorded')}`",
            ),
        ]
    if observable == "meson_quasi_da":
        rows.insert(2, ("DA flavor classes", f"`psi1={result.get('psi1_flavor_class', 'heavy')}`, `psi2={result.get('psi2_flavor_class', 'heavy')}`"))
        rows.insert(3, ("DA symmetry guarantee", f"`symmetry_guarantee={str(bool(result.get('symmetry_guarantee', False))).lower()}`"))
    header = "| Quantity | Value |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _artifact_field_table(kind: str, *, language: str, target_observable: str = "pdf") -> list[str]:
    if kind == "result":
        rows = [
            ("`values`", "Complex final Fourier samples after fit-model averaging or best-model selection, with dimensions `(resample, x)`."),
            ("coordinate `x`", "Fourier momentum-fraction grid."),
            ("attr `resample`", "Resampling mode recorded by `EnsembleData`."),
            ("attr `ft_re_mean` / `ft_im_mean`", "Final real/imaginary central values after fit-model averaging or best-model selection."),
            ("attr `ft_re_stat_sdev` / `ft_im_stat_sdev`", "Statistical standard deviations from bootstrap/jackknife samples."),
            ("attr `ft_re_sys_sdev` / `ft_im_sys_sdev`", "Weighted spread among fit-model candidates at fixed selected range."),
            ("attr `scheme_labels`", "Fit-model labels at the selected range."),
            ("attr `fit_failures`", "Number of failed resampled tail fits in each fit model."),
            ("attrs `fit_model_*`", "Per-sample fit-model weights and diagnostics for `(order, prior width)` candidates."),
            ("attrs `candidate_scheme_*`", "Sample-average range-scan diagnostics used before model averaging."),
            ("attr `selection_mode`", "Two-stage selection mode: range selection followed by fit-model averaging or best-model selection."),
            ("attrs `momentum_gev`, `final_momentum_gev`, `lattice_spacing_fm`", "Momentum and lattice-spacing metadata."),
            ("attrs `sector`, `gfix`, `order`, `observable`, `part`, `output_scale`, `symmetry_guarantee`, `psi1_flavor_class`, `psi2_flavor_class`", "Physics projection, gauge-link treatment, formula choices, execution channel, final output normalization, DA symmetry projection, and flavor-class metadata."),
        ]
        if target_observable in {"pdf", "gpd"}:
            rows[-1:] = [
                ("attrs `observable`, `observable_backend`, `parton`, `hadron`, `current_operator`, `polarization`, `sector`", "Resolved observable, numerical tail backend, operator provenance, and physics projection."),
                ("attrs `gfix`, `order`, `part`, `output_scale`, `symmetry_guarantee`, `psi1_flavor_class`, `psi2_flavor_class`", "Gauge-link treatment, formula choices, execution channel, final normalization, DA symmetry projection, and flavor-class metadata."),
            ]
        if target_observable == "gpd":
            rows.extend(
                [
                    ("attrs `bilocal_anchor`, `delta_momentum_gev`, `phase_momentum_source`", "Authored bilocal layout and the signed momentum transfer used to convert it to the centered convention."),
                    ("attrs `hermitian_partner_id`, `hermiticity_phase`, `gpd_completion_mode`", "Exchanged-flow provenance and the Hermiticity convention used to reconstruct negative z."),
                ]
            )
    else:
        rows = [
            ("`values`", "Fit-parameter samples with dimensions `(resample, scheme, parameter)`."),
            ("coordinates `scheme`, `parameter`", "Scheme labels and fitted parameter names."),
            ("attr `fit_params`", "Tail-fit parameters for every scheme and resample."),
            ("attr `fit_param_center` / `fit_param_sdev`", "Sample mean and statistical standard deviation of fit parameters."),
            ("attrs `fit_chi2`, `fit_dof`, `fit_q`, `fit_chi2_dof`", "Per-resample fit quality diagnostics."),
            ("attrs `fit_chi2_center`, `fit_chi2_dof_center`, `fit_q_center`", "Sample-averaged fit quality diagnostics for each scheme."),
            ("attrs `mean_fit_params`, `mean_fit_chi2`, `mean_fit_dof`, `mean_fit_q`, `mean_fit_log_gbf`", "Initial sample-average fit results used to seed resampled fits."),
            ("attrs `fit_model_*`", "Per-sample weights and diagnostics for fixed-range fit-model averaging."),
            ("attrs `candidate_scheme_*`, `selection_mode`", "Range-scan diagnostics and the two-stage selection mode."),
        ]
        if target_observable == "gpd":
            rows.append(
                ("attrs `bilocal_anchor`, `delta_momentum_gev`, `phase_momentum_source`, `hermitian_partner_id`, `hermiticity_phase`, `gpd_completion_mode`", "GPD bilocal-layout conversion and exchanged-flow completion provenance shared with the Fourier result.")
            )
    header = "| Field | Meaning |"
    lines = [header, "|---|---|"]
    for field, description in rows:
        lines.append(f"| {field} | {description} |")
    return lines


def _artifact_help(*, language: str, target_observable: str = "pdf") -> list[str]:
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
        *_artifact_field_table("result", language="en", target_observable=target_observable),
        "",
        "### `fourier_fit_info.nc` Field Reference",
        *_artifact_field_table("fit_info", language="en", target_observable=target_observable),
    ]


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    header = "| File | Description |"
    lines = [header, "|---|---|"]
    for key in FOURIER_ARTIFACT_ORDER:
        value = artifacts.get(key)
        if not value:
            continue
        desc = FOURIER_ARTIFACT_DESCRIPTIONS[key]
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
    gfix = str(result.get("gfix", "not recorded"))
    order = str(result.get("order", "not recorded"))
    y_grid = np.asarray(result.get("y_grid", []), dtype=float)
    schemes = list(result.get("scheme_results", []))
    selected_model = schemes[0] if schemes else {}
    fit_range_text = _format_fit_range(selected_model.get("fit_range"), language=language)
    z_ext_max = selected_model.get("z_ext_max", "not available")

    title = "# Fourier Transform Analysis Report"
    abstract = f"This report summarizes the Fourier transform for `{observable}` ({observable_text}) using `{gfix}` / `{order}` large-distance extrapolation."
    transform_text = _fourier_transform_text(result, language="en")
    lines = [
        title,
        "",
        "## Abstract",
        abstract,
        "",
        "## Analysis Setup",
        *_settings_table(result=result, observable=observable, observable_text=observable_text, gfix=gfix, order=order, fit_range_text=fit_range_text, z_ext_max=z_ext_max, y_grid=y_grid, language="en"),
        "",
        "### Field Definitions",
        *_field_definitions(result, language="en"),
        "",
        *_projection_text(result, language="en"),
        "",
        "## Large-Distance Extrapolation",
        _tail_formula_text(result, language="en"),
        "",
        "## Fourier Transform Setup",
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
        *_artifact_help(language="en", target_observable=str(result.get("target_observable", ""))),
    ]
    return "\n".join(lines) + "\n"


def write_fourier_report(
    *,
    result: dict[str, Any],
    summary: dict[str, Any] | None,
    artifacts: dict[str, Any] | None,
    path: str | Path,
    report_language: str = "en",
    backend: str = "",
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Path]:
    """Write one Fourier report and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_artifacts = markdown_artifact_paths(
        artifacts,
        base_dir=output.parent,
        path_keys=FOURIER_ARTIFACT_ORDER,
    )
    markdown = build_fourier_report_markdown(result=result, summary=summary, artifacts=report_artifacts, language="en")
    output.write_text(markdown, encoding="utf-8")
    if report_language.lower() == "ch":
        translated = translate_markdown_report(
            markdown,
            backend=backend,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        target, _language = _report_target(output, report_language)
        target.write_text(translated, encoding="utf-8")
        return {"report": target}
    return {"report": output}


def write_fourier_stage_report(
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
    """Write one report summarizing all Fourier jobs in a stage."""
    output = Path(path)
    target, language = _report_target(output, report_language)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output
    first = jobs[0]["result"]
    language = "en"

    def artifact_paths(item: dict[str, Any]) -> dict[str, str]:
        raw = item.get("artifacts", {})
        return markdown_artifact_paths(
            raw,
            base_dir=output.parent,
            path_keys=(*FOURIER_ARTIFACT_ORDER, *(key for key in raw if key.startswith("fourier_overlay_"))),
        )

    observable = str(first.get("observable", ""))
    observable_text = OBSERVABLE_TEXT.get(observable, observable or "not recorded")
    gfix = str(first.get("gfix", "not recorded"))
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
    fit_range_text = "see the per-momentum diagnostics below"
    z_ext_max = finite_z_ext[0] if same_z_ext else "see the per-momentum diagnostics below"
    transform_text = _fourier_transform_text(first, language=language)
    all_jobs = jobs + list(systematics_jobs or [])
    use_systematics_table = bool(systematics_jobs)
    momentum_labels = {}
    for item in all_jobs:
        initial = item["result"].get("momentum_gev")
        final = item["result"].get("final_momentum_gev")
        momentum_labels[item["job_id"]] = (
            "n/a"
            if initial is None
            else rf"$P^z={float(initial):.2f}$"
            if final is None
            else rf"$P_i^z={float(initial):.2f},\ P_f^z={float(final):.2f}$"
        )
    lines = [
        "# Fourier Transform Stage Report",
        "",
        f"This report summarizes all Fourier-transform jobs in this stage for `{observable}` ({observable_text}).",
        "",
        "## Job Summary",
        (
            "| job | momentum [GeV] | $z_s$ [fm] | selected range | output | plot |"
            if use_systematics_table
            else "| job | momentum [GeV] | selected range | output | plot |"
        ),
        (
            "|---|---:|---:|---|---|---|"
            if use_systematics_table
            else "|---|---:|---|---|---|"
        ),
    ]
    for item in all_jobs:
        result = item["result"]
        momentum_text = momentum_labels[item["job_id"]]
        artifacts = artifact_paths(item)
        if use_systematics_table:
            lines.append(
                f"| `{item['job_id']}` | {momentum_text} | {_fmt(result.get('zs_fm'))} | "
                f"{result.get('selected_range_label', 'n/a')} | "
                f"{artifacts.get('fourier_artifact', 'n/a')} | "
                f"{artifacts.get('fourier_plot', 'n/a')} |"
            )
        else:
            lines.append(
                f"| `{item['job_id']}` | {momentum_text} | "
                f"{result.get('selected_range_label', 'n/a')} | "
                f"{artifacts.get('fourier_artifact', 'n/a')} | "
                f"{artifacts.get('fourier_plot', 'n/a')} |"
            )
    stage_artifacts = artifact_paths(jobs[0])
    overlay_images = [value for key, value in sorted(stage_artifacts.items()) if key.startswith("fourier_overlay_image_")]
    if overlay_images:
        for image in overlay_images:
            stem = Path(image).stem
            label = stem[3:-5] if stem.startswith("ft_") and stem.endswith("_xdep") else stem
            title = f"{label} ensemble overview"
            lines.extend(["", f"## {title}", "", f"![{title}]({image})"])
    lines.extend(
        [
            "",
            "## Analysis Setup",
            *_settings_table(
                result=first,
                observable=observable,
                observable_text=observable_text,
                gfix=gfix,
                order=order,
                fit_range_text=fit_range_text,
                z_ext_max=z_ext_max,
                y_grid=y_grid,
                language=language,
            ),
            "",
            "### Field Definitions",
            *_field_definitions(first, language=language),
            "",
            *_projection_text(first, language=language),
            "",
            "## Large-Distance Extrapolation",
            _tail_formula_text(first, language=language),
            "",
            "## Fourier Transform Setup",
            transform_text,
            "",
            "## Fit Quality and Model Diagnostics",
            ]
        )
    lines.append(
        "This stage first scans `zmin_fm × zmax_fm` on the sample-average matrix element, selects the largest-`logGBF` range among candidates passing $Q\\ge0.05$, and falls back to the largest-$Q$ successful range if none passes. The selected range is then fixed; range variation is not part of model averaging."
    )
    lines.append(
        "With `model_average=true`, each resample sample refits the `(order, prior width)` candidates at fixed range and fixed gfix, then uses that sample's normalized evidence weight $w_{s,m}=\\exp(\\log\\mathrm{GBF}_{s,m}-\\max_n\\log\\mathrm{GBF}_{s,n})/\\sum_k\\exp(\\log\\mathrm{GBF}_{s,k}-\\max_n\\log\\mathrm{GBF}_{s,n})$. With `model_average=false`, each sample selects the largest-`logGBF` candidate after the $Q$ gate."
    )
    lines.extend(
        [
            "",
            "| job | momentum [GeV] | selected range [fm] | selected fit range [fm] | omitted short z [fm] | $\\chi^2/{\\rm dof}$ range | fit failures |",
            "|---|---:|---|---|---|---:|---:|",
        ]
    )
    for item in jobs:
        result = item["result"]
        momentum_text = momentum_labels[item["job_id"]]
        schemes = list(result.get("scheme_results", []))
        selected_model = schemes[0] if schemes else {}
        chi2 = np.asarray(result.get("fit_model_chi2_dof", []), dtype=float)
        finite = chi2[np.isfinite(chi2)]
        chi_text = "n/a" if finite.size == 0 else f"{_fmt(np.min(finite))} to {_fmt(np.max(finite))}"
        missing = result.get("missing_short_distance_coord", [])
        lines.append(
            f"| `{item['job_id']}` | {momentum_text} | "
            f"{result.get('selected_range_label', 'n/a')} | "
            f"{_format_fit_range(selected_model.get('fit_range'), language=language)} | "
            f"{missing if missing else 'none'} | "
            f"{chi_text} | "
            f"{int(np.sum(np.asarray(result.get('fit_failures', []), dtype=float)))} |"
        )
    lines.extend(["", *_smooth_explanation(first, language=language)])
    lines.extend(
        [
            "",
            "- The `range grid` denotes range candidates, not the model-averaging space; the model candidates are `(order, prior width)` at fixed range.",
            "- `gfix` is inherited from correlator provenance or declared for an external input, and is not model averaged.",
        ]
    )
    for item in jobs:
        result = item["result"]
        momentum_text = momentum_labels[item["job_id"]]
        lines.extend(
            [
                "",
                f"### `{item['job_id']}`: {momentum_text} GeV",
                "",
                *_range_selection_table(result, language=language),
                "",
                *_fit_model_table(result, language=language),
                "",
                *_fit_model_parameter_table(result, language=language),
            ]
        )
    lines.append("")
    lines.append("## Figures and Visual Assessment")
    for item in jobs:
        result = item["result"]
        momentum_text = momentum_labels[item["job_id"]]
        artifacts = artifact_paths(item)
        lines.extend(["", f"### `{item['job_id']}`: {momentum_text} GeV"])
        for key, title in (
            ("fourier_plot", "Fourier result"),
            ("extension_plot_re", "Real-part extension"),
            ("extension_plot_im", "Imaginary-part extension"),
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
                lines.append("Not available.")
    lines.extend(["", "## Output Artifacts"])
    lines.extend(
        [
            "| File | Description |",
            "|---|---|",
        ]
    )
    for item in jobs:
        artifacts = artifact_paths(item)
        for key in FOURIER_ARTIFACT_ORDER:
            value = artifacts.get(key)
            if value:
                desc = FOURIER_ARTIFACT_DESCRIPTIONS[key]
                lines.append(f"| [{Path(value).name}]({value}) | `{item['job_id']}`: {desc} |")
    stage_artifacts = artifact_paths(jobs[0])
    for key, value in sorted(stage_artifacts.items()):
        if not key.startswith("fourier_overlay_"):
            continue
        stem = Path(value).stem
        label = stem[3:-5] if stem.startswith("ft_") and stem.endswith("_xdep") else stem
        desc = f"Fourier overlay for ensemble {label}"
        lines.append(f"| [{Path(value).name}]({value}) | {desc} |")
    lines.extend(["", *_artifact_help(language=language, target_observable=str(first.get("target_observable", "")))])
    markdown = "\n".join(lines) + "\n"
    output.write_text(markdown, encoding="utf-8")
    if report_language.lower() == "ch":
        translated = translate_markdown_report(
            markdown,
            backend=backend,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        _target, _language = _report_target(output, report_language)
        _target.write_text(translated, encoding="utf-8")
        report_path = _target
    return {"report": report_path}
