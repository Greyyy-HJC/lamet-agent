"""Markdown reporting helpers for the correlator-analysis stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lamet_agent.core.reporting import (
    format_report_list,
    format_report_value,
    markdown_artifact_paths,
    resolve_report_target,
)


CORRELATOR_ARTIFACT_DESCRIPTIONS = {
    "bare_artifact": ("Bare matrix element samples (EnsembleData NetCDF)", "裸矩阵元样本（EnsembleData NetCDF）"),
    "summary_plot": ("PDF plot of the bare matrix element versus Wilson-line length", "裸矩阵元随 Wilson 线长度变化的 PDF 图"),
    "summary_plot_image": ("SVG companion for Markdown embedding", "供 Markdown 嵌入的裸矩阵元 SVG 图"),
    "tuning_log": ("Window tuning and sample-average fit-quality log", "窗口选择和样本平均拟合质量日志"),
    "sample_log": ("Per-sample and per-z fit-quality log", "逐样本、逐 z 拟合质量日志"),
    "E0_artifact": ("Stage-level dispersion-relation table (NetCDF)", "阶段级色散关系数据表（NetCDF）"),
    "dispersion_relation_plot": ("Stage-level dispersion-relation PDF", "阶段级色散关系 PDF 图"),
    "dispersion_relation_image": ("Stage-level dispersion-relation SVG", "阶段级色散关系 SVG 图"),
}

CORRELATOR_ARTIFACT_ORDER = (
    "bare_artifact",
    "summary_plot",
    "summary_plot_image",
    "tuning_log",
    "sample_log",
    "E0_artifact",
    "dispersion_relation_plot",
    "dispersion_relation_image",
)


def _job_settings_table(result: dict[str, Any], *, language: str) -> list[str]:
    if language == "zh":
        rows = [
            ("拟合形式", f"`{result.get('fitting_form', 'not recorded')}`"),
            ("拟合对象", f"`{result.get('fit_scope', 'not recorded')}`"),
            ("拟合策略", f"`{result.get('fit_strategy', 'not recorded')}`"),
            ("拟合模式", f"`{result.get('fit_mode', 'not recorded')}`"),
            ("Model average", f"`{result.get('model_average', 'not recorded')}`"),
            ("选择规则", f"`{result.get('selection_rule', 'not recorded')}`"),
            ("重采样", f"`{result.get('resample_mode', 'not recorded')}`，共 {result.get('n_samples', 'n/a')} 个样本"),
            ("z 网格", format_report_list(result.get("z_values", []))),
            ("调参 z", format_report_list(result.get("tune_z_values", [result.get("tune_z")] if result.get("tune_z") is not None else []))),
            ("correlator_rescale", format_report_value(result.get("correlator_rescale"))),
        ]
        header = "| 条目 | 数值或设置 |"
    else:
        rows = [
            ("Fitting form", f"`{result.get('fitting_form', 'not recorded')}`"),
            ("Fit scope", f"`{result.get('fit_scope', 'not recorded')}`"),
            ("Fit strategy", f"`{result.get('fit_strategy', 'not recorded')}`"),
            ("Fit mode", f"`{result.get('fit_mode', 'not recorded')}`"),
            ("Model average", f"`{result.get('model_average', 'not recorded')}`"),
            ("Selection rule", f"`{result.get('selection_rule', 'not recorded')}`"),
            ("Resampling", f"`{result.get('resample_mode', 'not recorded')}` with {result.get('n_samples', 'n/a')} samples"),
            ("z grid", format_report_list(result.get("z_values", []))),
            ("Tuning z values", format_report_list(result.get("tune_z_values", [result.get("tune_z")] if result.get("tune_z") is not None else []))),
            ("correlator_rescale", format_report_value(result.get("correlator_rescale"))),
        ]
        header = "| Quantity | Value |"
    lines = [header, "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def _window_text(result: dict[str, Any], *, language: str) -> list[str]:
    specs = result.get("shared_window_specs")
    if not specs:
        return ["No shared window metadata was recorded." if language == "en" else "未记录共享窗口信息。"]
    if not isinstance(specs, list):
        specs = [specs]
    lines = [
        "| scope | strategy | nstate | pt2 window | pt3 window | n_data | n_params |"
        if language == "en"
        else "| 拟合对象 | 策略 | nstate | 2pt 窗口 | 3pt 窗口 | n_data | n_params |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        pt2_window = spec.get("pt2_window", f"[{spec.get('tmin', 'n/a')},{spec.get('tmax', 'n/a')})")
        pt3_window = spec.get(
            "pt3_window",
            f"tsep={format_report_list(spec.get('tsep_ls', []))}, tau_cut={spec.get('tau_cut', 'n/a')}",
        )
        lines.append(
            f"| `{spec.get('fit_scope', spec.get('scope', 'n/a'))}` | "
            f"`{spec.get('fit_strategy', spec.get('strategy', 'n/a'))}` | "
            f"{spec.get('nstate', 'n/a')} | "
            f"{pt2_window} | "
            f"{pt3_window} | "
            f"{spec.get('n_data', 'n/a')} | "
            f"{spec.get('n_params', 'n/a')} |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _z_fit_table(result: dict[str, Any], *, language: str) -> list[str]:
    z_fits = result.get("z_fits") or []
    lines = [
        "| z | Q | chi2/dof | logGBF | failed samples | Re sys | Im sys |"
        if language == "en"
        else "| z | Q | chi2/dof | logGBF | 失败样本 | Re sys | Im sys |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fit in z_fits[:20]:
        if not isinstance(fit, dict):
            continue
        window = fit.get("window") if isinstance(fit.get("window"), dict) else {}
        lines.append(
            f"| {format_report_value(fit.get('z'))} | {format_report_value(fit.get('Q', window.get('Q')))} | "
            f"{format_report_value(fit.get('chi2_dof', fit.get('chi2/DOF', window.get('chi2_dof'))))} | "
            f"{format_report_value(fit.get('logGBF', window.get('logGBF')))} | {fit.get('n_failed_samples', 0)} | "
            f"{format_report_value(fit.get('real_sys_sdev'))} | {format_report_value(fit.get('imag_sys_sdev'))} |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return lines


def _outputs_table(artifacts: dict[str, Any], *, language: str) -> list[str]:
    header = "| Artifact | Description |" if language == "en" else "| 文件 | 说明 |"
    lines = [header, "|---|---|"]
    for key in CORRELATOR_ARTIFACT_ORDER:
        if key in {"E0_artifact", "dispersion_relation_plot", "dispersion_relation_image"}:
            continue
        value = artifacts.get(key)
        if not value:
            continue
        desc = CORRELATOR_ARTIFACT_DESCRIPTIONS[key][0 if language == "en" else 1]
        lines.append(f"| `{value}` | {desc} |")
    if len(lines) == 2:
        lines.append("| not available | not available |")
    return lines


def _fit_form_text(*, language: str) -> str:
    if language == "zh":
        return r"""
lamet-agent 在该阶段以同一批重采样样本构造 2pt/3pt 数据，并在选定时间窗口内提取裸矩阵元。2pt 输入为给定动量与 interpolator 的 $C_2(t)$；3pt 输入为各 $t_{\rm sep}$、插入时间 $\tau$ 和 Wilson 线长度 $z$ 的 $C_3(t_{\rm sep},\tau,z)$。对每个 job，调参步骤先在样本平均数据上确定窗口、态数、`fit_scope` 与 `fit_strategy`；随后固定这些选择，对所有 $z$ 和所有重采样样本执行拟合。

2pt 谱分解写为

$$
C_2^\alpha(t)=\sum_{n=0}^{N_{\rm st}-1}
\frac{z_{n,\alpha}^2}{2E_{n,\alpha}}
\left(e^{-E_{n,\alpha}t}+e^{-E_{n,\alpha}(L_t-t)}\right),
\qquad
E_{n,\alpha}=E_{0,\alpha}+\sum_{k=1}^{n}e^{\log\Delta E_{k,\alpha}} .
$$

这里 $\alpha$ 标记初态或末态动量通道；Breit/forward 情形只有一套 $\{E_n,z_n\}$，NonBreit 情形有初态 $i$ 与末态 $f$ 两套 $\{E_{n,i},z_{n,i}\}$、$\{E_{n,f},z_{n,f}\}$。2pt 拟合参数包括 $E_0$、各激发能隙 $\log\Delta E_k$ 和重叠因子 $z_n$；NonBreit 中这些参数分别属于 initial/final 两个 2pt。

Breit ratio 的 3pt 模型为

$$
R_{\rm B}(t,\tau,z)=\frac{C_3(t,\tau,z)}{C_2(t)}
=\frac{1}{C_2(t)}
\sum_{m,n}\frac{O^\Gamma_{mn}(z)z_mz_n}{(2E_m)(2E_n)}
e^{-E_m(t-\tau)}e^{-E_n\tau},
\qquad h_{\rm B}(z)=\frac{O_{00}(z)}{2E_0}.
$$

该形式输入同一动量下的 2pt 与 3pt ratio，拟合参数除 2pt 谱参数外，还包括每个 $z$ 的矩阵元 $O_{mn}(z)$。报告和 NetCDF 中输出的 Breit 裸矩阵元是基态归一化组合 $h_{\rm B}(z)$。

NonBreit ratio 使用初末态不同动量的对称化 ratio：

$$
R_{\rm NB}(t,\tau,z)=
\frac{C_3^{f\leftarrow i}(t,\tau,z)}{C_2^f(t)}
\left[
\frac{C_2^i(t-\tau)C_2^f(\tau)C_2^f(t)}
{C_2^f(t-\tau)C_2^i(\tau)C_2^i(t)}
\right]^{1/2},
\qquad
h_{\rm NB}(z)={\rm sign}(z_{0,i}z_{0,f})\frac{O_{00}(z)}{E_{0,i}+E_{0,f}} .
$$

该形式输入 initial 2pt、final 2pt 以及 non-forward 3pt；拟合参数包括两套 2pt 谱参数和每个 $z$ 的 transition matrix elements $O_{mn}(z)$。报告中 NonBreit 总览图的裸矩阵元对应 $O_{00}/(E_{0,i}+E_{0,f})$，并带有基态重叠符号约定。

若 `fit_scope` 包含 `FH` 或 `ratio+FH`，还会从 ratio 构造 summed-ratio/Feynman-Hellmann 约束：

$$
S(t)=\sum_{\tau=\tau_c}^{t-\tau_c}R(t,\tau),
\qquad
R_{\rm FH}(t)=\frac{S(t+\Delta t)-S(t)}{\Delta t}.
$$

`fit_strategy="joint"` 表示 2pt 与 3pt/FH 在同一个非线性拟合中共同约束，相关参数同时浮动。`fit_strategy="chained"` 表示先拟合 2pt 并把得到的能量与重叠因子作为后续 3pt/FH 拟合的锚定先验；Breit chained 主要锚定单套 $E_0,z_0$，NonBreit chained 锚定 initial/final 两套基态和激发态相关参数。`fit_scope="ratio"` 只用 ratio 数据，`fit_scope="FH"` 只用 summed-ratio/FH 数据，`fit_scope="ratio+FH"` 同时使用两类约束。
""".strip()
    return r"""
lamet-agent builds 2pt/3pt data from the same resampled ensemble and extracts bare matrix elements in the selected time windows. The 2pt input is $C_2(t)$ for the chosen momentum and interpolator; the 3pt input is $C_3(t_{\rm sep},\tau,z)$ for each source-sink separation, insertion time, and Wilson-line length. For each job, tuning first fixes the window, state count, `fit_scope`, and `fit_strategy` on sample-average data; those choices are then held fixed for all $z$ and all resampled samples.

The 2pt spectral form is

$$
C_2^\alpha(t)=\sum_{n=0}^{N_{\rm st}-1}
\frac{z_{n,\alpha}^2}{2E_{n,\alpha}}
\left(e^{-E_{n,\alpha}t}+e^{-E_{n,\alpha}(L_t-t)}\right),
\qquad
E_{n,\alpha}=E_{0,\alpha}+\sum_{k=1}^{n}e^{\log\Delta E_{k,\alpha}} .
$$

Here $\alpha$ labels the initial or final momentum channel. Breit/forward fits use one set of $\{E_n,z_n\}$, while NonBreit fits use separate initial and final sets. The 2pt parameters are $E_0$, the gaps $\log\Delta E_k$, and overlaps $z_n$.

For Breit kinematics the ratio model is

$$
R_{\rm B}(t,\tau,z)=\frac{C_3(t,\tau,z)}{C_2(t)}
=\frac{1}{C_2(t)}
\sum_{m,n}\frac{O^\Gamma_{mn}(z)z_mz_n}{(2E_m)(2E_n)}
e^{-E_m(t-\tau)}e^{-E_n\tau},
\qquad h_{\rm B}(z)=\frac{O_{00}(z)}{2E_0}.
$$

The inputs are the same-momentum 2pt and 3pt ratio. In addition to the 2pt spectral parameters, the fit determines the matrix elements $O_{mn}(z)$ for each Wilson-line length. The reported Breit bare matrix element is the ground-state normalized combination $h_{\rm B}(z)$.

For NonBreit kinematics the symmetrized non-forward ratio is

$$
R_{\rm NB}(t,\tau,z)=
\frac{C_3^{f\leftarrow i}(t,\tau,z)}{C_2^f(t)}
\left[
\frac{C_2^i(t-\tau)C_2^f(\tau)C_2^f(t)}
{C_2^f(t-\tau)C_2^i(\tau)C_2^i(t)}
\right]^{1/2},
\qquad
h_{\rm NB}(z)={\rm sign}(z_{0,i}z_{0,f})\frac{O_{00}(z)}{E_{0,i}+E_{0,f}} .
$$

The inputs are initial 2pt, final 2pt, and non-forward 3pt data. The fit parameters include both 2pt spectra and the transition matrix elements $O_{mn}(z)$. The reported NonBreit summary uses $O_{00}/(E_{0,i}+E_{0,f})$ with the ground-state overlap sign convention.

When `fit_scope` contains `FH` or `ratio+FH`, a summed-ratio/Feynman-Hellmann constraint is also formed:

$$
S(t)=\sum_{\tau=\tau_c}^{t-\tau_c}R(t,\tau),
\qquad
R_{\rm FH}(t)=\frac{S(t+\Delta t)-S(t)}{\Delta t}.
$$

`fit_strategy="joint"` fits 2pt and 3pt/FH constraints in one nonlinear fit with shared floating parameters. `fit_strategy="chained"` fits the 2pt data first and uses the resulting energies and overlaps as anchored priors for the following 3pt/FH fit. `fit_scope="ratio"` uses only ratio data, `fit_scope="FH"` uses only summed-ratio/FH data, and `fit_scope="ratio+FH"` uses both.
""".strip()


def _diagnostic_plots(artifacts: dict[str, Any], *, language: str) -> list[str]:
    plots = list(artifacts.get("sample0_fit_plots", [])) + list(artifacts.get("sample0_pt2_plots", []))
    plots = [plot for plot in plots if plot and str(plot).endswith(".svg")]
    if not plots:
        return ["No sample-0 diagnostic SVGs were recorded." if language == "en" else "未记录 sample-0 诊断 SVG。"]
    lines = ["Sample-0 diagnostic SVGs:" if language == "en" else "Sample-0 诊断 SVG："]
    for start in range(0, len(plots), 4):
        row = plots[start : start + 4]
        lines.append("<table><tr>")
        for plot in row:
            lines.append(f'<td><img src="{plot}" alt="{Path(str(plot)).stem}" width="230"><br><code>{Path(str(plot)).stem}</code></td>')
        lines.append("</tr></table>")
    return lines


def build_correlator_stage_report_markdown(
    *,
    jobs: list[dict[str, Any]],
    base_dir: Path,
    language: str = "en",
) -> str:
    """Build one Markdown report for all correlator-analysis jobs."""
    title = "# Correlator Analysis Stage Report" if language == "en" else "# Correlator Analysis 阶段报告"
    intro = (
        "This report summarizes correlator fits that extract bare matrix elements from 2pt/3pt data."
        if language == "en"
        else "本报告汇总从 2pt/3pt 关联函数中提取裸矩阵元的拟合结果。"
    )
    lines = [
        title,
        "",
        intro,
        "",
        "## Fitting Form" if language == "en" else "## 拟合形式",
        _fit_form_text(language=language),
        "",
        "## Job Summary" if language == "en" else "## Job 汇总",
        "| job | fit scope | strategy | output | plot |" if language == "en" else "| job | 拟合对象 | 策略 | 输出 | 图像 |",
        "|---|---|---|---|---|",
    ]
    markdown_jobs = []
    for item in jobs:
        result = item.get("result", {})
        artifacts = markdown_artifact_paths(
            item.get("artifacts", {}),
            base_dir=base_dir,
            path_keys=(
                *CORRELATOR_ARTIFACT_ORDER,
                *(key for key in item.get("artifacts", {}) if key.startswith("matrix_overlay_")),
            ),
            list_path_keys=("sample0_pt2_plots", "sample0_fit_plots"),
        )
        markdown_jobs.append((item, result, artifacts))
        lines.append(
            f"| `{item['job_id']}` | `{result.get('fit_scope', 'n/a')}` | "
            f"`{result.get('fit_strategy', 'n/a')}` | "
            f"{artifacts.get('bare_artifact', 'n/a')} | {artifacts.get('summary_plot', 'n/a')} |"
        )

    stage_artifacts = markdown_jobs[0][2] if markdown_jobs else {}
    if stage_artifacts.get("dispersion_relation_image"):
        lines.extend(
            [
                "",
                "## Dispersion Relation" if language == "en" else "## 色散关系",
                "",
                (
                    "The dispersion-relation plot is designed to check the dependence of $E_0^2$ on $p^2$ and shows the ground-state energy posterior obtained from 2pt correlator fits at different ensembles and momenta. "
                    r"The conversion to physical units uses $E_0^{\rm GeV}=E_0^{\rm lat}\hbar c/a$."
                    if language == "en"
                    else r"色散关系图旨在检查 $E_0^2$ 随 $p^2$ 的变化，展示了不同组态、不同动量下由 2pt correlator 拟合得到的基态能量后验值。能量按 $E_0^{\rm GeV}=E_0^{\rm lat}\hbar c/a$ 转换到物理单位。"
                ),
                "",
                f"![Dispersion relation]({stage_artifacts['dispersion_relation_image']})",
            ]
        )
    overlay_images = [
        value for key, value in sorted(stage_artifacts.items()) if key.startswith("matrix_overlay_") and "_image_" in key
    ]
    if overlay_images:
        overlay_groups: dict[str, list[str]] = {}
        for image in overlay_images:
            stem = Path(image).stem
            label = stem[3:] if stem.startswith("ca_") else stem
            label = label[:-3] if label.endswith(("_re", "_im")) else label
            overlay_groups.setdefault(label, []).append(image)
        for label, images in overlay_groups.items():
            title = f"{label} ensemble overview" if language == "en" else f"{label}组态总览图"
            lines.extend(["", f"## {title}", ""])
            images.sort(key=lambda image: 0 if Path(image).stem.endswith("_re") else 1)
            lines.extend(f"![{title}]({image})" for image in images)

    for item, result, artifacts in markdown_jobs:
        lines.extend(
            [
                "",
                f"## `{item['job_id']}`",
                "",
                "### Fit Setup" if language == "en" else "### 拟合设置",
                *_job_settings_table(result, language=language),
                "",
                "### Shared Windows" if language == "en" else "### 共享窗口",
                *_window_text(result, language=language),
                "",
                "### Per-z Fit Summary" if language == "en" else "### 逐 z 拟合摘要",
                *_z_fit_table(result, language=language),
                "",
                "### fit_logs" if language == "en" else "### fit_logs",
                (
                    "The `fit_logs` directory contains split logs: the tuning log records window selection and sample-average fit quality, while the sample log records per-sample and per-z fit quality, including failures."
                    if language == "en"
                    else "`fit_logs` 目录包含拆分日志：tuning log 记录窗口选择和样本平均拟合质量；sample log 记录逐样本、逐 z 的拟合质量以及失败信息。"
                ),
                "",
                *_outputs_table(artifacts, language=language),
                "",
                "### Diagnostic SVGs" if language == "en" else "### 诊断 SVG",
                *_diagnostic_plots(artifacts, language=language),
                "",
                "### Summary Figure" if language == "en" else "### 总览图",
                (
                    f"![Bare matrix element summary]({artifacts.get('summary_plot_image')})"
                    if artifacts.get("summary_plot_image")
                    else ("Not available." if language == "en" else "未生成。")
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def write_correlator_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
    report_language: str = "en",
) -> dict[str, Path]:
    """Write one report summarizing all correlator-analysis jobs."""
    output = Path(path)
    target, language = resolve_report_target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_correlator_stage_report_markdown(jobs=jobs, base_dir=target.parent, language=language),
        encoding="utf-8",
    )
    return {"report": target}
