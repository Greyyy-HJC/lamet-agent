"""Reporting helpers for extrapolation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gvar
import numpy as np
import xarray as xr

from lamet_agent.core.reporting import markdown_artifact_paths, resolve_report_target


def write_extrapolation_stage_report(
    *,
    jobs: list[dict[str, Any]],
    path: str | Path,
    report_language: str = "en",
) -> dict[str, Path]:
    """Write a compact stage-level extrapolation report."""
    target, language = resolve_report_target(Path(path), report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    zh = language == "zh"
    lines = ["# 外推阶段报告" if zh else "# Extrapolation Report", ""]
    for record in jobs:
        result = record.get("result", {})
        raw_artifacts = record.get("artifacts", {})
        artifacts = markdown_artifact_paths(
            raw_artifacts,
            base_dir=target.parent,
            path_keys=(
                "extrapolated_artifact",
                "fit_info_artifact",
                "extrapolated_plot",
                "extrapolated_plot_image",
                "adep_plot",
                "adep_plot_image",
                "pdep_plot",
                "pdep_plot_image",
            ),
        )
        a_orders = result.get("lattice_spacing_allow_order", [2])
        p_orders = result.get("momentum_allow_order", [2])
        a_text = ",".join(str(int(value)) for value in a_orders)
        p_text = ",".join(str(int(value)) for value in p_orders)
        formula = (
            rf"$h(x,p_z,a)=h(x,\infty,0)+\sum_{{i\in\{{{a_text}\}}}} c_{{a,i}}a^i"
            rf"+\sum_{{j\in\{{{p_text}\}}}}\frac{{c_{{p,j}}}}{{p_z^j}}$"
        )
        pdep_text = ", ".join(f"{float(value):.2f}" for value in result.get("pdep_gev", [])) or "not set"
        chi_text = f"{float(result.get('chi2_dof', 0.0)):.3g}"
        fit_columns: list[tuple[str, np.ndarray]] = []
        fit_x = np.asarray([], dtype=float)
        fit_indices: list[int] = []
        artifact_path = raw_artifacts.get("extrapolated_artifact")
        if artifact_path:
            with xr.open_dataset(artifact_path) as dataset:
                fit_x = np.asarray(dataset.coords.get("x", []), dtype=float)
                fit_indices = [index for index in (0, len(fit_x) // 2, len(fit_x) - 1) if 0 <= index < len(fit_x)]
                fit_indices = list(dict.fromkeys(fit_indices))
                for name in dataset.data_vars:
                    if not (name.startswith("c_a_") or name.startswith("c_p_")):
                        continue
                    label = rf"$c_{{a,{name.removeprefix('c_a_')}}}$" if name.startswith("c_a_") else rf"$c_{{p,{name.removeprefix('c_p_')}}}$"
                    fit_columns.append((label, np.asarray(dataset[name].values, dtype=float)))
        header = "| $x$ | " + " | ".join(label for label, _values in fit_columns) + " |" if fit_columns else "| $x$ |"
        divider = "|---" * (len(fit_columns) + 1) + "|"
        fit_rows = []
        for index in fit_indices:
            cells = [f"{float(fit_x[index]):.4g}"]
            for _label, values in fit_columns:
                samples = values[:, index]
                sdev = 0.0 if samples.size < 2 else float(np.std(samples, ddof=1))
                cells.append(str(gvar.gvar(float(np.mean(samples)), sdev)))
            fit_rows.append("| " + " | ".join(cells) + " |")
        if zh:
            lines.extend(
                [
                    f"## {record.get('job_id')}",
                    "",
                    "本报告汇总 perturbative matching 输出的光锥分布，并对格距和动量依赖进行联合或单变量外推。",
                    "",
                    "## 外推形式",
                    "",
                    formula,
                    "",
                    "## Job 汇总",
                    "| job | 模式 | 输入数 | 参数数 | $\\chi^2/\\mathrm{dof}$ | 输出 |",
                    "|---|---|---:|---:|---:|---|",
                    f"| `{record.get('job_id')}` | {result.get('mode')} | {result.get('n_inputs')} | {result.get('n_parameters')} | {chi_text} | {Path(str(artifacts.get('extrapolated_artifact'))).name if artifacts.get('extrapolated_artifact') else 'n/a'} |",
                    "",
                    "## 分析设置",
                    "| 条目 | 数值或设置 | 解释 |",
                    "|---|---|---|",
                    f"| `lattice_spacing_allow_order` | {a_orders} | 允许进入拟合的格距修正幂次；代码使用 $a^i$。 |",
                    f"| `momentum_allow_order` | {p_orders} | 允许进入拟合的有限动量修正幂次；代码使用 $1/p_z^j$。 |",
                    f"| `pdep_gev` | {pdep_text} | 额外动量依赖图中指定的 $p_z$（GeV）取值；缺省时不生成 `extrapolate_pdep` 图。 |",
                    "",
                    "## 拟合模型参数表",
                    header,
                    divider,
                    *fit_rows,
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"## {record.get('job_id')}",
                    "",
                    "This report summarizes the light-cone distributions from perturbative matching and extrapolates their lattice-spacing and momentum dependence.",
                    "",
                    "## Extrapolation Form",
                    "",
                    formula,
                    "",
                    "## Job Summary",
                    "| job | mode | inputs | parameters | $\\chi^2/\\mathrm{dof}$ | output |",
                    "|---|---|---:|---:|---:|---|",
                    f"| `{record.get('job_id')}` | {result.get('mode')} | {result.get('n_inputs')} | {result.get('n_parameters')} | {chi_text} | {Path(str(artifacts.get('extrapolated_artifact'))).name if artifacts.get('extrapolated_artifact') else 'n/a'} |",
                    "",
                    "## Analysis Settings",
                    "| Item | Value or setting | Explanation |",
                    "|---|---|---|",
                    f"| `lattice_spacing_allow_order` | {a_orders} | Allowed lattice-spacing correction powers; the code fits $a^i$ terms. |",
                    f"| `momentum_allow_order` | {p_orders} | Allowed finite-momentum correction powers; the code fits $1/p_z^j$ terms. |",
                    f"| `pdep_gev` | {pdep_text} | Requested $p_z$ values in GeV for the extra momentum-dependence figure; if unset, `extrapolate_pdep` is not generated. |",
                    "",
                    "## Fit Model Parameter Table",
                    header,
                    divider,
                    *fit_rows,
                    "",
                ]
            )
        if result.get("warning"):
            lines.append(("警告：" if zh else "Warning: ") + str(result["warning"]))
            lines.append("")
        if result.get("use_lattice_spacing_dependence") and not result.get("use_momentum_dependence"):
            lines.append("由于输入为多系综单动量，本阶段只能生成格距依赖图。" if zh else "The inputs contain multiple ensembles at a single momentum, so only the lattice-spacing-dependence figure can be generated.")
            lines.append("")
        if result.get("use_momentum_dependence") and not result.get("use_lattice_spacing_dependence"):
            lines.append("由于输入为单系综多动量，本阶段只能生成动量依赖图；若未设置 `pdep_gev`，则不生成该图。" if zh else "The inputs contain one ensemble at multiple momenta, so only the momentum-dependence figure can be generated; it is omitted unless `pdep_gev` is set.")
            lines.append("")
        title = "外推结果" if zh else "Extrapolated Result"
        if artifacts.get("extrapolated_plot_image"):
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['extrapolated_plot_image']})", ""])
        if artifacts.get("adep_plot_image"):
            title = "格距依赖图" if zh else "Lattice-Spacing Dependence"
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['adep_plot_image']})", ""])
        if artifacts.get("pdep_plot_image"):
            title = "动量依赖图" if zh else "Momentum Dependence"
            lines.extend([f"## {title}", "", f"![{title}]({artifacts['pdep_plot_image']})", ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    return {"report": target}
