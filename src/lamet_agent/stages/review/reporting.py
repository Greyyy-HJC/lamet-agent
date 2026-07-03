"""Paper-style review Markdown generated from stage reports."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np


def _target(path: Path, report_language: str) -> tuple[Path, str]:
    if report_language.lower() == "ch":
        return path.with_name(f"{path.stem}_CN{path.suffix or '.md'}"), "zh"
    return path, "en"


def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base) if path.is_absolute() else str(path)


def _fix_links(text: str, *, source_dir: Path, base_dir: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix, value, suffix = match.groups()
        path = Path(value)
        if not path.is_absolute() and not value.startswith(("http://", "https://", "#")):
            value = os.path.relpath(source_dir / path, base_dir)
        return f"{prefix}{value}{suffix}"

    text = re.sub(r'(<img[^>]+src=")([^"]+)(")', repl, text)
    return re.sub(r'(\]\()([^)]+)(\))', repl, text)


def _table_after(text: str, marker: str, *, max_rows: int = 8) -> str:
    idx = text.find(marker)
    if idx < 0:
        return ""
    rows = []
    for line in text[idx:].splitlines()[1:]:
        if line.startswith("#"):
            break
        if line.startswith("|"):
            rows.append(line)
            if len(rows) >= max_rows + 2:
                break
        elif rows:
            break
    return "\n".join(rows)


def _images(text: str, *, source_dir: Path, base_dir: Path, max_items: int = 4) -> list[str]:
    fixed = _fix_links(text, source_dir=source_dir, base_dir=base_dir)
    found = []
    for match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)|<img[^>]+src="([^"]+)"[^>]*alt="([^"]*)"[^>]*>|<img[^>]+alt="([^"]*)"[^>]*src="([^"]+)"[^>]*>|<img[^>]+src="([^"]+)"[^>]*>', fixed):
        if match.group(2):
            found.append(f"![{match.group(1)}]({match.group(2)})")
        else:
            src = match.group(3) or match.group(6) or match.group(7)
            alt = match.group(4) or match.group(5) or Path(src).stem
            found.append(f"![{alt}]({src})")
    return found[:max_items]


def _src(figure: str) -> str:
    md = re.search(r'\]\(([^)]+)\)', figure)
    return md.group(1) if md else ""


def _job_id_from_src(src: str) -> str | None:
    stem = Path(src).stem
    if stem.startswith(("ca_", "rn_", "mt_")):
        return stem
    if stem.startswith("ft_"):
        return re.sub(r"_(extension|extension_im)$", "", stem)
    return None


def _full_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return metadata or {}


def _stage_names(metadata: dict[str, Any] | None, reports: list[dict[str, Any]]) -> list[str]:
    data = _full_metadata(metadata)
    if isinstance(data.get("metadata"), dict):
        return list(data["metadata"].get("stages", []))
    return list(data.get("stages", [])) if data else [item["stage"] for item in reports]


def _manifest_job(metadata: dict[str, Any] | None, stage: str, job_id: str | None) -> dict[str, Any]:
    data = _full_metadata(metadata)
    config = data.get("stages", {}).get(stage, {}) if isinstance(data.get("stages"), dict) else {}
    for job in config.get("jobs", []):
        if job.get("id") == job_id:
            params = {}
            params.update(config.get("defaults", {}))
            params.update(job.get("params", {}))
            return {"job": job, "params": params}
    return {"job": {}, "params": {}}


def _correlators(metadata: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    data = _full_metadata(metadata)
    return {item.get("correlator_id"): item for item in data.get("inputs", {}).get("correlators", [])}


def _correlator_context(metadata: dict[str, Any] | None, job: dict[str, Any]) -> dict[str, Any]:
    lookup = _correlators(metadata)
    items = [lookup.get(cid, {}) for cid in job.get("correlator_ids", [])]
    items = [item for item in items if item]
    if not items:
        return {}
    pt3 = next((item for item in items if item.get("kind") == "3pt"), items[0])
    return {
        "momentum": pt3.get("momentum"),
        "pz_gev": pt3.get("pz_gev"),
        "pz_out_gev": pt3.get("pz_out_gev"),
    }


def _job_by_momentum(metadata: dict[str, Any] | None, momentum: str) -> tuple[str | None, dict[str, Any]]:
    data = _full_metadata(metadata)
    for job in data.get("stages", {}).get("correlator_analysis", {}).get("jobs", []):
        ctx = _correlator_context(metadata, job)
        if ctx.get("momentum") == momentum:
            return job.get("id"), ctx
    return None, {}


def _figure_context(stage: str, figure: str, *, report_text: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
    src = _src(figure)
    job_id = _job_id_from_src(src)
    ctx: dict[str, Any] = {"job_id": job_id, "z": None, "nonbreit": "`NonBreit`" in report_text or "NonBreit" in report_text}
    z_match = re.search(r"_z(\d+)_sample", src)
    if z_match:
        ctx["z"] = int(z_match.group(1))
    mom_match = re.search(r"(PX-?\d+PY-?\d+PZ-?\d+)", src)
    if stage == "correlator_analysis":
        if job_id:
            job = _manifest_job(metadata, stage, job_id)["job"]
            ctx.update(_correlator_context(metadata, job))
        elif mom_match:
            ctx["job_id"], found = _job_by_momentum(metadata, mom_match.group(1))
            ctx.update(found)
        if ctx.get("pz_gev") is None and mom_match:
            ctx["momentum"] = mom_match.group(1)
    elif stage == "renormalization":
        rn_job = _manifest_job(metadata, stage, job_id)
        target_id = rn_job["job"].get("inputs", {}).get("target")
        target = _manifest_job(metadata, "correlator_analysis", target_id)["job"]
        ctx.update(_correlator_context(metadata, target))
    elif stage in {"fourier_transform", "perturbative_matching"}:
        params = _manifest_job(metadata, stage, job_id)["params"]
        ctx.update({"pz_gev": params.get("pz_gev"), "pz_out_gev": params.get("pz_out_gev")})
    if ctx.get("pz_out_gev") is not None and ctx.get("pz_gev") is not None and ctx.get("pz_out_gev") != ctx.get("pz_gev"):
        ctx["nonbreit"] = True
    return ctx


def _kinematic_label(ctx: dict[str, Any], *, language: str) -> str:
    pz = ctx.get("pz_gev")
    pz_out = ctx.get("pz_out_gev")
    if ctx.get("nonbreit") and pz is not None and pz_out is not None:
        denom = float(pz) + float(pz_out)
        xi = float("nan") if denom == 0 else (float(pz) - float(pz_out)) / denom
        q2 = (float(pz_out) - float(pz)) ** 2
        return rf"$Q^2={q2:.4g}\,\mathrm{{GeV}}^2$, $\xi={xi:.4g}$"
    if pz is not None:
        return rf"$p={float(pz):.4g}\,\mathrm{{GeV}}$"
    if ctx.get("momentum"):
        return f"`{ctx['momentum']}`"
    return "the selected momentum" if language == "en" else "所选动量"


def _figure_note(stage: str, figure: str, *, language: str, report_text: str = "", metadata: dict[str, Any] | None = None) -> str:
    ctx = _figure_context(stage, figure, report_text=report_text, metadata=metadata)
    kin = _kinematic_label(ctx, language=language)
    if language == "zh":
        if "ca_" in figure and "fit_logs/" not in figure:
            return f"这张汇总图给出动量 {kin} 下裸矩阵元随 Wilson 线长度的坐标空间结构，是 LaMET 后续重整化和 Fourier reconstruction 的直接输入。它概括了所有重采样样本给出的均值和误差，因而比单个 ratio 诊断图更直接反映坐标空间矩阵元的物理形状。若大 $z$ 端信号快速衰减或误差增大，后续长程外推会成为 Fourier 变换系统误差的主要来源之一。图中实部和虚部的相对大小也提供了对所选 sector、动量方向和算符结构是否一致的快速检查。"
        if "fit_logs/" in figure:
            z = "n/a" if ctx.get("z") is None else str(ctx["z"])
            return f"该诊断图为动量 {kin}、Wilson 线长度 $z/a={z}$ 的裸矩阵元 sample-0 诊断图。它比较 ratio 数据、拟合带和基态 plateau，用来判断所选 $t_{{\\rm sep}}$、$\\tau$ 区间内的激发态污染是否被拟合 ansatz 吸收。若不同 $t_{{\\rm sep}}$ 数据在 plateau 附近相互一致，基态矩阵元的提取通常更稳定；若拟合带远宽于数据点，则应回看 2pt 约束、态数或先验宽度。该图不是最终统计误差，而是一个代表样本上的局部拟合质量检查。"
        if stage == "renormalization":
            return f"该图展示动量 {kin} 下 hybrid-ratio 重整化后的坐标空间矩阵元；平滑的 $z$ 依赖表示 Wilson 线线性发散和短距离因子已被非微扰地吸收。与裸矩阵元相比，重整化结果应更适合作为有限 Ioffe-time Fourier reconstruction 的输入。短距离区域主要检验 ratio normalization，长距离区域则检验固定 denominator 与指数质量修正是否引入不连续。若实部或虚部在切换点附近出现尖锐结构，通常需要重新检查 $z_s$ 或 denominator 选择。"
        if stage == "fourier_transform" and "extension" in figure:
            return f"该图检验动量 {kin} 下实测矩阵元与长程外推 ansatz 的衔接。LaMET 中有限 Ioffe-time 数据对大 $x$ 行为和振荡结构较敏感，因此这里是 Fourier reconstruction 的核心诊断。理想情况下，外推曲线应在选定 $z_{{\\min}}$ 到 $z_{{\\max}}$ 区间内平滑接续数据，而不应在数据末端产生突变。该图也帮助判断 LA/NLA 模型平均是否由数据约束，而不是由尾部 ansatz 主导。"
        if stage == "fourier_transform":
            return f"该图是动量 {kin} 下补全坐标空间矩阵元后的 quasi 分布，尚未经过微扰匹配，因而仍保留有限动量和重整化方案依赖。其峰值位置、支撑区间外尾部和误差带共同反映有限 Ioffe-time、长程外推和统计样本的综合影响。若 quasi 分布在物理区外有明显振荡，通常应结合 extension 图判断是否来自尾部截断。该结果是 matching kernel 的直接输入。"
        if stage == "perturbative_matching":
            return f"该图比较动量 {kin} 下 quasi 分布与匹配后的 light-cone 分布；二者的差异反映 NLO matching kernel 和有限动量修正的综合影响。匹配后结果应被理解为在所选 $\\mu$ 和 scheme 下的光锥分布估计。归一化变化较小时，说明微扰核主要重分布 $x$ 依赖而不是整体尺度；若变化很大，则需要检查 $x$ 网格、动量和 kernel 约定。该图是判断最终分布形状是否稳定的主要可视化输出。"
        return "该图作为本阶段主要物理输出或诊断，用于连接该阶段输入与下一步 LaMET 分析。"
    if "ca_" in figure and "fit_logs/" not in figure:
        return f"This summary plot shows the coordinate-space bare matrix element at {kin} as a function of Wilson-line separation. It is the direct input to renormalization and Fourier reconstruction, and summarizes the mean and uncertainty over all resampled matrix elements. The large-separation behavior indicates where tail reconstruction can become a relevant systematic effect."
    if "fit_logs/" in figure:
        z = "n/a" if ctx.get("z") is None else str(ctx["z"])
        return f"This sample-0 diagnostic is for {kin} and Wilson-line separation $z/a={z}$. It compares ratio data, fit bands, and the ground-state plateau, testing whether excited-state contamination is controlled in the chosen time window. It is a local fit-quality diagnostic rather than the final statistical error."
    if stage == "renormalization":
        return f"The plot displays the hybrid-ratio-renormalized coordinate-space matrix element at {kin}. Smooth behavior in $z$ indicates that Wilson-line and short-distance ultraviolet factors have been absorbed nonperturbatively. The behavior near the switching point checks whether the fixed denominator and exponential mass correction introduce visible discontinuities."
    if stage == "fourier_transform" and "extension" in figure:
        return f"This plot checks the matching between measured matrix elements and the long-distance tail ansatz at {kin}. It is the key diagnostic for finite-Ioffe-time Fourier reconstruction because tail behavior directly affects oscillations in momentum-fraction space. A smooth connection to the last reliable data points indicates that the ansatz is constrained by the measured matrix element."
    if stage == "fourier_transform":
        return f"The plot shows the quasi distribution at {kin} after coordinate-space completion and Fourier transformation, before perturbative matching. Its peak, support outside the physical region, and uncertainty band reflect finite momentum, finite Ioffe time, and tail-reconstruction effects. This distribution is the direct input to the matching kernel."
    if stage == "perturbative_matching":
        return f"The plot compares the quasi and matched light-cone distributions at {kin}. Their difference exposes the net effect of the NLO matching kernel and finite-momentum corrections. The normalization change is a useful check of the matching convention and grid implementation."
    return "This figure records the main physical output or diagnostic of the stage."


def _report_map(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["stage"]): item for item in reports}


def _flow(stages: list[str], *, language: str) -> str:
    labels = {
        "correlator_analysis": "bare matrix element" if language == "en" else "裸矩阵元",
        "renormalization": "renormalized matrix element" if language == "en" else "重整化矩阵元",
        "fourier_transform": "quasi distribution" if language == "en" else "quasi 分布",
        "perturbative_matching": "light-cone distribution" if language == "en" else "光锥分布",
        "extrapolation": "physical-limit extrapolation" if language == "en" else "物理点外推",
    }
    return " $\\rightarrow$ ".join(labels.get(stage, stage) for stage in stages)


def _as_numeric(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.fields and {"r", "i"}.issubset(arr.dtype.fields):
        arr = arr["r"] + 1j * arr["i"]
    return np.asarray(arr)


def _nc_summaries(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    try:
        import xarray as xr
    except Exception:
        return out
    for report in reports:
        stage = str(report.get("stage", ""))
        for path in sorted(Path(report["path"]).parent.glob("*.nc")):
            if path.name.endswith("_fit_info.nc"):
                continue
            try:
                ds = xr.open_dataset(path)
                name = next(iter(ds.data_vars))
                values = _as_numeric(ds[name].values)
                dims = dict(ds.sizes)
                coords = {key: ds[key].values for key in ds.coords if key in {"z", "x"}}
                mean = np.nanmean(values, axis=0) if values.ndim > 1 else values
                if np.iscomplexobj(mean):
                    center = float(np.nanmax(np.abs(mean)))
                    real_range = (float(np.nanmin(np.real(mean))), float(np.nanmax(np.real(mean))))
                    imag_range = (float(np.nanmin(np.imag(mean))), float(np.nanmax(np.imag(mean))))
                else:
                    center = float(np.nanmax(np.abs(mean)))
                    real_range = (float(np.nanmin(mean)), float(np.nanmax(mean)))
                    imag_range = None
                out.append(
                    {
                        "stage": stage,
                        "file": path.name,
                        "var": name,
                        "dims": dims,
                        "coord": {key: (float(np.nanmin(val)), float(np.nanmax(val)), int(len(val))) for key, val in coords.items()},
                        "max_abs_mean": center,
                        "real_range": real_range,
                        "imag_range": imag_range,
                    }
                )
                ds.close()
            except Exception:
                continue
    return out


def _numeric_summary(reports: list[dict[str, Any]], *, language: str) -> str:
    summaries = _nc_summaries(reports)
    if language == "zh":
        lines = [
            "## NetCDF 数值摘要",
            "",
            "下表提取各 stage report 同目录 NetCDF 文件中的样本维度、坐标范围和均值幅度，作为 SVG 图像和最终 LLM 总结的数值依据。",
        ]
        if summaries:
            lines.extend(["", "| stage | NetCDF | 变量 | 维度 | 坐标范围 | 均值幅度摘要 |", "|---|---|---|---|---|---|"])
            for item in summaries:
                coord = ", ".join(f"{key}: {lo:.4g}..{hi:.4g} ({n})" for key, (lo, hi, n) in item["coord"].items()) or "n/a"
                dims = ", ".join(f"{key}={value}" for key, value in item["dims"].items())
                if item["imag_range"] is None:
                    amp = f"range={item['real_range'][0]:.4g}..{item['real_range'][1]:.4g}, max|mean|={item['max_abs_mean']:.4g}"
                else:
                    amp = f"Re={item['real_range'][0]:.4g}..{item['real_range'][1]:.4g}, Im={item['imag_range'][0]:.4g}..{item['imag_range'][1]:.4g}, max|mean|={item['max_abs_mean']:.4g}"
                lines.append(f"| `{item['stage']}` | `{item['file']}` | `{item['var']}` | {dims} | {coord} | {amp} |")
        lines.extend(
            [
                "",
                "这些数值摘要说明 review 中的 SVG 图像是相应 NetCDF 样本均值和误差的可视化投影。correlator 与 renormalization 的 `z` 维度给出坐标空间矩阵元，Fourier 和 matching 的 `x` 维度给出动量分数空间分布；两者共同连接 Euclidean matrix element 与 light-cone observable。",
            ]
        )
        return "\n".join(lines)
    lines = [
        "## NetCDF Numerical Summary",
        "",
        "The table below extracts sample dimensions, coordinate ranges, and mean-level amplitudes from the NetCDF files next to the stage reports. These values provide numerical context for the SVG figures and for the final LLM summary.",
    ]
    if summaries:
        lines.extend(["", "| stage | NetCDF | variable | dimensions | coordinate range | mean-level summary |", "|---|---|---|---|---|---|"])
        for item in summaries:
            coord = ", ".join(f"{key}: {lo:.4g}..{hi:.4g} ({n})" for key, (lo, hi, n) in item["coord"].items()) or "n/a"
            dims = ", ".join(f"{key}={value}" for key, value in item["dims"].items())
            if item["imag_range"] is None:
                amp = f"range={item['real_range'][0]:.4g}..{item['real_range'][1]:.4g}, max|mean|={item['max_abs_mean']:.4g}"
            else:
                amp = f"Re={item['real_range'][0]:.4g}..{item['real_range'][1]:.4g}, Im={item['imag_range'][0]:.4g}..{item['imag_range'][1]:.4g}, max|mean|={item['max_abs_mean']:.4g}"
            lines.append(f"| `{item['stage']}` | `{item['file']}` | `{item['var']}` | {dims} | {coord} | {amp} |")
    lines.extend(
        [
            "",
            "These numerical summaries clarify that the SVG figures are projections of the same sampled NetCDF observables rather than standalone illustrations. The `z`-space outputs diagnose the Euclidean matrix element and its renormalization, while the `x`-space outputs diagnose the quasi and matched light-cone distributions.",
        ]
    )
    return "\n".join(lines)


def _stage_review(stage: str, report: dict[str, Any], *, base_dir: Path, language: str, metadata: dict[str, Any] | None = None) -> str:
    text = str(report.get("text", ""))
    source_dir = Path(report["path"]).parent
    figures = _images(text, source_dir=source_dir, base_dir=base_dir, max_items=80)
    evidence = "### Evidence from stage report" if language == "en" else "### 阶段报告证据"
    representative = "### Representative figures" if language == "en" else "### 代表性图像"
    if stage == "correlator_analysis":
        fixed = _fix_links(text, source_dir=source_dir, base_dir=base_dir)
        summaries = re.findall(r'!\[[^\]]*\]\([^)]*/?ca_[^)]*\.svg\)', fixed)
        figures = [item for item in figures if "fit_logs/" in item][:4] + summaries[:4]
        setup = _table_after(text, "### Fit Setup" if language == "en" else "### 拟合设置", max_rows=10)
        windows = _table_after(text, "### Shared Windows" if language == "en" else "### 共享窗口", max_rows=6)
        zfits = _table_after(text, "### Per-z Fit Summary" if language == "en" else "### 逐 z 拟合摘要", max_rows=6)
        if language == "zh":
            prose = "LaMET 计算的起点是 boosted hadron 态中等时、空间分离的 Euclidean 非局域算符矩阵元。为得到这些矩阵元，lamet-agent 对 2pt 和 3pt 关联函数进行一致的重采样，并用 Breit 或 NonBreit 的 ratio/FH ansatz 同时约束能量、重叠因子和基态矩阵元。时间窗口、态数和拟合策略先由样本平均诊断确定；一旦固定，同一组选择被用于所有 Wilson 线长度和所有重采样样本，从而保留不同 $z$ 点之间的关联结构。"
            title = "## 2pt/3pt 关联函数拟合与裸矩阵元"
            labels = ("#### 拟合设置", "#### 共享窗口", "#### 逐 z 拟合诊断")
        else:
            prose = "The LaMET construction starts from equal-time, spatially separated Euclidean operator matrix elements evaluated in boosted hadron states. lamet-agent obtains these matrix elements by coherently resampling the two- and three-point correlators and fitting them with the Breit or NonBreit ratio/FH ansatz, so that energies, overlap factors, and ground-state matrix elements are constrained in one common analysis. Time windows, state counts, and fit strategies are fixed from sample-average diagnostics; after this choice is made, the same setup is applied to all Wilson-line separations and resampled samples, preserving correlations among the $z$ points."
            title = "## 2pt/3pt Correlator Fits and Bare Matrix Elements"
            labels = ("#### Fit setup", "#### Shared window", "#### Per-z diagnostics")
        pieces = [title, "", prose, "", evidence, labels[0], setup, labels[1], windows, labels[2], zfits]
    elif stage == "renormalization":
        summary = _table_after(text, "## Job Summary" if language == "en" else "## Job 汇总", max_rows=6)
        params = _table_after(text, "### Scheme Parameters" if language == "en" else "### 方案参数", max_rows=10)
        if language == "zh":
            prose = "裸的非局域矩阵元含有 Wilson 线自能导致的线性发散以及端点相关的短距离重整化因子。lamet-agent 采用 hybrid-ratio 处方将这些紫外因子转化为逐样本的非微扰比值：短距离处直接除以参考矩阵元，超过切换距离 $z_s$ 后固定 denominator 并接入指数质量修正。该步骤不重新拟合矩阵元，而是在每个重采样样本上施加同一个 renormalization map，使后续 Fourier reconstruction 的输入为平滑的重整化坐标空间分布。"
            title = "## Hybrid-Ratio 重整化"
            labels = ("#### Job 汇总", "#### 方案参数")
        else:
            prose = "The bare nonlocal matrix element contains the Wilson-line self-energy divergence and endpoint short-distance renormalization factors. lamet-agent removes these ultraviolet factors with the hybrid-ratio prescription: the target matrix element is divided by the reference denominator at short distances, while beyond the switching distance $z_s$ the denominator is fixed at the nearest grid point and continued with the exponential mass correction. No new matrix-element fit is introduced at this stage; the same renormalization map is applied sample by sample, yielding a smooth coordinate-space distribution for the Fourier reconstruction."
            title = "## Hybrid-Ratio Renormalization"
            labels = ("#### Job summary", "#### Scheme parameters")
        pieces = [title, "", prose, "", evidence, labels[0], summary, labels[1], params]
    elif stage == "fourier_transform":
        settings = _table_after(text, "## Job Summary" if language == "en" else "## Job 汇总", max_rows=6)
        quality = _table_after(text, "## Fit Quality" if language == "en" else "## 拟合质量与模型诊断", max_rows=8)
        if language == "zh":
            prose = "由于格点数据只覆盖有限的 Ioffe-time 或坐标区间，直接截断 Fourier 变换会把长程尾部的不确定性折叠进动量分数空间。lamet-agent 因此先在样本平均数据上确定一个稳定的长程拟合区间，再在该固定区间内对 LA/NLA 与先验宽度候选逐样本执行模型平均或最优模型选择。外推 ansatz 只承担有限距离数据之外的 reconstruction，随后将补全后的坐标空间矩阵元按离散 Fourier 核映射为 quasi 分布。"
            title = "## 长程外推与 Fourier 变换"
            note = "长程外推和 Fourier 核的数学形式已在上文公式汇总中列出；本节只摘录该 run 的区间选择和模型诊断。完整的 LA/NLA 参数对应关系保留在 Fourier stage report 中。"
            labels = ("#### Job 汇总", "#### 区间与模型诊断")
        else:
            prose = "Since the lattice data cover only a finite Ioffe-time or coordinate range, a naive truncated Fourier transform would fold the uncertainty of the long-distance tail into momentum-fraction space. lamet-agent therefore fixes a stable tail-fit interval from sample-average diagnostics and, at that interval, performs per-sample model averaging or best-model selection over the configured LA/NLA and prior-width candidates. The tail ansatz is used only to reconstruct the coordinate-space distribution beyond the measured range; the completed matrix element is then mapped to the quasi distribution with the discrete Fourier kernel."
            title = "## Large-Distance Extrapolation and Fourier Transform"
            note = "The mathematical tail ansatz and Fourier kernel are summarized above. This section records only the run-specific interval selection and model diagnostics; the full LA/NLA parameter mapping remains in the Fourier stage report."
            labels = ("#### Job summary", "#### Range and model diagnostics")
        pieces = [title, "", prose, "", note, "", evidence, labels[0], settings, labels[1], quality]
    elif stage == "perturbative_matching":
        settings = _table_after(text, "## Analysis Settings" if language == "en" else "## 分析设置", max_rows=10)
        diag = _table_after(text, "## Diagnostics" if language == "en" else "## 诊断与一致性检查", max_rows=8)
        if language == "zh":
            prose = "LaMET 的最后一步利用大动量因子化关系，把等时 quasi 分布与 light-cone 分布相联系。lamet-agent 按 manifest 指定的 NLO kernel、重整化标度和 $x/y$ 网格构造离散匹配矩阵，并将其逐样本作用在 quasi 分布上；这样，来自关联函数拟合、重整化和长程 reconstruction 的统计涨落会贯穿到最终的 matched 分布。归一化比较提供了一个直接的 sanity check，用来评估微扰修正的整体大小。"
            title = "## 微扰匹配"
            note = "匹配卷积和离散矩阵形式已在上文公式汇总中给出；具体 kernel、scheme 和归一化诊断来自 matching stage report。"
            labels = ("#### 分析设置", "#### 归一化诊断")
        else:
            prose = "The final LaMET step uses the large-momentum factorization relation to connect the equal-time quasi distribution to the light-cone distribution. lamet-agent constructs the discrete matching matrix from the manifest-selected NLO kernel, renormalization scale, and $x/y$ grid, and applies it to every resampled quasi distribution. In this way, statistical fluctuations from correlator fits, renormalization, and long-distance reconstruction are propagated to the matched result. The normalization comparison provides a direct sanity check on the size of the perturbative correction."
            title = "## Perturbative Matching"
            note = "The matching convolution and its discrete matrix form are summarized above. The concrete kernel, scheme, and normalization diagnostics are inherited from the matching stage report."
            labels = ("#### Analysis settings", "#### Normalization diagnostics")
        pieces = [title, "", prose, "", note, "", evidence, labels[0], settings, labels[1], diag]
    else:
        title = "## Physical Extrapolation" if language == "en" else "## 物理点外推"
        pieces = [title, "", "not recorded" if language == "en" else "未记录。"]
    if figures:
        pieces.extend(["", representative, ""])
        for figure in figures:
            pieces.extend([figure, "", _figure_note(stage, figure, language=language, report_text=text, metadata=metadata), ""])
    return "\n".join(item for item in pieces if item).strip()


def _formula_block(*, language: str) -> str:
    if language == "zh":
        return r"""
### 关联函数拟合公式

现有 correlator stage report 记录窗口、$Q$、$\chi^2/{\rm dof}$、logGBF 和逐 $z$ 诊断；解析公式在 review 中按 lamet-agent 当前拟合约定给出，用于说明报告中这些诊断对应的模型：

$$
C_2^\alpha(t)=\sum_{n=0}^{N_{\rm st}-1}
\frac{z_{n,\alpha}^2}{2E_{n,\alpha}}
\left(e^{-E_{n,\alpha}t}+e^{-E_{n,\alpha}(L_t-t)}\right),
\qquad
E_{n,\alpha}=E_{0,\alpha}+\sum_{k=1}^{n}e^{\log\Delta E_{k,\alpha}} .
$$

$$
R_{\rm B}(t,\tau,z)=\frac{C_3(t,\tau,z)}{C_2(t)}
=\frac{1}{C_2(t)}
\sum_{m,n}\frac{O^\Gamma_{mn}(z)z_mz_n}{(2E_m)(2E_n)}
e^{-E_m(t-\tau)}e^{-E_n\tau},
\qquad h_B(z)=\frac{O_{00}(z)}{2E_0}.
$$

$$
R_{\rm NB}(t,\tau,z)=
\frac{C_3^{f\leftarrow i}(t,\tau,z)}{C_2^f(t)}
\left[
\frac{C_2^i(t-\tau)C_2^f(\tau)C_2^f(t)}
{C_2^f(t-\tau)C_2^i(\tau)C_2^i(t)}
\right]^{1/2},
\qquad
h_B^{\rm NB}(z)={\rm sign}(z_{0,i}z_{0,f})\frac{O_{00}(z)}{E_{0,i}+E_{0,f}} .
$$

$$
S(t)=\sum_{\tau=\tau_c}^{t-\tau_c}R(t,\tau),\qquad
R_{\rm FH}(t)=\frac{S(t+\Delta t)-S(t)}{\Delta t}.
$$

### 重整化公式

$$
N_s=\frac{h^{\rm den}_s(0)}{h^{\rm tar}_s(0)},\qquad
h^R_s(z)=
\begin{cases}
N_s h^{\rm tar}_s(z)/h^{\rm den}_s(z), & |z|_{\rm fm}\le z_s,\\
N_s e^{(\delta m+m_0)(|z|_{\rm fm}-z_s)/(\hbar c)}
h^{\rm tar}_s(z)/h^{\rm den}_s(z_s^{\rm grid}), & |z|_{\rm fm}>z_s .
\end{cases}
$$

### 长程外推与坐标空间 extension

Fourier stage report 中给出的具体 LA/NLA 文献公式和 lamet-agent 实际拟合公式是本综述的逐 run 来源。通用形式可写为

$$
h^{\rm tail}(z)=e^{-\Lambda |z|}
\left[
\sum_j A_j e^{i(\phi_j{\rm sign}(z)+\omega_j z)}
+\frac{1}{|z|}\sum_j A'_j e^{i(\phi'_j{\rm sign}(z)+\omega'_jz)}
\right],
$$

其中 $1/|z|$ 项只在 NLA 中启用。extension 平滑为

$$
h_{\rm ext}(z)=[1-w(z)]h_{\rm data}(z)+w(z)h_{\rm fit}(z),
\qquad
w(z)=\frac{z-z_{\min}}{z_{\max}-z_{\min}}
$$

在线性平滑区间内使用。

### Fourier 变换公式

$$
{\rm Re}\,q(x)=\frac{\Delta\lambda}{2\pi}\sum_\lambda
\left[\cos(x\lambda){\rm Re}\,h(\lambda)-\sin(x\lambda){\rm Im}\,h(\lambda)\right],
$$

$$
{\rm Im}\,q(x)=\frac{\Delta\lambda}{2\pi}\sum_\lambda
\left[\sin(x\lambda){\rm Re}\,h(\lambda)+\cos(x\lambda){\rm Im}\,h(\lambda)\right],
\qquad \lambda=P_z z .
$$

### 微扰匹配公式

$$
f(x,\mu)=\int\frac{dy}{|y|}\,
C^{-1}\!\left(\frac{x}{y},\frac{\mu}{|y|P_z}\right)\tilde f(y,P_z),
\qquad
f_i=\sum_j K_{ij}\tilde f_j .
$$

### 物理点外推公式

当前 extrapolation stage 仍是占位实现；若未来 report 提供连续极限、体积和手征外推，可在此处记录形如

$$
F(a,L,m_\pi)=F_{\rm phys}+c_a a^2+c_L e^{-m_\pi L}+c_\pi(m_\pi^2-m_{\pi,{\rm phys}}^2)
$$

的实际拟合 ansatz。本次 review 只在相应 report 存在时总结独立 extrapolation 结果。
""".strip()
    return r"""
### Correlator Fit Formulae

The correlator report records windows, $Q$, $\chi^2/{\rm dof}$, logGBF, and per-$z$ diagnostics. The following equations summarize the current lamet-agent fit convention behind those diagnostics:

$$
C_2^\alpha(t)=\sum_{n=0}^{N_{\rm st}-1}
\frac{z_{n,\alpha}^2}{2E_{n,\alpha}}
\left(e^{-E_{n,\alpha}t}+e^{-E_{n,\alpha}(L_t-t)}\right),
\qquad
E_{n,\alpha}=E_{0,\alpha}+\sum_{k=1}^{n}e^{\log\Delta E_{k,\alpha}} .
$$

$$
R_{\rm B}(t,\tau,z)=\frac{C_3(t,\tau,z)}{C_2(t)}
=\frac{1}{C_2(t)}
\sum_{m,n}\frac{O^\Gamma_{mn}(z)z_mz_n}{(2E_m)(2E_n)}
e^{-E_m(t-\tau)}e^{-E_n\tau},
\qquad h_B(z)=\frac{O_{00}(z)}{2E_0}.
$$

$$
R_{\rm NB}(t,\tau,z)=
\frac{C_3^{f\leftarrow i}(t,\tau,z)}{C_2^f(t)}
\left[
\frac{C_2^i(t-\tau)C_2^f(\tau)C_2^f(t)}
{C_2^f(t-\tau)C_2^i(\tau)C_2^i(t)}
\right]^{1/2},
\qquad
h_B^{\rm NB}(z)={\rm sign}(z_{0,i}z_{0,f})\frac{O_{00}(z)}{E_{0,i}+E_{0,f}} .
$$

$$
S(t)=\sum_{\tau=\tau_c}^{t-\tau_c}R(t,\tau),\qquad
R_{\rm FH}(t)=\frac{S(t+\Delta t)-S(t)}{\Delta t}.
$$

### Renormalization Formula

$$
N_s=\frac{h^{\rm den}_s(0)}{h^{\rm tar}_s(0)},\qquad
h^R_s(z)=
\begin{cases}
N_s h^{\rm tar}_s(z)/h^{\rm den}_s(z), & |z|_{\rm fm}\le z_s,\\
N_s e^{(\delta m+m_0)(|z|_{\rm fm}-z_s)/(\hbar c)}
h^{\rm tar}_s(z)/h^{\rm den}_s(z_s^{\rm grid}), & |z|_{\rm fm}>z_s .
\end{cases}
$$

### Large-Distance Tail and Coordinate-Space Extension

The run-specific LA/NLA article formulae and lamet-agent implementation formulae are taken from the Fourier report. A compact generic form is

$$
h^{\rm tail}(z)=e^{-\Lambda |z|}
\left[
\sum_j A_j e^{i(\phi_j{\rm sign}(z)+\omega_j z)}
+\frac{1}{|z|}\sum_j A'_j e^{i(\phi'_j{\rm sign}(z)+\omega'_jz)}
\right],
$$

where the $1/|z|$ terms are active for NLA. The linear extension rule is

$$
h_{\rm ext}(z)=[1-w(z)]h_{\rm data}(z)+w(z)h_{\rm fit}(z),
\qquad
w(z)=\frac{z-z_{\min}}{z_{\max}-z_{\min}} .
$$

### Fourier Transform Formulae

$$
{\rm Re}\,q(x)=\frac{\Delta\lambda}{2\pi}\sum_\lambda
\left[\cos(x\lambda){\rm Re}\,h(\lambda)-\sin(x\lambda){\rm Im}\,h(\lambda)\right],
$$

$$
{\rm Im}\,q(x)=\frac{\Delta\lambda}{2\pi}\sum_\lambda
\left[\sin(x\lambda){\rm Re}\,h(\lambda)+\cos(x\lambda){\rm Im}\,h(\lambda)\right],
\qquad \lambda=P_z z .
$$

### Perturbative Matching Formula

$$
f(x,\mu)=\int\frac{dy}{|y|}\,
C^{-1}\!\left(\frac{x}{y},\frac{\mu}{|y|P_z}\right)\tilde f(y,P_z),
\qquad
f_i=\sum_j K_{ij}\tilde f_j .
$$

### Physical Extrapolation Formula

The current extrapolation stage is a placeholder. Once an extrapolation report is produced, this review can record the actual continuum, volume, and pion-mass ansatz, for example

$$
F(a,L,m_\pi)=F_{\rm phys}+c_a a^2+c_L e^{-m_\pi L}+c_\pi(m_\pi^2-m_{\pi,{\rm phys}}^2).
$$
""".strip()


def build_review_markdown(
    *,
    reports: list[dict[str, Any]],
    missing_stages: list[str],
    metadata: dict[str, Any] | None = None,
    base_dir: Path,
    language: str,
) -> str:
    report_by_stage = _report_map(reports)
    stages = _stage_names(metadata, reports)
    title = "# LaMET/LQCD Analysis Review" if language == "en" else "# LaMET/LQCD 分析综述"
    intro = (
        "This review condenses the stage reports into a paper-style narrative. Numerical conclusions are inherited from the report files listed below."
        if language == "en"
        else "本综述将各阶段报告凝练为科学论文式叙述。所有数值性结论均来自下方列出的 report 文件。"
    )
    lines = [
        title,
        "",
        intro,
        "",
        "## Analysis Flow and Provenance" if language == "en" else "## 分析流程与来源",
        "",
        _flow(stages, language=language),
        "",
        "| stage | report |" if language == "en" else "| 阶段 | report |",
        "|---|---|",
    ]
    for item in reports:
        lines.append(f"| `{item['stage']}` | `{_rel(Path(item['path']), base_dir)}` |")
    for stage in missing_stages:
        lines.append(f"| `{stage}` | {'not available' if language == 'en' else '未生成'} |")

    lines.extend(
        [
            "",
            "## Formulae Used in the Review" if language == "en" else "## 本综述使用的公式",
            "",
            _formula_block(language=language),
        ]
    )

    for stage in stages:
        if stage not in report_by_stage:
            continue
        lines.extend(["", _stage_review(stage, report_by_stage[stage], base_dir=base_dir, language=language, metadata=metadata)])

    lines.extend(["", _numeric_summary(reports, language=language)])
    return "\n".join(lines) + "\n"


def write_review_report(
    *,
    reports: list[dict[str, Any]],
    missing_stages: list[str],
    path: str | Path,
    report_language: str = "en",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output = Path(path)
    target, language = _target(output, report_language)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_review_markdown(
            reports=reports,
            missing_stages=missing_stages,
            metadata=metadata,
            base_dir=target.parent,
            language=language,
        ),
        encoding="utf-8",
    )
    return {"report": target}


def append_llm_summary(path: str | Path, summary: str, *, report_language: str = "en") -> Path:
    target = Path(path)
    language = "zh" if report_language.lower() == "ch" else "en"
    heading = "## LLM总结" if language == "zh" else "## LLM Summary"
    text = target.read_text(encoding="utf-8").rstrip()
    target.write_text(f"{text}\n\n{heading}\n\n{summary.strip()}\n", encoding="utf-8")
    return target
