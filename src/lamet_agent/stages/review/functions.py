"""Review-stage utilities built from existing stage reports."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from lamet_agent.core.llm import request_llm_text
from lamet_agent.manifest import AnalysisManifest
from lamet_agent.manifest_params import merge_stage_params

STAGE_REPORTS = {
    "correlator_analysis": ("ca_report.md", "ca_report_CN.md"),
    "renormalization": ("renorm_report.md", "renorm_report_CN.md"),
    "fourier_transform": ("ft_report.md", "ft_report_CN.md"),
    "perturbative_matching": ("matching_report.md", "matching_report_CN.md"),
    "extrapolation": ("extrapolation_report.md", "extrapolation_report_CN.md"),
}


def _effective_params(manifest: AnalysisManifest, stage: str, job: Any) -> dict[str, Any]:
    return merge_stage_params(manifest.stages[stage].defaults, job.params)


def _zs_path(manifest: AnalysisManifest, stage: str, job: Any) -> str:
    jobs = manifest.stages[stage].jobs
    index = next(index for index, candidate in enumerate(jobs) if candidate.id == job.id)
    if "zs_fm" in job.params:
        return f"stages.{stage}.jobs[{index}].params.zs_fm"
    return f"stages.{stage}.defaults.zs_fm"


def hybrid_zs_consistency_checks(manifest: AnalysisManifest) -> list[dict[str, Any]]:
    """Compare hybrid matching and renormalization ``zs_fm`` along manifest DAG chains."""
    matching_stage = manifest.stages.get("perturbative_matching")
    if matching_stage is None:
        return []

    from lamet_agent.stages.matching.functions import is_hybrid_kernel, resolve_kernel_id

    jobs_by_id = {
        job.id: (stage, job)
        for stage, config in manifest.stages.items()
        for job in config.jobs
    }
    checks: list[dict[str, Any]] = []
    for matching_index, matching_job in enumerate(matching_stage.jobs):
        matching_params = _effective_params(manifest, "perturbative_matching", matching_job)
        kernel_id = matching_params.get("kernel_id")
        if kernel_id is None:
            matching_kernels = [
                item for item in manifest.kernels if item.stage == "perturbative_matching"
            ]
            if len(matching_kernels) == 1:
                kernel_id = matching_kernels[0].kernel_id
        declaration = next((item for item in manifest.kernels if item.kernel_id == kernel_id), None)
        is_hybrid = False
        if declaration is not None:
            try:
                is_hybrid = is_hybrid_kernel(resolve_kernel_id(declaration.kernel_id, declaration.scheme))
            except ValueError:
                is_hybrid = declaration.scheme == "hybrid_ratio" and "hybrid" in declaration.kernel_id.lower()

        base: dict[str, Any] = {
            "matching_job": matching_job.id,
            "matching_job_path": f"stages.perturbative_matching.jobs[{matching_index}]",
            "renormalization_job": None,
            "matching_zs_fm": matching_params.get("zs_fm"),
            "renormalization_zs_fm": None,
            "matching_zs_path": (
                _zs_path(manifest, "perturbative_matching", matching_job)
                if "zs_fm" in matching_params
                else None
            ),
            "renormalization_zs_path": None,
        }
        if not is_hybrid:
            checks.append({**base, "status": "not_applicable", "reason": "matching kernel is not hybrid"})
            continue

        quasi_ref = matching_job.inputs.get("quasi")
        fourier_entry = jobs_by_id.get(quasi_ref) if isinstance(quasi_ref, str) else None
        if fourier_entry is None or fourier_entry[0] != "fourier_transform":
            checks.append(
                {
                    **base,
                    "status": "unverifiable",
                    "reason": "matching quasi input does not resolve to an in-manifest Fourier job",
                }
            )
            continue
        fourier_job = fourier_entry[1]
        renorm_ref = fourier_job.inputs.get("input")
        renorm_entry = jobs_by_id.get(renorm_ref) if isinstance(renorm_ref, str) else None
        if renorm_entry is None or renorm_entry[0] != "renormalization":
            checks.append(
                {
                    **base,
                    "status": "unverifiable",
                    "reason": "Fourier input does not resolve to an in-manifest renormalization job",
                }
            )
            continue

        renorm_job = renorm_entry[1]
        renorm_params = _effective_params(manifest, "renormalization", renorm_job)
        compared = {
            **base,
            "renormalization_job": renorm_job.id,
            "renormalization_zs_fm": renorm_params.get("zs_fm"),
            "renormalization_zs_path": (
                _zs_path(manifest, "renormalization", renorm_job)
                if "zs_fm" in renorm_params
                else None
            ),
        }
        if renorm_params.get("scheme") != "hybrid_ratio":
            checks.append(
                {**compared, "status": "not_applicable", "reason": "upstream renormalization is not hybrid_ratio"}
            )
            continue
        try:
            matching_zs = float(matching_params["zs_fm"])
            renorm_zs = float(renorm_params["zs_fm"])
        except (KeyError, TypeError, ValueError):
            checks.append(
                {**compared, "status": "unverifiable", "reason": "one or both jobs lack a numeric zs_fm"}
            )
            continue
        status = "consistent" if math.isclose(matching_zs, renorm_zs, rel_tol=0.0, abs_tol=1e-12) else "mismatch"
        checks.append(
            {
                **compared,
                "status": status,
                "reason": "zs_fm values agree" if status == "consistent" else "zs_fm values differ",
                "recommended_path": f"stages.perturbative_matching.jobs[{matching_index}].params.zs_fm",
            }
        )
    return checks


def _format_manifest_consistency(checks: list[dict[str, Any]], *, language: str) -> str:
    if language == "zh":
        lines = [
            "## Manifest 参数一致性",
            "",
            "该检查沿 `matching.quasi → fourier.input → renormalization job` 追踪数据链；结果仅用于 review，不阻断运行。",
            "",
            "| Matching job | Renormalization job | Matching `zs_fm` | Renormalization `zs_fm` | 状态 |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "## Manifest Parameter Consistency",
            "",
            "This check follows `matching.quasi → fourier.input → renormalization job`; findings are review-only and do not block execution.",
            "",
            "| Matching job | Renormalization job | Matching `zs_fm` | Renormalization `zs_fm` | Status |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    if not checks:
        lines.append("| — | — | — | — | not applicable |")
    for check in checks:
        renorm_job = f"`{check['renormalization_job']}`" if check.get("renormalization_job") else "—"
        matching_zs = check.get("matching_zs_fm")
        renorm_zs = check.get("renormalization_zs_fm")
        lines.append(
            f"| `{check['matching_job']}` | "
            f"{renorm_job} | "
            f"{matching_zs if matching_zs is not None else '—'} | "
            f"{renorm_zs if renorm_zs is not None else '—'} | "
            f"`{check['status']}` |"
        )
    mismatches = [check for check in checks if check["status"] == "mismatch"]
    unverifiable = [check for check in checks if check["status"] == "unverifiable"]
    if mismatches:
        lines.extend(["", "### " + ("需要修改" if language == "zh" else "Required changes")])
        for check in mismatches:
            if language == "zh":
                lines.append(
                    f"- `{check['matching_job']}` 与上游 `{check['renormalization_job']}` 不一致：将 "
                    f"`{check['recommended_path']}` 设置为 `{check['renormalization_zs_fm']}`。"
                )
            else:
                lines.append(
                    f"- `{check['matching_job']}` differs from upstream `{check['renormalization_job']}`: set "
                    f"`{check['recommended_path']}` to `{check['renormalization_zs_fm']}`."
                )
    if unverifiable:
        lines.extend(["", "### " + ("无法核对" if language == "zh" else "Not verifiable")])
        for check in unverifiable:
            reason = check["reason"]
            if language == "zh":
                lines.append(f"- `{check['matching_job']}`：当前 manifest 内没有完整上游 job 链，无法核对 `zs_fm`。")
            else:
                lines.append(f"- `{check['matching_job']}`: {reason}.")
    return "\n".join(lines)


def write_review_from_manifest(
    manifest: AnalysisManifest,
    *,
    report_language: str = "en",
    backend: str = "",
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Ask the configured LLM to write the final review from reports and NetCDF summaries."""
    artifacts_dir = Path(output_dir) if output_dir is not None else manifest.artifacts_directory
    review_dir = artifacts_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    language = "zh" if report_language.lower() == "ch" else "en"
    target = review_dir / ("review_CN.md" if language == "zh" else "review.md")
    consistency_checks = hybrid_zs_consistency_checks(manifest)
    materials = []
    stages = [stage for stage in STAGE_REPORTS if (artifacts_dir / stage).is_dir() or stage in manifest.metadata.stages]
    for stage in stages:
        en_name, zh_name = STAGE_REPORTS[stage]
        stage_dir = artifacts_dir / stage
        report_path = stage_dir / (zh_name if language == "zh" else en_name)
        if not report_path.exists() and language == "zh":
            report_path = stage_dir / en_name
        item: dict[str, Any] = {
            "stage": stage,
            "artifact_stage_dir": str(stage_dir),
            "report": str(report_path),
            "report_text": "",
            "netcdf": [],
            "svg": [],
        }
        if report_path.exists():
            item["report_text"] = report_path.read_text(encoding="utf-8")
        for path in sorted(stage_dir.glob("*.nc")):
            if path.name.endswith("_fit_info.nc"):
                continue
            with xr.open_dataset(path) as ds:
                name = next(iter(ds.data_vars))
                values = np.asarray(ds[name].values)
                if values.dtype.fields and {"r", "i"}.issubset(values.dtype.fields):
                    values = values["r"] + 1j * values["i"]
                mean = np.nanmean(values, axis=0) if values.ndim > 1 else values
                summary: dict[str, Any] = {
                    "file": path.name,
                    "variable": name,
                    "dims": dict(ds.sizes),
                    "coords": {
                        key: [float(np.nanmin(ds[key].values)), float(np.nanmax(ds[key].values)), int(len(ds[key].values))]
                        for key in ds.coords
                        if key in {"z", "x"}
                    },
                    "max_abs_mean": float(np.nanmax(np.abs(mean))),
                    "real_mean_range": [float(np.nanmin(np.real(mean))), float(np.nanmax(np.real(mean)))],
                }
                if np.iscomplexobj(mean):
                    summary["imag_mean_range"] = [float(np.nanmin(np.imag(mean))), float(np.nanmax(np.imag(mean)))]
            item["netcdf"].append(summary)
        item["svg"] = [
            {
                "markdown_path": os.path.relpath(path, review_dir),
                "stage_subpath": os.path.relpath(path, stage_dir),
                "absolute_path": str(path),
            }
            for path in sorted(stage_dir.rglob("*.svg"))[:80]
        ]
        materials.append(item)
    lamet_review_rules_zh = (
        "你是一个专门从事 LaMET（大动量有效理论）格点数值分析的专家 AI。你的任务是根据用户提供的五步分析报告生成严格基于报告事实的 Review，并给出 Recommended Manifest Changes。\n"
        "领域背景：LaMET 通过大动量下准 PDF/TMD/DA/GPD 的傅里叶变换、微扰匹配和动量外推，从格点 QCD 提取光锥部分子分布。标准流程为：Step 1 correlator_analysis，通常从两点和三点关联函数拟合基态能谱、重叠因子和裸矩阵元 h(z,Pz)，关注拟合质量、激发态能隙、重叠因子相对误差、z=0 与最大 z 处信噪比；若 manifest 或报告显示 `fit_scope=\"qda_ratio\"`，则该步骤是 nonlocal 2pt z-ratio 分析，应描述为使用非局域两点函数比值提取裸矩阵元，不要强行写成 3pt/2pt ratio 或要求 overlap、三点函数、源汇间距、tau/current-insertion 诊断。Step 2 renormalization，消除紫外发散得到 h_R(z)，关注重整化常数、重整化前后误差放大因子、窗口依赖；Step 3 fourier_transform，对 h_R(z) 做离散傅里叶变换得到准分布，关注 zmin/zmax、震荡幅度、x/y 空间误差带、零阶矩；Step 4 perturbative_matching，用 LaMET 匹配核得到光锥分布，关注正定性、一阶矩及其与 1 的偏差、中间 x 区域误差、匹配阶数影响；Step 5 extrapolation，对不同 Pz 或格距结果做无限动量或连续极限外推，关注模型合理性、拟合质量和稳定性。\n"
        "数据提取规则：从 report 和 NetCDF 摘要逐字读取或自动计算指标；缺失时写“未提供”。Step1 对标准 3pt_ratio 提取拟合质量、激发态能隙、重叠因子相对误差、z=0 和最大 z 处 h(z) 信噪比；对 `fit_scope=\"qda_ratio\"` 只提取 nonlocal 2pt z-ratio 的拟合质量、2pt 拟合窗口、普通局域 2pt 分母、z 方向信噪比和长 Wilson 线噪声表现，不要把未提供的 3pt/overlap/tsep/tau 指标当作问题。Step2 提取重整化常数及误差、重整化前后统计误差放大因子；Step3 提取 zmin/zmax、准分布误差棒、零阶矩积分值；Step4 提取匹配后 q(x) 误差棒、一阶矩及其与 1 的差值；Step5 提取外推模型、外推拟合质量、最终物理量总误差。每个 stage 用连贯物理语言说明做了什么、得到什么关键结果、质量如何。\n"
        "Physical Summary 写作规则：每个 stage 的 Physical Summary 必须是可发表级别的学术正文段落，可直接插入论文 Results 或 Analysis 部分。使用第三人称被动语态或 “we” 为主语，描述已完成分析，不要写“根据报告”“Step1 的指标显示”“这里我们看到”等元语言。每段 3–5 句，紧凑专业；段首可用 **Correlator analysis.**、**Renormalization.**、**Fourier transform.**、**Perturbative matching.**、**Extrapolation.**。内容必须自然包含该步骤的物理目的、关键方法或设置、核心数值结果及统计/系统精度、结果质量的物理解读。物理量必须用 `$...$`，例如 `$\\chi^2/\\mathrm{dof}$`、`$h(z,P_z)$`、`$\\langle x \\rangle$`。若指标良好，使用 “demonstrates good convergence”“is well under control”“agrees with expectations” 等学术表述；若存在问题，如信噪比不足或过大的 `$\\chi^2$`，必须如实描述并使用 “shows a mild tension”“indicates potential systematic effects”“may require further investigation” 等学术措辞。\n"
        "Diagnostics 写作规则：Diagnostics 不是重复 Summary，也不能只根据 `$\\chi^2/\\mathrm{dof}$`、logGBF 或 job 是否成功来判断物理可靠性。必须区分三类结论：(1) report/NetCDF 直接支持的数值事实；(2) 可以通过 manifest 调整的数据分析问题；(3) 不能靠 lamet-agent 调参解决、需要新 LQCD 数据或外部计算条件改善的原始数据质量问题。若准分布在物理区间外振荡、正定性/归一化不合理、误差带过大、不同动量或格距不一致、重整化后动态范围异常放大、长距离矩阵元噪声主导，Diagnostics 必须明确说明“数值流程跑通不等于物理结果可靠”。可讨论的外部原因包括：胶子算符本征噪声大、三点函数统计量不足、源汇间距或激发态污染受限、最高动量下 overlap 变差、长 Wilson 线信噪比指数恶化、格距/体积/有限动量导致系统误差、组态自相关或有效独立样本不足、原始 2pt/3pt 构造或投影导致信号弱。若是 `fit_scope=\"qda_ratio\"`，外部原因应优先围绕 nonlocal 2pt z-ratio 的统计量、非局域两点函数长距离信噪比、pt2 拟合窗口、普通局域 2pt 分母、样本自相关和 Wilson 线长度展开，不要要求三点函数统计量、源汇间距、tau 覆盖、current insertion 或 overlap 诊断。所有这些原因必须写成“与观测到的异常相一致的物理解释”，不能当成已被证明的事实；如果报告没有给出对应诊断量，必须写明该原因需要额外检查原始 correlator、统计量或独立 LQCD 计算来确认。\n"
        "推荐修改判定标准：只有触发条件时才建议修改，否则写“当前设置合理，无需修改”。Step1 对标准 3pt_ratio，若拟合质量不好、重叠因子误差很大或 h(z) 信噪比 < 3，建议调整拟合区间、增加激发态数量或增大统计量；对 `fit_scope=\"qda_ratio\"`，若 nonlocal 2pt z-ratio 拟合质量不好或长 z 信噪比 < 3，优先建议调整 `pt2_windows`、`nstate`/`nstate_values`、`fit_strategy`、`prior_width`、`svdcut`，不要建议 `pt3_tau_cuts`。Step2 若误差放大因子 > 2.0 或窗口依赖显著，建议调整重整化窗口或方案；Step3 若准分布剧烈震荡且 zmax 处信噪比 < 3，或零阶矩偏差 > 10%，建议增大 zmax 或改进变换方法；Step4 若一阶矩偏离 1 超过统计误差 3 倍或 q(x) 明显非物理，建议增大 Pz 或使用更高阶匹配核；Step5 若外推拟合质量不好或剔除最高/最低动量点后变化超过 1σ，建议增加中间动量点或重新评估外推模型。任何显著异常也要指出。\n"
        "Recommended Manifest Changes 中每条建议必须包含 `parameter`、`current_value`、`recommended_change`、`evidence`、`expected_effect`。若不确定 manifest 参数名，使用 `related_parameter`。绝对禁止编造报告中没有的数值或现象；不要用“可能”“也许”等模糊词。若证据不足，直接写“指标正常，无明确修改依据”。\n"
    )
    lamet_review_rules_en = (
        "You are an expert AI specialized in LaMET lattice numerical analysis. Your task is to generate a fact-grounded Review from the supplied five-step analysis reports and provide Recommended Manifest Changes.\n"
        "Domain background: LaMET extracts light-cone PDFs/TMDs/DAs/GPDs from lattice QCD through Fourier reconstruction, perturbative matching, and momentum extrapolation of large-momentum quasi distributions. The standard flow is: Step 1 correlator_analysis usually fits two- and three-point correlators to obtain ground-state spectra, overlap factors, and bare matrix elements h(z,Pz), with diagnostics from fit quality, excited-state gaps, relative overlap errors, and signal-to-noise at z=0 and maximal z; if the manifest or report shows `fit_scope=\"qda_ratio\"`, this step is a nonlocal 2pt z-ratio analysis and must be described as extracting bare matrix elements from nonlocal two-point correlator ratios, without forcing 3pt/2pt ratio, overlap, source-sink separation, tau, or current-insertion diagnostics. Step 2 renormalization removes UV divergences and gives h_R(z), with diagnostics from renormalization constants, error amplification, and window dependence; Step 3 fourier_transform reconstructs quasi distributions from h_R(z), with diagnostics from zmin/zmax, oscillations, x/y-space errors, and the zeroth moment; Step 4 perturbative_matching applies LaMET kernels to obtain light-cone distributions, with diagnostics from positivity, first moment and deviation from 1, intermediate-x errors, and matching order; Step 5 extrapolation performs infinite-momentum or continuum extrapolation, with diagnostics from model reasonableness, fit quality, and stability.\n"
        "Data extraction rules: read or calculate only from the reports and NetCDF summaries; write 'not provided' when absent. For standard 3pt_ratio Step1, extract fit quality, excited-state gaps, overlap relative errors, and signal-to-noise at z=0 and maximal z. For `fit_scope=\"qda_ratio\"`, extract only nonlocal 2pt z-ratio fit quality, 2pt fit windows, the ordinary local 2pt denominator, z-dependent signal-to-noise, and long-Wilson-line noise behavior; do not treat missing 3pt/overlap/tsep/tau diagnostics as a problem. Step2 extract renormalization constants with errors and statistical-error amplification. Step3 extract zmin/zmax, quasi-distribution error bars, and zeroth moment. Step4 extract matched q(x) error bars, first moment, and deviation from 1. Step5 extract extrapolation model, fit quality, and final total uncertainty. For each stage, write one coherent physics summary of the operation, key result, and quality.\n"
        "Physical Summary writing rules: each stage's Physical Summary must read like publication-level prose that can be inserted directly into a paper's Results or Analysis section. Use third-person passive voice or 'we' as the subject, and describe the completed analysis rather than an ongoing process. Do not use meta-language such as 'according to the report', 'the Step 1 indicators show', or 'here we see'. Each paragraph must contain 3-5 compact professional sentences, optionally beginning with a short bold label such as **Correlator analysis.**, **Renormalization.**, **Fourier transform.**, **Perturbative matching.**, or **Extrapolation.** The paragraph must naturally include the physical purpose, key methods or settings, core numerical results with statistical/systematic precision, and a short physics interpretation of quality. All physics quantities must use `$...$`, for example `$\\chi^2/\\mathrm{dof}$`, `$h(z,P_z)$`, and `$\\langle x \\rangle$`. If the indicators are good, use scholarly language such as 'demonstrates good convergence', 'is well under control', or 'agrees with expectations'; if thresholds are approached or exceeded, state the issue faithfully with language such as 'shows a mild tension', 'indicates potential systematic effects', or 'may require further investigation'.\n"
        "Diagnostics writing rules: Diagnostics must not repeat the Summary and must not judge physics reliability only from `$\\chi^2/\\mathrm{dof}$`, logGBF, or job success. It must separate three kinds of statements: (1) numerical facts directly supported by reports/NetCDF summaries; (2) analysis issues that can be addressed through manifest changes; and (3) raw-data or external LQCD limitations that cannot be fixed by lamet-agent tuning and would require new measurements or improved external simulation conditions. If quasi distributions oscillate outside the physical region, positivity/normalization is unreasonable, error bands are large, different momenta or lattice spacings are inconsistent, renormalization produces an anomalously enlarged dynamic range, or long-distance matrix elements are noise dominated, Diagnostics must state that successful numerical execution does not imply a physically reliable result. External explanations may include the intrinsic noisiness of gluon operators, insufficient three-point statistics, limited source-sink separation or excited-state control, degraded overlap at large momentum, exponential signal-to-noise loss for long Wilson lines, lattice-spacing/volume/finite-momentum systematics, autocorrelations or too few effectively independent configurations, and weak signal from the original 2pt/3pt construction or projection. For `fit_scope=\"qda_ratio\"`, external explanations should instead focus on nonlocal 2pt z-ratio statistics, long-distance nonlocal two-point signal-to-noise, pt2 windows, the ordinary local 2pt denominator, autocorrelations, sample size, and Wilson-line length; do not require three-point statistics, source-sink separations, tau coverage, current insertion, or overlap diagnostics. These explanations must be phrased as physics interpretations consistent with the observed anomalies, not as proven facts; when the corresponding diagnostic is absent, state that confirmation requires checking the raw correlators, statistics, or independent LQCD inputs.\n"
        "Recommendation triggers: recommend manifest changes only when triggered; otherwise state that the current setting is reasonable and no change is justified. For standard 3pt_ratio Step1, poor fit quality, very large overlap errors, or h(z) signal-to-noise < 3 triggers fit-window, nstate, or statistics recommendations. For `fit_scope=\"qda_ratio\"`, poor nonlocal 2pt z-ratio fit quality or long-z signal-to-noise < 3 should prioritize `pt2_windows`, `nstate`/`nstate_values`, `fit_strategy`, `prior_width`, `svdcut`, and should not recommend `pt3_tau_cuts`. Step2 error amplification > 2.0 or significant window dependence triggers renormalization-window or scheme recommendations. Step3 strong quasi-distribution oscillations with zmax signal-to-noise < 3, or zeroth-moment deviation > 10%, triggers larger zmax or improved transform-method recommendations. Step4 first moment differs from 1 by more than 3 sigma, or q(x) shows clear unphysical values, triggers larger Pz or higher-order matching recommendations. Step5 poor extrapolation fit quality or leave-one-momentum-out changes above 1 sigma triggers adding intermediate momenta or reassessing the model. Explicitly flag any other significant anomaly.\n"
        "Each Recommended Manifest Changes item must contain `parameter`, `current_value`, `recommended_change`, `evidence`, and `expected_effect`; use `related_parameter` when the exact manifest key is uncertain. Never invent unreported numbers or phenomena. Do not use vague words such as 'maybe' or 'possibly'. If evidence is insufficient, state that the indicators are normal and there is no clear basis for a change.\n"
    )
    if language == "zh":
        system = lamet_review_rules_zh + "只根据用户提供的 stage reports、NetCDF 摘要、SVG 文件清单和 manifest 写详细科学综述，不编造未给出的数值；当设置或输出不符合真实 LaMET 场景时，必须给出可执行的 manifest 修改建议。"
        user = (
            "请直接生成完整的 `review_CN.md` 正文。请按 Stage materials 给出的顺序写；这些 stage 来自 `root_directory/artifacts_directory/<stage>` 中实际存在的 stage 子目录以及 manifest 中声明的 stage；例如 correlator_analysis 的诊断图也会从 `correlator_analysis/fit_logs` 子目录收集。"
            "每个有材料的 stage 写一个二级标题章节，并包含 `Physical Summary`、`Key figure`、`Diagnostics`、`Recommended Manifest Changes` 四个小节。"
            "`Physical Summary` 必须遵循 system prompt 中的论文正文写作规则，而不是 report 式罗列；只能使用 report 和 NetCDF 摘要中给出的数值。中文 review 的 `Physical Summary` 必须先写一个中文论文正文段落，再写一个内容对应的英文论文正文段落；两段都要保持 3–5 句、学术正文风格和 LaTeX 物理量格式。"
            "`Key figure` 中请你从该 stage 的 `svg` 列表里选择一张最能代表该 stage 质量或物理结果的 SVG；如果该列表包含组态总览图（如 `ca_<ensemble>_*.svg`、`rn_<ensemble>_*.svg`、`ft_<ensemble>_xdep.svg`、`mt_<ensemble>.svg`），必须优先选择组态总览图，否则按原规则选择单 job 图。用 Markdown 图片语法嵌入；必须原样复制该图条目里的 `markdown_path`，写成 `![说明](markdown_path)`，不要自己拼路径、不要只写文件名、不要使用 `absolute_path` 作为 Markdown 链接。图下必须用中文详细解释为什么选这张图、它应如何辅助判断该 stage；如果没有 SVG，用中文明确说明未生成可嵌入 SVG。"
            "`Diagnostics` 要用中文判断该 stage 是否自洽，尤其检查是否符合真实 LaMET 分析场景；必须遵循 system prompt 的 Diagnostics 规则，明确区分流程是否跑通、manifest 可调问题、以及无法靠 lamet-agent 调参解决的原始 LQCD 数据质量或外部计算条件问题。`Recommended Manifest Changes` 必须按上述字段格式给出；如果没有触发条件，写“当前设置合理，无需修改”。"
            "修改建议必须引用真实 manifest 路径和值，例如 `stages.<stage>.defaults.<key>`、`stages.<stage>.jobs[].params.<key>`、`inputs.kernels[].kernel_parameters.<key>`，并说明建议值或取值范围以及理由。"
            "优先讨论这些可调参数：correlator 的 `pt2_windows`、`nstate`、`fit_scope`、`fit_strategy`、`prior_width`、`svdcut`，仅当 fit_scope 使用三点函数时才讨论 `pt3_tau_cuts`；renormalization 的 `zs_fm`、`scheme_parameters.m0_gev`、`scheme_parameters.delta_m_gev`；fourier 的 `scheme_scan.zmin_values`、`zmax_values`、`z_ext_max`、`smooth`、`order`、`posterior_prior_error_scale`、`y_grid`；matching 的 `kernel_id`、`mu`、`momentum_gev`。"
            "如果 renormalization 章节已经说明 `zs_fm`，matching 章节不要重复描述同一个 `zs_fm`；只有当 manifest consistency checks 显示 matching 与 renormalization 的 `zs_fm` 不一致，或 matching 存在独立的 `zs_fm` 问题时，才在 matching 中讨论它。"
            "不要建议改 lamet-agent 代码。你不能查看 SVG 图像本身；SVG 清单只代表已生成图像的路径和 provenance，"
            "不得从 SVG 像素、path 几何、文件名臆测数值或曲线形状。图像相关判断只能来自 report 文本和 NetCDF 摘要。"
            "缺失 report、NetCDF 或 SVG 时要明确说明缺失，不能补数值。输出必须是 Markdown；除 `Physical Summary` 中额外给出的英文论文段落外，其余内容必须使用中文。\n\n"
            f"Manifest JSON:\n```json\n{json.dumps(manifest.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n```\n\n"
            f"Stage materials:\n```json\n{json.dumps(materials, ensure_ascii=False, indent=2)}\n```\n\n"
            f"Deterministic manifest consistency checks:\n```json\n{json.dumps(consistency_checks, ensure_ascii=False, indent=2)}\n```"
        )
    else:
        system = lamet_review_rules_en + "Write a detailed scientific review using only the supplied stage reports, NetCDF summaries, SVG file lists, and manifest. Do not invent unreported numbers; when settings or outputs do not match a realistic LaMET analysis scenario, give executable manifest-level recommendations."
        user = (
            "Generate the complete `review.md` body directly. Follow the order in Stage materials; these stages come from stage subdirectories under `root_directory/artifacts_directory/<stage>` plus stages declared in the manifest. For example, correlator diagnostics are also collected from the `correlator_analysis/fit_logs` subdirectory. "
            "Return normal Markdown only; do not wrap the whole answer in a fenced code block. "
            "Write one level-2 section for each stage with available material, and include `Physical Summary`, `Key figure`, `Diagnostics`, and `Recommended Manifest Changes` subsections. "
            "`Physical Summary` must follow the publication-style prose rules in the system prompt rather than report-like listing, and may only use numerical values supplied by the reports and NetCDF summaries. "
            "`Key figure` must choose one SVG from that stage's `svg` list; if the list contains an ensemble overview figure such as `ca_<ensemble>_*.svg`, `rn_<ensemble>_*.svg`, `ft_<ensemble>_xdep.svg`, or `mt_<ensemble>.svg`, choose that overview figure first, otherwise follow the usual single-job figure selection rule. Embed it with Markdown image syntax. You must copy the chosen entry's `markdown_path` exactly as `![description](markdown_path)`; do not invent paths, do not use only the basename, and do not use `absolute_path` as the Markdown link. Then give a detailed explanation below the figure stating why it was selected and how it helps assess the stage; if no SVG exists, say that no embeddable SVG was generated. "
            "`Diagnostics` must judge whether the stage is self-consistent and whether it matches a realistic LaMET analysis scenario; it must follow the Diagnostics rules in the system prompt and explicitly distinguish successful execution, manifest-tunable analysis issues, and raw-data or external LQCD limitations that lamet-agent tuning cannot fix. `Recommended Manifest Changes` must use the required field format above; if no trigger is met, state that the current setting is reasonable and no change is justified. "
            "Recommendations must cite real manifest paths and values such as `stages.<stage>.defaults.<key>`, `stages.<stage>.jobs[].params.<key>`, or `inputs.kernels[].kernel_parameters.<key>`, and state suggested values or ranges with reasons. "
            "Prioritize these tunable parameters: for correlator, `pt2_windows`, `nstate`, `fit_scope`, `fit_strategy`, `prior_width`, `svdcut`, and discuss `pt3_tau_cuts` only for three-point fit scopes; for renormalization, `zs_fm`, `scheme_parameters.m0_gev`, `scheme_parameters.delta_m_gev`; for Fourier, `scheme_scan.zmin_values`, `zmax_values`, `z_ext_max`, `smooth`, `order`, `posterior_prior_error_scale`, `y_grid`; for matching, `kernel_id`, `mu`, `momentum_gev`. "
            "If `zs_fm` has already been described in the renormalization section, do not repeat the same `zs_fm` discussion in the matching section; discuss it under matching only when the manifest consistency checks show a renormalization/matching mismatch or when there is an independent matching-specific `zs_fm` issue. "
            "Do not recommend changing lamet-agent source code. You cannot inspect SVG images; the SVG list only records figure paths and provenance. "
            "Do not infer numerical values or curve shapes from SVG pixels, path geometry, or filenames. Figure-related statements must come from report text and NetCDF summaries. "
            "State missing reports, NetCDF files, or SVG figures explicitly and do not fill in missing numbers. Output Markdown in English.\n\n"
            f"Manifest JSON:\n```json\n{json.dumps(manifest.model_dump(mode='json'), indent=2)}\n```\n\n"
            f"Stage materials:\n```json\n{json.dumps(materials, indent=2)}\n```\n\n"
            f"Deterministic manifest consistency checks:\n```json\n{json.dumps(consistency_checks, indent=2)}\n```"
        )
    review = request_llm_text(
        backend=backend,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        api_key=api_key,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )
    review = review.strip()
    if review.startswith("```"):
        lines = review.splitlines()
        if lines and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
            review = "\n".join(lines[1:-1]).strip()
    consistency_section = _format_manifest_consistency(consistency_checks, language=language)
    target.write_text(review + "\n\n" + consistency_section + "\n", encoding="utf-8")
    return {"review": str(target), "artifact": str(target), "n_stages": len(materials)}


def write_review(store: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Tool wrapper: write review from ``store['manifest']``."""
    result = write_review_from_manifest(store["manifest"], **kwargs)
    store["output"] = result["review"]
    return result


STAGE_TOOLS = {"write_review": write_review}
