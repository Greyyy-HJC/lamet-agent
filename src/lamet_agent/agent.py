"""Agent runtime loop for job-based LaMET workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .core.llm import LlmSession, format_api_model_spec, make_llm_session
from .core.data import EnsembleData
from .core.plotting import COLOR_CYCLE, ERRORBAR_STYLE, FONT_SIZE, LEGEND_SETS, default_plot
from .core.prompting import build_stage_static_prompt
from .core.tools import (
    filter_tool_kwargs,
    prepare_tool_args,
    required_job_tool_sequence,
    resolve_job_tools,
    resolve_stage_tools,
    validate_stage_inputs,
)
from .core.trace import AgentTrace
from .manifest import AnalysisManifest, ArtifactInput, StageJob, resolve_manifest_artifact_metadata

# Partial runs reference external artifacts by id; hydrate them before the LLM loop.
_STAGE_ARTIFACT_LOADERS: dict[str, dict[str, tuple[str, str]]] = {
    "renormalization": {
        "target": ("load_bare_matrix_element_grid", "target"),
        "denominator": ("load_bare_matrix_element_grid", "denominator"),
        "reference": ("load_bare_matrix_element", "reference"),
    },
    "fourier_transform": {
        "input": ("load_renormalized_matrix_element_samples", "matrix_element_data"),
    },
    "perturbative_matching": {
        "quasi": ("load_quasi_pdf", "quasi_ed"),
    },
}


def _hydrate_external_artifact_inputs(
    stage: str,
    job: StageJob,
    manifest: AnalysisManifest,
    store: dict[str, Any],
    *,
    effective_params: dict[str, Any],
    artifacts_dir: Path,
) -> None:
    """Load declared external artifacts into the job store before tool execution."""
    loaders = _STAGE_ARTIFACT_LOADERS.get(stage, {})
    if not loaders:
        return
    tools = resolve_stage_tools(stage)
    for role, (tool_name, data_key) in loaders.items():
        value = store.get(role)
        if not isinstance(value, ArtifactInput):
            continue
        tool = tools.get(tool_name)
        if tool is None:
            raise ValueError(f"stage {stage!r} missing hydration tool {tool_name!r}")
        args = prepare_tool_args(
            tool_name,
            {},
            manifest=manifest,
            stage=stage,
            job=job,
            effective_params=effective_params,
            artifacts_dir=artifacts_dir,
            store=store,
        )
        call_args, _ = filter_tool_kwargs(tool, args)
        tool(store, **call_args)
        loaded = store.get(data_key)
        if loaded is not None:
            if isinstance(loaded, EnsembleData):
                loaded.array.attrs.update(value.resolved_metadata)
            store[role] = loaded


@dataclass
class AgentState:
    """Structured state for one manifest run."""

    run_id: str
    completed_stages: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    stage_results: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    input_issues: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    pending_user_input: dict[str, dict[str, list[str]]] = field(default_factory=dict)


def _run_job(
    stage: str,
    job: StageJob,
    manifest: AnalysisManifest,
    state: AgentState,
    session: LlmSession,
    *,
    input_issues: list[str],
    max_tool_steps: int,
    backend: str,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model_spec: str | None,
    trace: AgentTrace,
    store: dict[str, Any],
    report_language: str,
) -> list[dict[str, Any]]:
    """Drive the LLM/tool loop for one isolated job store."""
    stage_tools = resolve_stage_tools(stage)
    observations: list[dict[str, Any]] = []
    stage_dir = manifest.artifacts_directory / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    if stage == "perturbative_matching":
        from lamet_agent.stages.matching.skills import effective_matching_params

        effective_params = effective_matching_params(manifest, job)
    else:
        effective_params = {**manifest.stages[stage].defaults, **job.params}

    _hydrate_external_artifact_inputs(
        stage,
        job,
        manifest,
        store,
        effective_params=effective_params,
        artifacts_dir=stage_dir,
    )
    tools = resolve_job_tools(
        stage,
        job,
        effective_params,
        stage_tools=stage_tools,
    )
    required_sequence = required_job_tool_sequence(stage, job, effective_params)
    required_index = 0

    if stage == "renormalization" and effective_params.get("normalization", True):
        from lamet_agent.stages.renorm.functions import normalize_bare_matrix_element_at_z0

        for role, value in list(store.items()):
            if isinstance(value, EnsembleData):
                store[role] = normalize_bare_matrix_element_at_z0(value)

    static_prompt = build_stage_static_prompt(
        stage,
        manifest,
        job=job,
        effective_params=effective_params,
        completed_stages=state.completed_stages.copy(),
        input_issues=input_issues,
        allowed_tool_names=sorted(tools),
    )
    session.begin_stage(static_prompt)
    trace.stage_context(static_prompt)

    if input_issues:
        action = {
            "action": "request_user_input",
            "reason": "This job's manifest inputs are incomplete.",
            "questions": input_issues,
        }
        trace.cycle_begin(1)
        trace.model_output(action)
        state.actions.append({"stage": stage, "job": job.id, "action": action})
        state.pending_user_input.setdefault(stage, {})[job.id] = input_issues
        return observations

    last_observation: dict[str, Any] | None = None
    for cycle in range(1, max_tool_steps + 1):
        trace.cycle_begin(cycle)
        if last_observation is not None:
            trace.prompt_delta(last_observation)
        trace.llm_call_begin(backend=backend, model_spec=model_spec)
        action = session.decide(last_observation=last_observation)
        trace.llm_call_end()
        trace.model_output(action)
        state.actions.append({"stage": stage, "job": job.id, "action": action})

        if action.get("action") != "call_tool":
            if required_index < len(required_sequence):
                expected = required_sequence[required_index]
                observation = {
                    "tool_name": expected,
                    "error": (
                        f"job {job.id!r} cannot {action.get('action', 'stop')} before "
                        f"required tool {expected!r} succeeds"
                    ),
                }
                observations.append(observation)
                trace.observation(observation)
                last_observation = observation
                continue
            break

        tool_name = action.get("tool_name", "")
        expected = required_sequence[required_index] if required_index < len(required_sequence) else None
        tool = tools.get(tool_name)
        if expected is not None and tool_name != expected:
            observation = {
                "tool_name": tool_name,
                "error": f"job {job.id!r} must call required tool {expected!r} next",
            }
        elif tool is None:
            error = "tool is not allowed for the current job" if tool_name in stage_tools else "unknown tool"
            observation = {"tool_name": tool_name, "error": error}
        else:
            args = prepare_tool_args(
                tool_name,
                action.get("args", {}) or {},
                manifest=manifest,
                stage=stage,
                job=job,
                effective_params=effective_params,
                artifacts_dir=stage_dir,
                store=store,
            )
            if (stage == "review" and tool_name == "write_review") or (
                stage == "perturbative_matching" and tool_name == "report_matching_result"
            ):
                # Both tools call an LLM. Hand them this run's resolved config rather than
                # letting them rediscover it from the environment.
                args["report_language"] = report_language
                args["backend"] = backend
                args["provider"] = provider
                args["model_name"] = model_name
                args["api_key"] = api_key
                args["base_url"] = base_url
            call_args, dropped_args = filter_tool_kwargs(tool, args)
            try:
                result = tool(store, **call_args)
                observation = {"tool_name": tool_name, "result": result}
                if dropped_args:
                    observation["ignored_args"] = dropped_args
            except (ValueError, TypeError, FileNotFoundError) as exc:
                observation = {"tool_name": tool_name, "error": str(exc)}
        if observation.get("result") is not None and tool_name == expected:
            required_index += 1
        observations.append(observation)
        trace.observation(observation)
        last_observation = observation

    if required_index < len(required_sequence):
        missing = list(required_sequence[required_index:])
        last_error = next(
            (str(item["error"]) for item in reversed(observations) if item.get("error")),
            "no tool error was recorded",
        )
        raise ValueError(
            f"job {stage}/{job.id} did not complete required tools {missing} "
            f"within {max_tool_steps} steps; last error: {last_error}"
        )
    if required_sequence and "output" not in store:
        raise ValueError(f"job {stage}/{job.id} completed its tool sequence without store['output']")

    trace.stage_end(f"{stage}/{job.id}", n_steps=cycle)
    return observations


def _last_tool_result(observations: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    """Return the last successful result for ``tool_name`` in a job observation list."""
    for observation in reversed(observations):
        if observation.get("tool_name") == tool_name and isinstance(observation.get("result"), dict):
            return observation["result"]
    return None


def _path_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(item) for item in value.values() if item]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def _correlator_sample0_plots(result: dict[str, Any]) -> list[str]:
    plots: list[str] = []
    for z_fit in result.get("z_fits", []) or []:
        if not isinstance(z_fit, dict):
            continue
        plots.extend(_path_values(z_fit.get("sample0_plot_paths")))
    return plots


def _write_correlator_energy_artifacts(records: list[dict[str, Any]], stage_dir: Path) -> dict[str, str]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("pt2_path"),
            record.get("ensemble"),
            record.get("hadron"),
            record.get("gfix"),
            record.get("volume"),
            record.get("lattice_spacing_fm"),
            record.get("source_operator"),
            record.get("sink_operator"),
            record.get("fitting_form"),
            record.get("channel"),
            record.get("momentum"),
        )
        deduped.setdefault(key, record)
    rows = list(deduped.values())
    if not rows:
        return {}

    stage_dir.mkdir(parents=True, exist_ok=True)
    row = np.arange(len(rows), dtype=int)
    numeric_keys = [
        "lattice_spacing_fm",
        "p_gev",
        "p2_gev2",
        "E0_lattice_mean",
        "E0_lattice_sdev",
        "E0_gev_mean",
        "E0_gev_sdev",
        "E0_gev2_mean",
        "E0_gev2_sdev",
    ]
    text_keys = [
        "job_id",
        "pt2_path",
        "ensemble",
        "hadron",
        "gfix",
        "volume",
        "source_operator",
        "sink_operator",
        "fitting_form",
        "channel",
        "momentum",
    ]
    data_vars = {
        key: ("row", np.asarray([float(item.get(key, np.nan)) for item in rows], dtype=float))
        for key in numeric_keys
    }
    data_vars.update({key: ("row", np.asarray([str(item.get(key, "")) for item in rows], dtype=object)) for key in text_keys})
    dataset = xr.Dataset(data_vars=data_vars, coords={"row": row})
    dataset.attrs.update(
        {
            "description": "Ground-state energies from final 2pt fit posteriors used in correlator_analysis.",
            "energy_unit": "GeV",
            "momentum_unit": "GeV",
            "dispersion_y": "E0_gev_mean^2 with propagated E0_gev_sdev",
        }
    )
    e0_path = stage_dir / "dispersion_relation.nc"
    dataset.to_netcdf(e0_path)

    fig, ax = default_plot()
    labels = sorted({str(item.get("ensemble", "")) for item in rows})
    for index, label in enumerate(labels):
        group = [item for item in rows if str(item.get("ensemble", "")) == label]
        group.sort(key=lambda item: (float(item["p2_gev2"]), str(item.get("channel", "")), str(item.get("momentum", ""))))
        ax.errorbar(
            [float(item["p2_gev2"]) for item in group],
            [float(item["E0_gev2_mean"]) for item in group],
            yerr=[float(item["E0_gev2_sdev"]) for item in group],
            label="_nolegend_",
            color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
            **ERRORBAR_STYLE,
        )
    p2_max = max(float(item["p2_gev2"]) for item in rows)
    p2_line = np.linspace(0.0, p2_max * 1.05 if p2_max > 0.0 else 1.0, 200)
    ax.plot(p2_line, p2_line, color="0.65", linestyle="--", linewidth=1.0, label=r"$E^2=p^2$")
    for index, label in enumerate(labels):
        zeros = [item for item in rows if str(item.get("ensemble", "")) == label and abs(float(item["p2_gev2"])) < 1e-12]
        if not zeros:
            continue
        mass2 = float(zeros[0]["E0_gev2_mean"])
        ax.plot(
            p2_line,
            mass2 + p2_line,
            color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
            linewidth=1.0,
            alpha=0.65,
            label=label or "ensemble",
        )
    ax.set_xlabel(r"$p^2\,[\mathrm{GeV}^2]$", **FONT_SIZE)
    ax.set_ylabel(r"$E_0^2\,[\mathrm{GeV}^2]$", **FONT_SIZE)
    ax.set_title("Dispersion relation", **FONT_SIZE)
    ax.legend(**{**LEGEND_SETS, "loc": "upper left"})
    fig.tight_layout()
    pdf_path = stage_dir / "dispersion_relation.pdf"
    svg_path = stage_dir / "dispersion_relation.svg"
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return {
        "E0_artifact": str(e0_path),
        "dispersion_relation_plot": str(pdf_path),
        "dispersion_relation_image": str(svg_path),
    }


def _normalize_report_language(report_language: str) -> str:
    language = report_language.lower()
    if language not in {"en", "ch"}:
        raise ValueError("report_language must be 'en' or 'ch'")
    return language


def run_agent(
    manifest: AnalysisManifest,
    *,
    backend: str,
    actions_path: str | Path | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tool_steps: int = 40,
    verbose: bool = False,
    report_language: str = "en",
) -> dict[str, Any]:
    """Execute the manifest's ordered stages and per-stage jobs."""
    report_language = _normalize_report_language(report_language)
    resolve_manifest_artifact_metadata(manifest)
    selected = list(manifest.metadata.stages)
    state = AgentState(run_id=manifest.run_id)
    session = make_llm_session(
        backend,
        actions_path,
        api_key=api_key,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
    )
    trace = AgentTrace(enabled=verbose, quiet_ui=not verbose)
    outputs: dict[str, Any] = {item.id: item for item in manifest.inputs.artifacts}
    stage_reports: dict[str, dict[str, str]] = {}
    model_spec = (
        format_api_model_spec(provider, model_name)
        if backend == "api" and provider and model_name
        else None
    )

    if verbose:
        trace.run_begin(
            run_id=manifest.run_id,
            backend=backend,
            stages=selected,
            model_spec=model_spec,
        )
    else:
        trace.run_banner(
            run_id=manifest.run_id,
            backend=backend,
            stages=selected,
            model_spec=model_spec,
        )
    for stage in selected:
        state.stage_results[stage] = {}
        stage_job_records: list[dict[str, Any]] = []
        for job in manifest.stages[stage].jobs:
            issues = validate_stage_inputs(stage, manifest, job)
            if issues:
                state.input_issues.setdefault(stage, {})[job.id] = issues
            if verbose:
                trace.stage_begin(f"{stage}/{job.id}", input_issues=issues or None)
            else:
                trace.job_begin(stage, job.id, input_issues=issues or None)

            store: dict[str, Any] = {}
            if stage == "review":
                store["manifest"] = manifest
            for role, value in job.inputs.items():
                if isinstance(value, list):
                    missing = [ref for ref in value if ref not in outputs]
                    if missing:
                        raise ValueError(f"job {job.id!r} has upstream jobs without output: {missing}")
                    store[role] = [outputs[ref] for ref in value]
                else:
                    if value not in outputs:
                        raise ValueError(f"job {job.id!r} has an upstream job without output: {value!r}")
                    store[role] = outputs[value]
            observations = _run_job(
                stage,
                job,
                manifest,
                state,
                session,
                input_issues=issues,
                max_tool_steps=max_tool_steps,
                backend=backend,
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                model_spec=model_spec,
                trace=trace,
                store=store,
                report_language=report_language,
            )
            state.stage_results[stage][job.id] = observations
            if job.id in state.pending_user_input.get(stage, {}):
                break
            if "output" in store:
                outputs[job.id] = store["output"]
            if stage == "review" and "output" in store:
                stage_reports[stage] = {"report": str(store["output"])}
            if stage == "correlator_analysis" and "output" in store:
                fit_result = _last_tool_result(observations, "fit_bare_matrix_grid")
                if fit_result is not None:
                    stage_job_records.append(
                        {
                            "job_id": job.id,
                            "result": fit_result,
                            "artifacts": {
                                "bare_artifact": fit_result.get("artifact") or fit_result.get("netcdf_path"),
                                "summary_plot": fit_result.get("plot_pdf"),
                                "summary_plot_image": fit_result.get("plot_svg"),
                                "tuning_log": fit_result.get("tuning_log_path"),
                                "sample_log": fit_result.get("sample_log_path"),
                                "sample0_pt2_plots": _path_values(fit_result.get("sample0_pt2_plot_paths")),
                                "sample0_fit_plots": _correlator_sample0_plots(fit_result),
                            },
                        }
                    )
            if stage == "renormalization" and "output" in store:
                apply_result = _last_tool_result(observations, "apply_self_renormalization")
                if apply_result is None:
                    apply_result = _last_tool_result(observations, "apply_ratio_scheme_renormalization")
                fit_result = _last_tool_result(observations, "fit_self_renormalization_factor")
                plot_result = _last_tool_result(observations, "plot_renormalized_matrix_element") or {}
                diag_result = _last_tool_result(observations, "plot_self_renormalization_diagnostics") or {}
                if apply_result is not None:
                    z_grid = []
                    matrix = store.get("matrix_element_data") or store.get("output")
                    if hasattr(matrix, "coords") and "z" in matrix.coords:
                        z_grid = np.asarray(matrix.coords["z"], dtype=float).tolist()
                    artifacts = {
                        "renormalized_artifact": apply_result.get("artifact")
                        or store.get("matrix_element_netcdf"),
                        "renormalized_plot": plot_result.get("plot"),
                        "renormalized_plot_image": plot_result.get("plot_image"),
                    }
                    for key, value in (diag_result.get("plots") or {}).items():
                        artifacts[f"diag_{key}"] = value
                    stage_job_records.append(
                        {
                            "job_id": job.id,
                            "result": {
                                **apply_result,
                                "scheme": apply_result.get("scheme", "hybrid_ratio"),
                                "z_grid": z_grid,
                                "diagnostic_plots": list((diag_result.get("plots") or {}).keys()),
                            },
                            "artifacts": artifacts,
                        }
                    )
                elif fit_result is not None:
                    artifacts = {
                        "zR_artifact": fit_result.get("artifact") or store.get("zR_netcdf"),
                    }
                    for key, value in (diag_result.get("plots") or {}).items():
                        artifacts[f"diag_{key}"] = value
                    stage_job_records.append(
                        {
                            "job_id": job.id,
                            "result": {
                                **fit_result,
                                "scheme": "hybrid_self_renormalization",
                                "job_kind": "fit",
                                "diagnostic_plots": list((diag_result.get("plots") or {}).keys()),
                            },
                            "artifacts": artifacts,
                        }
                    )
            if stage == "fourier_transform" and "fourier_result" in store:
                stage_job_records.append(
                    {
                        "job_id": job.id,
                        "result": store["fourier_result"],
                        "summary": store.get("fourier_summary"),
                        "artifacts": {
                            "fourier_artifact": store["fourier_result"].get("artifact"),
                            "fit_info_artifact": store["fourier_result"].get("fit_info_artifact"),
                            "fourier_plot": store.get("fourier_plot", {}).get("plot"),
                            "fourier_plot_image": store.get("fourier_plot", {}).get("plot_image"),
                            "extension_plot_re": store.get("fourier_extension_plot", {}).get("plot_re"),
                            "extension_plot_re_image": store.get("fourier_extension_plot", {}).get("plot_re_image"),
                            "extension_plot_im": store.get("fourier_extension_plot", {}).get("plot_im"),
                            "extension_plot_im_image": store.get("fourier_extension_plot", {}).get("plot_im_image"),
                        },
                    }
                )
            if stage == "perturbative_matching" and "lightcone_ed" in store and "quasi_ed" in store and "x_ls" in store:
                quasi_ed = store["quasi_ed"]
                lightcone_ed = store["lightcone_ed"]
                kernel_info = dict(store.get("matching_kernel_info", {}))
                x_ls = np.asarray(store["x_ls"], dtype=float)
                stage_job_records.append(
                    {
                        "job_id": job.id,
                        "result": {
                            **kernel_info,
                            "component": store.get("matching_component"),
                            "source": "job input",
                            "resample": lightcone_ed.resample,
                            "n_sample": int(lightcone_ed.n_sample),
                            "n_points": int(x_ls.size),
                            "x_grid": x_ls.tolist(),
                            "quasi_mean": [float(v) for v in np.asarray(quasi_ed.mean)],
                            "lightcone_mean": [float(v) for v in np.asarray(lightcone_ed.mean)],
                        },
                        "artifacts": {
                            "lightcone_artifact": store.get("matching_artifact"),
                            "matched_plot": store.get("matching_plot", {}).get("path"),
                            "matched_plot_image": store.get("matching_plot", {}).get("plot_image"),
                        },
                    }
                )
        if stage in state.pending_user_input:
            break
        if stage == "correlator_analysis" and stage_job_records:
            from lamet_agent.stages.correlator.reporting import write_correlator_stage_report

            energy_artifacts = _write_correlator_energy_artifacts(
                [
                    energy
                    for record in stage_job_records
                    for energy in (record.get("result", {}).get("pt2_energies") or [])
                    if isinstance(energy, dict)
                ],
                manifest.artifacts_directory / stage,
            )
            if energy_artifacts:
                for record in stage_job_records:
                    record.setdefault("artifacts", {}).update(energy_artifacts)
            paths = write_correlator_stage_report(
                jobs=stage_job_records,
                path=manifest.artifacts_directory / stage / "ca_report.md",
                report_language=report_language,
            )
            stage_reports[stage] = {"report": str(paths["report"])}
        if stage == "renormalization" and stage_job_records:
            from lamet_agent.stages.renorm.reporting import write_renorm_stage_report

            paths = write_renorm_stage_report(
                jobs=stage_job_records,
                path=manifest.artifacts_directory / stage / "renorm_report.md",
                report_language=report_language,
            )
            stage_reports[stage] = {"report": str(paths["report"])}
        if stage == "fourier_transform" and stage_job_records:
            from lamet_agent.stages.fourier.reporting import write_fourier_stage_report

            paths = write_fourier_stage_report(
                jobs=stage_job_records,
                path=manifest.artifacts_directory / stage / "ft_report.md",
                report_language=report_language,
            )
            stage_reports[stage] = {"report": str(paths["report"])}
        if stage == "perturbative_matching" and stage_job_records:
            from lamet_agent.stages.matching.reporting import FormulaLlm, write_matching_stage_report

            paths = write_matching_stage_report(
                jobs=stage_job_records,
                path=manifest.artifacts_directory / stage / "matching_report.md",
                report_language=report_language,
                llm=FormulaLlm(
                    backend=backend, provider=provider, api_key=api_key,
                    model_name=model_name, base_url=base_url,
                ),
            )
            stage_reports[stage] = {"report": str(paths["report"])}
        state.completed_stages.append(stage)

    trace.run_end(action_count=len(state.actions))
    status = "waiting_for_user_input" if state.pending_user_input else "completed"
    result: dict[str, Any] = {
        "run_id": manifest.run_id,
        "status": status,
        "backend": backend,
        "stages": selected,
        "completed_stages": state.completed_stages,
        "input_issues": state.input_issues,
        "pending_user_input": state.pending_user_input,
        "actions": state.actions,
        "stage_results": state.stage_results,
        "stage_reports": stage_reports,
        "outputs": sorted(outputs),
        "summary": json.dumps(
            {"run_id": manifest.run_id, "stage_count": len(state.completed_stages), "action_count": len(state.actions)}
        ),
    }
    if model_spec is not None:
        result["model"] = model_spec
    return result
