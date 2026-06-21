"""Agent runtime loop for job-based LaMET workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .core.llm import LlmSession, make_llm_session
from .core.prompting import build_stage_static_prompt
from .core.tools import filter_tool_kwargs, prepare_tool_args, resolve_stage_tools, validate_stage_inputs
from .core.trace import AgentTrace
from .manifest import AnalysisManifest, ArtifactInput, StageJob

# Partial runs reference external artifacts by id; hydrate them before the LLM loop.
_STAGE_ARTIFACT_LOADERS: dict[str, dict[str, tuple[str, str]]] = {
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
    model: str,
    trace: AgentTrace,
    store: dict[str, Any],
) -> list[dict[str, Any]]:
    """Drive the LLM/tool loop for one isolated job store."""
    tools = resolve_stage_tools(stage)
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

    static_prompt = build_stage_static_prompt(
        stage,
        manifest,
        job=job,
        effective_params=effective_params,
        completed_stages=state.completed_stages.copy(),
        input_issues=input_issues,
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
        trace.llm_call_begin(model=model)
        action = session.decide(last_observation=last_observation)
        trace.llm_call_end()
        trace.model_output(action)
        state.actions.append({"stage": stage, "job": job.id, "action": action})

        if action.get("action") != "call_tool":
            break

        tool_name = action.get("tool_name", "")
        tool = tools.get(tool_name)
        if tool is None:
            observation = {"tool_name": tool_name, "error": "unknown tool"}
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
            call_args, dropped_args = filter_tool_kwargs(tool, args)
            try:
                result = tool(store, **call_args)
                observation = {"tool_name": tool_name, "result": result}
                if dropped_args:
                    observation["ignored_args"] = dropped_args
            except (ValueError, TypeError, FileNotFoundError) as exc:
                observation = {"tool_name": tool_name, "error": str(exc)}
        observations.append(observation)
        trace.observation(observation)
        last_observation = observation

    trace.stage_end(f"{stage}/{job.id}", n_steps=cycle)
    return observations


def run_agent(
    manifest: AnalysisManifest,
    *,
    model: str = "mock",
    actions_path: str | Path | None = None,
    api_key: str | None = None,
    llm_model: str | None = None,
    base_url: str | None = None,
    max_tool_steps: int = 40,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute the manifest's ordered stages and per-stage jobs."""
    selected = list(manifest.metadata.stages)
    state = AgentState(run_id=manifest.run_id)
    session = make_llm_session(model, actions_path, api_key, llm_model, base_url)
    trace = AgentTrace(enabled=verbose)
    outputs: dict[str, Any] = {item.id: item for item in manifest.inputs.artifacts}
    stage_reports: dict[str, dict[str, str]] = {}

    trace.run_begin(run_id=manifest.run_id, model=model, stages=selected)
    for stage in selected:
        state.stage_results[stage] = {}
        stage_job_records: list[dict[str, Any]] = []
        for job in manifest.stages[stage].jobs:
            issues = validate_stage_inputs(stage, manifest, job)
            if issues:
                state.input_issues.setdefault(stage, {})[job.id] = issues
            trace.stage_begin(f"{stage}/{job.id}", input_issues=issues or None)

            store: dict[str, Any] = {}
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
                model=model,
                trace=trace,
                store=store,
            )
            state.stage_results[stage][job.id] = observations
            if job.id in state.pending_user_input.get(stage, {}):
                break
            if "output" in store:
                outputs[job.id] = store["output"]
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
                        },
                    }
                )
        if stage in state.pending_user_input:
            break
        if stage == "fourier_transform" and stage_job_records:
            from lamet_agent.stages.fourier.reporting import write_fourier_stage_report

            paths = write_fourier_stage_report(
                jobs=stage_job_records,
                path=manifest.artifacts_directory / stage / "ft_report.md",
            )
            stage_reports[stage] = {"report": str(paths["en"]), "report_cn": str(paths["zh"])}
        if stage == "perturbative_matching" and stage_job_records:
            from lamet_agent.stages.matching.reporting import write_matching_stage_report

            paths = write_matching_stage_report(
                jobs=stage_job_records,
                path=manifest.artifacts_directory / stage / "matching_report.md",
            )
            stage_reports[stage] = {"report": str(paths["en"]), "report_cn": str(paths["zh"])}
        state.completed_stages.append(stage)

    trace.run_end(action_count=len(state.actions))
    status = "waiting_for_user_input" if state.pending_user_input else "completed"
    return {
        "run_id": manifest.run_id,
        "status": status,
        "model": model,
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
