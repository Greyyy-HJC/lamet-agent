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
from .core.plotting import COLOR_CYCLE, ERRORBAR_STYLE, FIG_SIZE, FONT_SIZE, LABEL_SIZE, LEGEND_SETS, apply_plot_style, default_plot
from .core.prompting import build_stage_static_prompt
from .core.resampling import sample_mean_and_sdev
from .core.tools import (
    filter_tool_kwargs,
    prepare_tool_args,
    required_job_tool_sequence,
    resolve_job_tools,
    resolve_stage_tools,
    validate_stage_diagnostics,
    validate_stage_inputs,
)
from .core.trace import AgentTrace
from .manifest import AnalysisManifest, ArtifactInput, StageJob, is_constant_job_input, resolve_manifest_artifact_metadata
from .manifest_params import resolve_stage_params

# Partial runs reference external artifacts by id; hydrate them before the LLM loop.
_STAGE_ARTIFACT_LOADERS: dict[str, dict[str, tuple[str, str]]] = {
    "renormalization": {
        "target": ("load_bare_matrix_element_grid", "target"),
        "denominator": ("load_bare_matrix_element_grid", "denominator"),
        "reference": ("load_bare_matrix_element", "reference"),
    },
    "fourier_transform": {
        "input": ("load_renormalized_matrix_element_samples", "matrix_element_data"),
        "hermitian_partner": ("load_renormalized_matrix_element_samples", "matrix_element_data"),
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
        if stage == "fourier_transform":
            args["path"] = value.path
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
    is_systematics_job = (
        job.params.get("operation") == "systematics_budget"
        or job.id == "ex_other"
        or job.id.endswith(("_zs_low", "_zs_high", "_lambda_low", "_lambda_high", "_mu_low", "_mu_high", "_a_sym", "_p_sym", "_ap_sym", "_budget"))
    )
    stage_dir = manifest.artifacts_directory / stage / "sym" if is_systematics_job else manifest.artifacts_directory / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    if stage == "perturbative_matching":
        from lamet_agent.stages.matching.validation import effective_matching_params

        effective_params = effective_matching_params(manifest, job)
    else:
        effective_params = resolve_stage_params(stage, manifest.stages[stage].defaults, job.params)

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

    if stage == "renormalization" and effective_params.get("normalization") is True:
        from lamet_agent.stages.renorm.functions import normalize_bare_matrix_element_at_z0

        for role, value in list(store.items()):
            if role != "zR" and isinstance(value, EnsembleData):
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
            if action.get("action") == "request_user_input":
                args = action.get("args") or {}
                questions = action.get("questions")
                if questions is None:
                    prompt = args.get("prompt") or action.get("reason") or "User input requested."
                    questions = [prompt] if isinstance(prompt, str) else list(prompt)
                elif isinstance(questions, str):
                    questions = [questions]
                else:
                    questions = list(questions)
                state.pending_user_input.setdefault(stage, {})[job.id] = questions
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
            if (
                (stage == "review" and tool_name == "write_review")
                or (stage == "fourier_transform" and tool_name == "report_fourier_result")
                or (stage == "perturbative_matching" and tool_name == "report_matching_result")
            ):
                # These tools may call an LLM. Hand them this run's resolved config rather than
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

    if job.id not in state.pending_user_input.get(stage, {}):
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


def _ensemble_label(data: EnsembleData, fallback: str = "") -> str:
    value = fallback or data.attrs.get("ensemble")
    if not value and data.ensemble is not None:
        value = data.ensemble.id
    return str(value or "ensemble")


def _momentum_label(attrs: dict[str, Any], result: dict[str, Any]) -> str:
    form = str(attrs.get("fitting_form") or result.get("fitting_form") or "Breit")
    if form == "NonBreit" or result.get("final_momentum_gev", attrs.get("final_momentum_gev")) is not None:
        t_gev2 = result.get("t_gev2", attrs.get("t_gev2"))
        xi = result.get("xi", attrs.get("xi"))
        initial = result.get("initial_momentum_gev", attrs.get("initial_momentum_gev"))
        if initial is None:
            initial = result.get("momentum_gev", attrs.get("momentum_gev"))
        final = result.get("final_momentum_gev", attrs.get("final_momentum_gev"))
        if (t_gev2 is None or xi is None) and initial is not None and final is not None:
            t_gev2 = (float(final) - float(initial)) ** 2
            denominator = float(initial) + float(final)
            xi = None if denominator == 0.0 else (float(initial) - float(final)) / denominator
        t_text = "n/a" if t_gev2 is None else f"{float(t_gev2):.2f}"
        xi_text = "n/a" if xi is None else f"{float(xi):.2f}"
        initial_text = "n/a" if initial is None else f"{float(initial):.2f}"
        final_text = "n/a" if final is None else f"{float(final):.2f}"
        return (
            rf"$P_i^z={initial_text},\ P_f^z={final_text}\,\mathrm{{GeV}}$, "
            rf"$t={t_text}\,\mathrm{{GeV}}^2$, $\xi={xi_text}$"
        )
    momentum = attrs.get("momentum_gev") or result.get("momentum_gev")
    p_text = "n/a" if momentum in (None, "") else f"{float(momentum):.2f}"
    return rf"$p={p_text}\,\mathrm{{GeV}}$"


def _overlay_paths(stage_dir: Path, prefix: str, ensemble: str, suffix: str) -> tuple[Path, Path]:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in ensemble)
    stem = stage_dir / f"{prefix}_{safe}_{suffix}" if suffix else stage_dir / f"{prefix}_{safe}"
    return stem.with_suffix(".pdf"), stem.with_suffix(".svg")


def _write_matrix_overlay_artifacts(
    jobs: list[dict[str, Any]],
    stage_dir: Path,
    *,
    artifact_key: str,
    prefix: str,
    title_suffix: str,
    y_label: str,
    x_label: str = r"$z/a$",
) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    records = [record for record in jobs if record.get("artifacts", {}).get(artifact_key) and not record.get("is_systematics")]
    if len(records) <= 1:
        return {}
    for record in records:
        path = record.get("artifacts", {}).get(artifact_key)
        if not path:
            continue
        data = EnsembleData.from_netcdf(path)
        groups.setdefault(_ensemble_label(data, str(record.get("result", {}).get("ensemble") or "")), []).append(
            {"record": record, "data": data}
        )
    artifacts: dict[str, str] = {}
    for ensemble, items in groups.items():
        if len(items) <= 1:
            continue
        for part, attr_key, marker in (("re", "matrix_overlay_re", "o"), ("im", "matrix_overlay_im", "s")):
            fig, ax = default_plot()
            n_items = len(items)
            for index, item in enumerate(items):
                data: EnsembleData = item["data"]
                z = np.asarray(data.coords["z"], dtype=float)
                values = np.asarray(data.values)
                mode = "jk" if data.resample == "jackknife" else "bs"
                sample_error_mode = str(data.attrs.get("sample_error_mode", data.attrs.get("average_method", "covariance")))
                z_step = float(np.median(np.diff(np.unique(z)))) if len(np.unique(z)) > 1 else 1.0
                offset = 0.06 * z_step * (index - (n_items - 1) / 2.0)
                if part == "re":
                    mean, err = sample_mean_and_sdev(np.real(values), mode=mode, sample_error_mode=sample_error_mode, axis=0)
                else:
                    mean, err = sample_mean_and_sdev(np.imag(values), mode=mode, sample_error_mode=sample_error_mode, axis=0)
                ax.errorbar(
                    z + offset,
                    mean,
                    np.abs(err),
                    label=_momentum_label(dict(data.attrs), item["record"].get("result", {})),
                    color=COLOR_CYCLE[index % len(COLOR_CYCLE)],
                    marker=marker,
                    **ERRORBAR_STYLE,
                )
            ax.set_xlabel(x_label, **FONT_SIZE)
            ax.set_ylabel(y_label, **FONT_SIZE)
            ax.set_title(f"{ensemble} {title_suffix}", **FONT_SIZE)
            ax.legend(**LEGEND_SETS)
            fig.tight_layout()
            pdf, svg = _overlay_paths(stage_dir, prefix, ensemble, part)
            fig.savefig(pdf, bbox_inches="tight", transparent=True)
            fig.savefig(svg, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)
            artifacts[f"{attr_key}_{pdf.stem}"] = str(pdf)
            artifacts[f"{attr_key}_image_{svg.stem}"] = str(svg)
    return artifacts


def _write_fourier_overlay_artifacts(jobs: list[dict[str, Any]], stage_dir: Path) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    records = [record for record in jobs if record.get("artifacts", {}).get("fourier_artifact") and not record.get("is_systematics")]
    if len(records) <= 1:
        return {}
    for record in records:
        path = record.get("artifacts", {}).get("fourier_artifact")
        if not path:
            continue
        data = EnsembleData.from_netcdf(path)
        ensemble = _ensemble_label(data, str(record.get("result", {}).get("ensemble") or ""))
        groups.setdefault(ensemble, []).append({"record": record, "data": data})
    artifacts: dict[str, str] = {}
    for ensemble, items in groups.items():
        if len(items) <= 1:
            continue
        import matplotlib.pyplot as plt

        apply_plot_style()
        fig, (ax_re, ax_im) = plt.subplots(
            2,
            1,
            figsize=FIG_SIZE,
            gridspec_kw={"height_ratios": [1, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0)
        for ax in (ax_re, ax_im):
            ax.tick_params(direction="in", top=True, right=True, **LABEL_SIZE)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(linestyle=":")
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        for index, item in enumerate(items):
            data: EnsembleData = item["data"]
            result = item["record"].get("result", {})
            x = np.asarray(data.coords["x"], dtype=float)
            color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
            values = np.asarray(data.values)
            mode = "jk" if data.resample == "jackknife" else "bs"
            sample_error_mode = str(data.attrs.get("sample_error_mode", data.attrs.get("average_method", "covariance")))
            for ax, name in ((ax_re, "re"), (ax_im, "im")):
                if f"ft_{name}_mean" in data.attrs:
                    mean = np.asarray(json.loads(data.attrs[f"ft_{name}_mean"]), dtype=float)
                    stat = np.asarray(json.loads(data.attrs.get(f"ft_{name}_stat_sdev", "0.0")), dtype=float)
                    sys = np.asarray(json.loads(data.attrs.get(f"ft_{name}_sys_sdev", "0.0")), dtype=float)
                    err = np.sqrt(stat**2 + sys**2)
                else:
                    mean, err = sample_mean_and_sdev(
                        np.real(values) if name == "re" else np.imag(values),
                        mode=mode,
                        sample_error_mode=sample_error_mode,
                        axis=0,
                    )
                mean = np.where(np.abs(mean) < 1e-14, 0.0, mean)
                err = np.where(err < 1e-14, 0.0, err)
                ax.fill_between(x, mean - err, mean + err, color=color, alpha=0.28, linewidth=0, label=_momentum_label(dict(data.attrs), result))
                ax.plot(x, mean, color=color, linewidth=0.9, alpha=0.72)
        ax_re.set_xlim(-2.0, 2.0)
        ax_im.set_xlim(-2.0, 2.0)
        is_gpd = any(str(item["data"].attrs.get("target_observable", "")) == "gpd" for item in items)
        distribution = r"\widetilde H(y,\xi,t)" if is_gpd else r"\tilde q(x)"
        ax_re.set_ylabel(rf"$\mathrm{{Re}}\,{distribution}$", **FONT_SIZE)
        ax_im.set_ylabel(rf"$\mathrm{{Im}}\,{distribution}$", **FONT_SIZE)
        ax_re.yaxis.set_label_coords(-0.11, 0.5)
        ax_im.yaxis.set_label_coords(-0.11, 0.5)
        ax_im.set_xlabel(r"$y$" if is_gpd else r"$x$", **FONT_SIZE)
        ax_re.legend(**LEGEND_SETS)
        ax_im.legend(**LEGEND_SETS)
        ax_re.set_title(f"{ensemble} {'paired quasi-GPD' if is_gpd else 'quasi distribution'}", **FONT_SIZE)
        fig.tight_layout()
        pdf, svg = _overlay_paths(stage_dir, "ft", ensemble, "xdep")
        fig.savefig(pdf, bbox_inches="tight", transparent=True)
        fig.savefig(svg, bbox_inches="tight")

        plt.close(fig)
        artifacts[f"fourier_overlay_{pdf.stem}"] = str(pdf)
        artifacts[f"fourier_overlay_image_{svg.stem}"] = str(svg)
    return artifacts


def _write_matching_overlay_artifacts(jobs: list[dict[str, Any]], stage_dir: Path) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    records = [record for record in jobs if record.get("artifacts", {}).get("lightcone_artifact") and not record.get("is_systematics")]
    if len(records) <= 1:
        return {}
    for record in records:
        path = record.get("artifacts", {}).get("lightcone_artifact")
        if not path:
            continue
        lightcone = EnsembleData.from_netcdf(path)
        ensemble = _ensemble_label(lightcone, str(record.get("result", {}).get("ensemble") or ""))
        groups.setdefault(ensemble, []).append({"record": record, "lightcone": lightcone})
    artifacts: dict[str, str] = {}
    for ensemble, items in groups.items():
        if len(items) <= 1:
            continue
        fig, ax = default_plot()
        x_limits = []
        y_limits = []
        for index, item in enumerate(items):
            result = item["record"].get("result", {})
            color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
            p = result.get("momentum_gev")
            p_text = "n/a" if p in (None, "") else f"{float(p):.2f}"
            if result.get("matching_plot_xlim"):
                x_limits.append(result["matching_plot_xlim"])
            if result.get("matching_plot_ylim"):
                y_limits.append(result["matching_plot_ylim"])
            x_quasi = np.asarray(result.get("quasi_x_grid") or result.get("x_grid"), dtype=float)
            quasi_mean = np.asarray(result.get("quasi_mean"), dtype=float)
            quasi_err = np.asarray(result.get("quasi_sdev", np.zeros_like(quasi_mean)), dtype=float)
            lightcone: EnsembleData = item["lightcone"]
            x_lc = np.asarray(lightcone.coords.get("x", result.get("x_grid")), dtype=float)
            lc_mean = np.asarray(lightcone.mean, dtype=float)
            lc_err = np.asarray(lightcone.sdev, dtype=float)
            for x, mean, err, linestyle, label in (
                (x_quasi, quasi_mean, quasi_err, "--", rf"$p={p_text}\,\mathrm{{GeV}}$ quasi"),
                (x_lc, lc_mean, lc_err, "-", rf"$p={p_text}\,\mathrm{{GeV}}$ light-cone"),
            ):
                ax.fill_between(x, mean - err, mean + err, color=color, alpha=0.22, linewidth=0)
                ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.0, label=label)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_xlabel(r"$x$", **FONT_SIZE)
        ax.set_ylabel(r"$f(x)$", **FONT_SIZE)
        if x_limits:
            ax.set_xlim(float(min(v[0] for v in x_limits)), float(max(v[1] for v in x_limits)))
        if y_limits:
            ax.set_ylim(float(min(v[0] for v in y_limits)), float(max(v[1] for v in y_limits)))
        ax.set_title(ensemble, **FONT_SIZE)
        ax.legend(**LEGEND_SETS)
        fig.tight_layout()
        pdf, svg = _overlay_paths(stage_dir, "mt", ensemble, "")
        fig.savefig(pdf, bbox_inches="tight", transparent=True)
        fig.savefig(svg, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        artifacts[f"matching_overlay_{pdf.stem}"] = str(pdf)
        artifacts[f"matching_overlay_image_{svg.stem}"] = str(svg)
    return artifacts


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
    if backend == "api" and provider and model_name:
        model_spec = format_api_model_spec(provider, model_name)
    elif backend == "codex":
        model_spec = model_name
    else:
        model_spec = None

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
            concise_issues = validate_stage_inputs(stage, manifest, job)
            if concise_issues:
                diagnostics = validate_stage_diagnostics(stage, manifest, job)
                issues = [
                    " ".join(
                        part
                        for part in (
                            diagnostic.detailed_message(),
                            f"Suggested fix: {diagnostic.suggested_fix}" if diagnostic.suggested_fix else "",
                        )
                        if part
                    )
                    for diagnostic in diagnostics
                ]
            else:
                issues = []
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
                elif is_constant_job_input(value):
                    store[role] = float(value)
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
            if "output" not in store:
                raise ValueError(f"job {job.id!r} finished without store['output']")
            outputs[job.id] = store["output"]
            if stage == "review" and "output" in store:
                stage_reports[stage] = {"report": str(store["output"])}
            if stage == "correlator_analysis" and "output" in store:
                fit_result = _last_tool_result(observations, "fit_bare_matrix_grid")
                if fit_result is not None:
                    matrix_attrs = dict(getattr(store.get("bare_matrix_element_data"), "attrs", {}))
                    stage_job_records.append(
                        {
                            "job_id": job.id,
                            "result": {**matrix_attrs, **fit_result},
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
                    matrix_attrs = dict(getattr(matrix, "attrs", {}))
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
                                **matrix_attrs,
                                **apply_result,
                                "scheme": apply_result.get("scheme"),
                                "strategy": apply_result.get("strategy"),
                                "z_grid": z_grid,
                                "diagnostic_plots": list((diag_result.get("plots") or {}).keys()),
                            },
                            "artifacts": artifacts,
                            "is_systematics": job.id.endswith(("_zs_low", "_zs_high")),
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
                                "scheme": fit_result.get("scheme"),
                                "strategy": fit_result.get("strategy"),
                                "job_kind": "fit",
                                "diagnostic_plots": list((diag_result.get("plots") or {}).keys()),
                            },
                            "artifacts": artifacts,
                            "is_systematics": job.id.endswith(("_zs_low", "_zs_high")),
                        }
                    )
            if stage == "fourier_transform" and "fourier_result" in store:
                fourier_attrs = dict(getattr(store.get("matrix_element_data") or store.get("input"), "attrs", {}))
                stage_job_records.append(
                    {
                        "job_id": job.id,
                        "result": {**fourier_attrs, **store["fourier_result"]},
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
                        "is_systematics": job.id.endswith(("_zs_low", "_zs_high", "_lambda_low", "_lambda_high")),
                    }
                )
            if stage == "perturbative_matching" and "lightcone_ed" in store and "quasi_ed" in store and "x_ls" in store:
                quasi_ed = store["quasi_ed"]
                lightcone_ed = store["lightcone_ed"]
                kernel_info = dict(store.get("matching_kernel_info", {}))
                matching_attrs = {**dict(getattr(lightcone_ed, "attrs", {})), **dict(getattr(quasi_ed, "attrs", {}))}
                x_ls = np.asarray(store["x_ls"], dtype=float)
                quasi_x = np.asarray(quasi_ed.coords.get("x", store.get("quasi_y_ls", x_ls)), dtype=float)
                stage_job_records.append(
                    {
                        "job_id": job.id,
                        "result": {
                            **matching_attrs,
                            **kernel_info,
                            "component": store.get("matching_component"),
                            "source": "job input",
                            "resample": lightcone_ed.resample,
                            "n_sample": int(lightcone_ed.n_sample),
                            "n_points": int(x_ls.size),
                            "x_grid": x_ls.tolist(),
                            "quasi_x_grid": quasi_x.tolist(),
                            "quasi_mean": [float(v) for v in np.asarray(quasi_ed.mean)],
                            "quasi_sdev": [float(v) for v in np.asarray(quasi_ed.sdev)],
                            "lightcone_mean": [float(v) for v in np.asarray(lightcone_ed.mean)],
                            "matching_plot_xlim": store.get("matching_plot", {}).get("xlim"),
                            "matching_plot_ylim": store.get("matching_plot", {}).get("ylim"),
                        },
                        "artifacts": {
                            "lightcone_artifact": store.get("matching_artifact"),
                            "matched_plot": store.get("matching_plot", {}).get("path"),
                            "matched_plot_image": store.get("matching_plot", {}).get("plot_image"),
                        },
                        "is_systematics": job.id.endswith(("_zs_low", "_zs_high", "_lambda_low", "_lambda_high", "_mu_low", "_mu_high")),
                    }
                )
            if stage == "extrapolation":
                result = _last_tool_result(observations, "run_extrapolation")
                if result is not None:
                    lightcones = list(store.get("lightcone", []))
                    zs_values = []
                    range_values = []
                    mu_values = []
                    for item in lightcones:
                        attrs = dict(getattr(item, "attrs", {}))
                        zs_value = attrs.get("zs_fm")
                        range_value = attrs.get("selected_range_label")
                        mu_value = attrs.get("mu")
                        if zs_value not in {None, ""}:
                            zs_values.append(str(zs_value))
                        if range_value not in {None, ""}:
                            text = str(range_value)
                            range_values.append(text[1:-1] if text.startswith('"') and text.endswith('"') else text)
                        if mu_value not in {None, ""}:
                            mu_values.append(str(mu_value))
                    stage_job_records.append(
                        {
                            "job_id": job.id,
                            "result": {
                                **result,
                                "zs_fm": list(dict.fromkeys(zs_values)),
                                "selected_range_label": list(dict.fromkeys(range_values)),
                                "mu": list(dict.fromkeys(mu_values)),
                            },
                            "artifacts": {
                                "extrapolated_artifact": result.get("artifact"),
                                "fit_info_artifact": result.get("fit_info_artifact"),
                                "extrapolated_plot": result.get("plot"),
                                "extrapolated_plot_image": result.get("plot_image"),
                                "chi2_xdep_plot": result.get("chi2_xdep_plot"),
                                "chi2_xdep_plot_image": result.get("chi2_xdep_plot_image"),
                                "adep_plot": result.get("adep_plot"),
                                "adep_plot_image": result.get("adep_plot_image"),
                                "pdep_plot": result.get("pdep_plot"),
                                "pdep_plot_image": result.get("pdep_plot_image"),
                            },
                            "is_systematics": (
                                job.id == "ex_other"
                                or job.id.endswith(("_zs_low", "_zs_high", "_lambda_low", "_lambda_high", "_mu_low", "_mu_high", "_a_sym", "_p_sym", "_ap_sym"))
                            ),
                        }
                    )
                result = _last_tool_result(observations, "run_systematics_budget")
                if result is not None:
                    stage_job_records.append(
                        {
                            "job_id": job.id,
                            "result": result,
                            "artifacts": {
                                "budget_artifact": result.get("artifact"),
                                "budget_plot": result.get("plot"),
                                "budget_plot_image": result.get("plot_image"),
                                "final_artifact": result.get("final_artifact"),
                                "final_plot": result.get("final_plot"),
                                "final_plot_image": result.get("final_plot_image"),
                            },
                            "is_systematics": True,
                        }
                    )
        if stage in state.pending_user_input:
            break
        if stage == "correlator_analysis" and stage_job_records:
            from lamet_agent.stages.correlator.reporting import write_correlator_stage_report
            from lamet_agent.stages.correlator.functions import (
                write_correlator_energy_artifacts,
                write_correlator_sample_quality_artifacts,
            )

            energy_artifacts = write_correlator_energy_artifacts(
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
            overlay_artifacts = _write_matrix_overlay_artifacts(
                stage_job_records,
                manifest.artifacts_directory / stage,
                artifact_key="bare_artifact",
                prefix="ca",
                title_suffix="bare matrix elements",
                y_label=r"Bare matrix element",
            )
            if overlay_artifacts:
                stage_job_records[0].setdefault("artifacts", {}).update(overlay_artifacts)
            quality_artifacts = write_correlator_sample_quality_artifacts(
                stage_job_records,
                manifest.artifacts_directory / stage,
            )
            if quality_artifacts:
                stage_job_records[0].setdefault("artifacts", {}).update(quality_artifacts)
            paths = write_correlator_stage_report(
                jobs=stage_job_records,
                path=manifest.artifacts_directory / stage / "ca_report.md",
                report_language=report_language,
                backend=backend,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
            )
            stage_reports[stage] = {"report": str(paths["report"])}
        if stage == "renormalization" and stage_job_records:
            from lamet_agent.stages.renorm.reporting import write_renorm_stage_report

            main_job_records = [record for record in stage_job_records if not record.get("is_systematics")]
            sym_job_records = [record for record in stage_job_records if record.get("is_systematics")]
            for record in main_job_records:
                record["systematics"] = [
                    item for item in sym_job_records if item["job_id"].startswith(f"{record['job_id']}_")
                ]
            overlay_artifacts = _write_matrix_overlay_artifacts(
                main_job_records,
                manifest.artifacts_directory / stage,
                artifact_key="renormalized_artifact",
                prefix="rn",
                title_suffix="renormalized matrix elements",
                y_label=r"Renormalized matrix element",
                x_label=r"$z$ [fm]",
            )
            if overlay_artifacts and main_job_records:
                main_job_records[0].setdefault("artifacts", {}).update(overlay_artifacts)
            paths = write_renorm_stage_report(
                jobs=main_job_records,
                systematics_jobs=sym_job_records,
                path=manifest.artifacts_directory / stage / "renorm_report.md",
                report_language=report_language,
                backend=backend,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
            )
            stage_reports[stage] = {"report": str(paths["report"])}
        if stage == "fourier_transform" and stage_job_records:
            from lamet_agent.stages.fourier.reporting import write_fourier_stage_report

            main_job_records = [record for record in stage_job_records if not record.get("is_systematics")]
            sym_job_records = [record for record in stage_job_records if record.get("is_systematics")]
            overlay_artifacts = _write_fourier_overlay_artifacts(main_job_records, manifest.artifacts_directory / stage)
            if overlay_artifacts and main_job_records:
                main_job_records[0].setdefault("artifacts", {}).update(overlay_artifacts)
            paths = write_fourier_stage_report(
                jobs=main_job_records,
                systematics_jobs=sym_job_records,
                path=manifest.artifacts_directory / stage / "ft_report.md",
                report_language=report_language,
                backend=backend,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
            )
            stage_reports[stage] = {"report": str(paths["report"])}
        if stage == "perturbative_matching" and stage_job_records:
            from lamet_agent.stages.matching.reporting import FormulaLlm, write_matching_stage_report

            main_job_records = [record for record in stage_job_records if not record.get("is_systematics")]
            sym_job_records = [record for record in stage_job_records if record.get("is_systematics")]
            overlay_artifacts = _write_matching_overlay_artifacts(main_job_records, manifest.artifacts_directory / stage)
            if overlay_artifacts and main_job_records:
                main_job_records[0].setdefault("artifacts", {}).update(overlay_artifacts)
            paths = write_matching_stage_report(
                jobs=main_job_records,
                systematics_jobs=sym_job_records,
                path=manifest.artifacts_directory / stage / "matching_report.md",
                report_language=report_language,
                llm=FormulaLlm(
                    backend=backend, provider=provider, api_key=api_key,
                    model_name=model_name, base_url=base_url,
                ),
            )
            stage_reports[stage] = {"report": str(paths["report"])}
        if stage == "extrapolation" and stage_job_records:
            from lamet_agent.stages.extrapolation.reporting import write_extrapolation_stage_report

            main_job_records = [record for record in stage_job_records if not record.get("is_systematics")]
            sym_job_records = [record for record in stage_job_records if record.get("is_systematics")]
            paths = write_extrapolation_stage_report(
                jobs=main_job_records,
                systematics_jobs=sym_job_records,
                path=manifest.artifacts_directory / stage / "extrapolation_report.md",
                report_language=report_language,
                backend=backend,
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
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
