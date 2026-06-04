"""Agent runtime loop for staged LaMET workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.llm import LlmSession, make_llm_session
from .core.prompting import build_stage_static_prompt
from .core.stages import DEFAULT_STAGES, select_stage_sequence
from .core.tools import (
    filter_tool_kwargs,
    prepare_tool_args,
    resolve_stage_tools,
    validate_stage_inputs,
)
from .core.trace import AgentTrace
from .manifest import AnalysisManifest


@dataclass
class AgentState:
    """In-memory agent state for one run."""

    run_id: str
    completed_stages: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    stage_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    input_issues: dict[str, list[str]] = field(default_factory=dict)


def _run_stage(
    stage: str,
    manifest: AnalysisManifest,
    state: AgentState,
    session: LlmSession,
    *,
    max_tool_steps: int,
    model: str,
    trace: AgentTrace,
) -> None:
    """Run one stage: drive the session and execute tool calls."""
    tools = resolve_stage_tools(stage)
    store: dict[str, Any] = {}
    observations: list[dict[str, Any]] = []
    artifacts_dir = Path.cwd() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    static_prompt = build_stage_static_prompt(
        stage,
        manifest,
        completed_stages=state.completed_stages.copy(),
    )
    session.begin_stage(static_prompt)
    trace.stage_context(static_prompt)

    last_observation: dict[str, Any] | None = None
    cycles = 0

    for _ in range(max_tool_steps):
        cycles += 1
        trace.cycle_begin(cycles)
        if cycles > 1 and last_observation is not None:
            trace.prompt_delta(last_observation)
        trace.llm_call_begin(model=model)
        action = session.decide(last_observation=last_observation)
        trace.llm_call_end()
        trace.model_output(action)
        state.actions.append({"stage": stage, "action": action})

        if action.get("action") != "call_tool":
            break

        tool_name = action.get("tool_name", "")
        args = prepare_tool_args(
            tool_name,
            action.get("args", {}) or {},
            manifest=manifest,
            artifacts_dir=artifacts_dir,
            _store=store,
        )
        tool = tools.get(tool_name)
        if tool is None:
            observation = {"tool_name": tool_name, "error": "unknown tool"}
            observations.append(observation)
            trace.observation(observation)
            last_observation = observation
            continue
        call_args, dropped_args = filter_tool_kwargs(tool, args)
        try:
            result = tool(store, **call_args)
        except (ValueError, TypeError) as exc:
            observation = {"tool_name": tool_name, "error": str(exc)}
            observations.append(observation)
            trace.observation(observation)
            last_observation = observation
            continue
        observation: dict[str, Any] = {"tool_name": tool_name, "result": result}
        if dropped_args:
            observation["ignored_args"] = dropped_args
        observations.append(observation)
        trace.observation(observation)
        last_observation = observation

    state.stage_results[stage] = observations
    trace.stage_end(stage, n_steps=cycles)


def _resolve_stages(
    manifest: AnalysisManifest,
    stages: list[str] | None,
    resume_from: str | None,
) -> list[str]:
    """Resolve which stages to run.

    ``stages`` (an explicit ordered subset) takes precedence; otherwise the
    default sequence is used, optionally sliced from ``resume_from`` onward.
    Running a later stage on its own requires the user to supply that stage's
    inputs in the manifest (surfaced per stage as ``input_issues``).
    """
    if stages is not None:
        unknown = [stage for stage in stages if stage not in DEFAULT_STAGES]
        if unknown:
            raise ValueError(f"Unknown stage(s): {unknown}. Known stages: {DEFAULT_STAGES}")
        return list(stages)

    sequence = select_stage_sequence(manifest.goal)
    if resume_from is not None:
        if resume_from in sequence:
            return sequence[sequence.index(resume_from):]
        return [resume_from]
    return sequence


def run_agent(
    manifest: AnalysisManifest,
    *,
    stages: list[str] | None = None,
    resume_from: str | None = None,
    model: str = "mock",
    actions_path: str | Path | None = None,
    api_key: str | None = None,
    deepseek_model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    max_tool_steps: int = 40,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the stage loop and collect structured actions and tool results.

    When ``verbose`` is True, print each cycle's model action and tool
    observation to stdout in a ReAct-style trace before the final JSON summary.
    Static stage context is printed once per stage.
    """
    selected = _resolve_stages(manifest, stages, resume_from)

    state = AgentState(run_id=manifest.run_id)
    session = make_llm_session(model, actions_path, api_key, deepseek_model, base_url)
    trace = AgentTrace(enabled=verbose)

    trace.run_begin(run_id=manifest.run_id, model=model, stages=selected)

    for stage in selected:
        issues = validate_stage_inputs(stage, manifest)
        if issues:
            state.input_issues[stage] = issues
        trace.stage_begin(stage, input_issues=issues or None)
        _run_stage(
            stage,
            manifest,
            state,
            session,
            max_tool_steps=max_tool_steps,
            model=model,
            trace=trace,
        )
        state.completed_stages.append(stage)

    trace.run_end(action_count=len(state.actions))

    return {
        "run_id": manifest.run_id,
        "status": "completed",
        "model": model,
        "stages": selected,
        "completed_stages": state.completed_stages,
        "input_issues": state.input_issues,
        "actions": state.actions,
        "stage_results": state.stage_results,
        "summary": json.dumps(
            {
                "run_id": manifest.run_id,
                "stage_count": len(state.completed_stages),
                "action_count": len(state.actions),
            }
        ),
    }
