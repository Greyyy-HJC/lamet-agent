"""Agent runtime loop for staged LaMET workflows."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .core.prompting import (
    build_stage_static_prompt,
    format_tool_observation,
)
from .core.stages import DEFAULT_STAGES, select_stage_sequence
from .core.tools import resolve_plot_save_path, resolve_stage_tools, validate_stage_inputs
from .core.trace import AgentTrace
from .manifest import AnalysisManifest, resolve_data_path

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["call_tool", "request_user_input", "finish"]},
        "reason": {"type": "string"},
        "tool_name": {"type": "string"},
        "args": {"type": "object"},
    },
    "required": ["action", "reason"],
    "additionalProperties": True,
}

_MOCK_TOOL_ACTION: dict[str, Any] = {
    "action": "call_tool",
    "tool_name": "mock_tool",
    "args": {"note": "Replace with real tool execution."},
    "reason": "Scaffold mode: deterministic mock action.",
}


@dataclass
class AgentState:
    """In-memory agent state for one run."""

    run_id: str
    completed_stages: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    stage_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    input_issues: dict[str, list[str]] = field(default_factory=dict)


class LlmSession(Protocol):
    """Per-stage LLM conversation handle."""

    def begin_stage(self, static_user: str) -> None: ...

    def decide(self, *, last_observation: dict[str, Any] | None) -> dict[str, Any]: ...


def _post_chat_completion(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    deepseek_model: str,
    base_url: str,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    system = (
        "You are the decision layer of a LaMET analysis agent. Decide the single "
        "next action only. Do NOT run shell commands or edit files. Reply with "
        "exactly one JSON object matching this shape: " + json.dumps(ACTION_SCHEMA)
    )
    body = {
        "model": deepseek_model,
        "messages": [{"role": "system", "content": system}, *messages],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))

    content = payload["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if match is None:
            raise ValueError(f"DeepSeek returned no JSON action:\n{content}")
        return json.loads(match.group(0))


def _request_llm_action(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    deepseek_model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> dict[str, Any]:
    """Return one structured agent action from the configured LLM backend."""
    if model == "mock":
        return dict(_MOCK_TOOL_ACTION)
    if model == "deepseek":
        if not api_key:
            raise ValueError("model='deepseek' requires an API key.")
        return _post_chat_completion(
            messages=messages,
            api_key=api_key,
            deepseek_model=deepseek_model,
            base_url=base_url,
        )
    raise NotImplementedError(
        f"LLM backend {model!r} is not implemented. Add provider logic in `_request_llm_action`."
    )


def _mock_session() -> LlmSession:
    """Return a session that emits one call_tool then finishes each stage."""
    state = {"emitted_tool": False}

    class _MockSession:
        def begin_stage(self, static_user: str) -> None:
            state["emitted_tool"] = False

        def decide(self, *, last_observation: dict[str, Any] | None) -> dict[str, Any]:
            if not state["emitted_tool"]:
                state["emitted_tool"] = True
                return _request_llm_action(model="mock", messages=[])
            state["emitted_tool"] = False
            return {"action": "finish", "reason": "Scaffold mode: mock stage complete."}

    return _MockSession()


def _external_session(actions_path: str | Path) -> LlmSession:
    """Return a session that replays a JSONL action transcript in order."""
    lines = Path(actions_path).read_text(encoding="utf-8").splitlines()
    queue = [json.loads(line) for line in lines if line.strip()]

    class _ExternalSession:
        def begin_stage(self, static_user: str) -> None:
            pass

        def decide(self, *, last_observation: dict[str, Any] | None) -> dict[str, Any]:
            if queue:
                return queue.pop(0)
            return {"action": "finish", "reason": "External transcript exhausted."}

    return _ExternalSession()


def _deepseek_session(
    api_key: str,
    deepseek_model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> LlmSession:
    """Return a multi-turn session backed by the DeepSeek chat-completions API."""

    class _DeepSeekSession:
        def __init__(self) -> None:
            self._messages: list[dict[str, str]] = []

        def begin_stage(self, static_user: str) -> None:
            self._messages = [{"role": "user", "content": static_user}]

        def decide(self, *, last_observation: dict[str, Any] | None) -> dict[str, Any]:
            if last_observation is not None:
                self._messages.append(
                    {
                        "role": "user",
                        "content": format_tool_observation(last_observation),
                    }
                )
            action = _request_llm_action(
                model="deepseek",
                messages=self._messages,
                api_key=api_key,
                deepseek_model=deepseek_model,
                base_url=base_url,
            )
            self._messages.append(
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
            )
            return action

    return _DeepSeekSession()


def _make_session(
    model: str,
    actions_path: str | Path | None,
    api_key: str | None = None,
    deepseek_model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> LlmSession:
    if model == "external":
        if actions_path is None:
            raise ValueError("model='external' requires an actions_path transcript.")
        return _external_session(actions_path)
    if model == "deepseek":
        if not api_key:
            raise ValueError("model='deepseek' requires an API key.")
        return _deepseek_session(api_key, deepseek_model, base_url)
    return _mock_session()


def _resolve_tool_args(args: dict[str, Any], manifest: AnalysisManifest) -> dict[str, Any]:
    """Resolve manifest-relative file paths in tool arguments."""
    if manifest.manifest_dir is None or manifest.project_root is None:
        return args
    resolved = dict(args)
    path_value = resolved.get("path")
    if isinstance(path_value, str) and not Path(path_value).is_absolute():
        resolved["path"] = resolve_data_path(
            manifest.project_root,
            manifest.manifest_dir,
            path_value,
        )
    return resolved


def _prepare_tool_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    manifest: AnalysisManifest,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Resolve paths and force plot output under ``artifacts_dir``."""
    resolved = _resolve_tool_args(args, manifest)
    if tool_name == "plot_fit_on_data":
        raw_save = resolved.get("save_path")
        if isinstance(raw_save, str) or raw_save is None:
            resolved["save_path"] = resolve_plot_save_path(
                raw_save if isinstance(raw_save, str) else None,
                artifacts_dir=artifacts_dir,
            )
        resolved["artifacts_dir"] = str(artifacts_dir)
    return resolved


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
        args = _prepare_tool_args(
            tool_name,
            action.get("args", {}) or {},
            manifest=manifest,
            artifacts_dir=artifacts_dir,
        )
        tool = tools.get(tool_name)
        if tool is None:
            observation = {"tool_name": tool_name, "error": "unknown tool"}
            observations.append(observation)
            trace.observation(observation)
            last_observation = observation
            continue
        try:
            result = tool(store, **args)
        except ValueError as exc:
            observation = {"tool_name": tool_name, "error": str(exc)}
            observations.append(observation)
            trace.observation(observation)
            last_observation = observation
            continue
        observation = {"tool_name": tool_name, "result": result}
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
    max_tool_steps: int = 30,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the stage loop and collect structured actions and tool results.

    When ``verbose`` is True, print each cycle's model action and tool
    observation to stdout in a ReAct-style trace before the final JSON summary.
    Static stage context is printed once per stage.
    """
    selected = _resolve_stages(manifest, stages, resume_from)

    state = AgentState(run_id=manifest.run_id)
    session = _make_session(model, actions_path, api_key, deepseek_model, base_url)
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
