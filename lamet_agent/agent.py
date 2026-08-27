"""Ordered stage/job harness, lazy tool discovery, and prompt composition."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal, Mapping, get_args, get_origin, get_type_hints

import numpy as np
from tqdm import tqdm

from .banner import BANNER
from .llm import LlmBackend, Message
from .manifest import Job, Manifest, _load_stage_contract
from .parallel._pool import _ParallelPool
from .stages._reporting import StageReportRecord
from .structured import annotation_schema, json_compatible, validate_value
from .contract import (
    CheckContext,
    _apply_recommended_defaults,
    _unresolved_null_hooks,
    evaluate_checks,
    evaluate_rules,
)


_MAX_ASSISTANT_TURNS = 40
_LLM_TRANSCRIPT_FILENAME = "llm_transcript.md"
_SAFE_TOOL = re.compile(r"^[a-z][a-z0-9_]*$")
_WORKFLOW_STAGES = frozenset(
    {"correlator_analysis", "renormalization", "fourier_transform", "perturbative_matching", "extrapolation"}
)


def _resolve_progress_mode(mode: str, *, has_systematics: bool) -> str:
    """Resolve automatic progress display from the authored manifest shape."""
    if mode not in {"auto", "stage", "job", "none"}:
        raise ValueError("progress mode must be auto, stage, job, or none")
    if mode == "auto":
        return "stage" if has_systematics else "job"
    return mode


def _emit_progress(message: str = "") -> None:
    """Write one immediately visible runtime progress line to stdout."""
    print(message, flush=True)


def _message_payload(message: Message) -> dict[str, Any]:
    """Return the complete backend-neutral representation of one message."""
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.calls:
        payload["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": dict(call.arguments)} for call in message.calls
        ]
    return payload


def _write_transcript_header(
    path: Path,
) -> None:
    """Create one append-only document containing only LLM exchanges."""
    path.write_text("# LLM communications\n", encoding="utf-8")


def _append_transcript(path: Path, title: str, payload: Any) -> None:
    """Append one exact JSON payload as a Markdown transcript section."""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"\n## {title}\n\n```json\n" + json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n```\n"
        )


@dataclass
class LlmSession:
    """One generic recorded LLM channel bound to a job transcript."""

    backend: LlmBackend
    transcript_path: Path
    history: list[Message] = field(default_factory=list)
    calls: int = 0
    max_recommendation_calls: int = 2
    recommendation_calls: int = 0
    _context_keys: set[str] = field(default_factory=set, repr=False)
    _pending_context: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def has_context(self, key: str) -> bool:
        """Return whether one named context is pending or already in message history."""
        return key in self._context_keys

    def add_context(self, key: str, content: Mapping[str, Any]) -> None:
        """Queue one named context for inclusion in the next user message."""
        if not isinstance(key, str) or not key:
            raise ValueError("LLM context key must be a nonempty string")
        if key in self._context_keys:
            return
        self._context_keys.add(key)
        self._pending_context.append({"key": key, "content": json_compatible(dict(content))})

    def complete(
        self,
        *,
        label: str,
        messages: list[Message] | None = None,
        user_message: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: Mapping[str, Any] | None = None,
        prompt_digest: str | None = None,
    ) -> Any:
        """Record and execute one backend call without imposing response semantics."""
        if (messages is None) == (user_message is None):
            raise ValueError("complete requires exactly one of messages or user_message")
        if messages is not None and self._pending_context:
            raise RuntimeError("pending LLM context requires a user_message completion")
        if response_schema is not None:
            if self.recommendation_calls >= self.max_recommendation_calls:
                raise RuntimeError(
                    f"parameter recommendation limit exceeded: used {self.recommendation_calls}, "
                    f"allowed {self.max_recommendation_calls}"
                )
            self.recommendation_calls += 1
        retain_history = user_message is not None
        combined_user_message = user_message
        if user_message is not None and self._pending_context:
            try:
                request_content = json.loads(user_message)
            except json.JSONDecodeError:
                request_content = user_message
            combined_user_message = json.dumps(
                {"context": self._pending_context, "request": request_content},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        request_messages = (
            [*self.history, Message("user", combined_user_message)]
            if combined_user_message is not None
            else list(messages or [])
        )
        tool_schemas = [] if tools is None else tools
        self.calls += 1
        request_payload = {
            "messages": [_message_payload(message) for message in request_messages],
            "tools": tool_schemas,
            "response_schema": response_schema,
        }
        _append_transcript(
            self.transcript_path,
            f"{label}, request {self.calls}: sent to LLM",
            request_payload,
        )
        if prompt_digest is None:
            digest_payload = json.dumps(
                {"request": request_payload, "backend": self.backend.identity},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            prompt_digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
        _emit_progress(f"Calling LLM ({self.backend.identity}) for {label}...")
        response = self.backend.complete(
            messages=request_messages,
            tools=tool_schemas,
            prompt_digest=prompt_digest,
            response_schema=response_schema,
        )
        _emit_progress("LLM response received.")
        assistant_message = Message("assistant", response.text, tool_calls=response.calls)
        _append_transcript(
            self.transcript_path,
            f"{label}, request {self.calls}: received from LLM",
            _message_payload(assistant_message),
        )
        if retain_history:
            self.history.extend((request_messages[-1], assistant_message))
            self._pending_context.clear()
        return response


@dataclass
class ToolContext:
    """Mutable state and terminal fields for one isolated job."""

    manifest: Mapping[str, Any]
    manifest_path: Path
    stage_id: str
    job_id: str
    params: dict[str, Any]
    inputs: Mapping[str, Any]
    input_summaries: Mapping[str, Any]
    state: dict[str, Any]
    artifact_directory: Path
    rng: np.random.Generator
    output: Any | None = None
    summary: dict[str, Any] | None = None
    _param_rules: tuple[Any, ...] = field(default=(), repr=False)
    _checks: tuple[Callable[..., Any], ...] = field(default=(), repr=False)
    _parallel: _ParallelPool | None = field(default=None, repr=False)

    @property
    def workers(self) -> int:
        """Return the validated run-level sample-fit parallelism."""
        return int(self.manifest["metadata"]["workers"])

    def finish(self, output: Any, summary: dict[str, Any]) -> None:
        """Set one terminal result after validating the shared summary envelope."""
        if self.output is not None or self.summary is not None:
            raise RuntimeError("context.finish() may only be called once")
        if not isinstance(summary, dict):
            raise TypeError("finish summary must be an object")
        required = {"stage_id", "job_id", "result", "decisions", "diagnostics", "artifacts"}
        if set(summary) != required:
            raise ValueError(f"finish summary keys must be exactly {sorted(required)}")
        if summary["stage_id"] != self.stage_id or summary["job_id"] != self.job_id:
            raise ValueError("finish summary identifies a different stage or job")
        if (
            not isinstance(summary["result"], str)
            or not isinstance(summary["decisions"], dict)
            or not isinstance(summary["diagnostics"], dict)
            or not isinstance(summary["artifacts"], list)
        ):
            raise TypeError("finish summary has invalid envelope types")
        for artifact in summary["artifacts"]:
            if not isinstance(artifact, str) or Path(artifact).is_absolute() or ".." in Path(artifact).parts:
                raise ValueError("finish artifact paths must be relative to the job cell")
            if not (self.artifact_directory / artifact).is_file():
                raise FileNotFoundError(f"declared artifact does not exist: {artifact}")
        self.output = output
        self.summary = dict(summary)

    def _resolve_null_hook(self, path: str, value: Any) -> None:
        """Atomically set and validate one runtime-resolved dependency."""
        unresolved = {
            rule.path: rule
            for rule in _unresolved_null_hooks(
                self.params,
                self._param_rules,
                root_document=self.manifest,
            )
        }
        if path not in unresolved:
            raise ValueError(f"{path!r} does not have an unresolved null hook")
        parts = path.split(".")
        previous_params = copy.deepcopy(self.params)
        parent: dict[str, Any] = self.params
        for segment in parts[:-1]:
            child = parent.get(segment)
            if not isinstance(child, dict):
                raise ValueError(f"null-hook parent {'.'.join(parts[:-1])!r} is not an object")
            parent = child
        child_name = parts[-1]
        parent[child_name] = copy.deepcopy(value)
        applied_defaults = _apply_recommended_defaults(
            self.params,
            self._param_rules,
            root_document=self.manifest,
        )
        rule_issues = evaluate_rules(
            self.params,
            self._param_rules,
            complete=True,
            root_document=self.manifest,
        )
        remaining = frozenset(
            rule.path
            for rule in _unresolved_null_hooks(
                self.params,
                self._param_rules,
                root_document=self.manifest,
            )
        )
        check_issues = []
        if not rule_issues:
            check_issues = evaluate_checks(
                self._checks,
                CheckContext(
                    self.manifest,
                    self.stage_id,
                    self.job_id,
                    self.params,
                    self.inputs,
                    remaining,
                ),
            )
        issues = [*rule_issues, *check_issues]
        if issues:
            self.params.clear()
            self.params.update(previous_params)
            detail = "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
            raise ValueError(f"invalid null-hook value for {path}: {detail}")
        if applied_defaults:
            defaults = self.state.setdefault("recommended_defaults", {})
            defaults.update(applied_defaults)
        resolved = self.state.setdefault("resolved_null_hooks", {})
        resolved[path] = copy.deepcopy(value)


@dataclass(frozen=True)
class _Tool:
    name: str
    run: Any
    prompt: str
    schema: dict[str, Any]


def _stage_path(stage_id: str, stage_root: str | Path | None) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", stage_id):
        raise ValueError(f"Invalid stage id '{stage_id}'")
    return (
        Path(stage_root).expanduser().resolve() if stage_root is not None else Path(__file__).parent / "stages"
    ) / stage_id


def _load_tool_module(stage_id: str, tool_directory: Path) -> ModuleType:
    digest = hashlib.sha256(str(tool_directory.resolve()).encode("utf-8")).hexdigest()
    module_name = f"_lamet_agent_neo_tool_{stage_id}_{tool_directory.name}_{digest}"
    init_path = tool_directory / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name, init_path, submodule_search_locations=[str(tool_directory)]
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load tool '{tool_directory.name}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool_schema(name: str, function: Any) -> dict[str, Any]:
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    parameters: dict[str, Any] = {}
    required: list[str] = []
    entries = list(signature.parameters.values())
    if (
        not entries
        or entries[0].name != "context"
        or entries[0].kind not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ):
        raise TypeError("tool run() must start with a context parameter")
    for parameter in entries:
        if parameter.name == "context":
            continue
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError("model-visible tool arguments must be keyword-only")
        if parameter.name.startswith("_"):
            raise TypeError("model-visible tool arguments cannot be private")
        annotation = hints.get(parameter.name, parameter.annotation)
        schema, nullable = annotation_schema(annotation)
        parameters[parameter.name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Call stage tool {name}; follow its system-prompt tool guidance.",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _discover_tools(stage_id: str, *, stage_root: str | Path | None = None) -> list[_Tool]:
    """Discover immediate public tool directories for one entered stage."""
    tools_directory = _stage_path(stage_id, stage_root) / "tools"
    if not tools_directory.is_dir():
        raise ValueError(f"Stage '{stage_id}' has no tools directory")
    discovered: list[_Tool] = []
    for directory in sorted(tools_directory.iterdir(), key=lambda path: path.name):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        if not _SAFE_TOOL.fullmatch(directory.name):
            raise ValueError(f"Tool directory '{directory.name}' is not a safe public tool name")
        init_path = directory / "__init__.py"
        prompt_path = directory / "prompt.md"
        if not init_path.exists() and not prompt_path.exists():
            continue
        if not init_path.is_file():
            # Prompt-only directories describe typed parameter suggestions used
            # by stage workflows; they are not model-visible executable tools.
            if not prompt_path.is_file() or not prompt_path.read_text(encoding="utf-8").strip():
                raise ValueError(f"Parameter suggestion '{directory.name}' requires a nonempty prompt.md")
            continue
        if not prompt_path.is_file() or not prompt_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Tool '{directory.name}' requires a nonempty prompt.md")
        module = _load_tool_module(stage_id, directory)
        function = getattr(module, "run", None)
        if not callable(function):
            raise TypeError(f"Tool '{directory.name}' must export callable run")
        discovered.append(
            _Tool(
                directory.name,
                function,
                prompt_path.read_text(encoding="utf-8").strip(),
                _tool_schema(directory.name, function),
            )
        )
    return discovered


def _load_stage_reporter(stage_id: str, stage_root: str | Path | None) -> Callable[..., Path] | None:
    """Load an optional stage-owned deterministic reporting hook."""
    path = _stage_path(stage_id, stage_root) / "reporting.py"
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"Stage '{stage_id}' reporting.py is not a file")
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location(f"_lamet_agent_neo_reporting_{stage_id}_{digest}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load reporting hook for stage '{stage_id}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    writer = getattr(module, "write_stage_report", None)
    if not callable(writer):
        raise TypeError(f"Stage '{stage_id}' reporting.py must export write_stage_report")
    return writer


def _write_stage_report(
    stage_id: str,
    records: list[StageReportRecord],
    *,
    stage_root: str | Path | None,
) -> Path | None:
    """Invoke one stage reporter after every job in that stage has finished."""
    if not records:
        raise ValueError(f"Stage '{stage_id}' has no completed jobs to report")
    writer = _load_stage_reporter(stage_id, stage_root)
    if writer is None:
        return None
    artifact_directory = records[0].artifact_directory.parent
    if any(record.artifact_directory.parent != artifact_directory for record in records):
        raise ValueError(f"Stage '{stage_id}' jobs do not share one artifact directory")
    result = writer(records=tuple(records), artifact_directory=artifact_directory)
    if not isinstance(result, Path):
        raise TypeError(f"Stage '{stage_id}' reporter must return pathlib.Path")
    resolved = result.resolve()
    if resolved != (artifact_directory / "report.md").resolve() or not resolved.is_file():
        raise ValueError(f"Stage '{stage_id}' reporter must create its canonical report.md")
    return resolved


def _summarize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _summarize(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _summarize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_summarize(item) for item in value]
    if isinstance(value, tuple):
        return [_summarize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if value.__class__.__name__ == "EnsembleData":
        return {
            "type": "EnsembleData",
            "dims": list(value.dims),
            "n_sample": int(value.n_sample),
            "name": value.name,
            "attrs": _summarize(value.attrs),
        }
    return {"type": type(value).__name__}


def _read_stage_prompt(stage_id: str, stage_root: str | Path | None) -> str:
    prompt_path = _stage_path(stage_id, stage_root) / "prompt.md"
    if not prompt_path.is_file() or not prompt_path.read_text(encoding="utf-8").strip():
        raise ValueError(f"Stage '{stage_id}' requires a nonempty prompt.md")
    return prompt_path.read_text(encoding="utf-8").strip()


def _build_static_prompt(
    *,
    job: Job,
    tools: list[_Tool],
    stage_root: str | Path | None,
    backend_identity: str,
) -> tuple[str, str]:
    """Compose the immutable job prefix and its stable digest."""
    policy_path = Path(__file__).with_name("agent_prompt.md")
    policy = policy_path.read_text(encoding="utf-8").strip()
    stage_prompt = _read_stage_prompt(job.stage_id, stage_root)
    contract = _load_stage_contract(job.stage_id, stage_root)
    guidance_lines = []
    for rule in [*contract.PARAM_RULES, *contract.INPUT_RULES]:
        if hasattr(rule, "physics"):
            path = getattr(rule, "path", "")
            expected = getattr(rule, "expected", None)
            choices = get_args(expected) if get_origin(expected) is Literal else ()
            choice_text = f" Choices: {', '.join(repr(choice) for choice in choices)}." if choices else ""
            guidance_lines.append(f"- `{path}`: {rule.physics}{choice_text}")
    guidance = "\n".join(guidance_lines)
    tool_text = "\n\n".join(f"## Tool: {tool.name}\n{tool.prompt}" for tool in tools)
    static = "\n\n".join(
        [
            policy,
            f"# Stage: {job.stage_id}\n{stage_prompt}",
            f"# Physical contract guidance\n{guidance or 'Use the validated effective parameters supplied below.'}",
            f"# Available tools\n{tool_text}",
        ]
    )
    digest_payload = "\0".join(
        [
            static,
            json.dumps([tool.schema for tool in tools], sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            backend_identity,
        ]
    )
    return static, hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()


def _observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("stage tools must return an observation object")
    if "summary" not in value or not isinstance(value["summary"], str):
        raise ValueError("stage tool observation requires a string summary")
    return json_compatible(dict(value))


def _resolve_runtime_null_hooks(
    *,
    context: ToolContext,
    contract: ModuleType,
    session: LlmSession,
) -> None:
    """Apply static defaults, then run unresolved dependency hooks in order."""
    context._param_rules = tuple(contract.PARAM_RULES)
    context._checks = tuple(contract.CHECKS)
    applied_defaults = _apply_recommended_defaults(
        context.params,
        context._param_rules,
        root_document=context.manifest,
    )
    if applied_defaults:
        context.state["recommended_defaults"] = applied_defaults
    while True:
        unresolved = _unresolved_null_hooks(
            context.params,
            context._param_rules,
            root_document=context.manifest,
        )
        if not unresolved:
            return
        rule = unresolved[0]
        hook = rule.null_hook
        if hook is None:
            raise RuntimeError(f"unresolved dependency {rule.path} has no null hook")
        previous_calls = session.calls
        value = hook(context, session)
        context._resolve_null_hook(rule.path, value)
        provenance = context.state.setdefault("null_hook_provenance", {})
        provenance[rule.path] = {
            "backend": session.backend.identity,
            "hook": hook.__name__,
            "llm_requests": session.calls - previous_calls,
            "value": copy.deepcopy(value),
        }


def _invoke(tool: _Tool, context: ToolContext, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be an object")
    signature = inspect.signature(tool.run)
    hints = get_type_hints(tool.run)
    visible_parameters = [parameter for parameter in signature.parameters.values() if parameter.name != "context"]
    visible = {parameter.name for parameter in visible_parameters}
    unknown = set(arguments) - visible
    if unknown and visible:
        raise ValueError(f"unknown arguments for tool '{tool.name}': {sorted(unknown)}")
    ignored = sorted(unknown)
    arguments = {key: value for key, value in arguments.items() if key in visible}
    missing = [
        parameter.name
        for parameter in visible_parameters
        if parameter.default is inspect.Parameter.empty and parameter.name not in arguments
    ]
    if missing:
        raise ValueError(f"missing arguments for tool '{tool.name}': {missing}")
    for parameter in visible_parameters:
        if parameter.name in arguments:
            validate_value(hints.get(parameter.name, parameter.annotation), arguments[parameter.name], parameter.name)
    observation = _observation(tool.run(context, **dict(arguments)))
    if ignored:
        observation["ignored_arguments"] = ignored
    return observation


@dataclass
class _AgentSession:
    """One ordered run with fresh state and conversation per job."""

    backend: LlmBackend
    stage_root: str | Path | None = None
    max_tool_steps: int = _MAX_ASSISTANT_TURNS
    progress_mode: Literal["auto", "stage", "job", "none"] = "auto"
    _outputs: dict[str, Any] = field(default_factory=dict, init=False)
    _summaries: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _stage_bundles: dict[str, tuple[list[_Tool], str, str]] = field(default_factory=dict, init=False)

    @staticmethod
    def _finish_context(context: ToolContext, *, llm_turns: int) -> tuple[Any, dict[str, Any]]:
        if context.summary is None or context.output is None:
            raise RuntimeError(f"job '{context.job_id}' deterministic workflow did not finish")
        remaining = _unresolved_null_hooks(
            context.params,
            context._param_rules,
            root_document=context.manifest,
        )
        if remaining:
            paths = [rule.path for rule in remaining]
            raise RuntimeError(f"job '{context.job_id}' finished with unresolved null hooks: {paths}")
        (context.artifact_directory / "summary.json").write_text(
            json.dumps(context.summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        _emit_progress(f"Job {context.stage_id}/{context.job_id} finished with {llm_turns} LLM turn(s).")
        return context.output, context.summary

    def _run_context(
        self,
        context: ToolContext,
        tools: list[_Tool],
        static_prompt: str,
        digest: str,
    ) -> tuple[Any, dict[str, Any]]:
        transcript_path = context.artifact_directory / _LLM_TRANSCRIPT_FILENAME
        _write_transcript_header(transcript_path)
        history = [Message("system", static_prompt)] if static_prompt else []
        retry_limit = int(context.manifest["metadata"].get("parameter_recommendation_retries", 1))
        llm_session = LlmSession(
            self.backend,
            transcript_path,
            history=history,
            max_recommendation_calls=1 + retry_limit,
        )
        contract = _load_stage_contract(context.stage_id, self.stage_root)
        _resolve_runtime_null_hooks(
            context=context,
            contract=contract,
            session=llm_session,
        )
        if context.stage_id in _WORKFLOW_STAGES:
            workflow = importlib.import_module(f"lamet_agent.stages.{context.stage_id}.workflow")
            workflow.run(context, llm_session)
            return self._finish_context(context, llm_turns=llm_session.calls)

        dynamic_job = {
            "stage_id": context.stage_id,
            "job_id": context.job_id,
            "params": _summarize(context.params),
            "inputs": _summarize(context.inputs),
            "input_summaries": _summarize(context.input_summaries),
            "artifact_directory": str(context.artifact_directory),
        }
        messages = [
            Message("system", static_prompt),
            Message(
                "user",
                "Begin by inspecting the supplied inputs and follow the stage decision policy.\n\n"
                + json.dumps(dynamic_job, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            ),
        ]
        tool_schemas = [tool.schema for tool in tools]
        tool_map = {tool.name: tool for tool in tools}
        tool_steps = 0
        try:
            for turn in range(1, self.max_tool_steps + 1):
                response = llm_session.complete(
                    label=f"{context.stage_id}/{context.job_id} turn {turn}",
                    messages=messages,
                    tools=tool_schemas,
                    prompt_digest=digest,
                )
                calls = response.calls
                assistant_message = Message("assistant", response.text, tool_calls=calls)
                messages.append(assistant_message)
                if not calls:
                    raise RuntimeError(f"job '{context.job_id}' returned no tool call")
                unavailable = [call.name for call in calls if call.name not in tool_map]
                if unavailable:
                    raise ValueError(f"model requested unavailable tool '{unavailable[0]}'")
                for call in calls:
                    tool_steps += 1
                    if tool_steps > self.max_tool_steps:
                        raise RuntimeError(f"job '{context.job_id}' exceeded {self.max_tool_steps} tool steps")
                    _emit_progress(f"Running tool: {call.name}...")
                    try:
                        observation = _invoke(tool_map[call.name], context, call.arguments)
                    except Exception:
                        _emit_progress(f"Tool failed: {call.name}.")
                        raise
                    _emit_progress(f"Tool completed: {call.name}.")
                    tool_message = Message(
                        "tool",
                        json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                        tool_call_id=call.id,
                    )
                    messages.append(tool_message)
                    if context.summary is not None:
                        remaining = _unresolved_null_hooks(
                            context.params,
                            context._param_rules,
                            root_document=context.manifest,
                        )
                        if remaining:
                            paths = [rule.path for rule in remaining]
                            raise RuntimeError(f"job '{context.job_id}' finished with unresolved null hooks: {paths}")
                        (context.artifact_directory / "summary.json").write_text(
                            json.dumps(context.summary, indent=2, sort_keys=True), encoding="utf-8"
                        )
                        _emit_progress(f"Job {context.stage_id}/{context.job_id} finished after {turn} turn(s).")
                        return context.output, context.summary
            raise RuntimeError(
                f"job '{context.job_id}' did not call a terminal tool within {self.max_tool_steps} turns"
            )
        except Exception:
            raise

    def run_manifest(self, manifest: Manifest) -> dict[str, Any]:
        """Validate and run one already-loaded manifest in authored order."""
        if not isinstance(manifest, Manifest):
            raise TypeError("run_manifest requires a loaded Manifest")
        self._outputs.clear()
        self._summaries.clear()
        self._stage_bundles.clear()
        issues = manifest.validate(stage_root=self.stage_root)
        if issues:
            raise ValueError("\n".join(f"{issue.path}: {issue.message}" for issue in issues))
        document = manifest.document
        jobs = list(manifest.jobs)
        jobs_by_stage = manifest.jobs_by_stage
        progress_mode = _resolve_progress_mode(self.progress_mode, has_systematics=manifest.has_systematics)
        collisions = [job.artifact_directory for job in jobs if job.artifact_directory.exists()]
        if collisions:
            raise FileExistsError(f"selected job artifact directory already exists: {collisions[0]}")
        artifact_base = jobs[0].artifact_directory.parent
        artifact_base.mkdir(parents=True, exist_ok=True)
        (artifact_base / "resolved_manifest.json").write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest_file = manifest.path
        metadata = document["metadata"]
        stage_ids = [str(stage_id) for stage_id in document["stages"]]
        _emit_progress(BANNER)
        _emit_progress("")
        _emit_progress(f"Run: {metadata['run_id']}  backend={self.backend.identity}")
        _emit_progress(f"Stages: {', '.join(stage_ids)}")
        _emit_progress("")
        stage_reports: dict[str, str] = {}
        stage_records: list[StageReportRecord] = []
        stage_progress = None
        parallel = _ParallelPool(int(metadata["workers"]))
        try:
            for job in jobs:
                stage_id = job.stage_id
                stage_jobs = jobs_by_stage.get(stage_id)
                if not stage_jobs:
                    raise ValueError(f"Stage '{stage_id}' has no indexed jobs")
                if job is stage_jobs[0]:
                    if stage_records:
                        raise RuntimeError("stage report records leaked across a stage boundary")
                    _emit_progress("")
                    _emit_progress(f"Stage: {stage_id}")
                    if progress_mode == "stage":
                        stage_progress = tqdm(total=len(stage_jobs), desc=stage_id, unit="job")
                _emit_progress(f"Job: {stage_id}/{job.job_id}")
                job.artifact_directory.mkdir(parents=True, exist_ok=False)
                resolved_inputs: dict[str, Any] = {}
                input_summaries: dict[str, Any] = {}
                for role, source in job.inputs.items():
                    resolved_inputs[role], input_summaries[role] = manifest._resolve_source(
                        source,
                        outputs=self._outputs,
                        summaries=self._summaries,
                    )
                seed_sequence = np.random.SeedSequence(
                    [int(metadata["random_seed"]), job.stage_index - 1, job.job_index]
                )
                context = ToolContext(
                    document,
                    manifest_file,
                    stage_id,
                    job.job_id,
                    copy.deepcopy(dict(job.params)),
                    resolved_inputs,
                    input_summaries,
                    {"show_job_progress": progress_mode == "job"},
                    job.artifact_directory,
                    np.random.default_rng(seed_sequence),
                    _parallel=parallel,
                )
                bundle = self._stage_bundles.get(stage_id)
                if bundle is None:
                    tools = _discover_tools(stage_id, stage_root=self.stage_root)
                    static_prompt, digest = _build_static_prompt(
                        job=job,
                        tools=tools,
                        stage_root=self.stage_root,
                        backend_identity=self.backend.identity,
                    )
                    self._stage_bundles[stage_id] = (tools, static_prompt, digest)
                else:
                    tools, static_prompt, digest = bundle
                output, summary = self._run_context(context, tools, static_prompt, digest)
                self._outputs[job.job_id] = output
                self._summaries[job.job_id] = summary
                stage_records.append(
                    StageReportRecord(
                        job_id=job.job_id,
                        params=copy.deepcopy(context.params),
                        inputs=dict(context.inputs),
                        output=output,
                        summary=copy.deepcopy(summary),
                        artifact_directory=job.artifact_directory,
                    )
                )
                if stage_progress is not None:
                    stage_progress.update(1)
                if job is stage_jobs[-1]:
                    _emit_progress(f"Writing stage report: {stage_id}...")
                    report = _write_stage_report(stage_id, stage_records, stage_root=self.stage_root)
                    if report is not None:
                        stage_reports[stage_id] = str(report)
                    _emit_progress(f"Stage {stage_id} finished.")
                    stage_records = []
                    if stage_progress is not None:
                        stage_progress.close()
                        stage_progress = None
            if stage_records:
                raise RuntimeError("final stage did not reach its indexed report boundary")
        finally:
            if stage_progress is not None:
                stage_progress.close()
            parallel.close()
        _emit_progress("=" * 60)
        _emit_progress(f"Agent run complete ({len(jobs)} job(s)).")
        _emit_progress("=" * 60)
        return {"outputs": dict(self._outputs), "summaries": dict(self._summaries), "stage_reports": stage_reports}


def create_session(
    backend: LlmBackend,
    *,
    progress_mode: Literal["auto", "stage", "job", "none"] = "auto",
) -> _AgentSession:
    """Create an isolated workflow session for one resolved LLM backend."""
    return _AgentSession(backend, progress_mode=progress_mode)


__all__ = [
    "ToolContext",
    "create_session",
]
