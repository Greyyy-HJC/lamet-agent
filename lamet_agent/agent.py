"""Ordered stage/job harness, lazy tool discovery, and prompt composition."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal, Mapping, Union, get_args, get_origin, get_type_hints, is_typeddict

import numpy as np

from .llm import LlmBackend, Message
from .manifest import Job, Manifest, _load_stage_contract
from .parallel._pool import _ParallelPool
from .stages._reporting import StageReportRecord
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


def _annotation_schema(annotation: Any) -> tuple[dict[str, Any], bool]:
    """Return a JSON schema and whether the annotation is nullable."""
    if annotation is Any or annotation is inspect.Parameter.empty:
        raise TypeError("tool arguments need supported annotations")
    if is_typeddict(annotation):
        hints = get_type_hints(annotation)
        properties = {}
        for name, child_annotation in hints.items():
            child_schema, _nullable = _annotation_schema(child_annotation)
            properties[name] = child_schema
        required_keys = sorted(getattr(annotation, "__required_keys__", hints))
        return {
            "type": "object",
            "properties": properties,
            "required": required_keys,
            "additionalProperties": False,
        }, False
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, getattr(__import__("types"), "UnionType", object)):
        if len(args) != 2 or type(None) not in args:
            raise TypeError("only T | None tool annotations are supported")
        other = args[0] if args[1] is type(None) else args[1]
        schema, _ = _annotation_schema(other)
        return {"anyOf": [schema, {"type": "null"}]}, True
    if origin is Literal:
        values = list(args)
        if not values:
            raise TypeError("Literal tool annotations cannot be empty")
        type_name = (
            "string"
            if all(isinstance(value, str) for value in values)
            else "integer"
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            else "number"
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
            else "boolean"
            if all(isinstance(value, bool) for value in values)
            else None
        )
        if type_name is None:
            raise TypeError("Literal values must share a JSON scalar type")
        return {"type": type_name, "enum": values}, False
    if annotation is str:
        return {"type": "string"}, False
    if annotation is bool:
        return {"type": "boolean"}, False
    if annotation is int:
        return {"type": "integer"}, False
    if annotation is float:
        return {"type": "number"}, False
    if annotation in (list, dict):
        return (
            ({"type": "array", "items": {}}, False)
            if annotation is list
            else ({"type": "object", "additionalProperties": True}, False)
        )
    if origin is list:
        item = args[0] if args else Any
        if item is Any:
            return {"type": "array", "items": {}}, False
        item_schema, _ = _annotation_schema(item)
        return {"type": "array", "items": item_schema}, False
    if origin is dict:
        if len(args) != 2 or args[0] is not str:
            raise TypeError("tool mappings must be dict[str, T]")
        if args[1] is Any:
            return {"type": "object", "additionalProperties": True}, False
        value_schema, _ = _annotation_schema(args[1])
        return {"type": "object", "additionalProperties": value_schema}, False
    raise TypeError(f"unsupported tool annotation {annotation!r}")


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
        schema, nullable = _annotation_schema(annotation)
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


def _validate_argument(annotation: Any, value: Any, path: str) -> None:
    """Validate one already-decoded JSON argument against a tool annotation."""
    if is_typeddict(annotation):
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise TypeError(f"tool argument '{path}' must be an object")
        hints = get_type_hints(annotation)
        required_keys = set(getattr(annotation, "__required_keys__", hints))
        unknown = set(value) - set(hints)
        missing = required_keys - set(value)
        if unknown:
            raise TypeError(f"tool argument '{path}' has unknown keys {sorted(unknown)}")
        if missing:
            raise TypeError(f"tool argument '{path}' is missing keys {sorted(missing)}")
        for key, child in value.items():
            _validate_argument(hints[key], child, f"{path}.{key}")
        return
    origin = get_origin(annotation)
    args = get_args(annotation)
    union_type = getattr(__import__("types"), "UnionType", object)
    if origin in (Union, union_type):
        if type(None) in args and value is None:
            return
        non_null = [candidate for candidate in args if candidate is not type(None)]
        if len(non_null) == 1:
            _validate_argument(non_null[0], value, path)
            return
        raise TypeError(f"unsupported union for tool argument '{path}'")
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in args):
            raise TypeError(f"tool argument '{path}' is not one of the declared literal values")
        return
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"tool argument '{path}' must be a string")
        return
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"tool argument '{path}' must be a boolean")
        return
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"tool argument '{path}' must be an integer")
        return
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"tool argument '{path}' must be a number")
        return
    if annotation in (list, dict):
        if not isinstance(value, annotation):
            raise TypeError(f"tool argument '{path}' has the wrong container type")
        return
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"tool argument '{path}' must be a list")
        if args and args[0] is not Any:
            for index, item in enumerate(value):
                _validate_argument(args[0], item, f"{path}[{index}]")
        return
    if origin is dict:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise TypeError(f"tool argument '{path}' must be a string-keyed object")
        if len(args) == 2 and args[1] is not Any:
            for key, item in value.items():
                _validate_argument(args[1], item, f"{path}.{key}")
        return
    if annotation is Any:
        return
    raise TypeError(f"unsupported tool annotation for '{path}'")


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
            raise ValueError(f"Tool '{directory.name}' is missing __init__.py")
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
    return _json_compatible(dict(value))


def _json_compatible(value: Any) -> Any:
    """Convert a tool observation to the supported JSON-compatible surface."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def _ask_for_parameter(
    *,
    backend: LlmBackend,
    transcript_path: Path,
    path: str,
    physics: str,
    expected: Any,
    instruction: str,
    evidence: Mapping[str, Any],
    request_index: int,
) -> Any:
    """Perform one backend-neutral structured parameter-estimation turn."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"null hook for {path} requires an instruction")
    if not isinstance(evidence, Mapping):
        raise TypeError(f"null hook for {path} requires object evidence")
    value_schema, nullable = _annotation_schema(expected)
    if nullable:
        raise TypeError(f"parameter estimate type for {path} cannot be nullable")
    tool_name = "return_parameter_estimate"
    tool_schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Return one structured parameter estimate.",
            "parameters": {
                "type": "object",
                "properties": {"value": value_schema},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    evidence_payload = _json_compatible(dict(evidence))
    messages = [
        Message(
            "system",
            "Estimate the requested value from the supplied scientific evidence. "
            "Return exactly one call to return_parameter_estimate and no other tool. "
            f"Parameter: {path}. Physics: {physics}",
        ),
        Message(
            "user",
            json.dumps(
                {"instruction": instruction.strip(), "evidence": evidence_payload},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        ),
    ]
    request_payload = {
        "messages": [_message_payload(message) for message in messages],
        "tools": [tool_schema],
    }
    _append_transcript(
        transcript_path,
        f"Null hook {path}, request {request_index}: sent to LLM",
        request_payload,
    )
    digest_payload = json.dumps(
        {"request": request_payload, "backend": backend.identity},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    _emit_progress(f"Requesting parameter estimate for {path} ({backend.identity})...")
    response = backend.complete(
        messages=messages,
        tools=[tool_schema],
        prompt_digest=hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
    )
    assistant_message = Message(
        "assistant",
        response.text,
        tool_calls=response.calls,
    )
    _append_transcript(
        transcript_path,
        f"Null hook {path}, request {request_index}: received from LLM",
        _message_payload(assistant_message),
    )
    if len(response.calls) != 1 or response.calls[0].name != tool_name:
        raise RuntimeError(f"parameter estimate for {path} must return exactly one {tool_name} call")
    arguments = dict(response.calls[0].arguments)
    if set(arguments) != {"value"}:
        raise ValueError(f"parameter estimate for {path} must contain exactly the value field")
    value = arguments["value"]
    _validate_argument(expected, value, path)
    _emit_progress(f"Parameter estimate received for {path}.")
    return copy.deepcopy(value)


def _resolve_runtime_null_hooks(
    *,
    context: ToolContext,
    contract: ModuleType,
    backend: LlmBackend,
    transcript_path: Path,
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
        hints = get_type_hints(hook)
        expected = hints.get("return")
        if expected is None or expected is Any:
            raise TypeError(f"null hook {hook.__name__} needs a concrete return annotation")
        request_count = 0

        def ask(
            *,
            instruction: str,
            evidence: Mapping[str, Any],
            response_type: Any | None = None,
        ) -> Any:
            nonlocal request_count
            request_count += 1
            estimate_type = expected if response_type is None else response_type
            if estimate_type is Any:
                raise TypeError("ask response_type must be a concrete supported type")
            return _ask_for_parameter(
                backend=backend,
                transcript_path=transcript_path,
                path=rule.path,
                physics=rule.physics,
                expected=estimate_type,
                instruction=instruction,
                evidence=evidence,
                request_index=request_count,
            )

        value = hook(context, ask)
        _validate_argument(expected, value, rule.path)
        context._resolve_null_hook(rule.path, value)
        provenance = context.state.setdefault("null_hook_provenance", {})
        provenance[rule.path] = {
            "backend": backend.identity,
            "hook": hook.__name__,
            "llm_requests": request_count,
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
            _validate_argument(
                hints.get(parameter.name, parameter.annotation), arguments[parameter.name], parameter.name
            )
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
    _outputs: dict[str, Any] = field(default_factory=dict, init=False)
    _summaries: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _stage_bundles: dict[str, tuple[list[_Tool], str, str]] = field(default_factory=dict, init=False)

    def _run_context(
        self,
        context: ToolContext,
        tools: list[_Tool],
        static_prompt: str,
        digest: str,
    ) -> tuple[Any, dict[str, Any]]:
        transcript_path = context.artifact_directory / _LLM_TRANSCRIPT_FILENAME
        _write_transcript_header(transcript_path)
        contract = _load_stage_contract(context.stage_id, self.stage_root)
        _resolve_runtime_null_hooks(
            context=context,
            contract=contract,
            backend=self.backend,
            transcript_path=transcript_path,
        )
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
                request_payload = {
                    "messages": [_message_payload(message) for message in messages],
                    "tools": tool_schemas,
                }
                _append_transcript(transcript_path, f"Turn {turn}: sent to LLM", request_payload)
                _emit_progress(
                    f"Calling LLM ({self.backend.identity}) for {context.stage_id}/{context.job_id} [turn {turn}]..."
                )
                response = self.backend.complete(
                    messages=messages,
                    tools=tool_schemas,
                    prompt_digest=digest,
                )
                _emit_progress("LLM response received.")
                calls = response.calls
                assistant_message = Message("assistant", response.text, tool_calls=calls)
                messages.append(assistant_message)
                _append_transcript(
                    transcript_path,
                    f"Turn {turn}: received from LLM",
                    _message_payload(assistant_message),
                )
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
        _emit_progress("=" * 60)
        _emit_progress(f"Agent run: {metadata['run_id']}  (backend={self.backend.identity})")
        _emit_progress(f"Stages: {', '.join(stage_ids)}")
        _emit_progress("=" * 60)
        stage_reports: dict[str, str] = {}
        stage_records: list[StageReportRecord] = []
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
                    {},
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
                if job is stage_jobs[-1]:
                    _emit_progress(f"Writing stage report: {stage_id}...")
                    report = _write_stage_report(stage_id, stage_records, stage_root=self.stage_root)
                    if report is not None:
                        stage_reports[stage_id] = str(report)
                    _emit_progress(f"Stage {stage_id} finished.")
                    stage_records = []
            if stage_records:
                raise RuntimeError("final stage did not reach its indexed report boundary")
        finally:
            parallel.close()
        _emit_progress("=" * 60)
        _emit_progress(f"Agent run complete ({len(jobs)} job(s)).")
        _emit_progress("=" * 60)
        return {"outputs": dict(self._outputs), "summaries": dict(self._summaries), "stage_reports": stage_reports}


def create_session(backend: LlmBackend) -> _AgentSession:
    """Create an isolated workflow session for one resolved LLM backend."""
    return _AgentSession(backend)


__all__ = [
    "ToolContext",
    "create_session",
]
