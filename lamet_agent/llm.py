"""Backend-neutral messages and unified provider-backed LLM construction."""

from __future__ import annotations

import ipaddress
import json
import os
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse


_OPENAI_COMPATIBLE_API = {
    "openai": ("https://api.openai.com/v1/", "OPENAI_API_KEY", "gpt-5.6-luna"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-haiku-4-5"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.7-flash"),
    "grok": ("https://api.x.ai/v1", "GROK_API_KEY", "grok-4.6"),
    "deepseek": ("https://api.deepseek.com/", "DEEPSEEK_API_KEY", "deepseek-v4-flash"),
}


@dataclass(frozen=True)
class _ResolvedProvider:
    """Internal provider selection derived from the public inputs."""

    kind: str
    provider: str
    model: str | None
    base_url: str | None = None
    key_env: str | None = None


def _is_local_url(value: str) -> bool:
    hostname = urlparse(value).hostname
    if hostname is None:
        return False
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _resolve_provider(provider: str, model: str | None = None) -> _ResolvedProvider:
    """Resolve a registered CLI/API provider or a custom compatible URL."""
    name = provider.strip()
    selected_model = model.strip() if model and model.strip() else None
    if not name:
        raise ValueError("provider must not be empty")
    if name == "codex":
        return _ResolvedProvider("cli", name, selected_model)
    if name in _OPENAI_COMPATIBLE_API:
        base_url, key_env, default_model = _OPENAI_COMPATIBLE_API[name]
        return _ResolvedProvider("api", name, selected_model or default_model, base_url, key_env)
    parsed = urlparse(name)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        if selected_model is None and not _is_local_url(name):
            raise ValueError("a custom OpenAI-compatible API URL requires a model")
        return _ResolvedProvider("api", name, selected_model, name)
    registered = sorted(["codex", *_OPENAI_COMPATIBLE_API])
    raise ValueError(f"unknown provider {name!r}; use one of {registered} or an HTTP(S) OpenAI-compatible API URL")


def _validate_provider_model(provider: _ResolvedProvider, api_key: str) -> _ResolvedProvider:
    """Validate the selected API model against the provider's models endpoint."""
    if provider.kind != "api" or provider.base_url is None:
        raise ValueError("model validation requires an API provider")
    request = urllib.request.Request(
        f"{provider.base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        response_context = (
            urllib.request.urlopen(request)
            if _is_local_url(provider.base_url)
            else urllib.request.urlopen(request, timeout=180)
        )
        with response_context as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to query available models from {provider.base_url!r}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"{provider.base_url!r}/models returned an invalid response")
    available = sorted(
        item["id"]
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    )
    if not available:
        raise ValueError(f"{provider.base_url!r}/models returned no model IDs")
    if provider.model is None:
        if not _is_local_url(provider.base_url):
            raise ValueError("model may be omitted only for a local provider URL")
        if len(available) != 1:
            raise ValueError(
                f"model is required because the local provider exposes multiple models: {', '.join(available)}"
            )
        return replace(provider, model=available[0])
    if provider.model not in available:
        raise ValueError(
            f"model {provider.model!r} is not available from {provider.base_url!r}; available models: {', '.join(available)}"
        )
    return provider


@dataclass(frozen=True)
class _ToolCall:
    """One already-parsed model tool call."""

    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("tool call id must be a nonempty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool call name must be a nonempty string")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be an object")


@dataclass(frozen=True)
class Message:
    """Neutral transcript message."""

    role: str
    content: str
    tool_call_id: str | None = None
    tool_call: _ToolCall | None = None
    tool_calls: tuple[_ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role '{self.role}'")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if not isinstance(self.tool_calls, tuple) or any(not isinstance(call, _ToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must be a tuple of internal tool-call values")
        if self.tool_call is None and len(self.tool_calls) == 1:
            object.__setattr__(self, "tool_call", self.tool_calls[0])
            object.__setattr__(self, "tool_calls", ())
        if self.tool_call is not None and self.tool_calls:
            raise ValueError("assistant messages cannot mix tool_call and tool_calls")
        if len({call.id for call in self.calls}) != len(self.calls):
            raise ValueError("tool call ids must be unique within one assistant message")
        if self.role in {"system", "user"} and (self.tool_call_id is not None or self.calls):
            raise ValueError("system/user messages cannot contain tool fields")
        if self.role == "assistant" and self.tool_call_id is not None:
            raise ValueError("assistant messages cannot contain tool_call_id")
        if self.role == "tool" and (not self.tool_call_id or self.calls):
            raise ValueError("tool messages require only tool_call_id")
        if self.role == "tool":
            try:
                observation = json.loads(self.content)
            except json.JSONDecodeError as exc:
                raise ValueError("tool message content must be canonical JSON") from exc
            if not isinstance(observation, dict):
                raise ValueError("tool message content must encode an object")
            canonical = json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if canonical != self.content:
                raise ValueError("tool message content must use canonical JSON")

    @property
    def calls(self) -> tuple[_ToolCall, ...]:
        """Return the assistant calls in provider order."""
        return self.tool_calls or ((self.tool_call,) if self.tool_call is not None else ())


@dataclass(frozen=True)
class _AssistantResponse:
    """One assistant turn with zero or more ordered tool calls."""

    text: str
    tool_call: _ToolCall | None = None
    tool_calls: tuple[_ToolCall, ...] = ()
    structured: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("assistant text must be a string")
        if not isinstance(self.tool_calls, tuple) or any(not isinstance(call, _ToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must be a tuple of internal tool-call values")
        if self.structured is not None and not isinstance(self.structured, Mapping):
            raise TypeError("structured response must be an object")
        if self.tool_call is None and len(self.tool_calls) == 1:
            object.__setattr__(self, "tool_call", self.tool_calls[0])
            object.__setattr__(self, "tool_calls", ())
        if self.tool_call is not None and self.tool_calls:
            raise ValueError("assistant responses cannot mix tool_call and tool_calls")
        if self.structured is not None and self.calls:
            raise ValueError("assistant responses cannot mix structured output and tool calls")
        if len({call.id for call in self.calls}) != len(self.calls):
            raise ValueError("tool call ids must be unique within one assistant response")

    @property
    def calls(self) -> tuple[_ToolCall, ...]:
        """Return all calls while preserving the single-call test interface."""
        return self.tool_calls or ((self.tool_call,) if self.tool_call is not None else ())


class LlmBackend(Protocol):
    """Synchronous interface used by an agent workflow session."""

    identity: str

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]],
        prompt_digest: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> _AssistantResponse: ...


def _chat_message(message: Message) -> dict[str, Any]:
    if message.role in {"system", "user"}:
        return {"role": message.role, "content": message.content}
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments), separators=(",", ":"))},
            }
            for call in message.calls
        ]
    return payload


class _OpenAICompatibleBackend:
    """One explicit non-streaming OpenAI-compatible chat adapter."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        if not base_url or not model or not api_key:
            raise ValueError("base_url, model, and api_key are required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.identity = f"openai-compatible:{self.base_url}:{self.model}"

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]],
        prompt_digest: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> _AssistantResponse:
        provider_messages = [_chat_message(message) for message in messages]
        body = {
            "model": self.model,
            "messages": provider_messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["parallel_tool_calls"] = False
        if response_schema is not None:
            if tools:
                raise ValueError("structured responses cannot be combined with tools")
            provider_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return only a JSON object matching this schema exactly:\n"
                        + json.dumps(response_schema["schema"], separators=(",", ":"), ensure_ascii=False)
                    ),
                }
            )
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_protocol_error: Exception | None = None
        for _attempt in range(3):
            with urllib.request.urlopen(request) as response:
                raw = response.read().decode("utf-8")
            try:
                payload = json.loads(raw)
                choice = payload["choices"][0]
                message = choice["message"]
                provider_calls = message.get("tool_calls") or []
                if not isinstance(provider_calls, list):
                    raise ValueError("provider returned malformed tool_calls")
                calls = []
                for provider_call in provider_calls:
                    if not isinstance(provider_call, dict) or not isinstance(provider_call.get("function"), dict):
                        raise ValueError("provider returned a malformed tool call")
                    function = provider_call["function"]
                    if (
                        not isinstance(provider_call.get("id"), str)
                        or not provider_call["id"]
                        or not isinstance(function.get("name"), str)
                        or not function["name"]
                        or not isinstance(function.get("arguments"), str)
                    ):
                        raise ValueError("provider returned a malformed tool call")
                    arguments = json.loads(function["arguments"])
                    if not isinstance(arguments, dict):
                        raise TypeError("provider tool arguments must decode to an object")
                    calls.append(_ToolCall(provider_call["id"], function["name"], arguments))
                text = str(message.get("content") or "")
                structured = None
                if response_schema is not None:
                    structured = json.loads(text)
                    if not isinstance(structured, dict):
                        raise TypeError("provider structured response must decode to an object")
                return _AssistantResponse(text, tool_calls=tuple(calls), structured=structured)
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_protocol_error = exc
        raise ValueError(
            f"provider returned malformed tool JSON after 3 attempts: {last_protocol_error}"
        ) from last_protocol_error


class _CodexBackend:
    """Stateless adapter for the installed openai-codex thread SDK."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.identity = f"codex:{model or 'default'}"
        self._turn = 0

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]],
        prompt_digest: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> _AssistantResponse:
        transcript = []
        start = 1 if messages and messages[0].role == "system" else 0
        for message in messages[start:]:
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if len(message.calls) == 1:
                call = message.calls[0]
                item["tool_call"] = {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
            elif message.calls:
                item["tool_calls"] = [
                    {"id": call.id, "name": call.name, "arguments": dict(call.arguments)} for call in message.calls
                ]
            transcript.append(item)
        task = {
            "messages": transcript,
            "tools": tools,
            "prompt_digest": prompt_digest,
            "response_schema": response_schema,
        }
        try:
            from openai_codex import Codex, Sandbox  # type: ignore
        except ImportError as exc:
            raise RuntimeError("the codex provider requires the openai-codex package") from exc

        if response_schema is not None:
            if tools:
                raise ValueError("structured responses cannot be combined with tools")
            output_constraint = "Return exactly one JSON object matching this schema and no other text:\n" + json.dumps(
                response_schema["schema"], separators=(",", ":"), ensure_ascii=False
            )
        else:
            output_constraint = (
                "Return exactly one JSON object with keys 'text' and 'tool_calls'. "
                "'text' must be a string. 'tool_calls' must be a list of objects containing exactly "
                "'name' and object 'arguments'. Do not call tools yourself."
            )
        task_input = "\n\n".join(
            [
                "<TASK_INPUT>",
                json.dumps(task, separators=(",", ":"), ensure_ascii=False),
                "</TASK_INPUT>",
                "<OUTPUT_CONSTRAINT>",
                output_constraint,
                "Do not use markdown. Do not run shell commands. Do not edit files.",
                "</OUTPUT_CONSTRAINT>",
            ]
        )
        developer_instructions = messages[0].content if messages and messages[0].role == "system" else ""
        with Codex() as codex:
            thread = codex.thread_start(
                developer_instructions=developer_instructions,
                sandbox=Sandbox.read_only,
                ephemeral=True,
                model=self.model,
            )
            result = thread.run(task_input, sandbox=Sandbox.read_only)
        raw = result.final_response
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError(f"Codex returned no final response: {result}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Codex returned malformed JSON: {raw}") from exc

        self._turn += 1
        if response_schema is not None:
            if not isinstance(payload, dict):
                raise TypeError("Codex structured response must decode to an object")
            return _AssistantResponse(raw, structured=payload)

        if not isinstance(payload, dict) or set(payload) != {"text", "tool_calls"}:
            raise ValueError("Codex response must contain exactly text and tool_calls")
        if not isinstance(payload["text"], str) or not isinstance(payload["tool_calls"], list):
            raise TypeError("Codex text must be a string and tool_calls must be a list")
        known_names = {
            item.get("function", {}).get("name")
            for item in tools
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        calls = []
        for index, tool_payload in enumerate(payload["tool_calls"], start=1):
            if (
                not isinstance(tool_payload, dict)
                or set(tool_payload) != {"name", "arguments"}
                or not isinstance(tool_payload["name"], str)
                or not isinstance(tool_payload["arguments"], dict)
            ):
                raise ValueError("Codex tool calls must contain exactly name and object arguments")
            if tool_payload["name"] not in known_names:
                raise ValueError(f"Codex response requested unavailable tool '{tool_payload['name']}'")
            calls.append(_ToolCall(f"turn-{self._turn}-{index}", tool_payload["name"], tool_payload["arguments"]))
        return _AssistantResponse(payload["text"], tool_calls=tuple(calls))


def create_backend(
    provider: str,
    model: str | None = None,
    api_key_file: str | Path | None = None,
) -> LlmBackend:
    """Create the one public LLM interface from provider-neutral inputs.

    Registered API providers obtain their key from ``api_key_file`` or their
    registered environment variable. Custom OpenAI-compatible URLs require a
    key file. Provider-specific authentication, model discovery, and adapter
    selection stay internal to this module.
    """
    resolved = _resolve_provider(provider, model)
    if resolved.kind == "cli":
        if api_key_file is not None:
            raise ValueError("api_key_file is only valid for API providers")
        return _CodexBackend(resolved.model)

    if api_key_file is not None:
        key_path = Path(api_key_file)
        if not key_path.is_file():
            raise ValueError(f"api_key_file {str(key_path)!r} does not exist or is not a file")
        key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"api_key_file {str(key_path)!r} is empty")
    else:
        key = (os.environ.get(resolved.key_env, "") if resolved.key_env else "").strip()
    if not key:
        if resolved.key_env:
            raise ValueError(
                f"API provider {resolved.provider!r} requires api_key_file or "
                f"the {resolved.key_env} environment variable"
            )
        raise ValueError(f"API provider {resolved.provider!r} requires api_key_file")
    resolved = _validate_provider_model(resolved, key)
    return _OpenAICompatibleBackend(resolved.base_url or "", resolved.model or "", key)


__all__ = [
    "Message",
    "LlmBackend",
    "create_backend",
]
