"""LLM session backends for the staged agent loop."""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .prompting import format_tool_observation

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

# OpenAI-compatible chat-completions providers. DeepSeek and OpenAI share the same
# request/response shape, so they only differ by base URL, default model, and the
# environment variable used to read the API key.
PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
}


def provider_config(model: str) -> dict[str, str] | None:
    """Return the OpenAI-compatible provider config for a backend name, if any."""
    return PROVIDERS.get(model)


class LlmSession(Protocol):
    """Per-stage LLM conversation handle."""

    def begin_stage(self, static_user: str) -> None: ...

    def decide(self, *, last_observation: dict[str, Any] | None) -> dict[str, Any]: ...


def _parse_json_action(content: str, *, label: str) -> dict[str, Any]:
    """Parse one JSON action from provider text, including fenced/explanatory replies."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as direct_exc:
        match = re.search(r"\{.*\}", content, re.S)
        if match is None:
            raise ValueError(f"{label} returned no JSON action:\n{content}") from direct_exc
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as extracted_exc:
            raise ValueError(
                f"{label} returned malformed JSON action: {extracted_exc.msg} "
                f"at line {extracted_exc.lineno} column {extracted_exc.colno}. Raw content:\n{content}"
            ) from extracted_exc


def _post_chat_completion(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    model_name: str,
    base_url: str,
    provider: str = "deepseek",
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    system = (
        "You are the decision layer of a LaMET analysis agent. Decide the single "
        "next action only. Do NOT run shell commands or edit files. Reply with "
        "exactly one JSON object matching this shape: " + json.dumps(ACTION_SCHEMA)
    )
    request_messages = [{"role": "system", "content": system}, *messages]
    label = provider.capitalize()
    last_parse_error: ValueError | None = None

    for parse_attempt in range(3):
        body = {
            "model": model_name,
            "messages": request_messages,
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
        payload = None
        last_error: BaseException | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (TimeoutError, urllib.error.URLError, ssl.SSLError) as exc:
                last_error = exc
                if attempt == 2:
                    raise RuntimeError(
                        f"{label} API request failed after 3 attempts. "
                        "This is usually a transient HTTPS/network/proxy issue; retry the command or check network/proxy settings."
                    ) from exc
                time.sleep(2**attempt)

        if payload is None:
            raise RuntimeError(f"{label} API request failed before returning a response.") from last_error

        content = payload["choices"][0]["message"]["content"]
        try:
            return _parse_json_action(content, label=label)
        except ValueError as exc:
            last_parse_error = exc
            if parse_attempt == 2:
                raise
            request_messages = [
                *request_messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON for the required action object. "
                        f"Parser error: {exc}. Return exactly one valid JSON object and no other text."
                    ),
                },
            ]

    raise RuntimeError(f"{label} failed to return a parseable JSON action.") from last_parse_error


def _request_llm_action(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return one structured agent action from the configured LLM backend."""
    if model == "mock":
        return dict(_MOCK_TOOL_ACTION)
    config = provider_config(model)
    if config is not None:
        if not api_key:
            raise ValueError(f"model={model!r} requires an API key.")
        return _post_chat_completion(
            messages=messages,
            api_key=api_key,
            model_name=model_name or config["default_model"],
            base_url=base_url or config["base_url"],
            provider=model,
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
                return dict(_MOCK_TOOL_ACTION)
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


def _openai_compatible_session(
    provider: str,
    api_key: str,
    model_name: str,
    base_url: str,
) -> LlmSession:
    """Return a multi-turn session backed by an OpenAI-compatible chat API."""

    class _ChatSession:
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
                model=provider,
                messages=self._messages,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
            )
            self._messages.append(
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
            )
            return action

    return _ChatSession()


def make_llm_session(
    model: str,
    actions_path: str | Path | None,
    api_key: str | None = None,
    llm_model: str | None = None,
    base_url: str | None = None,
) -> LlmSession:
    """Build an LLM session for mock, external, DeepSeek, or OpenAI backends.

    ``model`` selects the backend (``mock``/``external``/``deepseek``/``openai``);
    ``llm_model`` and ``base_url`` override the provider defaults when given.
    """
    if model == "external":
        if actions_path is None:
            raise ValueError("model='external' requires an actions_path transcript.")
        return _external_session(actions_path)
    config = provider_config(model)
    if config is not None:
        if not api_key:
            raise ValueError(f"model={model!r} requires an API key.")
        return _openai_compatible_session(
            model,
            api_key,
            llm_model or config["default_model"],
            base_url or config["base_url"],
        )
    return _mock_session()
