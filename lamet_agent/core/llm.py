"""LLM session backends for the staged agent loop."""

from __future__ import annotations

import http.client
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

_SYSTEM_PROMPT = (
    "You are the decision layer of a LaMET analysis agent. Decide the single "
    "next action only. Do NOT run shell commands or edit files. Reply with "
    "exactly one JSON object matching this shape: " + json.dumps(ACTION_SCHEMA)
)

_API_REQUEST_TIMEOUT_SECONDS = 60
_API_REQUEST_ATTEMPTS = 3

# OpenAI-compatible chat-completions providers. DeepSeek and OpenAI share the same
# request/response shape, so they only differ by base URL, default model, and the
# environment variable used to read the API key.
PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
}


def provider_config(provider: str) -> dict[str, str] | None:
    """Return the OpenAI-compatible provider config for a provider name, if any."""
    return PROVIDERS.get(provider)


def supports_temperature(model_name: str) -> bool:
    """Return whether chat-completions requests may send ``temperature``.

    OpenAI GPT-5+ and o-series reasoning models reject custom sampling params
    (400). DeepSeek and GPT-4o-class models still accept ``temperature: 0``.
    """
    name = model_name.strip().lower()
    if name.startswith("gpt-5"):
        return False
    if name.startswith(("o1", "o3", "o4")):
        return False
    return True


def _chat_completion_body(
    *,
    model_name: str,
    messages: list[dict[str, str]],
    response_format: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a chat-completions request body, omitting unsupported sampling params."""
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }
    if response_format is not None:
        body["response_format"] = response_format
    if supports_temperature(model_name):
        body["temperature"] = 0.0
    return body


def parse_api_model(spec: str) -> tuple[str, str]:
    """Parse ``provider/model_id`` or shorthand ``provider`` into provider and model name."""
    text = spec.strip()
    if not text:
        raise ValueError(
            "API model spec must be non-empty, e.g. 'deepseek/deepseek-v4-flash' or 'openai/gpt-4o-mini'."
        )
    if "/" in text:
        provider, model_name = text.split("/", 1)
        provider = provider.strip()
        model_name = model_name.strip()
        if not provider or not model_name:
            raise ValueError(
                f"Invalid API model spec {spec!r}; use 'provider/model_id', e.g. 'deepseek/deepseek-v4-flash'."
            )
    else:
        provider = text
        model_name = ""
    config = provider_config(provider)
    if config is None:
        raise ValueError(
            f"Unknown API provider {provider!r}; use one of {sorted(PROVIDERS)}."
        )
    if not model_name:
        model_name = config["default_model"]
    return provider, model_name


def format_api_model_spec(provider: str, model_name: str) -> str:
    """Return the canonical provider/model_id string for trace and run summaries."""
    return f"{provider}/{model_name}"


def _codex_decide(
    messages: list[dict],
    *,
    model_name: str | None = None,
) -> dict:
    try:
        from openai_codex import Codex, Sandbox
    except ImportError as exc:
        raise RuntimeError(
            "backend='codex' requires the openai-codex Python SDK. "
            "Install the project's codex extra before using this backend."
        ) from exc

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_parts = [m["content"] for m in messages if m.get("role") == "user"]

    developer_instructions = "\n\n".join(system_parts)

    task_input = "\n\n".join(
        [
            "<TASK_INPUT>",
            "\n\n".join(user_parts),
            "</TASK_INPUT>",
            "",
            "<OUTPUT_CONSTRAINT>",
            "Return exactly one JSON object only.",
            "Do not use markdown.",
            "Do not explain.",
            "Do not run shell commands.",
            "Do not edit files.",
            "</OUTPUT_CONSTRAINT>",
        ]
    )

    with Codex() as codex:
        thread = codex.thread_start(
            developer_instructions=developer_instructions,
            sandbox=Sandbox.read_only,
            ephemeral=True,
            model=model_name,
        )

        result = thread.run(
            task_input,
            sandbox=Sandbox.read_only,
        )

    if not result.final_response:
        raise RuntimeError(f"Codex returned no final response: {result}")

    return _parse_json_action(result.final_response, label="Codex")


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
    request_messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *messages]
    label = provider.capitalize()
    last_parse_error: ValueError | None = None

    for parse_attempt in range(3):
        body = _chat_completion_body(
            model_name=model_name,
            messages=request_messages,
            response_format={"type": "json_object"},
        )
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
        for attempt in range(_API_REQUEST_ATTEMPTS):
            try:
                with urllib.request.urlopen(
                    request, timeout=_API_REQUEST_TIMEOUT_SECONDS
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (TimeoutError, urllib.error.URLError, ssl.SSLError, http.client.IncompleteRead) as exc:
                last_error = exc
                if attempt == _API_REQUEST_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"{label} API request failed after {_API_REQUEST_ATTEMPTS} attempts "
                        f"with a {_API_REQUEST_TIMEOUT_SECONDS}-second timeout per attempt. "
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


def _post_chat_text_completion(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    model_name: str,
    base_url: str,
    provider: str = "deepseek",
    request_timeout_seconds: int = _API_REQUEST_TIMEOUT_SECONDS,
    request_attempts: int = _API_REQUEST_ATTEMPTS,
) -> str:
    if type(request_timeout_seconds) is not int or request_timeout_seconds < 1:
        raise ValueError("request_timeout_seconds must be a positive integer")
    if type(request_attempts) is not int or request_attempts < 1:
        raise ValueError("request_attempts must be a positive integer")
    url = base_url.rstrip("/") + "/chat/completions"
    label = provider.capitalize()
    body = _chat_completion_body(model_name=model_name, messages=messages)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_error: BaseException | None = None
    for attempt in range(request_attempts):
        try:
            with urllib.request.urlopen(
                request, timeout=request_timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (TimeoutError, urllib.error.URLError, ssl.SSLError, http.client.IncompleteRead) as exc:
            last_error = exc
            if attempt == request_attempts - 1:
                raise RuntimeError(
                    f"{label} API text request failed after {request_attempts} attempts "
                    f"with a {request_timeout_seconds}-second timeout per attempt. "
                    "Retry the command or check network/proxy settings."
                ) from exc
            time.sleep(2**attempt)
    raise RuntimeError(f"{label} API text request failed before returning a response.") from last_error


def request_llm_text(
    *,
    backend: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    request_timeout_seconds: int = _API_REQUEST_TIMEOUT_SECONDS,
    request_attempts: int = _API_REQUEST_ATTEMPTS,
) -> str:
    """Return free-form text from the configured LLM backend."""
    if backend == "api":
        if not provider:
            raise ValueError("backend='api' requires a provider.")
        config = provider_config(provider)
        if config is None:
            raise ValueError(f"Unknown API provider {provider!r}; use one of {sorted(PROVIDERS)}.")
        if not api_key:
            raise ValueError(f"backend='api' provider={provider!r} requires an API key.")
        return _post_chat_text_completion(
            messages=messages,
            api_key=api_key,
            model_name=model_name or config["default_model"],
            base_url=base_url or config["base_url"],
            provider=provider,
            request_timeout_seconds=request_timeout_seconds,
            request_attempts=request_attempts,
        )
    if backend == "codex":
        try:
            from openai_codex import Codex, Sandbox
        except ImportError as exc:
            raise RuntimeError(
                "backend='codex' requires the openai-codex Python SDK. "
                "Install the project's codex extra before using this backend."
            ) from exc
        with Codex() as codex:
            thread = codex.thread_start(
                developer_instructions=messages[0]["content"] if messages else "",
                sandbox=Sandbox.read_only,
                ephemeral=True,
                model=model_name,
            )
            result = thread.run("\n\n".join(item["content"] for item in messages[1:]), sandbox=Sandbox.read_only)
        if not result.final_response:
            raise RuntimeError(f"Codex returned no final response: {result}")
        return result.final_response.strip()
    raise ValueError(f"backend={backend!r} cannot generate free-form LLM text.")


def _request_llm_action(
    *,
    backend: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return one structured agent action from the configured LLM backend."""
    if backend == "mock":
        return dict(_MOCK_TOOL_ACTION)
    if backend == "codex":
        return _codex_decide(
            [{"role": "system", "content": _SYSTEM_PROMPT}, *messages],
            model_name=model_name,
        )
    if backend == "api":
        if not provider:
            raise ValueError("backend='api' requires a provider.")
        config = provider_config(provider)
        if config is None:
            raise ValueError(
                f"Unknown API provider {provider!r}; use one of {sorted(PROVIDERS)}."
            )
        if not api_key:
            raise ValueError(f"backend='api' provider={provider!r} requires an API key.")
        return _post_chat_completion(
            messages=messages,
            api_key=api_key,
            model_name=model_name or config["default_model"],
            base_url=base_url or config["base_url"],
            provider=provider,
        )
    raise ValueError(
        f"Unknown LLM backend {backend!r}; use one of mock, external, api, codex."
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


def _codex_session(model_name: str | None = None) -> LlmSession:
    """Return a multi-turn session backed by the Codex Python SDK."""

    class _CodexSession:
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
                backend="codex",
                messages=self._messages,
                model_name=model_name,
            )
            self._messages.append(
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
            )
            return action

    return _CodexSession()


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
                backend="api",
                messages=self._messages,
                api_key=api_key,
                provider=provider,
                model_name=model_name,
                base_url=base_url,
            )
            self._messages.append(
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
            )
            return action

    return _ChatSession()


def make_llm_session(
    backend: str,
    actions_path: str | Path | None = None,
    *,
    api_key: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> LlmSession:
    """Build an LLM session for mock, external, api, or codex backends.

    ``backend`` selects the integration (``mock``/``external``/``api``/``codex``).
    For ``api``, pass ``provider`` and ``model_name``; ``base_url`` overrides the
    provider default when given. For ``codex``, ``model_name`` optionally selects
    the Codex model; omitting it preserves the SDK default.
    """
    if backend == "mock":
        return _mock_session()
    if backend == "external":
        if actions_path is None:
            raise ValueError("backend='external' requires an actions_path transcript.")
        return _external_session(actions_path)
    if backend == "codex":
        return _codex_session(model_name)
    if backend == "api":
        if not provider:
            raise ValueError("backend='api' requires a provider.")
        config = provider_config(provider)
        if config is None:
            raise ValueError(
                f"Unknown API provider {provider!r}; use one of {sorted(PROVIDERS)}."
            )
        if not api_key:
            raise ValueError(f"backend='api' provider={provider!r} requires an API key.")
        return _openai_compatible_session(
            provider,
            api_key,
            model_name or config["default_model"],
            base_url or config["base_url"],
        )
    raise ValueError(
        f"Unknown LLM backend {backend!r}; use one of mock, external, api, codex."
    )
