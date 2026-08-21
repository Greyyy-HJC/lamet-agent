"""LLM provider resolution and sessions for the staged agent loop."""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
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

_SYSTEM_PROMPT = (
    "You are the decision layer of a LaMET analysis agent. Decide the single "
    "next action only. Do NOT run shell commands or edit files. Reply with "
    "exactly one JSON object matching this shape: " + json.dumps(ACTION_SCHEMA)
)

_API_REQUEST_TIMEOUT_SECONDS = 180
_API_REQUEST_ATTEMPTS = 6

_OPENAI_COMPATIBLE_API = {
    "openai": ("https://api.openai.com/v1/", "OPENAI_API_KEY", "gpt-5.6-luna"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", "claude-haiku-4-5"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.7-flash"),
    "grok": ("https://api.x.ai/v1", "GROK_API_KEY", "grok-4.6"),
    "deepseek": ("https://api.deepseek.com/", "DEEPSEEK_API_KEY", "deepseek-v4-flash"),
}

_AGENT_CLI = frozenset({"codex"})


@dataclass(frozen=True)
class ResolvedLlmProvider:
    """Normalized CLI or OpenAI-compatible API provider configuration."""

    kind: str
    provider: str
    model_name: str | None
    base_url: str | None = None
    key_env: str | None = None


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_local_url(value: str) -> bool:
    hostname = urllib.parse.urlparse(value).hostname
    if hostname is None:
        return False
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _urlopen(request: urllib.request.Request, *, timeout: int):
    """Open local endpoints without a timeout and bound all remote requests."""
    if _is_local_url(request.full_url):
        return urllib.request.urlopen(request)
    return urllib.request.urlopen(request, timeout=timeout)


def list_available_models(*, base_url: str, api_key: str) -> list[str]:
    """Return model IDs from an OpenAI-compatible ``GET /models`` endpoint."""
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    last_error: BaseException | None = None
    for attempt in range(_API_REQUEST_ATTEMPTS):
        try:
            with _urlopen(request, timeout=_API_REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError(
                    f"Model-list endpoint {url!r} returned no OpenAI-compatible 'data' list."
                )
            return sorted(
                {
                    str(item["id"])
                    for item in data
                    if isinstance(item, dict) and item.get("id")
                }
            )
        except (TimeoutError, urllib.error.URLError, ssl.SSLError, http.client.IncompleteRead) as exc:
            last_error = exc
            if attempt == _API_REQUEST_ATTEMPTS - 1:
                timeout_detail = (
                    "without a request timeout"
                    if _is_local_url(url)
                    else f"with a {_API_REQUEST_TIMEOUT_SECONDS}-second timeout per attempt"
                )
                raise RuntimeError(
                    f"Model-list request to {url!r} failed after "
                    f"{_API_REQUEST_ATTEMPTS} attempts {timeout_detail}."
                ) from exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Model-list request to {url!r} failed.") from last_error


def validate_api_model(
    resolved: ResolvedLlmProvider,
    *,
    api_key: str,
) -> ResolvedLlmProvider:
    """Validate or infer an API model against the provider's ``/models`` list."""
    if resolved.kind != "api" or resolved.base_url is None:
        raise ValueError("Model-list validation requires an API provider.")
    available = list_available_models(base_url=resolved.base_url, api_key=api_key)
    available_text = ", ".join(available) if available else "(none)"
    model_name = resolved.model_name
    if model_name is None:
        if _is_local_url(resolved.base_url) and len(available) == 1:
            return replace(resolved, model_name=available[0])
        raise ValueError(
            "--model is required because the local provider did not return exactly "
            f"one model. Available models: {available_text}"
        )
    if model_name not in available:
        raise ValueError(
            f"Model {model_name!r} is not available from {resolved.base_url!r}. "
            f"Available models: {available_text}"
        )
    return resolved


def resolve_llm_provider(
    provider: str,
    model_name: str | None = None,
) -> ResolvedLlmProvider:
    """Resolve a registered CLI, registered API, or custom API URL."""
    name = provider.strip()
    selected_model = model_name.strip() if model_name and model_name.strip() else None
    if not name:
        raise ValueError("--provider must not be empty.")

    if name in _AGENT_CLI:
        return ResolvedLlmProvider(
            kind="cli",
            provider=name,
            model_name=selected_model,
        )

    api_config = _OPENAI_COMPATIBLE_API.get(name)
    if api_config is not None:
        base_url, key_env, default_model = api_config
        return ResolvedLlmProvider(
            kind="api",
            provider=name,
            model_name=selected_model or default_model,
            base_url=base_url,
            key_env=key_env,
        )

    if _is_http_url(name):
        if selected_model is None and not _is_local_url(name):
            raise ValueError(
                "A custom OpenAI-compatible API URL passed to --provider requires --model."
            )
        return ResolvedLlmProvider(
            kind="api",
            provider=name,
            model_name=selected_model,
            base_url=name,
        )

    registered = sorted([*_AGENT_CLI, *_OPENAI_COMPATIBLE_API])
    raise ValueError(
        f"Unknown provider {name!r}; use one of {registered} or an HTTP(S) "
        "OpenAI-compatible API URL."
    )


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


def _codex_decide(
    messages: list[dict],
    *,
    model_name: str | None = None,
) -> dict:
    try:
        from openai_codex import Codex, Sandbox
    except ImportError as exc:
        raise RuntimeError(
            "provider='codex' requires the openai-codex Python SDK. "
            "Install the project's codex extra before using this provider."
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
    local_url = _is_local_url(url)
    timeout_detail = (
        "without a request timeout"
        if local_url
        else f"with a {_API_REQUEST_TIMEOUT_SECONDS}-second timeout per attempt"
    )
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
                with _urlopen(
                    request, timeout=_API_REQUEST_TIMEOUT_SECONDS
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (TimeoutError, urllib.error.URLError, ssl.SSLError, http.client.IncompleteRead) as exc:
                last_error = exc
                if attempt == _API_REQUEST_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"{label} API request failed after {_API_REQUEST_ATTEMPTS} attempts "
                        f"{timeout_detail}. "
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
    local_url = _is_local_url(url)
    timeout_detail = (
        "without a request timeout"
        if local_url
        else f"with a {request_timeout_seconds}-second timeout per attempt"
    )
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
            with _urlopen(request, timeout=request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (TimeoutError, urllib.error.URLError, ssl.SSLError, http.client.IncompleteRead) as exc:
            last_error = exc
            if attempt == request_attempts - 1:
                raise RuntimeError(
                    f"{label} API text request failed after {request_attempts} attempts "
                    f"{timeout_detail}. "
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
    """Return free-form text from the configured LLM provider."""
    if backend == "api":
        if not provider:
            raise ValueError("API requests require a provider.")
        resolved = resolve_llm_provider(provider, model_name)
        if resolved.kind != "api":
            raise ValueError(f"provider={provider!r} is registered as a CLI, not an API.")
        if not api_key:
            raise ValueError(f"API provider {provider!r} requires an API key.")
        return _post_chat_text_completion(
            messages=messages,
            api_key=api_key,
            model_name=resolved.model_name or "",
            base_url=resolved.base_url or "",
            provider=resolved.provider,
            request_timeout_seconds=request_timeout_seconds,
            request_attempts=request_attempts,
        )
    if backend == "cli":
        if not provider:
            raise ValueError("CLI requests require a provider.")
        resolved = resolve_llm_provider(provider, model_name)
        if resolved.kind != "cli":
            raise ValueError(f"provider={provider!r} is registered as an API, not a CLI.")
        if resolved.provider != "codex":
            raise ValueError(f"Unsupported agent CLI provider {resolved.provider!r}.")
        try:
            from openai_codex import Codex, Sandbox
        except ImportError as exc:
            raise RuntimeError(
                "provider='codex' requires the openai-codex Python SDK. "
                "Install the project's codex extra before using this provider."
            ) from exc
        with Codex() as codex:
            thread = codex.thread_start(
                developer_instructions=messages[0]["content"] if messages else "",
                sandbox=Sandbox.read_only,
                ephemeral=True,
                model=resolved.model_name,
            )
            result = thread.run("\n\n".join(item["content"] for item in messages[1:]), sandbox=Sandbox.read_only)
        if not result.final_response:
            raise RuntimeError(f"Codex returned no final response: {result}")
        return result.final_response.strip()
    raise ValueError(f"Unknown LLM provider kind {backend!r}; use 'cli' or 'api'.")


def _request_llm_action(
    *,
    backend: str,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return one structured agent action from the configured LLM provider."""
    if backend == "cli":
        if not provider:
            raise ValueError("CLI requests require a provider.")
        resolved = resolve_llm_provider(provider, model_name)
        if resolved.kind != "cli":
            raise ValueError(f"provider={provider!r} is registered as an API, not a CLI.")
        if resolved.provider != "codex":
            raise ValueError(f"Unsupported agent CLI provider {resolved.provider!r}.")
        return _codex_decide(
            [{"role": "system", "content": _SYSTEM_PROMPT}, *messages],
            model_name=resolved.model_name,
        )
    if backend == "api":
        if not provider:
            raise ValueError("API requests require a provider.")
        resolved = resolve_llm_provider(provider, model_name)
        if resolved.kind != "api":
            raise ValueError(f"provider={provider!r} is registered as a CLI, not an API.")
        if not api_key:
            raise ValueError(f"API provider {provider!r} requires an API key.")
        return _post_chat_completion(
            messages=messages,
            api_key=api_key,
            model_name=resolved.model_name or "",
            base_url=resolved.base_url or "",
            provider=resolved.provider,
        )
    raise ValueError(f"Unknown LLM provider kind {backend!r}; use 'cli' or 'api'.")


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
                backend="cli",
                messages=self._messages,
                provider="codex",
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
    *,
    api_key: str | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> LlmSession:
    """Build an LLM session for a registered CLI or compatible API provider."""
    if backend == "cli":
        if not provider:
            raise ValueError("CLI sessions require a provider.")
        resolved = resolve_llm_provider(provider, model_name)
        if resolved.kind != "cli":
            raise ValueError(f"provider={provider!r} is registered as an API, not a CLI.")
        if resolved.provider == "codex":
            return _codex_session(resolved.model_name)
        raise ValueError(f"Unsupported agent CLI provider {resolved.provider!r}.")
    if backend == "api":
        if not provider:
            raise ValueError("API sessions require a provider.")
        resolved = resolve_llm_provider(provider, model_name)
        if resolved.kind != "api":
            raise ValueError(f"provider={provider!r} is registered as a CLI, not an API.")
        if not api_key:
            raise ValueError(f"API provider {provider!r} requires an API key.")
        return _openai_compatible_session(
            resolved.provider,
            api_key,
            resolved.model_name or "",
            resolved.base_url or "",
        )
    raise ValueError(f"Unknown LLM provider kind {backend!r}; use 'cli' or 'api'.")
