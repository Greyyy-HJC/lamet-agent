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
                    "DeepSeek API request failed after 3 attempts. "
                    "This is usually a transient HTTPS/network/proxy issue; retry the command or check network/proxy settings."
                ) from exc
            time.sleep(2**attempt)

    if payload is None:
        raise RuntimeError("DeepSeek API request failed before returning a response.") from last_error

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


def make_llm_session(
    model: str,
    actions_path: str | Path | None,
    api_key: str | None = None,
    deepseek_model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> LlmSession:
    """Build an LLM session for mock, external transcript, or DeepSeek backends."""
    if model == "external":
        if actions_path is None:
            raise ValueError("model='external' requires an actions_path transcript.")
        return _external_session(actions_path)
    if model == "deepseek":
        if not api_key:
            raise ValueError("model='deepseek' requires an API key.")
        return _deepseek_session(api_key, deepseek_model, base_url)
    return _mock_session()
