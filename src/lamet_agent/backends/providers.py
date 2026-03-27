"""Built-in backend descriptors used for future LLM provider integrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentBackendDescriptor:
    """Describe a future agent backend and its preferred authentication mode."""

    provider: str
    display_name: str
    auth_mode: str
    notes: str


BUILTIN_AGENT_BACKENDS = {
    "codex": AgentBackendDescriptor(
        provider="codex",
        display_name="OpenAI Codex",
        auth_mode="oauth",
        notes="OAuth provider configuration must be supplied via LAMET_AGENT_CODEX_* environment variables.",
    ),
    "claude_code": AgentBackendDescriptor(
        provider="claude_code",
        display_name="Claude Code",
        auth_mode="oauth",
        notes="OAuth provider configuration must be supplied via LAMET_AGENT_CLAUDE_CODE_* environment variables.",
    ),
}


def get_backend_descriptor(provider: str) -> AgentBackendDescriptor:
    """Return the backend descriptor for a supported provider."""
    return BUILTIN_AGENT_BACKENDS[provider]
