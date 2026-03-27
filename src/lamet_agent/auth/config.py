"""Provider configuration resolution for built-in OAuth-capable backends."""

from __future__ import annotations

import json
import os

from lamet_agent.auth.models import OAuthProviderConfig
from lamet_agent.errors import OAuthConfigurationError

OAUTH_PROVIDER_NAMES = ("codex", "claude_code")


def resolve_provider_configurations() -> dict[str, OAuthProviderConfig]:
    """Load configured OAuth providers from environment variables."""
    resolved: dict[str, OAuthProviderConfig] = {}
    for provider in OAUTH_PROVIDER_NAMES:
        config = load_provider_configuration(provider)
        if config is not None:
            resolved[provider] = config
    return resolved


def load_provider_configuration(provider: str) -> OAuthProviderConfig | None:
    """Load one provider configuration if enough values are present."""
    provider = provider.lower()
    if provider not in OAUTH_PROVIDER_NAMES:
        raise OAuthConfigurationError(
            f"Unsupported OAuth provider {provider!r}. Expected one of {list(OAUTH_PROVIDER_NAMES)}."
        )
    env_prefix = f"LAMET_AGENT_{provider.upper()}"
    client_id = os.getenv(f"{env_prefix}_CLIENT_ID")
    authorization_url = os.getenv(f"{env_prefix}_AUTH_URL")
    token_url = os.getenv(f"{env_prefix}_TOKEN_URL")
    if not any((client_id, authorization_url, token_url)):
        return None
    missing = [
        name
        for name, value in {
            "CLIENT_ID": client_id,
            "AUTH_URL": authorization_url,
            "TOKEN_URL": token_url,
        }.items()
        if not value
    ]
    if missing:
        raise OAuthConfigurationError(
            f"Provider {provider!r} is partially configured. Missing environment variables: "
            f"{', '.join(f'{env_prefix}_{item}' for item in missing)}."
        )
    scopes = _split_scopes(os.getenv(f"{env_prefix}_SCOPES", ""))
    extra_auth_params = _load_json_mapping(os.getenv(f"{env_prefix}_EXTRA_AUTH_PARAMS_JSON", "{}"))
    try:
        redirect_port = int(os.getenv(f"{env_prefix}_REDIRECT_PORT", "8765"))
    except ValueError as exc:
        raise OAuthConfigurationError(
            f"{env_prefix}_REDIRECT_PORT must be an integer."
        ) from exc
    redirect_host = os.getenv(f"{env_prefix}_REDIRECT_HOST", "127.0.0.1")
    return OAuthProviderConfig(
        provider=provider,
        client_id=client_id,
        authorization_url=authorization_url,
        token_url=token_url,
        scopes=scopes,
        client_secret=os.getenv(f"{env_prefix}_CLIENT_SECRET"),
        audience=os.getenv(f"{env_prefix}_AUDIENCE"),
        extra_auth_params=extra_auth_params,
        redirect_host=redirect_host,
        redirect_port=redirect_port,
    )


def _split_scopes(raw_scopes: str) -> list[str]:
    """Parse a comma- or space-delimited scope string into a stable list."""
    normalized = raw_scopes.replace(",", " ")
    return [scope for scope in normalized.split() if scope]


def _load_json_mapping(raw_json: str) -> dict[str, str]:
    """Parse a JSON object of extra OAuth authorization parameters."""
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise OAuthConfigurationError(
            f"Failed to parse OAuth extra auth params JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise OAuthConfigurationError("OAuth extra auth params must decode to a JSON object.")
    return {str(key): str(value) for key, value in payload.items()}
