"""OAuth flow manager for provider-backed agent integrations."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

from lamet_agent.auth.callback import OAuthCallbackServer
from lamet_agent.auth.config import load_provider_configuration, resolve_provider_configurations
from lamet_agent.auth.models import OAuthProviderConfig, OAuthToken
from lamet_agent.auth.pkce import build_code_challenge, generate_code_verifier, generate_state
from lamet_agent.auth.storage import FileTokenStore
from lamet_agent.errors import OAuthConfigurationError, OAuthFlowError


class OAuthLoginManager:
    """High-level helper for listing providers and running login flows."""

    def __init__(self, token_store: FileTokenStore | None = None) -> None:
        self.token_store = token_store or FileTokenStore()

    def list_provider_statuses(self) -> list[dict[str, Any]]:
        """Return configured-provider status for CLI inspection."""
        configs = resolve_provider_configurations()
        statuses: list[dict[str, Any]] = []
        for provider_name in ("codex", "claude_code"):
            token = self.token_store.load(provider_name)
            config = configs.get(provider_name)
            statuses.append(
                {
                    "provider": provider_name,
                    "configured": config is not None,
                    "redirect_uri": config.redirect_uri if config is not None else None,
                    "token_present": token is not None,
                    "token_expired": token.is_expired() if token is not None else None,
                }
            )
        return statuses

    def get_status(self, provider: str) -> dict[str, Any]:
        """Return detailed status for one provider."""
        config = load_provider_configuration(provider)
        token = self.token_store.load(provider)
        return {
            "provider": provider,
            "configured": config is not None,
            "redirect_uri": config.redirect_uri if config is not None else None,
            "scopes": config.scopes if config is not None else [],
            "token_present": token is not None,
            "token_expired": token.is_expired() if token is not None else None,
            "token_path": str(self.token_store.token_path(provider)),
        }

    def logout(self, provider: str) -> dict[str, Any]:
        """Delete the stored token for a provider."""
        removed = self.token_store.delete(provider)
        return {"provider": provider, "removed": removed}

    def begin_login(self, provider: str, redirect_port: int | None = None) -> dict[str, Any]:
        """Prepare a browser login flow and return the authorize URL."""
        config = load_provider_configuration(provider)
        if config is None:
            raise OAuthConfigurationError(
                f"Provider {provider!r} is not configured. Set the LAMET_AGENT_{provider.upper()}_* environment variables first."
            )
        if redirect_port is not None:
            config = OAuthProviderConfig(
                provider=config.provider,
                client_id=config.client_id,
                authorization_url=config.authorization_url,
                token_url=config.token_url,
                scopes=list(config.scopes),
                client_secret=config.client_secret,
                audience=config.audience,
                extra_auth_params=dict(config.extra_auth_params),
                redirect_host=config.redirect_host,
                redirect_port=redirect_port,
            )
        state = generate_state()
        code_verifier = generate_code_verifier()
        code_challenge = build_code_challenge(code_verifier)
        authorization_url = self.build_authorization_url(config, state, code_challenge)
        return {
            "provider": provider,
            "authorization_url": authorization_url,
            "redirect_uri": config.redirect_uri,
            "state": state,
            "code_verifier": code_verifier,
        }

    def login(self, provider: str, redirect_port: int | None = None, timeout_seconds: int = 180) -> dict[str, Any]:
        """Run an interactive local-browser OAuth login flow and persist the token."""
        prepared = self.begin_login(provider, redirect_port=redirect_port)
        return self.complete_login(
            provider=provider,
            state=prepared["state"],
            code_verifier=prepared["code_verifier"],
            authorization_url=prepared["authorization_url"],
            redirect_port=redirect_port,
            timeout_seconds=timeout_seconds,
        )

    def complete_login(
        self,
        provider: str,
        state: str,
        code_verifier: str,
        authorization_url: str,
        redirect_port: int | None = None,
        timeout_seconds: int = 180,
    ) -> dict[str, Any]:
        """Wait for the callback, exchange the code, and persist the token."""
        config = load_provider_configuration(provider)
        if config is None:
            raise OAuthConfigurationError(f"Provider {provider!r} is not configured.")
        if redirect_port is not None:
            config.redirect_port = redirect_port
        callback_server = OAuthCallbackServer(config)
        callback_server.start()
        callback = callback_server.wait_for_callback(timeout_seconds=timeout_seconds)
        if callback.state != prepared["state"]:
            raise OAuthFlowError("OAuth callback state did not match the login session state.")
        token = self.exchange_code(
            config=config,
            code=callback.code,
            code_verifier=prepared["code_verifier"],
        )
        token_path = self.token_store.save(token)
        return {
            "provider": provider,
            "redirect_uri": config.redirect_uri,
            "authorization_url": authorization_url,
            "token_path": str(token_path),
        }

    def exchange_code(self, config: OAuthProviderConfig, code: str, code_verifier: str) -> OAuthToken:
        """Exchange an authorization code for an access token."""
        form_data = {
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "code": code,
            "redirect_uri": config.redirect_uri,
            "code_verifier": code_verifier,
        }
        if config.client_secret:
            form_data["client_secret"] = config.client_secret
        response_payload = self._post_form(config.token_url, form_data)
        expires_in = response_payload.get("expires_in")
        expires_at = time.time() + float(expires_in) if expires_in is not None else None
        access_token = response_payload.get("access_token")
        if not access_token:
            raise OAuthFlowError("OAuth token response did not include 'access_token'.")
        return OAuthToken(
            provider=config.provider,
            access_token=str(access_token),
            token_type=str(response_payload.get("token_type", "Bearer")),
            refresh_token=response_payload.get("refresh_token"),
            scope=response_payload.get("scope"),
            expires_at=expires_at,
            id_token=response_payload.get("id_token"),
            raw=response_payload,
        )

    def build_authorization_url(
        self,
        config: OAuthProviderConfig,
        state: str,
        code_challenge: str,
    ) -> str:
        """Construct the provider authorization URL."""
        query_params = {
            "response_type": "code",
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if config.scopes:
            query_params["scope"] = " ".join(config.scopes)
        if config.audience:
            query_params["audience"] = config.audience
        query_params.update(config.extra_auth_params)
        return f"{config.authorization_url}?{urllib.parse.urlencode(query_params)}"

    def _post_form(self, url: str, form_data: dict[str, str]) -> dict[str, Any]:
        """POST a form-encoded request and parse the JSON response."""
        body = urllib.parse.urlencode(form_data).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw_response = response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - depends on live provider behavior.
            raise OAuthFlowError(f"Failed to exchange OAuth code for a token: {exc}") from exc
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise OAuthFlowError(f"OAuth token response was not valid JSON: {exc}") from exc
        if "error" in payload:
            raise OAuthFlowError(f"OAuth token endpoint returned an error: {payload['error']}")
        return payload
