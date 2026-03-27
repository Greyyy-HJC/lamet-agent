"""Models used by the OAuth scaffolding for supported agent providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class OAuthProviderConfig:
    """Configuration required to run an OAuth authorization-code flow with PKCE."""

    provider: str
    client_id: str
    authorization_url: str
    token_url: str
    scopes: list[str] = field(default_factory=list)
    client_secret: str | None = None
    audience: str | None = None
    extra_auth_params: dict[str, str] = field(default_factory=dict)
    redirect_host: str = "127.0.0.1"
    redirect_port: int = 8765

    @property
    def redirect_uri(self) -> str:
        """Return the local callback URI used for browser-based login."""
        return f"http://{self.redirect_host}:{self.redirect_port}/callback"


@dataclass(slots=True)
class OAuthToken:
    """Persisted OAuth token bundle."""

    provider: str
    access_token: str
    token_type: str = "Bearer"
    refresh_token: str | None = None
    scope: str | None = None
    expires_at: float | None = None
    id_token: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: datetime | None = None) -> bool:
        """Return whether the token expiry has elapsed."""
        if self.expires_at is None:
            return False
        current = now or datetime.now(timezone.utc)
        return current.timestamp() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize the token for JSON storage."""
        return {
            "provider": self.provider,
            "access_token": self.access_token,
            "token_type": self.token_type,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
            "expires_at": self.expires_at,
            "id_token": self.id_token,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthToken":
        """Hydrate a token model from persisted JSON."""
        return cls(
            provider=str(data["provider"]),
            access_token=str(data["access_token"]),
            token_type=str(data.get("token_type", "Bearer")),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
            expires_at=data.get("expires_at"),
            id_token=data.get("id_token"),
            raw=dict(data.get("raw", {})),
        )
