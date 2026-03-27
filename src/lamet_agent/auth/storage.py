"""Filesystem-backed token storage for OAuth credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path

from lamet_agent.auth.models import OAuthToken
from lamet_agent.errors import TokenStoreError


class FileTokenStore:
    """Persist OAuth tokens under the user's config directory."""

    def __init__(self, root_directory: Path | None = None) -> None:
        base_dir = root_directory or self._default_root_directory()
        self.root_directory = self._ensure_writable_directory(base_dir)

    def _default_root_directory(self) -> Path:
        """Choose a sensible default token directory for the current environment."""
        explicit_dir = os.getenv("LAMET_AGENT_OAUTH_DIR")
        if explicit_dir:
            return Path(explicit_dir)
        xdg_config_home = os.getenv("XDG_CONFIG_HOME")
        if xdg_config_home:
            return Path(xdg_config_home) / "lamet-agent" / "oauth"
        return Path.home() / ".config" / "lamet-agent" / "oauth"

    def _ensure_writable_directory(self, preferred_directory: Path) -> Path:
        """Create a writable token directory or fall back to /tmp when needed."""
        for candidate in (preferred_directory, Path("/tmp/lamet-agent/oauth")):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError:
                continue
        raise TokenStoreError(
            f"Failed to create a writable OAuth token directory. Tried: {preferred_directory} and /tmp/lamet-agent/oauth."
        )

    def token_path(self, provider: str) -> Path:
        """Return the on-disk JSON file used for one provider."""
        return self.root_directory / f"{provider}.json"

    def load(self, provider: str) -> OAuthToken | None:
        """Load a token if it exists."""
        path = self.token_path(provider)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TokenStoreError(f"Failed to parse token file {path}: {exc}") from exc
        return OAuthToken.from_dict(payload)

    def save(self, token: OAuthToken) -> Path:
        """Persist a token bundle to disk."""
        path = self.token_path(token.provider)
        path.write_text(json.dumps(token.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def delete(self, provider: str) -> bool:
        """Delete a persisted provider token if it exists."""
        path = self.token_path(provider)
        if not path.exists():
            return False
        path.unlink()
        return True
