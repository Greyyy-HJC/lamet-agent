"""Tests for OAuth URL construction and token exchange helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.auth.models import OAuthProviderConfig
from lamet_agent.auth.oauth import OAuthLoginManager
from lamet_agent.auth.storage import FileTokenStore


class OAuthLoginManagerTests(unittest.TestCase):
    """Verify non-network OAuth helper behavior."""

    def test_build_authorization_url_contains_pkce_and_scope(self) -> None:
        manager = OAuthLoginManager(token_store=FileTokenStore(Path(tempfile.mkdtemp())))
        config = OAuthProviderConfig(
            provider="codex",
            client_id="client-id",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
            scopes=["openid", "offline_access"],
            audience="lamet-agent",
            extra_auth_params={"prompt": "consent"},
        )
        url = manager.build_authorization_url(config, state="state123", code_challenge="challenge456")
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(params["client_id"], ["client-id"])
        self.assertEqual(params["state"], ["state123"])
        self.assertEqual(params["code_challenge"], ["challenge456"])
        self.assertEqual(params["scope"], ["openid offline_access"])
        self.assertEqual(params["audience"], ["lamet-agent"])
        self.assertEqual(params["prompt"], ["consent"])

    def test_exchange_code_builds_token_model(self) -> None:
        manager = OAuthLoginManager(token_store=FileTokenStore(Path(tempfile.mkdtemp())))
        config = OAuthProviderConfig(
            provider="claude_code",
            client_id="client-id",
            authorization_url="https://example.com/authorize",
            token_url="https://example.com/token",
        )
        with patch.object(
            manager,
            "_post_form",
            return_value={
                "access_token": "access",
                "refresh_token": "refresh",
                "token_type": "Bearer",
                "scope": "openid",
                "expires_in": 3600,
            },
        ):
            token = manager.exchange_code(config, code="abc", code_verifier="verifier")
        self.assertEqual(token.provider, "claude_code")
        self.assertEqual(token.access_token, "access")
        self.assertEqual(token.refresh_token, "refresh")
        self.assertFalse(token.is_expired())


if __name__ == "__main__":
    unittest.main()
