"""Tests for OAuth provider configuration resolution."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.auth.config import load_provider_configuration, resolve_provider_configurations
from lamet_agent.errors import OAuthConfigurationError


class OAuthConfigTests(unittest.TestCase):
    """Verify environment-driven provider configuration handling."""

    def test_unconfigured_provider_returns_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(load_provider_configuration("codex"))

    def test_configured_provider_loads_expected_fields(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LAMET_AGENT_CODEX_CLIENT_ID": "codex-client",
                "LAMET_AGENT_CODEX_AUTH_URL": "https://example.com/auth",
                "LAMET_AGENT_CODEX_TOKEN_URL": "https://example.com/token",
                "LAMET_AGENT_CODEX_SCOPES": "openid profile offline_access",
                "LAMET_AGENT_CODEX_AUDIENCE": "lamet-agent",
                "LAMET_AGENT_CODEX_EXTRA_AUTH_PARAMS_JSON": "{\"prompt\": \"consent\"}",
            },
            clear=True,
        ):
            config = load_provider_configuration("codex")
            assert config is not None
            self.assertEqual(config.client_id, "codex-client")
            self.assertEqual(config.scopes, ["openid", "profile", "offline_access"])
            self.assertEqual(config.audience, "lamet-agent")
            self.assertEqual(config.extra_auth_params["prompt"], "consent")

    def test_partial_configuration_raises(self) -> None:
        with patch.dict(
            os.environ,
            {"LAMET_AGENT_CLAUDE_CODE_CLIENT_ID": "claude-client"},
            clear=True,
        ):
            with self.assertRaises(OAuthConfigurationError):
                load_provider_configuration("claude_code")

    def test_resolve_provider_configurations_filters_missing_entries(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LAMET_AGENT_CODEX_CLIENT_ID": "codex-client",
                "LAMET_AGENT_CODEX_AUTH_URL": "https://example.com/auth",
                "LAMET_AGENT_CODEX_TOKEN_URL": "https://example.com/token",
            },
            clear=True,
        ):
            providers = resolve_provider_configurations()
            self.assertEqual(sorted(providers.keys()), ["codex"])


if __name__ == "__main__":
    unittest.main()
