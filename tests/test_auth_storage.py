"""Tests for OAuth token persistence."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamet_agent.auth.models import OAuthToken
from lamet_agent.auth.storage import FileTokenStore


class FileTokenStoreTests(unittest.TestCase):
    """Verify token storage round-trips and cleanup."""

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileTokenStore(Path(tmpdir))
            token = OAuthToken(provider="codex", access_token="abc123", refresh_token="refresh")
            store.save(token)
            loaded = store.load("codex")
            assert loaded is not None
            self.assertEqual(loaded.access_token, "abc123")
            self.assertEqual(loaded.refresh_token, "refresh")

    def test_delete_reports_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileTokenStore(Path(tmpdir))
            self.assertFalse(store.delete("claude_code"))
            store.save(OAuthToken(provider="claude_code", access_token="token"))
            self.assertTrue(store.delete("claude_code"))


if __name__ == "__main__":
    unittest.main()
