"""Minimal local callback server for OAuth browser redirects."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from lamet_agent.auth.models import OAuthProviderConfig
from lamet_agent.errors import OAuthFlowError


@dataclass(slots=True)
class OAuthCallbackPayload:
    """Authorization callback payload captured from the browser redirect."""

    code: str
    state: str


class OAuthCallbackServer:
    """One-shot local HTTP server that captures an OAuth callback."""

    def __init__(self, provider_config: OAuthProviderConfig) -> None:
        self.provider_config = provider_config
        self._queue: queue.Queue[OAuthCallbackPayload | Exception] = queue.Queue(maxsize=1)
        self._httpd = HTTPServer((provider_config.redirect_host, provider_config.redirect_port), self._handler_factory())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> None:
        """Start the local callback server."""
        self._thread.start()

    def wait_for_callback(self, timeout_seconds: int = 180) -> OAuthCallbackPayload:
        """Block until the browser redirect arrives or timeout expires."""
        try:
            payload = self._queue.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            raise OAuthFlowError("Timed out waiting for the OAuth callback.") from exc
        finally:
            self.shutdown()
        if isinstance(payload, Exception):
            raise payload
        return payload

    def shutdown(self) -> None:
        """Stop the callback server if it is still running."""
        self._httpd.shutdown()
        self._httpd.server_close()

    def _handler_factory(self):
        queue_ref = self._queue

        class CallbackHandler(BaseHTTPRequestHandler):
            """HTTP handler used to capture the authorization callback parameters."""

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/callback":
                    self.send_response(404)
                    self.end_headers()
                    return
                params = parse_qs(parsed.query)
                code = params.get("code", [None])[0]
                state = params.get("state", [None])[0]
                error = params.get("error", [None])[0]
                if error:
                    queue_ref.put(OAuthFlowError(f"OAuth provider returned an error: {error}"))
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"OAuth login failed. You can close this window.")
                    return
                if not code or not state:
                    queue_ref.put(OAuthFlowError("OAuth callback did not include both 'code' and 'state'."))
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"OAuth callback was incomplete. You can close this window.")
                    return
                queue_ref.put(OAuthCallbackPayload(code=code, state=state))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OAuth login succeeded. You can close this window.")

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        return CallbackHandler
