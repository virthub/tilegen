# tests/test_llm.py
#
# Tests for the OpenAI-compatible chat client against a local HTTP server.

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tilegen.config import ApiConfig
from tilegen.llm import LLMClient, LLMError

_captured = {}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self): # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        _captured["path"] = self.path
        _captured["auth"] = self.headers.get("Authorization")
        _captured["body"] = json.loads(body)
        mode = _captured.get("mode", "ok")
        if mode == "error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "boom"}')
        elif mode == "slow":
            import time

            time.sleep(1.0)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"choices": [{"message": {"content": "hello"}}]}')

    def log_message(self, *args):
        del args # silence test-server logging


@pytest.fixture()
def server():
    _captured.clear()
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    _captured["base"] = f"http://127.0.0.1:{srv.server_port}/v1"
    yield srv
    srv.shutdown()
    thread.join(timeout=2)


def _client(server, **overrides):
    cfg = ApiConfig.from_dict(
        {"provider": "openai", "model": "m",
         "api_key": "sk-test", "base_url": _captured["base"],
         **overrides}, env={})
    return LLMClient(cfg)


class TestChat:
    def test_success_roundtrip(self, server):
        client = _client(server)
        resp = client.chat([{"role": "user", "content": "hi"}])
        assert resp["choices"][0]["message"]["content"] == "hello"
        assert _captured["path"] == "/v1/chat/completions"
        assert _captured["auth"] == "Bearer sk-test"
        assert _captured["body"]["model"] == "m"
        assert _captured["body"]["temperature"] == 0.2

    def test_complete_convenience(self, server):
        client = _client(server)
        assert client.complete("hi") == "hello"

    def test_http_error_raises_without_key(self, server):
        _captured["mode"] = "error"
        client = _client(server)
        with pytest.raises(LLMError, match="HTTP 500"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_no_auth_header_for_ollama(self, server):
        cfg = ApiConfig.from_dict({
            "provider": "ollama", "api_key": "not-needed",
            "base_url": _captured["base"]}, env={})
        LLMClient(cfg).chat([{"role": "user", "content": "hi"}])
        assert _captured["auth"] is None

    def test_timeout_raises(self, server):
        _captured["mode"] = "slow"
        client = _client(server, timeout=1)
        with pytest.raises(LLMError, match="timed out"):
            client.chat([{"role": "user", "content": "hi"}], timeout=0.2)

    def test_repr_hides_key(self, server):
        client = _client(server)
        assert "sk-test" not in repr(client) # the key is omitted entirely
