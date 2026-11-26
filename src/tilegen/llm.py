# src/tilegen/llm.py
#
# Minimal OpenAI-compatible chat client over ApiConfig (tilegen.config):
# one /chat/completions endpoint covers ollama, openai, and deepseek.
# Standard-library only (no third-party HTTP dependency), so it works in the
# runtime-only install. API keys are never logged.

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence

from tilegen.config import ApiConfig, load_config

_ROLE = "user"


class LLMError(RuntimeError):
    """Raised for transport, HTTP, and timeout failures (never logs keys)."""


class LLMClient:
    """OpenAI-compatible chat client.

    ``config`` defaults to the effective global configuration
    (conf/tilegen.yaml + provider env vars + built-in defaults).
    """

    def __init__(self, config: Optional[ApiConfig] = None):
        self.config = config if config is not None else load_config().api_config
        self.config.validate()

    @property
    def endpoint(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _payload(
        self,
        messages: Sequence[dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
        }
        payload["temperature"] = (
            self.config.temperature if temperature is None else temperature)
        payload["max_tokens"] = (
            self.config.max_tokens if max_tokens is None else max_tokens)
        return payload

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """POST /chat/completions; returns the parsed JSON response dict."""
        timeout = self.config.timeout if timeout is None else timeout
        body = json.dumps(
            self._payload(messages, temperature, max_tokens)).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=body, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMError(
                f"chat request failed: HTTP {exc.code} "
                f"({self.config.provider}); {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"chat request failed: {exc.reason} "
                f"({self.config.provider})") from exc
        except TimeoutError as exc:
            raise LLMError(
                f"chat request timed out after {timeout}s "
                f"({self.config.provider})") from exc
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"chat response was not valid JSON ({self.config.provider})"
            ) from exc

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Single-user-message convenience; returns the assistant text."""
        response = self.chat(
            [{"role": _ROLE, "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, timeout=timeout,
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "chat response missing choices[0].message.content") from exc

    def __repr__(self) -> str:
        masked = self.config.masked()
        return (f"LLMClient(provider={masked.provider!r}, "
                f"model={masked.model!r}, base_url={masked.base_url!r})")


__all__ = ["LLMClient", "LLMError"]
