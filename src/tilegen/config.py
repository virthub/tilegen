# src/tilegen/config.py
#
# Global configuration loader: the OPTIONAL conf/tilegen.yaml api_config
# block, provider environment variables, and built-in defaults, with
# explicit overrides. Precedence: explicit overrides > conf file > provider
# env vars > defaults. Missing conf files degrade gracefully — the core
# commands never require this file.

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional

DEFAULT_CONF_PATH = "conf/tilegen.yaml"

# Provider -> (api_key env var, base_url env var) namespaces.
PROVIDER_ENV: dict[str, tuple[str, str]] = {
    "openai": ("OPENAI_API_KEY", "OPENAI_API_BASE"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE"),
    "ollama": ("OLLAMA_API_KEY", "OLLAMA_API_BASE"),
}

DEFAULTS: dict[str, Any] = {
    "provider": "ollama",
    "model": "qwen3.5:27b",
    "api_key": "not-needed",
    "base_url": "http://localhost:11434/v1",
    "temperature": 0.2,
    "max_tokens": 131072,
    "timeout": 3600,
}


@dataclass
class ApiConfig:
    """LLM access configuration for one provider."""

    provider: str = "ollama"
    model: str = "qwen3.5:27b"
    api_key: str = "not-needed"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.2
    max_tokens: int = 131072
    timeout: int = 3600

    @classmethod
    def from_dict(
        cls,
        data: Optional[Mapping[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "ApiConfig":
        """Resolve the effective api_config.

        Precedence: ``overrides`` > ``data`` (conf file) > provider env vars
        > ``DEFAULTS``.  ``env`` defaults to ``os.environ``.
        """
        data = dict(data or {})
        env = os.environ if env is None else env
        overrides = dict(overrides or {})

        valid = {f.name for f in fields(cls)}
        provider = str(data.get("provider") or overrides.get("provider")
                       or DEFAULTS["provider"])
        merged = dict(DEFAULTS)

        key_env, base_env = PROVIDER_ENV.get(provider, ("", ""))
        if key_env and env.get(key_env):
            merged["api_key"] = env[key_env]
        if base_env and env.get(base_env):
            merged["base_url"] = env[base_env]

        merged.update({k: v for k, v in data.items()
                       if k in valid and v is not None})
        merged.update({k: v for k, v in overrides.items()
                       if k in valid and v is not None})

        cfg = cls(**{k: merged[k] for k in valid})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.provider not in PROVIDER_ENV and self.provider != "custom":
            raise ValueError(
                f"unknown provider {self.provider!r}; expected one of "
                f"{sorted(set(PROVIDER_ENV) | {'custom'})}")
        if not isinstance(self.temperature, (int, float)) \
                or not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ValueError("timeout must be a positive integer (seconds)")
        if not self.base_url:
            raise ValueError("base_url is required")

    def masked(self) -> "ApiConfig":
        """A copy with the api key hidden (safe for logs)."""
        if not self.api_key:
            return replace(self)
        key = self.api_key
        shown = key[:4] + "***" if len(key) > 4 else "***"
        return replace(self, api_key=shown)


@dataclass
class Config:
    """The effective global configuration."""

    api_config: ApiConfig = field(default_factory=ApiConfig)

    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "Config":
        path = path or DEFAULT_CONF_PATH
        api_data: dict[str, Any] = {}
        if Path(path).exists():
            import yaml # type: ignore

            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"config {path} must be a mapping")
            block = raw.get("api_config")
            if block is None:
                pass # file exists but has no api_config: defaults + env
            elif isinstance(block, dict):
                api_data = block
            else:
                raise ValueError(f"config {path}: api_config must be a mapping")
        api = ApiConfig.from_dict(api_data, env=env, overrides=overrides)
        return cls(api_config=api)


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Config:
    """Convenience wrapper around ``Config.load``."""
    return Config.load(path=path, overrides=overrides, env=env)


__all__ = ["ApiConfig", "Config", "load_config", "DEFAULTS", "PROVIDER_ENV",
           "DEFAULT_CONF_PATH"]
