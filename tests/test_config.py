# tests/test_config.py
#
# Tests for the global configuration loader: conf/tilegen.yaml (optional),
# provider environment variables, defaults, and explicit overrides.

import pytest

from tilegen.config import ApiConfig, Config, load_config


class TestApiConfig:
    def test_defaults_without_env(self):
        cfg = ApiConfig.from_dict({}, env={})
        assert cfg.provider == "ollama"
        assert cfg.model == "qwen3.5:27b"
        assert cfg.base_url == "http://localhost:11434/v1"

    def test_env_fallback_openai(self):
        cfg = ApiConfig.from_dict(
            {"provider": "openai", "model": "gpt-4o-mini"},
            env={"OPENAI_API_KEY": "sk-env", "OPENAI_API_BASE": "https://x/v1"},
        )
        assert cfg.api_key == "sk-env"
        assert cfg.base_url == "https://x/v1"

    def test_env_fallback_deepseek(self):
        cfg = ApiConfig.from_dict(
            {"provider": "deepseek"},
            env={"DEEPSEEK_API_KEY": "sk-ds", "DEEPSEEK_API_BASE": "https://d/v1"},
        )
        assert cfg.api_key == "sk-ds"
        assert cfg.base_url == "https://d/v1"

    def test_conf_beats_env(self):
        cfg = ApiConfig.from_dict(
            {"provider": "openai", "api_key": "conf-key", "base_url": "https://c/v1"},
            env={"OPENAI_API_KEY": "sk-env", "OPENAI_API_BASE": "https://x/v1"},
        )
        assert cfg.api_key == "conf-key"
        assert cfg.base_url == "https://c/v1"

    def test_overrides_beat_conf(self):
        cfg = ApiConfig.from_dict(
            {"provider": "openai", "temperature": 0.2},
            env={}, overrides={"temperature": 1.0},
        )
        assert cfg.temperature == 1.0

    def test_partial_conf_fills_defaults(self):
        cfg = ApiConfig.from_dict({"model": "custom-model"}, env={})
        assert cfg.model == "custom-model"
        assert cfg.timeout == 3600

    def test_masked_hides_key(self):
        cfg = ApiConfig.from_dict(
            {"provider": "openai", "api_key": "sk-abcdefgh"}, env={})
        masked = cfg.masked()
        assert masked.api_key == "sk-a***"
        assert "abcdefgh" not in masked.api_key

    def test_validation(self):
        with pytest.raises(ValueError, match="unknown provider"):
            ApiConfig.from_dict({"provider": "nope"}, env={})
        with pytest.raises(ValueError, match="temperature"):
            ApiConfig.from_dict({"temperature": 9.0}, env={})
        with pytest.raises(ValueError, match="max_tokens"):
            ApiConfig.from_dict({"max_tokens": 0}, env={})


class TestConfigFile:
    def test_missing_file_uses_defaults(self, tmp_path):
        cfg = Config.load(path=str(tmp_path / "nope.yaml"), env={})
        assert cfg.api_config.provider == "ollama"

    def test_file_without_block(self, tmp_path, monkeypatch):
        p = tmp_path / "tilegen.yaml"
        p.write_text("budget: {}\n", encoding="utf-8")
        cfg = Config.load(path=str(p), env={})
        assert cfg.api_config.model == "qwen3.5:27b"

    def test_file_provides_block(self, tmp_path):
        p = tmp_path / "tilegen.yaml"
        p.write_text(
            "api_config:\n"
            "  provider: deepseek\n"
            "  model: deepseek-chat\n"
            "  api_key: file-key\n"
            "  base_url: https://api.deepseek.com/v1\n"
            "  temperature: 0.5\n"
            "  max_tokens: 8192\n"
            "  timeout: 300\n", encoding="utf-8")
        cfg = Config.load(path=str(p), env={})
        assert cfg.api_config.provider == "deepseek"
        assert cfg.api_config.model == "deepseek-chat"
        assert cfg.api_config.max_tokens == 8192
        assert cfg.api_config.temperature == 0.5

    def test_bad_block_raises(self, tmp_path):
        p = tmp_path / "tilegen.yaml"
        p.write_text("api_config: [1, 2]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="api_config"):
            Config.load(path=str(p), env={})

    def test_load_config_wrapper(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-w")
        monkeypatch.setenv("OPENAI_API_BASE", "https://w/v1")
        cfg = load_config(path="/nonexistent", overrides={"provider": "openai"})
        assert cfg.api_config.api_key == "sk-w"
        assert cfg.api_config.base_url == "https://w/v1"
