"""Tests for the shared pydantic models and provider settings."""

# pylint: disable=missing-function-docstring
import json

import pytest
from pydantic import ValidationError

from models import AppConfig, ChatChunk, ChatMessage, GenerationOptions, ModelInfo, load_app_config
from providers.settings import GeminiSettings, OllamaSettings, WatsonxSettings


def test_chat_message_shown_and_payload():
    msg = ChatMessage(role="user", content="expanded", display="short")
    assert msg.shown == "short"
    assert msg.payload() == {"role": "user", "content": "expanded"}
    assert ChatMessage(role="assistant", content="reply").shown == "reply"


def test_chat_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="x")


def test_chat_chunk_of():
    assert ChatChunk.of("hi").message.content == "hi"
    assert ChatChunk.of(None).message.content == ""


def test_generation_options_defaults_and_bounds():
    opts = GenerationOptions()
    assert (opts.temperature, opts.top_p, opts.max_tokens) == (0.7, 0.9, 1024)
    with pytest.raises(ValidationError):
        GenerationOptions(temperature=1.5)


def test_model_info_ignores_extra_fields():
    info = ModelInfo.model_validate({"model": "llama3", "size": 1024, "details": {"family": "llama", "x": 1}})
    assert info.size == 1024
    assert info.details.family == "llama"
    assert info.details.parameter_size is None


def test_app_config_provider_defaults():
    config = AppConfig.model_validate(
        {"default_model": "global", "providers": {"ollama": {"default_model": "llama3"}, "gemini": {}}}
    )
    assert config.default_model_for("ollama") == "llama3"
    assert config.default_model_for("gemini") == "global"
    assert config.default_model_for("missing") == "global"
    assert config.provider("missing").models == {}


def test_load_app_config_valid(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"default_provider": "gemini", "templates": {"A": "do a"}}))
    config = load_app_config(str(path))
    assert config.default_provider == "gemini"
    assert config.templates == {"A": "do a"}


@pytest.mark.parametrize("content", ["{not json", json.dumps({"templates": ["not", "a", "dict"]})])
def test_load_app_config_invalid_falls_back(tmp_path, content):
    path = tmp_path / "config.json"
    path.write_text(content)
    assert load_app_config(str(path)) == AppConfig()


def test_load_app_config_missing(tmp_path):
    assert load_app_config(str(tmp_path / "nope.json")) == AppConfig()


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("WATSONX_API_KEY", "secret")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    monkeypatch.delenv("WATSONX_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    wx = WatsonxSettings()
    assert wx.api_key.get_secret_value() == "secret"
    assert "secret" not in repr(wx)
    assert wx.project_id == "proj"
    assert wx.url == "https://eu-gb.ml.cloud.ibm.com"
    assert OllamaSettings().enabled is False
    assert GeminiSettings().api_key is None
