"""Tests for default provider and model settings from config.json."""

# pylint: disable=redefined-outer-name, unused-argument, missing-function-docstring, duplicate-code
import json
from unittest.mock import mock_open, patch

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def mock_dual_provider_env():
    """Setup environment where both providers are available."""
    with (
        patch("ollama.list") as mock_list,
        patch("ollama.chat") as mock_chat,
        patch("providers.watsonx_provider.WatsonxProvider.is_available") as mock_wx_avail,
        patch.dict("os.environ", {"OLLAMA_ENABLED": "true"}),
    ):
        mock_list.return_value = {"models": [{"model": "llama3"}, {"model": "mistral"}]}
        mock_wx_avail.return_value = True  # Both are available

        # Mocking chat for both
        def stream_response(*args, **kwargs):
            yield {"message": {"content": "Response"}}

        mock_chat.side_effect = stream_response

        yield mock_list, mock_chat


def test_default_provider_ollama(mock_dual_provider_env):
    """Test that default_provider: ollama works."""
    config_data = {"default_provider": "ollama", "providers": {"ollama": {"default_model": "mistral"}}, "templates": {}}

    with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
        at = AppTest.from_file("app.py").run()
        # Find provider selectbox (second one in sidebar)
        provider_selector = at.sidebar.selectbox[1]
        assert provider_selector.value == "Ollama (Local)"

        # Check if default model was also applied
        model_selector = at.sidebar.selectbox[2]
        assert model_selector.value == "mistral"


def test_default_provider_watsonx(mock_dual_provider_env):
    """Test that default_provider: watsonx works."""
    # We need to mock the models for watsonx specifically if it's selected
    # but the app calls list_models() on the selected provider.

    config_data = {
        "default_provider": "watsonx",
        "providers": {"watsonx": {"default_model": "meta-llama/llama-3-3-70b-instruct"}},
        "templates": {},
    }

    # We also need to mock WatsonxProvider.list_models to return the model we expect
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        patch("providers.watsonx_provider.WatsonxProvider.list_models") as mock_wx_models,
    ):
        mock_wx_models.return_value = ["meta-llama/llama-3-3-70b-instruct", "google/flan-t5-xl"]

        at = AppTest.from_file("app.py").run()

        # Find provider selectbox (second one in sidebar)
        provider_selector = at.sidebar.selectbox[1]
        assert provider_selector.value == "IBM watsonx"

        # Check if default model was also applied
        model_selector = at.sidebar.selectbox[2]
        assert model_selector.value == "meta-llama/llama-3-3-70b-instruct"


def test_fallback_when_default_provider_invalid(mock_dual_provider_env):
    """Test that it falls back to first provider if default_provider is invalid."""
    config_data = {"default_provider": "nonexistent", "providers": {}, "templates": {}}

    with patch("builtins.open", mock_open(read_data=json.dumps(config_data))):
        at = AppTest.from_file("app.py").run()
        provider_selector = at.sidebar.selectbox[1]
        # Should fall back to the first available (Ollama (Local))
        assert provider_selector.value == "Ollama (Local)"
