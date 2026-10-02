"""Tests for new LLM provider implementations (OpenRouter, Gemini)."""

import os
from unittest.mock import MagicMock, patch

from models import AppConfig, ChatMessage, GenerationOptions
from providers import GeminiProvider, OpenRouterProvider
from providers.settings import GeminiSettings, OpenRouterSettings


class TestOpenRouterProvider:
    """Test suite for OpenRouterProvider."""

    def test_initialization(self):
        """Test authentication checks."""
        # Without key
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenRouterProvider()
            assert provider.is_available() is False
            assert provider.get_name() == "OpenRouter"

        # With key
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            with patch("providers.openrouter_provider.OpenAI") as mock_openai:
                provider = OpenRouterProvider()
                assert provider.is_available() is True
                mock_openai.assert_called_once()

    def test_initialization_with_settings(self):
        """Test the API key from a settings object is unwrapped before reaching the client."""
        with patch("providers.openrouter_provider.OpenAI") as mock_openai:
            provider = OpenRouterProvider(OpenRouterSettings(api_key="sk-secret"))
            assert provider.is_available() is True
            assert mock_openai.call_args[1]["api_key"] == "sk-secret"

        with patch("providers.openrouter_provider.OpenAI"):
            disabled = OpenRouterProvider(OpenRouterSettings(api_key="sk-secret", enabled=False))
            assert disabled.is_available() is False

    def test_config_loading(self):
        """Test models and display names are read from config via load_app_config."""
        config = AppConfig.model_validate({"providers": {"openrouter": {"models": {"test/model": "Test Model"}}}})
        with patch("providers.openrouter_provider.OpenAI"):
            with patch("providers.openrouter_provider.load_app_config", return_value=config) as mock_load:
                provider = OpenRouterProvider(OpenRouterSettings(api_key="sk-test"))

                mock_load.assert_called_once()
                assert provider.list_models() == ["test/model"]
                assert provider.get_model_info("test/model").details.display_name == "Test Model"
                # Unknown models fall back to their id as display name
                assert provider.get_model_info("other/model").details.display_name == "other/model"

    def test_default_models_without_config(self):
        """Test the fallback model list is used when config has no OpenRouter models."""
        with patch("providers.openrouter_provider.OpenAI"):
            with patch("providers.openrouter_provider.load_app_config", return_value=AppConfig()):
                provider = OpenRouterProvider(OpenRouterSettings(api_key="sk-test"))
                assert "openai/gpt-4o-mini" in provider.list_models()

    def test_chat(self):
        """Test chat interaction."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            with patch("providers.openrouter_provider.OpenAI") as mock_openai:
                # Mock client instance
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                # Mock streaming response
                mock_chunk = MagicMock()
                mock_chunk.choices[0].delta.content = "Hello"
                mock_client.chat.completions.create.return_value = [mock_chunk]

                provider = OpenRouterProvider()
                messages = [ChatMessage(role="user", content="Hi", display="UI only")]
                response = list(provider.chat("model", messages))

                assert len(response) == 1
                assert response[0].message.content == "Hello"

                # Messages are sent as role/content payloads with default sampling options
                kwargs = mock_client.chat.completions.create.call_args[1]
                assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]
                assert kwargs["temperature"] == 0.7
                assert kwargs["top_p"] == 0.9

    def test_chat_with_options(self):
        """Test chat passes GenerationOptions through to the client."""
        with patch("providers.openrouter_provider.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.return_value = []

            provider = OpenRouterProvider(OpenRouterSettings(api_key="sk-test"))
            options = GenerationOptions(temperature=0.2, top_p=0.5)
            list(provider.chat("model", [ChatMessage(role="user", content="Hi")], options=options))

            kwargs = mock_client.chat.completions.create.call_args[1]
            assert kwargs["temperature"] == 0.2
            assert kwargs["top_p"] == 0.5


class TestGeminiProvider:
    """Test suite for GeminiProvider."""

    def test_initialization(self):
        """Test authentication checks."""
        # Without key
        with patch.dict(os.environ, {}, clear=True):
            provider = GeminiProvider()
            assert provider.is_available() is False
            assert provider.get_name() == "Google Gemini"

        # With key
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaTest"}):
            with patch("google.genai.Client"):
                provider = GeminiProvider()
                assert provider.is_available() is True
                # Just checking initialization happens

    def test_initialization_with_settings(self):
        """Test the API key from a settings object is unwrapped before reaching the client."""
        with patch("google.genai.Client") as mock_client_cls:
            provider = GeminiProvider(GeminiSettings(api_key="AIzaSecret"))
            assert provider.is_available() is True
            mock_client_cls.assert_called_once_with(api_key="AIzaSecret")

            disabled = GeminiProvider(GeminiSettings(api_key="AIzaSecret", enabled=False))
            assert disabled.is_available() is False

    def test_list_models(self):
        """Test model listing."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaTest"}):
            with patch("google.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client

                # Test Success
                mock_model1 = MagicMock()
                mock_model1.name = "models/gemini-pro"
                mock_model2 = MagicMock()
                mock_model2.name = "models/gemini-vision-pro"  # Should be filtered out

                mock_client.models.list.return_value = [mock_model1, mock_model2]

                provider = GeminiProvider()
                models = provider.list_models()
                assert "gemini-pro" in models
                assert "gemini-vision-pro" not in models

                # Test Fallback (Exception)
                mock_client.models.list.side_effect = Exception("API Error")
                models_fallback = provider.list_models()
                assert "gemini-2.0-flash" in models_fallback

    def test_chat(self):
        """Test chat interaction."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaTest"}):
            with patch("google.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client

                # Mock stream response (empty chunks are skipped)
                mock_chunk = MagicMock()
                mock_chunk.text = "Hello"
                mock_empty = MagicMock()
                mock_empty.text = None
                mock_client.models.generate_content_stream.return_value = [mock_chunk, mock_empty]

                provider = GeminiProvider()
                messages = [
                    ChatMessage(role="system", content="Be brief"),
                    ChatMessage(role="user", content="Hi"),
                    ChatMessage(role="assistant", content="Hello"),
                    ChatMessage(role="user", content="Again"),
                ]

                response = list(provider.chat("gemini-2.0-flash", messages))

                assert len(response) == 1
                assert response[0].message.content == "Hello"

                # System prompt moves to config; history maps assistant to "model"
                kwargs = mock_client.models.generate_content_stream.call_args[1]
                assert kwargs["config"].system_instruction == "Be brief"
                assert [c.role for c in kwargs["contents"]] == ["user", "model", "user"]
                # Gemini's own defaults apply when options is None
                assert kwargs["config"].temperature == 0.7
                assert kwargs["config"].top_p == 0.95

    def test_chat_with_options(self):
        """Test chat uses GenerationOptions values when provided."""
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content_stream.return_value = []

            provider = GeminiProvider(GeminiSettings(api_key="AIzaTest"))
            options = GenerationOptions(temperature=0.1, top_p=0.4)
            list(provider.chat("gemini-2.0-flash", [ChatMessage(role="user", content="Hi")], options=options))

            config = mock_client.models.generate_content_stream.call_args[1]["config"]
            assert config.temperature == 0.1
            assert config.top_p == 0.4
