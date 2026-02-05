"""Tests for new LLM provider implementations (OpenRouter, Gemini)."""
import os
from unittest.mock import patch, MagicMock
from providers import OpenRouterProvider, GeminiProvider

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
            with patch('providers.openrouter_provider.OpenAI') as mock_openai:
                provider = OpenRouterProvider()
                assert provider.is_available() is True
                mock_openai.assert_called_once()

    def test_config_loading(self):
        """Test loading models from config.json."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            with patch('providers.openrouter_provider.OpenAI'):
                with patch('builtins.open', new_callable=MagicMock) as mock_open:
                    with patch('json.load') as mock_json:
                        mock_json.return_value = {
                            "providers": {
                                "openrouter": {
                                    "models": {
                                        "test/model": "Test Model"
                                    }
                                }
                            }
                        }
                        # Setup mock file context manager
                        mock_file = MagicMock()
                        mock_open.return_value.__enter__.return_value = mock_file

                        with patch('os.path.exists', return_value=True):
                            provider = OpenRouterProvider()
                            assert "test/model" in provider.list_models()
                            info = provider.get_model_info("test/model")
                            assert info['details']['display_name'] == "Test Model"

    def test_chat(self):
        """Test chat interaction."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            with patch('providers.openrouter_provider.OpenAI') as mock_openai:
                # Mock client instance
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                # Mock streaming response
                mock_chunk = MagicMock()
                mock_chunk.choices[0].delta.content = "Hello"
                mock_client.chat.completions.create.return_value = [mock_chunk]

                provider = OpenRouterProvider()
                messages = [{"role": "user", "content": "Hi"}]
                response = list(provider.chat("model", messages))

                assert len(response) == 1
                assert response[0]['message']['content'] == "Hello"


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
            with patch('google.genai.Client'):
                provider = GeminiProvider()
                assert provider.is_available() is True
                # Just checking initialization happens

    def test_list_models(self):
        """Test model listing."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaTest"}):
            with patch('google.genai.Client') as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client

                # Test Success
                mock_model1 = MagicMock()
                mock_model1.name = "models/gemini-pro"
                mock_model2 = MagicMock()
                mock_model2.name = "models/gemini-vision-pro" # Should be filtered out

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
            with patch('google.genai.Client') as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client

                # Mock stream response
                mock_chunk = MagicMock()
                mock_chunk.text = "Hello"
                mock_client.models.generate_content_stream.return_value = [mock_chunk]

                provider = GeminiProvider()
                messages = [{"role": "user", "content": "Hi"}]

                response = list(provider.chat("gemini-2.0-flash", messages))

                assert len(response) == 1
                assert response[0]['message']['content'] == "Hello"
