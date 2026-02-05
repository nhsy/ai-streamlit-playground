"""Shared test fixtures for the AI Streamlit Playground."""
from unittest.mock import patch
import pytest

@pytest.fixture
def mock_app_env():
    """Mock the app environment for testing."""
    with patch('ollama.list') as mock_list, \
         patch('ollama.chat') as mock_chat, \
         patch('providers.watsonx_provider.WatsonxProvider.is_available') as mock_wx_avail, \
         patch('providers.gemini_provider.GeminiProvider.is_available') as mock_gemini_avail, \
         patch('providers.openrouter_provider.OpenRouterProvider.is_available') as mock_or_avail, \
         patch('app.load_config') as mock_load_config, \
         patch.dict('os.environ', {"OLLAMA_ENABLED": "true"}):

        # Setup default mock behavior
        mock_list.return_value = {'models': [{'model': 'llama3'}, {'model': 'mistral'}]}
        mock_wx_avail.return_value = False
        mock_gemini_avail.return_value = False
        mock_or_avail.return_value = False

        # Mock load_config to return stable defaults for testing
        mock_load_config.return_value = {
            "default_provider": "ollama",
            "providers": {
                "ollama": {"default_model": "llama3"}
            },
            "templates": {
                "Summarize": "Summarize the following text:",
                "Fix Grammar": "Fix grammar:",
                "Rewrite Professionally": "Rewrite:",
                "Email": "Write a professional email based on:",
            }
        }

        # Mock chat to return a generator for streaming
        def stream_response(*_args, **_kwargs):
            yield {'message': {'content': 'This is a '}}
            yield {'message': {'content': 'mock response.'}}

        mock_chat.side_effect = stream_response

        yield mock_list, mock_chat
