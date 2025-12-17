"""Tests for LLM provider implementations."""
import os
from unittest.mock import patch, MagicMock
import pytest
from providers import OllamaProvider, WatsonxProvider


class TestOllamaProvider:
    """Test suite for OllamaProvider."""

    def test_initialization(self):
        """Test that OllamaProvider initializes correctly."""
        provider = OllamaProvider()
        assert provider.get_name() == "Ollama (Local)"

    @patch('providers.ollama_provider.ollama.list')
    def test_is_available_when_ollama_running(self, mock_list):
        """Test is_available returns True when Ollama is running."""
        mock_list.return_value = {'models': []}
        with patch.dict(os.environ, {"OLLAMA_ENABLED": "true"}):
            provider = OllamaProvider()
            assert provider.is_available() is True

    @patch('providers.ollama_provider.ollama.list')
    def test_is_available_when_ollama_not_running(self, mock_list):
        """Test is_available returns False when Ollama is not running."""
        mock_list.side_effect = Exception("Connection refused")
        with patch.dict(os.environ, {"OLLAMA_ENABLED": "true"}):
            provider = OllamaProvider()
            assert provider.is_available() is False

    @patch('providers.ollama_provider.ollama.list')
    def test_list_models(self, mock_list):
        """Test list_models returns correct model names."""
        mock_list.return_value = {
            'models': [
                {'model': 'llama2:latest'},
                {'model': 'mistral:latest'}
            ]
        }
        provider = OllamaProvider()
        models = provider.list_models()
        assert models == ['llama2:latest', 'mistral:latest']

    @patch('providers.ollama_provider.ollama.chat')
    def test_chat_streaming(self, mock_chat):
        """Test chat method with streaming."""
        # Mock streaming response
        mock_chat.return_value = [
            {'message': {'content': 'Hello'}},
            {'message': {'content': ' world'}}
        ]

        provider = OllamaProvider()
        messages = [{'role': 'user', 'content': 'Hi'}]
        options = {'temperature': 0.7, 'top_p': 0.9}

        response = list(provider.chat(
            model='llama2:latest',
            messages=messages,
            stream=True,
            options=options
        ))

        assert len(response) == 2
        assert response[0]['message']['content'] == 'Hello'
        assert response[1]['message']['content'] == ' world'

        # Verify ollama.chat was called with correct parameters
        mock_chat.assert_called_once_with(
            model='llama2:latest',
            messages=messages,
            stream=True,
            options=options
        )


class TestWatsonxProvider:
    """Test suite for WatsonxProvider."""

    def test_initialization_without_credentials(self):
        """Test WatsonxProvider initialization without credentials."""
        with patch.dict(os.environ, {}, clear=True):
            provider = WatsonxProvider()
            assert provider.get_name() == "IBM watsonx"
            assert provider.is_available() is False

    def test_initialization_with_credentials(self):
        """Test WatsonxProvider initialization with credentials."""
        # Mock the ibm_watsonx_ai module with foundation_models submodule
        mock_credentials = MagicMock()
        mock_model_inference = MagicMock()
        mock_foundation_models = MagicMock()
        mock_foundation_models.ModelInference = mock_model_inference

        mock_ibm_module = MagicMock()
        mock_ibm_module.Credentials = mock_credentials
        mock_ibm_module.foundation_models = mock_foundation_models

        with patch.dict('sys.modules', {
            'ibm_watsonx_ai': mock_ibm_module,
            'ibm_watsonx_ai.foundation_models': mock_foundation_models
        }):
            with patch.dict(os.environ, {
                'WATSONX_API_KEY': 'test-key',
                'WATSONX_PROJECT_ID': 'test-project'
            }):
                provider = WatsonxProvider()
                assert provider.is_available() is True

    def test_default_url_is_uk(self):
        """Test that default URL is UK region."""
        mock_credentials = MagicMock()
        with patch.dict('sys.modules', {'ibm_watsonx_ai': MagicMock(Credentials=mock_credentials)}):
            with patch.dict(os.environ, {
                'WATSONX_API_KEY': 'test-key',
                'WATSONX_PROJECT_ID': 'test-project'
            }):
                provider = WatsonxProvider()
                # pylint: disable=protected-access
                assert provider._url == "https://eu-gb.ml.cloud.ibm.com"

    def test_custom_url(self):
        """Test custom URL from environment variable."""
        mock_credentials = MagicMock()
        with patch.dict('sys.modules', {'ibm_watsonx_ai': MagicMock(Credentials=mock_credentials)}):
            with patch.dict(os.environ, {
                'WATSONX_API_KEY': 'test-key',
                'WATSONX_PROJECT_ID': 'test-project',
                'WATSONX_URL': 'https://us-south.ml.cloud.ibm.com'
            }):
                provider = WatsonxProvider()
                # pylint: disable=protected-access
                assert provider._url == "https://us-south.ml.cloud.ibm.com"

    def test_is_available_without_api_key(self):
        """Test is_available returns False without API key."""
        with patch.dict(os.environ, {
            'WATSONX_PROJECT_ID': 'test-project'
        }, clear=True):
            provider = WatsonxProvider()
            assert provider.is_available() is False

    def test_is_available_without_project_id(self):
        """Test is_available returns False without project ID."""
        with patch.dict(os.environ, {
            'WATSONX_API_KEY': 'test-key'
        }, clear=True):
            provider = WatsonxProvider()
            assert provider.is_available() is False

    def test_list_models(self):
        """Test list_models returns expected models."""
        mock_credentials = MagicMock()
        mock_model_inference = MagicMock()
        mock_foundation_models = MagicMock()
        mock_foundation_models.ModelInference = mock_model_inference

        # Mock APIClient and its foundation_models.get_entities
        mock_api_client = MagicMock()
        mock_api_client.foundation_models.get_entities.return_value = [
            {'model_id': 'ibm/granite-13b-chat-v2'},
            {'model_id': 'meta-llama/llama-3-70b-instruct'}
        ]

        mock_ibm_module = MagicMock()
        mock_ibm_module.Credentials = mock_credentials
        mock_ibm_module.foundation_models = mock_foundation_models
        mock_ibm_module.APIClient = MagicMock(return_value=mock_api_client)

        with patch.dict('sys.modules', {
            'ibm_watsonx_ai': mock_ibm_module,
            'ibm_watsonx_ai.foundation_models': mock_foundation_models
        }):
            with patch.dict(os.environ, {
                'WATSONX_API_KEY': 'test-key',
                'WATSONX_PROJECT_ID': 'test-project'
            }):
                provider = WatsonxProvider()
                models = provider.list_models()

                # Check that common models are in the list
                assert 'ibm/granite-13b-chat-v2' in models
                assert 'meta-llama/llama-3-70b-instruct' in models
                assert len(models) > 0

    def test_list_models_without_credentials(self):
        """Test list_models raises exception without credentials."""
        with patch.dict(os.environ, {}, clear=True):
            provider = WatsonxProvider()
            with pytest.raises(RuntimeError, match="watsonx credentials not configured"):
                provider.list_models()

    def test_messages_to_prompt_conversion(self):
        """Test conversion of messages to prompt format."""
        mock_credentials = MagicMock()
        with patch.dict('sys.modules', {'ibm_watsonx_ai': MagicMock(Credentials=mock_credentials)}):
            with patch.dict(os.environ, {
                'WATSONX_API_KEY': 'test-key',
                'WATSONX_PROJECT_ID': 'test-project'
            }):
                provider = WatsonxProvider()

                messages = [
                    {'role': 'system', 'content': 'You are helpful'},
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hi there'},
                    {'role': 'user', 'content': 'How are you?'}
                ]

                # pylint: disable=protected-access
                prompt = provider._messages_to_prompt(messages)

                assert 'System: You are helpful' in prompt
                assert 'User: Hello' in prompt
                assert 'Assistant: Hi there' in prompt
                assert 'User: How are you?' in prompt
                assert prompt.endswith('Assistant:')

    def test_chat_streaming(self):
        """Test chat method with streaming."""
        # Mock the entire ibm_watsonx_ai module
        mock_model = MagicMock()
        mock_model.generate_text_stream.return_value = ['Hello', ' world', '!']

        mock_model_inference = MagicMock(return_value=mock_model)
        mock_credentials = MagicMock()
        mock_gen_params = MagicMock()
        mock_gen_params.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_params.TEMPERATURE = 'temperature'
        mock_gen_params.TOP_P = 'top_p'

        mock_ibm_module = MagicMock()
        mock_ibm_module.Credentials = mock_credentials
        mock_ibm_module.foundation_models.ModelInference = mock_model_inference
        mock_ibm_module.metanames.GenTextParamsMetaNames = mock_gen_params

        with patch.dict('sys.modules', {
            'ibm_watsonx_ai': mock_ibm_module,
            'ibm_watsonx_ai.foundation_models': mock_ibm_module.foundation_models,
            'ibm_watsonx_ai.metanames': mock_ibm_module.metanames
        }):
            with patch.dict(os.environ, {
                'WATSONX_API_KEY': 'test-key',
                'WATSONX_PROJECT_ID': 'test-project'
            }):
                provider = WatsonxProvider()
                messages = [{'role': 'user', 'content': 'Hi'}]

                response = list(provider.chat(
                    model='ibm/granite-13b-chat-v2',
                    messages=messages,
                    stream=True,
                    options={'temperature': 0.7, 'top_p': 0.9}
                ))

                # Check response format
                assert len(response) == 3
                assert response[0]['message']['content'] == 'Hello'
                assert response[1]['message']['content'] == ' world'
                assert response[2]['message']['content'] == '!'

    def test_chat_non_streaming(self):
        """Test chat method without streaming."""
        # Mock the entire ibm_watsonx_ai module
        mock_model = MagicMock()
        mock_model.generate_text.return_value = 'Hello world!'

        mock_model_inference = MagicMock(return_value=mock_model)
        mock_credentials = MagicMock()
        mock_gen_params = MagicMock()
        mock_gen_params.MAX_NEW_TOKENS = 'max_new_tokens'
        mock_gen_params.TEMPERATURE = 'temperature'
        mock_gen_params.TOP_P = 'top_p'

        mock_ibm_module = MagicMock()
        mock_ibm_module.Credentials = mock_credentials
        mock_ibm_module.foundation_models.ModelInference = mock_model_inference
        mock_ibm_module.metanames.GenTextParamsMetaNames = mock_gen_params

        with patch.dict('sys.modules', {
            'ibm_watsonx_ai': mock_ibm_module,
            'ibm_watsonx_ai.foundation_models': mock_ibm_module.foundation_models,
            'ibm_watsonx_ai.metanames': mock_ibm_module.metanames
        }):
            with patch.dict(os.environ, {
                'WATSONX_API_KEY': 'test-key',
                'WATSONX_PROJECT_ID': 'test-project'
            }):
                provider = WatsonxProvider()
                messages = [{'role': 'user', 'content': 'Hi'}]

                response = list(provider.chat(
                    model='ibm/granite-13b-chat-v2',
                    messages=messages,
                    stream=False,
                    options={'temperature': 0.7}
                ))

                # Check response format
                assert len(response) == 1
                assert response[0]['message']['content'] == 'Hello world!'

    def test_chat_without_credentials(self):
        """Test chat raises exception without credentials."""
        with patch.dict(os.environ, {}, clear=True):
            provider = WatsonxProvider()
            messages = [{'role': 'user', 'content': 'Hi'}]

            with pytest.raises(RuntimeError, match="watsonx credentials not configured"):
                list(provider.chat(
                    model='ibm/granite-13b-chat-v2',
                    messages=messages,
                    stream=True
                ))


class TestProviderInterface:
    """Test that both providers implement the required interface."""

    def test_ollama_implements_interface(self):
        """Test OllamaProvider implements all required methods."""
        provider = OllamaProvider()
        assert hasattr(provider, 'is_available')
        assert hasattr(provider, 'list_models')
        assert hasattr(provider, 'chat')
        assert hasattr(provider, 'get_name')
        assert callable(provider.is_available)
        assert callable(provider.list_models)
        assert callable(provider.chat)
        assert callable(provider.get_name)

    def test_watsonx_implements_interface(self):
        """Test WatsonxProvider implements all required methods."""
        provider = WatsonxProvider()
        assert hasattr(provider, 'is_available')
        assert hasattr(provider, 'list_models')
        assert hasattr(provider, 'chat')
        assert hasattr(provider, 'get_name')
        assert callable(provider.is_available)
        assert callable(provider.list_models)
        assert callable(provider.chat)
        assert callable(provider.get_name)
