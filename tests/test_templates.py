"""Tests for the JSON template configuration."""
# pylint: disable=redefined-outer-name, unused-argument, missing-function-docstring
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

# Mock dependencies to avoid actual API calls (reusing logic from test_app.py)
@pytest.fixture
def mock_app_env():
    """Mock the app environment."""
    with patch('ollama.list') as mock_list, \
         patch('ollama.chat') as mock_chat, \
         patch('providers.watsonx_provider.WatsonxProvider.is_available') as mock_wx_avail, \
         patch.dict('os.environ', {"OLLAMA_ENABLED": "true"}):
        mock_list.return_value = {'models': [{'model': 'llama3'}]}
        mock_wx_avail.return_value = False
        mock_chat.return_value = iter([]) # Empty stream
        yield mock_list, mock_chat

def test_json_config_templates_loaded(mock_app_env):
    """Verify that templates defined in template_config.json are available."""
    at = AppTest.from_file("app.py").run()

    # Switch to Text Transformation mode
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Get options from the template selector
    options = at.selectbox[0].options

    # Assert keys from our known json file exist
    expected_templates = [
        "Summarize",
        "Fix Grammar",
        "Rewrite Professionally",
        "Explain to a 5-year old"
    ]

    for template in expected_templates:
        assert template in options

def test_template_text_correctness(mock_app_env):
    """Verify that selecting a JSON template loads the correct prompt text."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Select specific template
    target_template = "Translate to Spanish"
    at.selectbox[0].select(target_template).run()

    # In the app, the text is not directly shown in a widget until we transform,
    # OR we can inspect the app's internal 'templates' variable if we could access it,
    # but AppTest is black-box.
    # However, we can run a transformation and check the prompt sent to the model.

    at.text_area[0].input("Hello world").run()
    at.button[0].click().run()

    # Verify mock call arguments
    _, mock_chat = mock_app_env
    call_args = mock_chat.call_args[1]
    sent_content = call_args['messages'][0]['content']

    # The prompt should contain the template text and the user text
    assert "Translate the following text to Spanish:" in sent_content
    assert "Hello world" in sent_content
