"""Tests for the Ollama Streamlit app."""
# pylint: disable=redefined-outer-name, unused-argument, missing-function-docstring
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

# Mock the ollama library and watsonx provider
@pytest.fixture
def mock_app_env():
    with patch('ollama.list') as mock_list, \
         patch('ollama.chat') as mock_chat, \
         patch('providers.watsonx_provider.WatsonxProvider.is_available') as mock_wx_avail, \
         patch.dict('os.environ', {"OLLAMA_ENABLED": "true"}):

        # Setup default mock behavior
        mock_list.return_value = {'models': [{'model': 'llama3'}, {'model': 'mistral'}]}
        mock_wx_avail.return_value = False # Keep watsonx disabled by default for these tests

        # Mock chat to return a generator for streaming
        def stream_response(*args, **kwargs):
            yield {'message': {'content': 'This is a '}}
            yield {'message': {'content': 'mock response.'}}

        mock_chat.side_effect = stream_response

        yield mock_list, mock_chat

def test_app_starts_smoke_test(mock_app_env):
    """Test that the app starts up without errors."""
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    assert "Ollama Playground" in at.title[0].value

def test_sidebar_defaults(mock_app_env):
    """Test sidebar loads with defaults."""
    at = AppTest.from_file("app.py").run()

    # Check default mode is Chat
    assert at.sidebar.selectbox[0].value == "Chat"

    # Check provider selector exists
    assert at.sidebar.selectbox[1].value == "Ollama (Local)"

    # Check model selector is populated (mocked)
    # The third selectbox in sidebar is model selector
    assert at.sidebar.selectbox[2].options == ["llama3", "mistral"]

def test_switch_to_transformation_mode(mock_app_env):
    """Test switching modes changes the UI."""
    at = AppTest.from_file("app.py").run()

    # Change first selectbox (Mode) to Text Transformation
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Title or subheader should change/exist
    # Our app sets subheader "Text Transformation" in that mode
    assert "Text Transformation" in [h.body for h in at.subheader]

    # Should have a template selector and text area
    assert len(at.selectbox) >= 1 # Template selector in main area
    assert len(at.text_area) >= 2  # System Prompt (sidebar) + Input area (main)

def test_custom_template_load(mock_app_env):
    """Test that custom templates from filesystem are loaded."""
    at = AppTest.from_file("app.py").run()

    # Switch to Text Transformation
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Check that "Email" is in the options of the template selector
    # template selector is the first selectbox in main area (index 0)
    template_options = at.selectbox[0].options
    assert "Email" in template_options

def test_prompt_file_expansion(mock_app_env):
    """Test standard prompt file expansion @[path]."""
    _, mock_chat = mock_app_env
    at = AppTest.from_file("app.py").run()

    # Use Chat Mode
    at.sidebar.selectbox[0].select("Chat").run()

    # Input with file reference
    at.chat_input[0].set_value("Hello @[templates/email.txt]").run()

    # Verify mock called with expanded text
    mock_chat.assert_called()
    last_message = mock_chat.call_args[1]['messages'][-1]['content']
    assert "Hello" in last_message
    assert "professional email" in last_message # Content from email.txt
    assert "@[" not in last_message # Should be fully expanded

def test_transformation_execution(mock_app_env):
    """Test running a transformation."""
    _, mock_chat = mock_app_env

    at = AppTest.from_file("app.py").run()

    # Switch to Transformation mode
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Select a template
    at.selectbox[0].select("Summarize").run()

    # Enter text (Find correct text area by label)
    txt_area = next(t for t in at.text_area if t.label == "Enter text to transform:")
    txt_area.input("Execute this text").run()

    # Click Transform (Find button by label)
    btn_transform = next(b for b in at.button if b.label == "Transform")
    btn_transform.click().run()

    # Verify mock was called
    mock_chat.assert_called()
    call_args = mock_chat.call_args[1]
    assert call_args['model'] == 'llama3' # Default first model
    assert "Summarize" in call_args['messages'][0]['content']
    assert "Execute this text" in call_args['messages'][0]['content']

    # Verify output
    assert "This is a mock response." in at.markdown[1].value # Result markdown

def test_chat_execution(mock_app_env):
    """Test sending a chat message."""
    _, mock_chat = mock_app_env

    at = AppTest.from_file("app.py").run()

    # Verify we are in chat mode (default)
    # Ensure a model is selected (mock default is index 0)

    # Input chat message
    at.chat_input[0].set_value("Hello").run()

    # Verify mock called
    mock_chat.assert_called()
    assert mock_chat.call_args[1]['messages'][-1]['content'] == "Hello"

    # Check message history in session state
    assert len(at.session_state["messages"]) == 2 # User + Assistant
    assert at.session_state["messages"][0]["role"] == "user"
    assert at.session_state["messages"][1]["role"] == "assistant"

def test_reset_functionality(mock_app_env):
    """Test that resetting clears text boxes and uploader keys."""
    at = AppTest.from_file("app.py").run()

    # Chat Mode Reset
    # Set system prompt
    sp_area = next(t for t in at.text_area if t.label == "System Prompt")
    sp_area.input("System prompt content").run()
    assert at.session_state["system_prompt_input"] == "System prompt content"

    # Click Reset
    btn_reset = next(b for b in at.button if b.label == "🗑️ Reset")
    btn_reset.click().run()
    assert at.session_state["system_prompt_input"] == ""
    assert at.session_state["uploader_key"] == 1

    # Switch to Text Transformation
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Set transformation text
    tr_area = next(t for t in at.text_area if t.label == "Enter text to transform:")
    tr_area.input("Text to transform").run()
    assert at.session_state["transformation_text"] == "Text to transform"

    # Click Reset
    btn_reset_tr = next(b for b in at.button if b.label == "🗑️ Reset")
    btn_reset_tr.click().run()
    assert at.session_state["transformation_text"] == ""
    assert at.session_state["uploader_key"] == 2
