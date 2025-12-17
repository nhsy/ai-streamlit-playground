"""Tests for the Ollama Streamlit app."""
# pylint: disable=redefined-outer-name, unused-argument, missing-function-docstring
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

# Mock the ollama library at the top level so it doesn't try to connect during import/execution
@pytest.fixture
def mock_ollama():
    with patch('ollama.list') as mock_list, \
         patch('ollama.chat') as mock_chat:

        # Setup default mock behavior
        mock_list.return_value = {'models': [{'model': 'llama3'}, {'model': 'mistral'}]}

        # Mock chat to return a generator for streaming
        def stream_response(*args, **kwargs):
            yield {'message': {'content': 'This is a '}}
            yield {'message': {'content': 'mock response.'}}

        mock_chat.side_effect = stream_response

        yield mock_list, mock_chat

def test_app_starts_smoke_test(mock_ollama):
    """Test that the app starts up without errors."""
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    assert "Ollama Playground" in at.title[0].value

def test_sidebar_defaults(mock_ollama):
    """Test sidebar loads with defaults."""
    at = AppTest.from_file("app.py").run()

    # Check default mode is Chat
    # Note: Streamlit test API for selectbox might need index or value check
    # We can check specific session state or UI presence
    assert at.sidebar.selectbox[0].value == "Chat"

    # Check model selector is populated (mocked)
    # The second selectbox in sidebar is model selector
    assert at.sidebar.selectbox[1].options == ["llama3", "mistral"]

def test_switch_to_transformation_mode(mock_ollama):
    """Test switching modes changes the UI."""
    at = AppTest.from_file("app.py").run()

    # Change first selectbox (Mode) to Text Transformation
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Title or subheader should change/exist
    # Our app sets subheader "Text Transformation" in that mode
    assert "Text Transformation" in [h.body for h in at.subheader]

    # Should have a template selector and text area
    assert len(at.selectbox) >= 1 # Template selector in main area
    assert len(at.text_area) >= 1  # Input area

def test_custom_template_load(mock_ollama):
    """Test that custom templates from filesystem are loaded."""
    at = AppTest.from_file("app.py").run()

    # Switch to Text Transformation
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Check that "Email" is in the options of the template selector
    # template selector is the first selectbox in main area (index 0)
    template_options = at.selectbox[0].options
    assert "Email" in template_options

def test_prompt_file_expansion(mock_ollama):
    """Test standard prompt file expansion @[path]."""
    _, mock_chat = mock_ollama
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

def test_transformation_execution(mock_ollama):
    """Test running a transformation."""
    _, mock_chat = mock_ollama

    at = AppTest.from_file("app.py").run()

    # Switch to Transformation mode
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Select a template
    at.selectbox[0].select("Summarize").run()

    # Enter text
    at.text_area[0].input("Execute this text").run()

    # Click Transform
    at.button[0].click().run()

    # Verify mock was called
    mock_chat.assert_called()
    call_args = mock_chat.call_args[1]
    assert call_args['model'] == 'llama3' # Default first model
    assert "Summarize" in call_args['messages'][0]['content']
    assert "Execute this text" in call_args['messages'][0]['content']

    # Verify output
    assert "This is a mock response." in at.markdown[1].value # Result markdown

def test_chat_execution(mock_ollama):
    """Test sending a chat message."""
    _, mock_chat = mock_ollama

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
