"""UI tests for the AI Streamlit Playground."""

# pylint: disable=redefined-outer-name, unused-argument, missing-function-docstring
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

# The mock_app_env fixture is in conftest.py

TRANSFORM_PAGE = "views/transform.py"


def open_transform(at):
    return at.switch_page(TRANSFORM_PAGE).run()


def test_app_starts_smoke_test(mock_app_env):
    """Test that the app starts on the Chat page without errors."""
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    assert at.title[0].value == "Chat"


def test_sidebar_defaults(mock_app_env):
    """Test sidebar loads with defaults."""
    at = AppTest.from_file("app.py").run()

    assert at.sidebar.selectbox[0].value == "ollama"
    assert at.sidebar.selectbox[1].options == ["llama3", "mistral"]


def test_switch_to_transformation_page(mock_app_env):
    """Test navigating to the transformation page."""
    at = open_transform(AppTest.from_file("app.py").run())

    assert not at.exception
    assert at.title[0].value == "Text Transformation"
    assert len(at.selectbox) >= 1  # Template selector in main area
    assert len(at.text_area) >= 2  # System Prompt (sidebar) + Input area (main)


def test_custom_template_load(mock_app_env):
    """Test that custom templates are listed."""
    at = open_transform(AppTest.from_file("app.py").run())
    assert "Email" in at.selectbox[0].options


def test_prompt_file_expansion(mock_app_env):
    """Test standard prompt file expansion @[path]."""
    _, mock_chat = mock_app_env
    at = AppTest.from_file("app.py").run()

    at.chat_input[0].set_value("Hello @[templates/email.txt]").run()

    mock_chat.assert_called()
    last_message = mock_chat.call_args[1]["messages"][-1]["content"]
    assert "Hello" in last_message
    assert "professional email" in last_message  # Content from email.txt
    assert "@[" not in last_message


def test_prompt_file_expansion_refuses_dotfiles(mock_app_env):
    """@[.env] must never be sent to a provider."""
    _, mock_chat = mock_app_env
    at = AppTest.from_file("app.py").run()

    at.chat_input[0].set_value("Leak @[.env]").run()

    last_message = mock_chat.call_args[1]["messages"][-1]["content"]
    assert "[Access denied: .env]" in last_message


def test_transformation_execution(mock_app_env):
    """Test running a transformation streams the result."""
    _, mock_chat = mock_app_env
    at = open_transform(AppTest.from_file("app.py").run())

    at.selectbox[0].select("Summarize").run()
    txt_area = next(t for t in at.text_area if t.label == "Enter text to transform:")
    txt_area.input("Execute this text").run()
    next(b for b in at.button if b.label == "Transform").click().run()

    mock_chat.assert_called()
    call_args = mock_chat.call_args[1]
    assert call_args["model"] == "llama3"  # Default model from config
    assert "Summarize" in call_args["messages"][0]["content"]
    assert "Execute this text" in call_args["messages"][0]["content"]
    assert any("This is a mock response." in m.value for m in at.markdown)


def test_chat_execution(mock_app_env):
    """Test sending a chat message."""
    _, mock_chat = mock_app_env
    at = AppTest.from_file("app.py").run()

    at.chat_input[0].set_value("Hello").run()

    mock_chat.assert_called()
    assert mock_chat.call_args[1]["messages"][-1]["content"] == "Hello"
    assert len(at.session_state["messages"]) == 2  # User + Assistant
    assert at.session_state["messages"][0]["role"] == "user"
    assert at.session_state["messages"][1] == {"role": "assistant", "content": "This is a mock response."}


def test_chat_history_keeps_expanded_content(mock_app_env):
    """Follow-up turns resend the expanded content of earlier turns, not the bare prompt."""
    _, mock_chat = mock_app_env
    at = AppTest.from_file("app.py").run()

    at.chat_input[0].set_value("Read @[templates/email.txt]").run()
    at.chat_input[0].set_value("And again?").run()

    first_turn = mock_chat.call_args[1]["messages"][0]["content"]
    assert "professional email" in first_turn


def test_chat_error_not_added_to_history(mock_app_env):
    """A provider error shows a message and leaves the history unchanged."""
    _, mock_chat = mock_app_env
    mock_chat.side_effect = RuntimeError("boom")
    at = AppTest.from_file("app.py").run()

    at.chat_input[0].set_value("Hello").run()

    assert at.session_state["messages"] == []
    assert any("boom" in e.value for e in at.error)


def test_reset_functionality(mock_app_env):
    """Test that resetting clears the system prompt and transformation text."""
    at = AppTest.from_file("app.py").run()

    sp_area = next(t for t in at.text_area if t.label == "System Prompt")
    sp_area.input("System prompt content").run()
    assert at.session_state["system_prompt_input"] == "System prompt content"

    next(b for b in at.button if b.label == "Reset").click().run()
    assert at.session_state["system_prompt_input"] == ""
    assert at.session_state["reset_key"] == 1

    at = open_transform(at)
    tr_area = next(t for t in at.text_area if t.label == "Enter text to transform:")
    tr_area.input("Text to transform").run()
    assert at.session_state["transformation_text"] == "Text to transform"

    next(b for b in at.button if b.label == "Reset").click().run()
    assert at.session_state["transformation_text"] == ""
    assert at.session_state["reset_key"] == 2


def test_no_providers_shows_setup_help(mock_app_env):
    """With no provider available the sidebar explains how to configure one."""
    mock_list, _ = mock_app_env
    mock_list.side_effect = ConnectionError("down")
    at = AppTest.from_file("app.py").run()

    assert not at.exception
    assert at.sidebar.error[0].value == "No providers available."
    assert any("GEMINI_API_KEY" in i.value for i in at.sidebar.info)


def test_pull_model(mock_app_env):
    """Pulling a library model streams progress from Ollama."""
    with patch("ollama.pull") as mock_pull:
        mock_pull.return_value = iter([{"status": "downloading", "completed": 5, "total": 10}, {"status": "success"}])
        at = AppTest.from_file("app.py").run()
        next(b for b in at.button if b.label == "Pull model").click().run()

    mock_pull.assert_called_once_with("llama3.2:latest", stream=True)
    assert not at.exception


def test_pull_model_error(mock_app_env):
    """A failed pull is reported in the status box."""
    with patch("ollama.pull", side_effect=RuntimeError("no such model")):
        at = AppTest.from_file("app.py").run()
        next(b for b in at.button if b.label == "Pull model").click().run()

    assert any("no such model" in s.label for s in at.status)
