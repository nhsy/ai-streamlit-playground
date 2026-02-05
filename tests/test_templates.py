"""Template integration tests for the AI Streamlit Playground."""
# pylint: disable=redefined-outer-name, unused-argument, missing-function-docstring
from streamlit.testing.v1 import AppTest

# Mock dependencies to avoid actual API calls (reusing logic from test_app.py)
# The mock_app_env fixture is now in conftest.py

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
        "Rewrite Professionally"
    ]

    for template in expected_templates:
        assert template in options

def test_template_text_correctness(mock_app_env):
    """Verify that selecting a JSON template loads the correct prompt text."""
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].select("Text Transformation").run()

    # Select specific template
    target_template = "Summarize"
    at.selectbox[0].select(target_template).run()

    # In the app, the text is not directly shown in a widget until we transform,
    # OR we can inspect the app's internal 'templates' variable if we could access it,
    # but AppTest is black-box.
    # However, we can run a transformation and check the prompt sent to the model.

    # Enter text (Find correct text area by label)
    txt_area = next(t for t in at.text_area if t.label == "Enter text to transform:")
    txt_area.input("Hello world").run()

    # Click Transform (Find button by label)
    btn_transform = next(b for b in at.button if b.label == "Transform")
    btn_transform.click().run()

    # Verify mock call arguments
    _, mock_chat = mock_app_env
    call_args = mock_chat.call_args[1]
    sent_content = call_args['messages'][0]['content']

    # The prompt should contain the template text and the user text
    assert "Summarize the following text:" in sent_content
    assert "Hello world" in sent_content
