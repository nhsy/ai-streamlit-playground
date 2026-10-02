"""
AI Streamlit Playground
A Streamlit app to chat with and transform text using Ollama, watsonx, OpenRouter and Gemini models.
"""

import os

import streamlit as st
from dotenv import load_dotenv

import core

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="AI Streamlit Playground", page_icon=":material/neurology:", layout="wide")

if "reset_key" not in st.session_state:
    st.session_state["reset_key"] = 0
if "system_prompt_input" not in st.session_state:
    st.session_state["system_prompt_input"] = ""


def show_setup_help():
    """Explain how to enable a provider when none is available."""
    st.error("No providers available.")
    if os.getenv("OLLAMA_ENABLED", "true").lower() != "true":
        st.info("**Ollama**: Disabled via OLLAMA_ENABLED environment variable.")
    else:
        st.info("**Ollama**: Make sure Ollama is running locally and OLLAMA_ENABLED is not set to 'false'.")
    st.info("**watsonx**: Set WATSONX_API_KEY and WATSONX_PROJECT_ID in .env file.")
    st.info("**OpenRouter**: Set OPENROUTER_API_KEY in .env file.")
    st.info("**Google Gemini**: Set GEMINI_API_KEY in .env file.")


def pull_model_ui(provider):
    """Download a model from the Ollama library with live progress."""
    with st.expander("Pull new model", icon=":material/download:"):
        library_models = [
            "llama3.2:latest (3B)",
            "llama3.1:8b",
            "mistral-nemo:latest (12B)",
            "gemma2:9b",
            "phi3:medium (14B)",
            "qwen2.5:7b",
            "moondream:latest (Vision)",
            "Other (Enter name...)",
        ]
        selection = st.selectbox("Download from library", library_models)

        if selection == "Other (Enter name...)":
            pull_target = st.text_input(
                "Enter model name", placeholder="e.g., llama3, mistral", key="pull_model_custom"
            ).strip()
        else:
            # "llama3.2:latest (3B)" -> "llama3.2:latest"
            pull_target = selection.split(" ")[0]

        if not st.button("Pull model", width="stretch"):
            return
        if not pull_target:
            st.warning("Please specify a model name.")
            return

        with st.status(f"Pulling {pull_target}…", expanded=True) as status:
            progress_bar = st.progress(0.0)
            try:
                for progress in provider.pull_model(pull_target):
                    completed, total = progress.get("completed"), progress.get("total")
                    if completed and total:
                        progress_bar.progress(completed / total, text=progress.get("status", ""))
            except Exception as e:  # pylint: disable=broad-exception-caught
                status.update(label=f"Error pulling model: {e}", state="error")
                return
            status.update(label=f"Pulled {pull_target}", state="complete", expanded=False)
        core.list_models.clear()
        st.rerun()


def select_model(provider_key, provider, config):
    """Model picker for the chosen provider, preselecting the configured default."""
    try:
        model_names = core.list_models(provider_key)
    except Exception as e:  # pylint: disable=broad-exception-caught
        st.error(f"Error loading models: {e}")
        return None

    if not model_names:
        st.warning(f"No models found for {core.PROVIDERS[provider_key][0]}.")
        return None

    default_model = config.get("providers", {}).get(provider_key, {}).get("default_model") or config.get(
        "default_model"
    )
    default_index = model_names.index(default_model) if default_model in model_names else 0
    widget_key = f"selected_model_{provider_key}"
    current = st.session_state.get(widget_key) or model_names[default_index]

    return st.selectbox(
        "Model",
        model_names,
        index=default_index,
        help=core.model_help(provider, current),
        key=widget_key,
    )


def sidebar_settings():
    """Render the sidebar and return the selections, or stop the run if nothing is usable."""
    st.header("Settings")

    provider_keys = core.available_provider_keys()
    if not provider_keys:
        show_setup_help()
        st.stop()

    config = core.get_config()
    default_provider = config.get("default_provider", "")
    provider_key = st.selectbox(
        "Provider",
        provider_keys,
        index=provider_keys.index(default_provider) if default_provider in provider_keys else 0,
        format_func=lambda key: core.PROVIDERS[key][0],
        help="Local Ollama or a cloud provider configured in .env",
    )
    provider = core.get_provider(provider_key)
    label = core.PROVIDERS[provider_key][0]

    model = select_model(provider_key, provider, config)

    if provider_key == "ollama":
        pull_model_ui(provider)

    with st.expander("Parameters", icon=":material/tune:"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness: higher values make outputs more random, lower values more deterministic.",
        )
        top_p = st.slider(
            "Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1, help="Controls diversity via nucleus sampling."
        )

    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state["system_prompt_input"],
        placeholder="You are a helpful assistant...",
        help="Instructions that apply to the entire conversation.",
        key=f"system_prompt_widget_{st.session_state['reset_key']}",
    )
    st.session_state["system_prompt_input"] = system_prompt

    return {
        "provider": provider,
        "provider_label": label,
        "model": model,
        "system_prompt": system_prompt,
        "options": {"temperature": temperature, "top_p": top_p},
    }


page = st.navigation(
    [
        st.Page("views/chat.py", title="Chat", icon=":material/chat:", default=True),
        st.Page("views/transform.py", title="Text Transformation", icon=":material/auto_fix_high:"),
    ]
)

with st.sidebar:
    st.session_state["settings"] = sidebar_settings()

page.run()
