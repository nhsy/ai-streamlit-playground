"""
Ollama Streamlit Playground
A Streamlit app to interact with Ollama and watsonx models.
"""
import streamlit as st
import os
import re
import json
import pypdf
from dotenv import load_dotenv
from providers import OllamaProvider, WatsonxProvider

# Load environment variables from .env file
load_dotenv()

def process_prompt(text):
    """
    Process the prompt to expand file references.
    Syntax: @[path/to/file]
    """
    if not text:
        return text

    def replace_match(match):
        path = match.group(1)
        # Security: Prevent escaping directory or absolute paths if desired,
        # but for this local app, basic existence check is sufficient.
        # We assume path is relative to current working directory.
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return f.read().strip()
            except Exception:
                return f"[Error reading {path}]"
        else:
            return f"[File not found: {path}]"

    # Recursive replacement to handle nested includes (up to a limit)
    for _ in range(3):
        new_text = re.sub(r'@\[([^]]+)\]', replace_match, text)
        if new_text == text:
            break
        text = new_text

    return text

def load_config():
    """Load configuration from config.json."""
    config_path = "config.json"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {config_path}: {e}")

    return {"default_model": None, "templates": {}, "providers": {}}

def load_templates():
    """Load templates from json config and filesystem."""
    config = load_config()
    templates = config["templates"].copy()

    # Load custom templates from 'templates' folder
    template_dir = "templates"
    if os.path.exists(template_dir):
        for filename in os.listdir(template_dir):
            if filename.endswith(".txt"):
                template_name = os.path.splitext(filename)[0].replace("_", " ").title()
                try:
                    with open(os.path.join(template_dir, filename), "r") as f:
                        templates[template_name] = f.read().strip()
                except Exception as e:
                    st.error(f"Error loading template {filename}: {e}")

    return templates

st.set_page_config(page_title="Ollama Playground", page_icon="🦙")

st.title("🦙 Ollama Playground")

# Sidebar for configuration
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "system_prompt_input" not in st.session_state:
    st.session_state["system_prompt_input"] = ""
with st.sidebar:
    st.header("Settings")

    # Mode selection
    mode = st.selectbox("App Mode", ["Chat", "Text Transformation"])
    st.divider()

    # Provider selection
    st.subheader("🔌 Provider")

    # Initialize providers
    ollama_provider = OllamaProvider()
    watsonx_provider = WatsonxProvider()

    # Determine available providers
    available_providers = {}
    if ollama_provider.is_available():
        available_providers["Ollama (Local)"] = ollama_provider
    if watsonx_provider.is_available():
        available_providers["IBM watsonx"] = watsonx_provider

    if not available_providers:
        st.error("⚠️ No providers available!")
        if not ollama_provider._enabled:
            st.info("**Ollama**: Disabled via OLLAMA_ENABLED environment variable.")
        else:
            st.info("**Ollama**: Make sure Ollama is running locally and OLLAMA_ENABLED is not set to 'false'.")

        st.info("**watsonx**: Set WATSONX_API_KEY and WATSONX_PROJECT_ID in .env file.")

        selected_provider = None
        selected_model = None
        st.stop()
    else:
        provider_names = list(available_providers.keys())
        selected_provider_name = st.selectbox(
            "Select Provider",
            provider_names,
            help="Choose between local Ollama or cloud-based watsonx"
        )
        selected_provider = available_providers[selected_provider_name]

        # Show provider status
        st.success(f"✓ {selected_provider_name} connected")

        st.divider()

        # Model selection based on provider
        st.subheader("🤖 Model")

        try:
            model_names = selected_provider.list_models()

            if not model_names:
                st.warning(f"No models found for {selected_provider_name}.")
                selected_model = None
            else:
                # Load config to get default model
                config = load_config()

                # Try provider-specific default first
                provider_key = "ollama" if "Ollama" in selected_provider_name else "watsonx"
                default_model = config.get("providers", {}).get(provider_key, {}).get("default_model")

                # Fallback to top-level default if provider-specific not found
                if not default_model:
                    default_model = config.get("default_model")

                # Try to use default model if it exists in available models
                default_index = 0
                if default_model and default_model in model_names:
                    default_index = model_names.index(default_model)

                selected_model = st.selectbox(
                    "Select a model",
                    model_names,
                    index=default_index,
                    help=f"Available models from {selected_provider_name}"
                )
        except Exception as e:
            st.error(f"Error loading models: {e}")
            selected_model = None

    st.divider()

    # Model Parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Controls randomness: higher values make outputs more random, lower values more deterministic.")
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1, help="Controls diversity via nucleus sampling.")

    st.divider()

    # System Prompt
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state["system_prompt_input"],
        placeholder="You are a helpful assistant...",
        help="Instructions that apply to the entire conversation.",
        key=f"system_prompt_widget_{st.session_state['uploader_key']}"
    )
    st.session_state["system_prompt_input"] = system_prompt

    st.divider()


def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        return f"[Error reading {uploaded_file.name}: {e}]"

# Chat Mode
if mode == "Chat":
    st.subheader("Chat Interface")

    # File Uploader in main content area
    uploaded_files = st.file_uploader(
        "📎 Upload context files (PDF, TXT, CSV, etc.)",
        type=["txt", "md", "py", "json", "yml", "yaml", "csv", "pdf"],
        accept_multiple_files=True,
        help="Upload files to provide additional context for your chat",
        key=f"chat_uploader_{st.session_state['uploader_key']}"
    )

    # Reset button
    if st.button("🗑️ Reset", help="Clear chat history, system prompt, and uploaded files"):
        st.session_state["messages"] = []
        st.session_state["uploader_key"] += 1
        st.session_state["system_prompt_input"] = ""
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter text"):
        if not selected_model:
            st.error("Please select a model to continue.")
            st.stop()

        # Process prompt aliases
        processed_prompt = process_prompt(prompt)

        # Append uploaded files content if any
        if uploaded_files:
            file_contents = "\n\n--- Uploaded Files ---\n"
            for uploaded_file in uploaded_files:
                content = read_uploaded_file(uploaded_file)
                file_contents += f"\nFile: {uploaded_file.name}\nContent:\n{content}\n"
            file_contents += "\n----------------------\n"
            processed_prompt += file_contents
            # Also show in UI that files were attached
            prompt += f" *({len(uploaded_files)} files attached)*"

        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt}) # Show original prompt

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Prepare messages with system prompt if exists
                messages_payload = []
                if system_prompt:
                    messages_payload.append({"role": "system", "content": system_prompt})

                # Reconstruct history with processing
                # Note: We are re-processing history here. In a real app we might cache this.
                # For the current message, we use the already processed version with file content.

                # Add history
                # We need to be careful not to re-process the current message again from st.session_state if we haven't saved the processed version there.
                # In this flow:
                # 1. We saved 'prompt' (original) to session_state
                # 2. We have 'processed_prompt' (expanded) for the current turn.

                for m in st.session_state["messages"][:-1]:
                    messages_payload.append({"role": m["role"], "content": process_prompt(m["content"])})

                messages_payload.append({"role": "user", "content": processed_prompt})

                stream = selected_provider.chat(
                    model=selected_model,
                    messages=messages_payload,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                )

                for chunk in stream:
                    if chunk['message']['content']:
                        full_response += chunk['message']['content']
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

        # Add assistant response to chat history
        st.session_state["messages"].append({"role": "assistant", "content": full_response})

# Text Transformation Mode
elif mode == "Text Transformation":
    st.subheader("Text Transformation")

    # File Uploader in main content area
    uploaded_files = st.file_uploader(
        "📎 Upload context files (PDF, TXT, CSV, etc.)",
        type=["txt", "md", "py", "json", "yml", "yaml", "csv", "pdf"],
        accept_multiple_files=True,
        help="Upload files to provide additional context for transformation",
        key=f"transform_uploader_{st.session_state['uploader_key']}"
    )

    templates = load_templates()

    selected_template = st.selectbox("Choose a transformation template", list(templates.keys()))

    # Initialize session state for transformation text if not exists
    if "transformation_text" not in st.session_state:
        st.session_state["transformation_text"] = ""

    user_text = st.text_area(
        "Enter text to transform:",
        height=200,
        value=st.session_state["transformation_text"],
        key=f"text_input_{st.session_state['uploader_key']}"
    )

    # Update session state when text changes
    st.session_state["transformation_text"] = user_text

    # Buttons in columns
    col1, col2 = st.columns([1, 1])
    with col1:
        transform_button = st.button("Transform", use_container_width=True)
    with col2:
        if st.button("🗑️ Reset", use_container_width=True, help="Clear input text, system prompt, and uploaded files"):
            st.session_state["transformation_text"] = ""
            st.session_state["system_prompt_input"] = ""
            st.session_state["uploader_key"] += 1
            st.rerun()

    if transform_button:
        if not selected_model:
            st.error("Please select a model first.")
        elif not user_text:
            st.warning("Please enter some text to transform.")
        else:
            with st.spinner("Processing..."):
                try:
                    # Process inputs
                    template_text = process_prompt(templates[selected_template])
                    processed_user_text = process_prompt(user_text)

                    prompt = f"{template_text}\n\n{processed_user_text}"

                    # Append uploaded files content if any
                    if uploaded_files:
                        file_contents = "\n\n--- Uploaded Files ---\n"
                        for uploaded_file in uploaded_files:
                            content = read_uploaded_file(uploaded_file)
                            file_contents += f"\nFile: {uploaded_file.name}\nContent:\n{content}\n"
                        file_contents += "\n----------------------\n"
                        prompt += file_contents

                    response_placeholder = st.empty()
                    full_response = ""

                    # Prepare messages
                    messages_payload = []
                    if system_prompt:
                        messages_payload.append({"role": "system", "content": system_prompt})
                    messages_payload.append({"role": "user", "content": prompt})

                    stream = selected_provider.chat(
                        model=selected_model,
                        messages=messages_payload,
                        stream=True,
                        options={
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                    )

                    for chunk in stream:
                        if chunk['message']['content']:
                            full_response += chunk['message']['content']
                            # Simple streaming effect in a customized way if desired,
                            # but for transformation, standard markdown update is fine
                            # response_placeholder.markdown(full_response + "▌")

                    st.success("Transformation Complete!")
                    st.markdown("### Result:")
                    st.markdown(full_response)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
