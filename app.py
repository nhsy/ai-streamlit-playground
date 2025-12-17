import streamlit as st
import ollama
import os
import re

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

def load_templates():
    """Load templates from code and filesystem."""
    # Default templates
    templates = {
        "Summarize": "Summarize the following text:",
        "Fix Grammar": "Fix the grammar and spelling in the following text:",
        "Rewrite Professionally": "Rewrite the following text to sound more professional:",
        "Explain to a 5-year old": "Explain the following text as if I am 5 years old:",
        "Translate to Spanish": "Translate the following text to Spanish:",
        "Extract Keywords": "Extract the main keywords from the following text:",
        "Rewrite as a DevOps SME": "Rewrite the following text as a DevOps SME:",
        "Rewrite as a HashiCorp SME": "Rewrite the following text as a HashiCorp SME:"    
    }
    
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
with st.sidebar:
    st.header("Settings")
    
    # Mode selection
    mode = st.selectbox("App Mode", ["Chat", "Text Transformation"])
    st.divider()

    try:
        models_info = ollama.list()
        # Adjust based on the actual structure of the response from ollama.list()
        # The library usually returns a dict with 'models' key which is a list of objects
        model_names = [m['model'] for m in models_info['models']]
        if not model_names:
            st.warning("No models found. Please run `ollama pull <model>` in your terminal.")
            selected_model = None
        else:
            selected_model = st.selectbox("Select a model", model_names)
    except Exception as e:
        st.error(f"Could not connect to Ollama. Make sure it is running. Error: {e}")
        selected_model = None

    st.divider()
    
    # Model Parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Controls randomness: higher values make outputs more random, lower values more deterministic.")
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1, help="Controls diversity via nucleus sampling.")

    st.divider()

    # System Prompt
    system_prompt = st.text_area("System Prompt", value="", placeholder="You are a helpful assistant...", help="Instructions that apply to the entire conversation.")

    st.divider()

    # File Uploader
    uploaded_files = st.file_uploader("Upload context files", type=["txt", "md", "py", "json", "yml", "yaml"], accept_multiple_files=True)


def read_uploaded_file(uploaded_file):
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        return f"[Error reading {uploaded_file.name}: {e}]"

# Chat Mode
if mode == "Chat":
    st.subheader("Chat Interface")
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

                stream = ollama.chat(
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
    
    templates = load_templates()
    
    selected_template = st.selectbox("Choose a transformation template", list(templates.keys()))
    
    user_text = st.text_area("Enter text to transform:", height=200)
    
    if st.button("Transform"):
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
                    
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Prepare messages
                    messages_payload = []
                    if system_prompt:
                        messages_payload.append({"role": "system", "content": system_prompt})
                    messages_payload.append({"role": "user", "content": prompt})

                    stream = ollama.chat(
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
