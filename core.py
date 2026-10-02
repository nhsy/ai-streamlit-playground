"""Shared helpers for the AI Streamlit Playground: config, providers, prompts and exports."""

import html as html_module
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import docx
import pypdf
import streamlit as st

from providers import GeminiProvider, OllamaProvider, OpenRouterProvider, WatsonxProvider

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.json"
TEMPLATE_DIR = "templates"
MAX_INCLUDE_BYTES = 200_000
UPLOAD_TYPES = ["txt", "md", "py", "json", "yml", "yaml", "csv", "pdf", "docx"]

# Provider key (as used in config.json) -> (display label, class)
PROVIDERS = {
    "ollama": ("Ollama (Local)", OllamaProvider),
    "watsonx": ("IBM watsonx", WatsonxProvider),
    "openrouter": ("OpenRouter", OpenRouterProvider),
    "gemini": ("Google Gemini", GeminiProvider),
}


# --- Config & templates -----------------------------------------------------


def load_config():
    """Load configuration from config.json, falling back to empty defaults."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Error loading %s: %s", CONFIG_PATH, e)

    return {"default_model": None, "templates": {}, "providers": {}}


@st.cache_data(ttl=10, show_spinner=False)
def get_config():
    """Cached config, re-read at most every 10 seconds."""
    return load_config()


def load_templates():
    """Load templates from the config and the templates folder."""
    templates = get_config().get("templates", {}).copy()

    if os.path.exists(TEMPLATE_DIR):
        for filename in sorted(os.listdir(TEMPLATE_DIR)):
            if filename.endswith(".txt"):
                template_name = os.path.splitext(filename)[0].replace("_", " ").title()
                try:
                    with open(os.path.join(TEMPLATE_DIR, filename), "r", encoding="utf-8") as f:
                        templates[template_name] = f.read().strip()
                except OSError as e:
                    logger.warning("Error loading template %s: %s", filename, e)

    return templates


# --- Providers --------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_provider(key):
    """One shared provider instance per key."""
    return PROVIDERS[key][1]()


@st.cache_data(ttl=30, show_spinner=False)
def available_provider_keys():
    """Keys of providers that are enabled and reachable, re-checked every 30 seconds."""
    return [key for key in PROVIDERS if get_provider(key).is_available()]


@st.cache_data(ttl=60, show_spinner=False)
def list_models(key):
    """Model ids for a provider, re-fetched every 60 seconds."""
    return get_provider(key).list_models()


def model_help(provider, model):
    """Tooltip text describing a model, from whatever metadata the provider returns."""
    info = provider.get_model_info(model) or {}
    details = info.get("details", {})

    if "size" in info:
        return (
            f"**{model}**\n\n"
            f"- **Size:** {info['size'] / (1024**3):.2f} GB\n"
            f"- **Params:** {details.get('parameter_size', 'Unknown')}\n"
            f"- **Quant:** {details.get('quantization_level', 'Unknown')}\n"
            f"- **Family:** {details.get('family', 'Unknown')}"
        )
    display_name = details.get("display_name")
    if display_name and display_name != model:
        return f"**{display_name}**\n\nID: `{model}`"
    return f"Available models from {provider.get_name()}"


def text_chunks(stream):
    """Adapt a provider chat stream to plain text chunks for st.write_stream."""
    for chunk in stream:
        content = chunk["message"]["content"]
        if content:
            yield content


# --- Prompts & files --------------------------------------------------------


def _read_included_file(raw_path, base):
    """Read a file referenced by @[path], restricted to non-hidden files under base."""
    try:
        path = (base / raw_path).resolve()
        relative = path.relative_to(base)
    except OSError, ValueError:
        return f"[Access denied: {raw_path}]"

    if any(part.startswith(".") for part in relative.parts):
        return f"[Access denied: {raw_path}]"
    if not path.is_file():
        return f"[File not found: {raw_path}]"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read(MAX_INCLUDE_BYTES + 1)
    except OSError, UnicodeDecodeError:
        return f"[Error reading {raw_path}]"

    if len(content) > MAX_INCLUDE_BYTES:
        content = content[:MAX_INCLUDE_BYTES] + "\n[truncated]"
    return content.strip()


def process_prompt(text):
    """
    Expand file references in a prompt.
    Syntax: @[path/to/file], relative to the working directory. Paths outside it and hidden files are refused.
    """
    if not text:
        return text

    base = Path.cwd().resolve()

    # Nested includes are expanded up to three levels deep
    for _ in range(3):
        new_text = re.sub(r"@\[([^]]+)\]", lambda m: _read_included_file(m.group(1), base), text)
        if new_text == text:
            break
        text = new_text

    return text


def read_uploaded_file(file):
    """Read content from an uploaded file (PDF, DOCX or text)."""
    try:
        name = file.name.lower()
        if name.endswith(".pdf"):
            reader = pypdf.PdfReader(file)
            return "\n".join(page.extract_text() for page in reader.pages)
        if name.endswith(".docx"):
            return "\n".join(paragraph.text for paragraph in docx.Document(file).paragraphs)
        return file.getvalue().decode("utf-8")
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"[Error reading {file.name}: {e}]"


def build_user_message(text, files=None):
    """
    Build a user chat message.
    'content' is what the model sees (file refs and uploads expanded) and is kept in history,
    so later turns still have the file context. 'display' is what the chat shows.
    """
    content = process_prompt(text or "")
    display = text or ""

    if files:
        parts = ["\n\n--- Uploaded Files ---"]
        for file in files:
            parts.append(f"\nFile: {file.name}\nContent:\n{read_uploaded_file(file)}")
        parts.append("\n----------------------")
        content += "\n".join(parts)
        names = ", ".join(f"`{file.name}`" for file in files)
        display = f"{display}\n\n*Attached: {names}*".strip()

    return {"role": "user", "content": content, "display": display}


# --- Export -----------------------------------------------------------------


def _shown(msg):
    return msg.get("display", msg["content"])


def format_chat_as_markdown(messages):
    """Convert chat messages to a markdown string."""
    lines = [f"# Chat Export\n\n_Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"---\n\n**{role}:**\n\n{_shown(msg)}\n")
    return "\n".join(lines)


def format_chat_as_html(messages):
    """Convert chat messages to a styled HTML document."""
    msg_blocks = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        css_class = "user" if msg["role"] == "user" else "assistant"
        escaped = html_module.escape(_shown(msg)).replace("\n", "<br>")
        msg_blocks.append(f'<div class="message {css_class}"><strong>{role}</strong><p>{escaped}</p></div>')
    body = "\n".join(msg_blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chat Export</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .timestamp {{ color: #888; font-size: 0.9em; margin-bottom: 20px; }}
  .message {{ padding: 12px 16px; margin: 10px 0; border-radius: 8px; }}
  .message strong {{ display: block; margin-bottom: 4px; }}
  .message p {{ margin: 0; white-space: pre-wrap; }}
  .user {{ background: #e3f2fd; border-left: 4px solid #1976d2; }}
  .assistant {{ background: #fff; border-left: 4px solid #43a047; }}
</style>
</head>
<body>
<h1>Chat Export</h1>
<p class="timestamp">Exported on {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
{body}
</body>
</html>"""


# --- Session ----------------------------------------------------------------


def settings():
    """Sidebar selections shared with the pages (set by app.py on every run)."""
    return st.session_state["settings"]


def build_payload(history, system_prompt):
    """Messages to send to the provider: optional system prompt plus the stored history."""
    payload = [{"role": "system", "content": system_prompt}] if system_prompt else []
    payload.extend({"role": m["role"], "content": m["content"]} for m in history)
    return payload
