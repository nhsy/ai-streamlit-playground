"""Chat page: multi-turn conversation with file attachments."""

# pylint: disable=invalid-name  # page scripts run top-level; names are not constants
import json
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

import core

cfg = core.settings()
st.session_state.setdefault("messages", [])
messages = st.session_state["messages"]

col_title, col_export, col_reset = st.columns([5, 1.4, 1.2], vertical_alignment="bottom")
col_title.title("Chat")
col_title.caption(f"{cfg['provider_label']} · {cfg['model'] or 'no model selected'}")

with col_export.popover("Export", icon=":material/ios_share:", width="stretch", disabled=not messages):
    md_text = core.format_chat_as_markdown(messages)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if st.button("Copy as Markdown", icon=":material/content_copy:", width="stretch"):
        st.session_state["_copy_chat"] = md_text
    st.download_button(
        "Markdown",
        data=md_text,
        file_name=f"chat_{timestamp}.md",
        mime="text/markdown",
        icon=":material/description:",
        width="stretch",
    )
    st.download_button(
        "HTML",
        data=core.format_chat_as_html(messages),
        file_name=f"chat_{timestamp}.html",
        mime="text/html",
        icon=":material/html:",
        width="stretch",
    )

if col_reset.button("Reset", icon=":material/delete:", width="stretch", help="Clear chat history and system prompt"):
    st.session_state["messages"] = []
    st.session_state["system_prompt_input"] = ""
    st.session_state["reset_key"] += 1
    st.rerun()

if copy_text := st.session_state.pop("_copy_chat", None):
    # json.dumps gives a safe JS string literal; "</" is escaped so the text can't close the script tag
    js_text = json.dumps(copy_text).replace("</", "<\\/")
    components.html(f"<script>navigator.clipboard.writeText({js_text});</script>", height=0)
    st.toast("Copied to clipboard", icon=":material/check:")

if not messages:
    st.caption("Ask anything. Attach files with the paperclip, or reference local files with `@[path/to/file]`.")

AVATARS = {"user": ":material/person:", "assistant": ":material/smart_toy:"}

for message in messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message.get("display", message["content"]))

prompt = st.chat_input("Message", accept_file="multiple", file_type=core.UPLOAD_TYPES)
if prompt:
    if not cfg["model"]:
        st.error("Please select a model to continue.")
        st.stop()

    user_message = core.build_user_message(getattr(prompt, "text", prompt), getattr(prompt, "files", None))
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_message["display"])

    reply = ""
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            stream = cfg["provider"].chat(
                model=cfg["model"],
                messages=core.build_payload([*messages, user_message], cfg["system_prompt"]),
                stream=True,
                options=cfg["options"],
            )
            reply = st.write_stream(core.text_chunks(stream))
            if not (isinstance(reply, str) and reply.strip()):
                reply = ""
                st.warning("No response received.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            st.error(f"An error occurred: {e}")

    # Only keep the turn if the model answered; a failed turn leaves history unchanged so it can be retried
    if reply:
        messages.extend([user_message, {"role": "assistant", "content": reply}])
        st.rerun()
