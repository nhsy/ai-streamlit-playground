"""Text Transformation page: apply a prompt template to a block of text."""

# pylint: disable=invalid-name  # page scripts run top-level; names are not constants
import streamlit as st

import core
from models import ChatMessage

cfg = core.settings()
st.session_state.setdefault("transformation_text", "")

st.title("Text Transformation")
st.caption(f"{cfg.provider_label} · {cfg.model or 'no model selected'}")

templates = core.load_templates()
selected_template = st.selectbox("Template", list(templates.keys()))

user_text = st.text_area(
    "Enter text to transform:",
    height=200,
    value=st.session_state["transformation_text"],
    key=f"text_input_{st.session_state['reset_key']}",
)
st.session_state["transformation_text"] = user_text

col_transform, col_reset = st.columns(2)
transform_clicked = col_transform.button("Transform", type="primary", icon=":material/auto_fix_high:", width="stretch")
if col_reset.button("Reset", icon=":material/delete:", width="stretch", help="Clear input text and system prompt"):
    st.session_state["transformation_text"] = ""
    st.session_state["system_prompt_input"] = ""
    st.session_state["reset_key"] += 1
    st.rerun()

if transform_clicked:
    if not cfg.model:
        st.error("Please select a model first.")
    elif not user_text:
        st.warning("Please enter some text to transform.")
    else:
        prompt = f"{core.process_prompt(templates[selected_template])}\n\n{core.process_prompt(user_text)}"
        with st.container(border=True):
            st.caption("Result")
            try:
                stream = cfg.provider.chat(
                    model=cfg.model,
                    messages=core.build_payload([ChatMessage(role="user", content=prompt)], cfg.system_prompt),
                    stream=True,
                    options=cfg.options,
                )
                st.write_stream(core.text_chunks(stream))
            except Exception as e:  # pylint: disable=broad-exception-caught
                st.error(f"An error occurred: {e}")
