"""
Chat UI (README §4, §8: Streamlit). Wires the dialogue manager, inference
backend, and conversation logger together into a minimal chat window.

Run: streamlit run src/app/ui.py
"""
import sys
from pathlib import Path

import streamlit as st

# allow `streamlit run src/app/ui.py` (not `python -m`) to resolve the
# `src` package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.dialogue_manager import DialogueManager  # noqa: E402
from src.app.inference_engine import select_backend  # noqa: E402
from src.app.logging import ConversationLogger  # noqa: E402

st.set_page_config(page_title="Naija-Switch", page_icon="🇳🇬")

st.title("🇳🇬 Naija-Switch")
st.caption(
    "A code-switch-aware chatbot for English-Nigerian Pidgin conversation. "
    "General conversation only -- **not** a source of medical, legal, or financial advice."
)

if "dialogue_manager" not in st.session_state:
    st.session_state.dialogue_manager = DialogueManager()
if "backend" not in st.session_state:
    st.session_state.backend = select_backend()
if "logger" not in st.session_state:
    st.session_state.logger = ConversationLogger()

backend = st.session_state.backend
dm: DialogueManager = st.session_state.dialogue_manager

with st.sidebar:
    st.subheader("Backend")
    st.write(f"Active: `{backend.name}`")
    if backend.name == "mock":
        st.info(
            "No HF_TOKEN configured, so responses come from a deterministic "
            "mock backend -- proves the app is wired correctly end-to-end. "
            "Add HF_TOKEN to .env to get live model responses."
        )
    st.subheader("Session")
    st.write(f"`{st.session_state.logger.session_id[:8]}`")
    if st.button("Reset conversation"):
        dm.reset()
        st.rerun()

for turn in dm.history:
    with st.chat_message(turn.role):
        st.markdown(turn.content)

user_input = st.chat_input("Type in English, Pidgin, or mix both...")
if user_input:
    dm.add_user_turn(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    messages = dm.build_messages()
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                response = backend.generate(messages)
            except Exception as e:  # noqa: BLE001 - surface backend errors in the UI
                response = f"(backend error: {e})"
        st.markdown(response)

    dm.add_assistant_turn(response)
    st.session_state.logger.log_exchange(user_input, response, backend.name)
