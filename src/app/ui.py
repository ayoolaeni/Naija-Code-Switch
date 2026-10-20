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

THINKING = "_Thinking... on a computer without a graphics card a reply can take a little while._"
EMPTY_REPLY = "Sorry, I couldn't come up with a reply -- please try again."

st.set_page_config(page_title="Naija-Switch", page_icon="🇳🇬")

st.title("🇳🇬 Naija-Switch")
st.caption(
    "A code-switch-aware chatbot for English-Nigerian Pidgin conversation. "
    "General conversation only -- **not** a source of medical, legal, or financial advice."
)


@st.cache_resource(show_spinner="Starting up the chatbot -- the first start can take a few minutes...")
def get_backend():
    # Once per app process, shared by every browser tab: a local model is
    # several GB of RAM, and loading it per session meant every page
    # refresh loaded another full copy.
    return select_backend()


backend = get_backend()

if "dialogue_manager" not in st.session_state:
    st.session_state.dialogue_manager = DialogueManager()
if "logger" not in st.session_state:
    st.session_state.logger = ConversationLogger()
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

dm: DialogueManager = st.session_state.dialogue_manager
logger = st.session_state.logger
pending = st.session_state.pending_input

with st.sidebar:
    st.subheader("Backend")
    st.write(f"Active: `{backend.name}`")
    if backend.detail:
        st.caption(backend.detail)
    if backend.name == "mock":
        st.info(
            "No HF_TOKEN configured, so responses come from a deterministic "
            "mock backend -- proves the app is wired correctly end-to-end. "
            "Add HF_TOKEN to .env to get live model responses."
        )
    st.subheader("Session")
    st.write(f"`{logger.session_id[:8]}`")
    if st.button("Reset conversation"):
        dm.reset()
        st.session_state.pending_input = None
        st.rerun()

for turn in dm.history:
    with st.chat_message(turn.role):
        st.markdown(turn.content)

user_input = st.chat_input("Type in English, Pidgin, or mix both...", disabled=pending is not None)

if user_input and pending is None:
    dm.add_user_turn(user_input)
    st.session_state.pending_input = user_input
    # Rerun so the input box is disabled *before* the slow reply starts:
    # otherwise someone who thinks it's stuck sends another message, which
    # interrupts and restarts the reply, and it never finishes.
    st.rerun()

if pending is not None:
    messages = dm.build_messages()
    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown(THINKING)

        def stream():
            shown = False
            for chunk in backend.generate_stream(messages):
                if chunk and not shown:
                    thinking.empty()
                    shown = True
                yield chunk
            if not shown:
                thinking.empty()

        try:
            response = st.write_stream(stream())
            if not isinstance(response, str) or not response.strip():
                response = EMPTY_REPLY
                st.markdown(response)
        except Exception as e:  # noqa: BLE001 - surface backend errors in the UI
            thinking.empty()
            response = f"(backend error: {e})"
            st.markdown(response)

    dm.add_assistant_turn(response)
    st.session_state.pending_input = None
    try:
        # Logging is for traceability, not core to the chat working -- it
        # must never take an already-delivered reply down with it.
        logger.log_exchange(pending, response, backend.name)
    except Exception:  # noqa: BLE001
        pass
    st.rerun()
