import asyncio

import streamlit as st

from core import DynaAgent

st.set_page_config(page_title="DynaAgent", layout="centered")
st.title("DynaAgent")

if "msgs" not in st.session_state:
    st.session_state.msgs = []
    st.session_state.agent = DynaAgent()  # keeps chat history and generated tools across turns

AVATARS = {"You": "🙂", "Agent": "🧠", "Budget": "⏱️"}


def language(text):
    """Guess the code block language."""
    raw = text.strip()
    if raw[:1] in ("{", "["):
        return "json"
    if raw.startswith(("def ", "import ")) or "\ndef " in raw:
        return "python"
    return "text"


def render(item):
    """Draw one chat message."""
    with st.chat_message("user" if item["who"] == "You" else "assistant", avatar=AVATARS.get(item["who"], "🤖")):
        if item["who"] not in ("You", "Agent"):
            st.markdown(f"**{item['who']}**")
        st.markdown(item["text"])
        if item["code"]:
            with st.expander("View Details"):
                st.code(item["code"], language=language(item["code"]), wrap_lines=True)


for item in st.session_state.msgs:
    render(item)

if question := st.chat_input("Ask anything..."):

    def update(who, text, code=""):
        """Store and draw a live agent event."""
        item = {"who": who, "text": text, "code": code}
        st.session_state.msgs.append(item)
        render(item)

    update("You", question)
    st.session_state.agent.on = update
    with st.spinner("Working..."):
        asyncio.run(st.session_state.agent.execute(question))
