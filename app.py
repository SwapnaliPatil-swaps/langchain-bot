"""Customer chat app (spec 3.2, guide steps 3-15).

Run:  python -m uv run streamlit run app.py          (http://localhost:8501)

Thin shell: authenticates, keeps (user_email, conversation_id) in session state, calls the single
agent, and redraws the conversation from the checkpointer with agent.get_state(config).
"""

from __future__ import annotations

import streamlit as st

from langchain_bot.agent import chat, get_agent, get_pending_interrupt_requests, get_thread_config
from langchain_bot.auth import authenticate_user
from langchain_bot.config import get_logger
from langchain_bot.context import make_thread_id
from langchain_bot.hitl_utils import handle_interrupt, has_pending_action_for_thread
from langchain_bot.rag_tool import initialize_vector_store
from langchain_bot.threads import load_threads, new_conversation

log = get_logger()

GREETING = ("Hi! I'm your support assistant. Ask me about your orders, payments and returns, "
            "our policies, or ask me to cancel an order or start a return.")


def init_session() -> None:
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("user_name", None)
    st.session_state.setdefault("user_role", None)
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("vector_store_ready", False)
    st.session_state.setdefault("flash", None)


def start_new_conversation() -> None:
    st.session_state.conversation_id = new_conversation(st.session_state.user_email)


def load_conversation(conv_id: str) -> None:
    st.session_state.conversation_id = conv_id


def _text(content) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def render_history() -> None:
    """Draw the active thread straight from the checkpointer (not from session state)."""
    with st.chat_message("assistant"):
        st.markdown(GREETING)
    config = get_thread_config(st.session_state.user_email, st.session_state.conversation_id)
    snapshot = get_agent().get_state(config)
    for msg in snapshot.values.get("messages", []) if snapshot.values else []:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(_text(msg.content))
        elif msg.type == "ai" and _text(msg.content).strip():  # skip tool-call-only turns
            with st.chat_message("assistant"):
                st.markdown(_text(msg.content))


def chat_round(user_input: str) -> None:
    email, conv_id = st.session_state.user_email, st.session_state.conversation_id
    result = chat(email, conv_id, user_input)
    notice = handle_interrupt(result, email, conv_id)  # writes pending_actions when paused
    if notice:
        st.session_state.flash = notice


def login_view() -> None:
    st.subheader("Sign in")
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")
    if submitted:
        user = authenticate_user(email, password, role="customer")
        if user is None:
            st.error("Invalid customer email or password. (Admins use the admin dashboard on port 8502.)")
            return
        st.session_state.user_email = user["email"]
        st.session_state.user_name = user["full_name"]
        st.session_state.user_role = user["role"]
        threads = load_threads(user["email"])
        if threads:
            st.session_state.conversation_id = threads[0]["id"]
        else:
            start_new_conversation()
        log.info("LOGIN | user=%s | role=customer", user["email"])
        st.rerun()
    st.caption("Demo customers: swapnali@gmail.com / swap@123 · ss@example.com /desp@123 · "
               "charme@example.com / char123")


def sidebar() -> None:
    with st.sidebar:
        st.markdown(f"**{st.session_state.user_name}**  \n{st.session_state.user_email}")
        st.header("Conversations")
        if st.button("➕ Start new conversation", use_container_width=True):
            start_new_conversation()
            st.rerun()

        threads = load_threads(st.session_state.user_email)
        ids = [t["id"] for t in threads]
        labels = {t["id"]: t["label"] for t in threads}
        if ids:
            current = st.session_state.conversation_id
            choice = st.selectbox("Your conversations", ids,
                                  index=ids.index(current) if current in ids else 0,
                                  format_func=lambda i: labels.get(i, i[:8]))
            if choice != current:
                load_conversation(choice)
                st.rerun()
        else:
            st.selectbox("Your conversations", ["(no threads yet)"], disabled=True)

        st.divider()
        if st.button("🔄 Refresh (check for admin replies)", use_container_width=True):
            st.rerun()
        if st.button("Log out", use_container_width=True):
            for key in ("user_email", "user_name", "user_role", "conversation_id", "flash"):
                st.session_state[key] = None
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="ShopBot Support", page_icon="🛒")
    st.title("🛒 ShopBot Support")
    st.caption("AI customer support · orders, returns, policies")
    init_session()

    if not st.session_state.vector_store_ready:
        with st.spinner("Initializing knowledge base..."):
            try:
                initialize_vector_store()
                st.session_state.vector_store_ready = True
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not initialise the policy knowledge base: {exc}")
                st.stop()

    if not st.session_state.user_email:
        login_view()
        return

    with st.spinner("Starting the support agent (first start launches the Gmail MCP server)..."):
        get_agent()

    sidebar()
    email, conv_id = st.session_state.user_email, st.session_state.conversation_id
    st.info(f"Logged in as **{email}** · role **{st.session_state.user_role}** · conversation `{conv_id}`")

    render_history()

    thread_id = make_thread_id(email, conv_id)
    paused = bool(get_pending_interrupt_requests(email, conv_id)) or has_pending_action_for_thread(thread_id)
    if st.session_state.flash:
        st.warning(st.session_state.flash)
        st.session_state.flash = None
    elif paused:
        st.warning("⏳ A cancellation/return request in this conversation is waiting for human review. "
                   "You'll get a reply here and by email once an admin decides. "
                   "Start a new conversation to ask something else meanwhile.")

    prompt = st.chat_input("Ask about orders, returns, refunds, shipping…", disabled=paused)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_round(prompt)
                except Exception as exc:  # noqa: BLE001
                    log.error("CHAT ERROR | user=%s | %s: %s", email, type(exc).__name__, exc)
                    st.error(f"Sorry, something went wrong: {exc}")
                    st.stop()
        st.rerun()


if __name__ == "__main__":
    main()
