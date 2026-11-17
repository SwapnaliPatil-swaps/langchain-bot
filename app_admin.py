"""Admin dashboard (spec 2.2 + 3.4, guide step 15.7).

Run:  python -m uv run streamlit run app_admin.py --server.port 8502

Decisions resume the customer's paused agent thread (user_email:conversation_id) with the same
agent, so the action tool, the SQL updates and the Gmail notification all run through the agent.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from langchain_bot.agent import get_agent, get_thread_config, last_ai_text, resume_with_decision
from langchain_bot.auth import authenticate_user
from langchain_bot.config import get_logger
from langchain_bot.context import split_thread_id
from langchain_bot.hitl_utils import list_pending_actions, list_return_tickets, update_pending_action
from langchain_bot.rag_tool import initialize_vector_store

log = get_logger()
STATUS_OPTIONS = ["PENDING", "APPROVED", "REJECTED", "ALL"]


def init_session() -> None:
    st.session_state.setdefault("admin_email", None)
    st.session_state.setdefault("services_ready", False)
    st.session_state.setdefault("last_result", None)


def login_view() -> None:
    st.subheader("Admin sign in")
    with st.form("admin_login"):
        email = st.text_input("Admin email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")
    if submitted:
        user = authenticate_user(email, password, role="admin")
        if user is None:
            st.error("Invalid admin credentials.")
            return
        st.session_state.admin_email = user["email"]
        log.info("LOGIN | user=%s | role=admin", user["email"])
        st.rerun()
    st.caption("Demo admin: admin@example.com / admin123")


def filters(key: str):
    c1, c2, c3, c4 = st.columns([1, 1.4, 1, 1])
    status = c1.selectbox("Status", STATUS_OPTIONS, key=f"{key}_status")
    email = c2.text_input("Customer email contains", key=f"{key}_email")
    d_from = c3.date_input("From", value=None, key=f"{key}_from")
    d_to = c4.date_input("To", value=None, key=f"{key}_to")
    iso = lambda d: d.isoformat() if isinstance(d, date) else None  # noqa: E731
    return status, email or None, iso(d_from), iso(d_to)


def conversation_preview(thread_id: str) -> None:
    user_email, conv_id = split_thread_id(thread_id)
    snapshot = get_agent().get_state(get_thread_config(user_email, conv_id))
    msgs = [m for m in (snapshot.values or {}).get("messages", [])
            if m.type in ("human", "ai") and isinstance(m.content, str) and m.content.strip()]
    if not msgs:
        st.caption("No messages in this thread.")
    for m in msgs[-6:]:
        who = "🧑 Customer" if m.type == "human" else "🤖 Agent"
        st.markdown(f"**{who}:** {m.content}")


def decide(row: dict, decision: str, edited: dict | None = None, message: str | None = None) -> None:
    with st.spinner("Resuming the customer's conversation with the agent..."):
        try:
            result, decided_orders = resume_with_decision(
                row["thread_id"], decision, row["order_id"], st.session_state.admin_email,
                edited_args=edited, message=message)
        except Exception as exc:  # noqa: BLE001
            log.error("HITL RESUME FAILED | thread=%s | %s: %s", row["thread_id"], type(exc).__name__, exc)
            st.session_state.last_result = ("error", f"Could not resume thread: {exc}")
            return

    status = "REJECTED" if decision == "reject" else "APPROVED"
    for pending in list_pending_actions("PENDING"):
        if pending["thread_id"] == row["thread_id"] and pending["order_id"] in decided_orders:
            if pending["id"] == row["id"] and edited:
                update_pending_action(pending["id"], status, edited.get("product_name"), edited.get("reason"))
            else:
                update_pending_action(pending["id"], status)
    st.session_state.last_result = ("ok", f"**{status}** · order #{row['order_id']}\n\n"
                                          f"Agent reply to customer:\n\n> {last_ai_text(result) or '(no text)'}")


def review_queue() -> None:
    status, email, d_from, d_to = filters("pa")
    rows = list_pending_actions(status, email, d_from, d_to)
    if not rows:
        st.info("No requests match. New cancellation/return requests appear here when customers submit them.")
        return
    st.caption(f"{len(rows)} request(s)")
    for row in rows:
        label = "Cancel order" if row["action_type"] == "CANCEL_ORDER" else f"Return · {row['product_name']}"
        with st.expander(f"#{row['id']} · {row['status']} · {label} · order #{row['order_id']} · {row['user_email']}",
                         expanded=row["status"] == "PENDING"):
            user_email, conv_id = split_thread_id(row["thread_id"])
            st.markdown(
                f"- **Customer:** {user_email}\n- **Order:** #{row['order_id']}\n"
                f"- **Action:** {row['action_type']}\n- **Reason:** {row['reason'] or '—'}\n"
                f"- **Conversation id:** `{conv_id}`\n- **Created:** {row['created_at']} · "
                f"**Updated:** {row['updated_at']}")
            with st.popover("View conversation"):
                conversation_preview(row["thread_id"])

            if row["status"] != "PENDING":
                continue

            edited = None
            if row["action_type"] == "CREATE_RETURN":
                new_product = st.text_input("Product name", row["product_name"] or "", key=f"p{row['id']}")
                new_reason = st.text_area("Reason", row["reason"] or "", key=f"r{row['id']}", height=70)
                if new_product != (row["product_name"] or "") or new_reason != (row["reason"] or ""):
                    edited = {"product_name": new_product, "reason": new_reason}
            reject_msg = st.text_input("Rejection reason (sent to the customer)",
                                       "Your request does not meet our policy.", key=f"m{row['id']}")

            b1, b2 = st.columns(2)
            if b1.button("✅ Approve" + (" with edits" if edited else ""), key=f"a{row['id']}", type="primary"):
                decide(row, "edit" if edited else "approve", edited=edited)
                st.rerun()
            if b2.button("❌ Reject", key=f"x{row['id']}"):
                decide(row, "reject", message=reject_msg)
                st.rerun()


def tickets_view() -> None:
    status, email, d_from, d_to = filters("tk")
    rows = list_return_tickets(status, email, d_from, d_to)
    if not rows:
        st.info("No tickets match these filters.")
        return
        # Plain markdown table: st.dataframe needs pyarrow, whose DLL can be blocked by Windows
    # Application Control / Smart App Control.
    cols = ["ticket_id", "subject", "ticket_status", "user_email", "order_id", "return_id",
            "return_status", "reason", "conversation_id", "created_at", "updated_at"]
    cell = lambda v: "—" if v is None else str(v).replace("|", "/").replace("\n", " ")  # noqa: E731
    table = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    table += "\n".join("| " + " | ".join(cell(r.get(c)) for c in cols) + " |" for r in rows)
    st.markdown(table)


def main() -> None:
    st.set_page_config(page_title="ShopBot Admin", page_icon="🛠️", layout="wide")
    st.title("🛠️ ShopBot Admin Dashboard")
    init_session()

    if not st.session_state.admin_email:
        login_view()
        return

    if not st.session_state.services_ready:
        with st.spinner("Loading services (RAG index, SQL toolkit, Gmail MCP) for resumed agents..."):
            initialize_vector_store()
            get_agent()
        st.session_state.services_ready = True

    with st.sidebar:
        st.markdown(f"Signed in as **{st.session_state.admin_email}**")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        if st.button("Log out", use_container_width=True):
            st.session_state.admin_email = None
            st.rerun()

    if st.session_state.last_result:
        kind, text = st.session_state.last_result
        (st.success if kind == "ok" else st.error)(text)
        st.session_state.last_result = None

    tab1, tab2 = st.tabs(["Review queue (cancellations & returns)", "Return tickets"])
    with tab1:
        review_queue()
    with tab2:
        tickets_view()


if __name__ == "__main__":
    main()
