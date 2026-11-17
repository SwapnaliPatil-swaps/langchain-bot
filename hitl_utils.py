"""Human-in-the-loop helpers (guide step 15.5).

pending_actions is business data (the admin's work queue), not a chat transcript.
resume_with_decision lives in agent.py, next to the agent it resumes.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Optional

from langchain_bot.action_tools import CANCEL_TOOL, RETURN_TOOL
from langchain_bot.config import DB_PATH, get_logger
from langchain_bot.context import make_thread_id

log = get_logger()

ACTION_TYPES = {CANCEL_TOOL: "CANCEL_ORDER", RETURN_TOOL: "CREATE_RETURN"}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _interrupt_values(result: Any) -> list[dict]:
    if not isinstance(result, dict):
        return []
    values = []
    for item in result.get("__interrupt__", []) or []:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            values.append(value)
    return values


def get_action_requests(result: Any) -> list[dict]:
    """All interrupted tool calls ({name, args, ...}) in an agent result."""
    requests: list[dict] = []
    for value in _interrupt_values(result):
        requests.extend(value.get("action_requests", []) or [])
    return requests


def is_interrupted(result: Any) -> bool:
    return bool(get_action_requests(result))


def handle_interrupt(result: Any, user_email: str, conversation_id: str) -> Optional[str]:
    """Record each interrupted action in pending_actions; return the markdown the chat shows.

    Returns None when the result was not interrupted.
    """
    requests = get_action_requests(result)
    if not requests:
        return None

    thread_id = make_thread_id(user_email, conversation_id)
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    lines: list[str] = []
    with _connect() as conn:
        for req in requests:
            action_type = ACTION_TYPES.get(req.get("name"))
            if action_type is None:
                continue
            args = req.get("args", {}) or {}
            order_id = int(args.get("order_id"))
            exists = conn.execute(
                """SELECT id FROM pending_actions WHERE thread_id = ? AND order_id = ?
                   AND action_type = ? AND status = 'PENDING'""",
                (thread_id, order_id, action_type),
            ).fetchone()
            if exists is None:
                conn.execute(
                    """INSERT INTO pending_actions (thread_id, user_email, action_type, order_id, product_name,
                       reason, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)""",
                    (thread_id, user_email, action_type, order_id, args.get("product_name"),
                     args.get("reason"), now, now),
                )
                log.info("HITL PENDING | user=%s | action=%s | order_id=%s | thread=%s",
                         user_email, action_type, order_id, thread_id)
            if action_type == "CANCEL_ORDER":
                lines.append(f"Cancellation request for order **#{order_id}**")
            else:
                lines.append(f"Return request for **{args.get('product_name')}** from order **#{order_id}**")

    return ("⏳ **Waiting for human review.** " + "; ".join(lines) +
            " has been sent to our support team. Nothing has been changed yet; an admin will approve or "
            "reject it and you will get an email and a reply in this conversation.")


def has_pending_action_for_thread(thread_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM pending_actions WHERE thread_id = ? AND status = 'PENDING' LIMIT 1",
                           (thread_id,)).fetchone()
    return row is not None


def list_pending_actions(status: Optional[str] = None, user_email: Optional[str] = None,
                         date_from: Optional[str] = None, date_to: Optional[str] = None) -> list[dict]:
    query = "SELECT * FROM pending_actions WHERE 1=1"
    params: list[Any] = []
    if status and status != "ALL":
        query += " AND status = ?"
        params.append(status)
    if user_email:
        query += " AND user_email LIKE ?"
        params.append(f"%{user_email.strip().lower()}%")
    if date_from:
        query += " AND substr(created_at, 1, 10) >= ?"
        params.append(date_from)
    if date_to:
        query += " AND substr(created_at, 1, 10) <= ?"
        params.append(date_to)
    query += " ORDER BY created_at DESC, id DESC"
    with _connect() as conn:
        return [dict(r) for r in conn.execute(query, params)]


def update_pending_action(action_id: int, status: str, product_name: Optional[str] = None,
                          reason: Optional[str] = None) -> None:
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with _connect() as conn:
        conn.execute(
            """UPDATE pending_actions SET status = ?, product_name = COALESCE(?, product_name),
               reason = COALESCE(?, reason), updated_at = ? WHERE id = ?""",
            (status, product_name, reason, now, action_id),
        )


def list_return_tickets(status: Optional[str] = None, user_email: Optional[str] = None,
                        date_from: Optional[str] = None, date_to: Optional[str] = None) -> list[dict]:
    """Tickets with their return rows, for the admin's history view (spec 3.4)."""
    query = """SELECT t.id AS ticket_id, t.subject, t.status AS ticket_status, t.user_email,
                      r.order_id, r.id AS return_id, r.status AS return_status, r.reason,
                      t.thread_id AS conversation_id, t.created_at, t.updated_at
               FROM tickets t LEFT JOIN returns r ON r.id = t.return_id WHERE 1=1"""
    params: list[Any] = []
    if status and status != "ALL":
        query += " AND r.status = ?"
        params.append(status)
    if user_email:
        query += " AND t.user_email LIKE ?"
        params.append(f"%{user_email.strip().lower()}%")
    if date_from:
        query += " AND substr(t.created_at, 1, 10) >= ?"
        params.append(date_from)
    if date_to:
        query += " AND substr(t.created_at, 1, 10) <= ?"
        params.append(date_to)
    query += " ORDER BY t.created_at DESC"
    with _connect() as conn:
        return [dict(r) for r in conn.execute(query, params)]
