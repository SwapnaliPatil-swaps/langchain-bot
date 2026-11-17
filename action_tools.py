"""Guarded action tools (guide step 15.3).

Both tools are interrupted by HumanInTheLoopMiddleware, so their bodies only run
after an admin approves (or edits then approves) the request from the dashboard.
Identity comes from ToolRuntime[SessionContext], never from model-supplied args.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from langchain.tools import ToolRuntime, tool

from langchain_bot.config import DB_PATH
from langchain_bot.context import SessionContext

CANCEL_TOOL = "cancel_order_action"
RETURN_TOOL = "create_return_action"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _admin_id(conn: sqlite3.Connection, admin_email: str | None) -> int | None:
    if not admin_email:
        return None
    row = conn.execute("SELECT id FROM users WHERE email = ? AND role = 'admin'", (admin_email,)).fetchone()
    return row["id"] if row else None


@tool(CANCEL_TOOL)
def cancel_order_action(order_id: int, runtime: ToolRuntime[SessionContext]) -> str:
    """Cancel one of the logged-in customer's orders. Requires admin approval (human-in-the-loop).

    Only orders in PLACED status (not yet shipped) can be cancelled. For SHIPPED or DELIVERED
    orders, the customer must request a return instead. Verify ownership and status with the
    SQL tools before calling this tool. Pass only the numeric order_id.
    """
    ctx = runtime.context
    now = _now()
    with _connect() as conn:
        order = conn.execute(
            """SELECT o.id, o.status, o.total_amount, u.id AS user_id
               FROM orders o JOIN users u ON u.id = o.user_id
               WHERE o.id = ? AND u.email = ?""",
            (order_id, ctx.user_email),
        ).fetchone()
        if order is None:
            return f"FAILED: order #{order_id} was not found for {ctx.user_email}. Nothing was changed."
        if order["status"] != "PLACED":
            return (f"FAILED: order #{order_id} is {order['status']}; only PLACED orders can be cancelled. "
                    "Nothing was changed. For SHIPPED or DELIVERED orders, offer a return instead.")

        conn.execute("UPDATE orders SET status = 'CANCELLED' WHERE id = ?", (order_id,))
        conn.execute("UPDATE payments SET status = 'REFUNDED' WHERE order_id = ?", (order_id,))
        cur = conn.execute(
            """INSERT INTO tickets (user_id, return_id, subject, status, thread_id, user_email, created_at, updated_at)
               VALUES (?, NULL, ?, 'RESOLVED', ?, ?, ?, ?)""",
            (order["user_id"], f"Cancellation of order #{order_id}", ctx.conversation_id,
             ctx.user_email, now, now),
        )
        ticket_id = cur.lastrowid

    return (f"SUCCESS: order #{order_id} is now CANCELLED and its payment of {order['total_amount']:.2f} "
            f"is marked REFUNDED. Ticket #{ticket_id} recorded as RESOLVED.")


@tool(RETURN_TOOL)
def create_return_action(order_id: int, product_name: str, reason: str,
                         runtime: ToolRuntime[SessionContext]) -> str:
    """Create a return request for one item of the logged-in customer's order and open a ticket.
    Requires admin approval (human-in-the-loop); never tell the customer it is approved before then.

    Only SHIPPED or DELIVERED orders can be returned; PLACED orders must be cancelled instead.
    product_name must be the exact product name found with the SQL tools; reason is the customer's reason.
    """
    ctx = runtime.context
    now = _now()
    with _connect() as conn:
        order = conn.execute(
            """SELECT o.id, o.status, u.id AS user_id
               FROM orders o JOIN users u ON u.id = o.user_id
               WHERE o.id = ? AND u.email = ?""",
            (order_id, ctx.user_email),
        ).fetchone()
        if order is None:
            return f"FAILED: order #{order_id} was not found for {ctx.user_email}. Nothing was changed."
        if order["status"] == "PLACED":
            return (f"FAILED: order #{order_id} has not shipped yet (PLACED), so it cannot be returned. "
                    "Offer to cancel the order instead. Nothing was changed.")
        if order["status"] not in ("SHIPPED", "DELIVERED"):
            return f"FAILED: order #{order_id} is {order['status']} and cannot be returned. Nothing was changed."

        item = conn.execute(
            """SELECT oi.id, oi.quantity, oi.unit_price, p.name
               FROM order_items oi JOIN products p ON p.id = oi.product_id
               WHERE oi.order_id = ? AND LOWER(p.name) = LOWER(?)""",
            (order_id, product_name.strip()),
        ).fetchone()
        if item is None:
            names = [r["name"] for r in conn.execute(
                "SELECT p.name FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE oi.order_id = ?",
                (order_id,))]
            return (f"FAILED: '{product_name}' is not in order #{order_id}. Items in this order: "
                    f"{', '.join(names)}. Nothing was changed.")

        existing = conn.execute(
            "SELECT id, status FROM returns WHERE order_item_id = ? AND status IN ('PENDING', 'APPROVED')",
            (item["id"],),
        ).fetchone()
        if existing is not None:
            return (f"FAILED: {item['name']} in order #{order_id} already has return #{existing['id']} "
                    f"({existing['status']}). Nothing was changed.")

        admin_id = _admin_id(conn, ctx.admin_email)
        cur = conn.execute(
            """INSERT INTO returns (order_id, order_item_id, user_id, reason, status, requested_at, resolved_at, admin_id)
               VALUES (?, ?, ?, ?, 'APPROVED', ?, ?, ?)""",
            (order_id, item["id"], order["user_id"], reason, now, now, admin_id),
        )
        return_id = cur.lastrowid
        cur = conn.execute(
            """INSERT INTO tickets (user_id, return_id, subject, status, thread_id, user_email, created_at, updated_at)
               VALUES (?, ?, ?, 'RESOLVED', ?, ?, ?, ?)""",
            (order["user_id"], return_id, f"Return request for {item['name']} (order #{order_id})",
             ctx.conversation_id, ctx.user_email, now, now),
        )
        ticket_id = cur.lastrowid

        refund = round(item["quantity"] * item["unit_price"], 2)
        method_row = conn.execute("SELECT payment_method FROM payments WHERE order_id = ? LIMIT 1",
                                  (order_id,)).fetchone()
        conn.execute(
            """INSERT INTO payments (order_id, amount, status, payment_method, transaction_reference, paid_at)
               VALUES (?, ?, 'REFUNDED', ?, ?, ?)""",
            (order_id, refund, method_row["payment_method"] if method_row else "CARD",
             f"RFND-{datetime.now():%Y%m%d}-R{return_id}", now),
        )

    return (f"SUCCESS: return #{return_id} for {item['name']} (order #{order_id}) is APPROVED. "
            f"Ticket #{ticket_id} recorded as RESOLVED. Refund of {refund:.2f} to the original payment method "
            "is recorded as REFUNDED (processed 5-10 business days after the item is received).")


def get_action_tools():
    return [cancel_order_action, create_return_action]
