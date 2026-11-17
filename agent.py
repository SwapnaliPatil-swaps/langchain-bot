"""The single main support agent (spec 4.1, guide steps 9, 11, 12, 13, 14, 15).

Tools on ONE agent:
  * rag_policy_search            - ChromaDB policy/FAQ retrieval
  * SQL Database Toolkit         - sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker
  * Gmail MCP server tools       - send_email (loaded from @gongrzhe/server-gmail-autoauth-mcp)
  * cancel_order_action / create_return_action - interrupted by HumanInTheLoopMiddleware

Conversation state lives in a SqliteSaver checkpointer keyed by "user_email:conversation_id".
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, dynamic_prompt
from langchain_core.tools import StructuredTool
from langchain_bot.llm import get_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from langchain_bot.action_tools import CANCEL_TOOL, RETURN_TOOL, get_action_tools
from langchain_bot.config import CHECKPOINTS_DB_PATH, GMAIL_MCP_ENABLED, OPENAI_MODEL, get_logger
from langchain_bot.context import SessionContext, make_thread_id, split_thread_id
from langchain_bot.logging_middleware import get_logging_middleware
from langchain_bot.rag_tool import search_policies
from langchain_bot.sql_tools import get_sql_tools

log = get_logger()

REFERENCE_DATE = os.getenv("REFERENCE_DATE", "2026-09-30")  # "today" for the seed data (see ecommerce_setup.sql)
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "40"))

# ---------------------------------------------------------------------------
# Gmail MCP server (guide step 14)
# ---------------------------------------------------------------------------
# The key "gmail" only names the server; tool names are NOT prefixed (send_email, not gmail_send_email).
GMAIL_CONFIG_DIR = Path.home() / ".gmail-mcp"
GMAIL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

GMAIL_MCP_CONFIG = {
    "mcpServers": {
        "gmail": {
            # Same pattern as the course's Gmail MCP lab: on Windows npx is a .cmd script,
            # so it is started through cmd.exe /c.
            "command": "cmd.exe" if os.name == "nt" else "npx",
            "args": (["/c", "npx", "-y", "@gongrzhe/server-gmail-autoauth-mcp"] if os.name == "nt"
                     else ["-y", "@gongrzhe/server-gmail-autoauth-mcp"]),
            "cwd": str(GMAIL_CONFIG_DIR),
        },
    }
}
# The server also exposes read/search/delete/label/filter tools. The support agent only needs
# to send notifications, so only these are handed to the model (the prompt also forbids the rest).
GMAIL_ALLOWED_TOOLS = {"send_email"}


# MCP clients are bound to the event loop that opened them. A fresh asyncio.run() per call
# closes that loop, so the next call fails with "Event loop is closed". Instead, one background
# loop lives for the whole process: the MCP session is opened there once and every Gmail call
# is scheduled onto it from the synchronous agent.invoke.
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_adapter: Any = None
_MCP_TIMEOUT_S = 120


def _get_mcp_loop() -> asyncio.AbstractEventLoop:
    global _mcp_loop
    if _mcp_loop is None:
        _mcp_loop = asyncio.new_event_loop()
        threading.Thread(target=_mcp_loop.run_forever, name="gmail-mcp-loop", daemon=True).start()
    return _mcp_loop


def _run_on_mcp_loop(coro):
    return asyncio.run_coroutine_threadsafe(coro, _get_mcp_loop()).result(timeout=_MCP_TIMEOUT_S)


async def load_gmail_tools() -> list:
    """Start the Gmail MCP server once (kept open for the process lifetime) and list its tools."""
    global _mcp_adapter
    try:
        from langchain.mcp import MCPAdapter  # the adapter named in the spec / guide
    except ImportError:
        MCPAdapter = None  # noqa: N806

    if MCPAdapter is not None:
        adapter = MCPAdapter(GMAIL_MCP_CONFIG)
        await adapter.__aenter__()  # deliberately not closed: tools reuse this session
        _mcp_adapter = adapter
        return await adapter.list_tools()
    """Start the Gmail MCP server once and list its tools."""
    try:
        from langchain.mcp import MCPAdapter  # the adapter named in the spec / guide
    except ImportError:
        MCPAdapter = None  # noqa: N806

    if MCPAdapter is not None:
        async with MCPAdapter(GMAIL_MCP_CONFIG) as adapter:
            return await adapter.list_tools()

    # Fallback for installs where langchain[mcp] exposes the adapter as langchain-mcp-adapters
    from langchain_mcp_adapters.client import MultiServerMCPClient

    servers = {name: {**cfg, "transport": "stdio"} for name, cfg in GMAIL_MCP_CONFIG["mcpServers"].items()}
    return await MultiServerMCPClient(servers).get_tools()


def _sync_mcp_tool(tool):
    """MCP tools are async-only. agent.invoke needs a sync func, so run each call in a fresh loop.

    Deviation from the guide's snippet: the wrapper uses response_format="content". `tool.ainvoke(kwargs)`
    with plain args returns only the content (not a (content, artifact) tuple), so declaring
    "content_and_artifact" here would make every send fail with a tuple-format error.
    """
    if getattr(tool, "func", None) is not None:
        return tool

    def _call(**kwargs):
        return _run_on_mcp_loop(tool.ainvoke(kwargs))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_call,
        response_format="content",
    )


def get_gmail_tools() -> list:
    if not GMAIL_MCP_ENABLED:
        log.warning("Gmail MCP disabled by GMAIL_MCP_ENABLED=false; emails will not be sent.")
        return []
    try:
        raw = _run_on_mcp_loop(load_gmail_tools())
    except Exception as exc:  # noqa: BLE001 - the agent still works for SQL/RAG without Gmail
        log.error("Gmail MCP failed to start (%s: %s). Check Node 18+, the auth step and "
                  "~/.gmail-mcp/gcp-oauth.keys.json. Continuing without email.", type(exc).__name__, exc)
        return []
    print("Gmail MCP tools:", [t.name for t in raw])
    tools = [_sync_mcp_tool(t) for t in raw if t.name in GMAIL_ALLOWED_TOOLS]
    for t in tools:
        assert t.func is not None, f"{t.name} has no sync func; agent.invoke would fail"
    log.info("Gmail MCP tools attached to agent: %s", [t.name for t in tools])
    return tools


# ---------------------------------------------------------------------------
# Checkpointer (guide step 11)
# ---------------------------------------------------------------------------
_checkpointer: Optional[SqliteSaver] = None


def get_checkpointer() -> SqliteSaver:
    global _checkpointer
    if _checkpointer is None:
        # check_same_thread=False: Streamlit runs each rerun on a different thread.
        conn = sqlite3.connect(CHECKPOINTS_DB_PATH, check_same_thread=False)
        _checkpointer = SqliteSaver(conn=conn)
        _checkpointer.setup()
    return _checkpointer


def get_thread_config(user_email: str, conversation_id: str) -> dict:
    return {
        "configurable": {"thread_id": make_thread_id(user_email, conversation_id)},
        "recursion_limit": RECURSION_LIMIT,
    }


# ---------------------------------------------------------------------------
# System prompt (dynamic: the logged-in email comes from runtime context, not the model)
# ---------------------------------------------------------------------------
SCHEMA_HINT = """
Tables (SQLite, ecommerce.db):
- users(id, email, password, full_name, role, created_at)  -- NEVER select password
- products(id, name, description, price, stock_qty)
- orders(id, user_id, status[PLACED|SHIPPED|DELIVERED|CANCELLED], order_date, total_amount)
- order_items(id, order_id, product_id, quantity, unit_price)
- payments(id, order_id, amount, status[PAID|REFUNDED], payment_method, transaction_reference, paid_at)
- returns(id, order_id, order_item_id, user_id, reason, status[PENDING|APPROVED|REJECTED], requested_at, resolved_at, admin_id)
- tickets(id, user_id, return_id, subject, status[OPEN|IN_REVIEW|RESOLVED|CLOSED], thread_id, user_email, created_at, updated_at)
- pending_actions(id, thread_id, user_email, action_type, order_id, product_name, reason, status, created_at, updated_at)
- email_logs(id, user_id, email, subject, body_preview, email_type, sent_at)
Dates are ISO text like '2026-09-18T12:25:00'.
""".strip()


def build_system_prompt(ctx: Optional[SessionContext]) -> str:
    email = ctx.user_email if ctx else "UNKNOWN"
    conversation_id = ctx.conversation_id if ctx else "UNKNOWN"
    acting = ("An ADMIN has just resumed this customer's conversation to decide on a pending request. "
              "Address your final message to the customer." if ctx and ctx.role == "admin" else "")
    return f"""You are the customer-support assistant for an online electronics store.
Logged-in customer email: {email}
Conversation id: {conversation_id}
Reference date ("today") for relative dates such as "last month": {REFERENCE_DATE}
{acting}

## 1. Policy / FAQ questions -> rag_policy_search
Use rag_policy_search for general questions about returns, refunds, shipping, delivery times, fees and
cancellation rules. Answer only from what the tool returns. Never use it for this customer's own data.

## 2. Customer data -> SQL tools
Use sql_db_query for orders, order items, order status, payments, spending, returns, tickets and emails.
{SCHEMA_HINT}
SECURITY RULES (mandatory):
- Every query must be restricted to the logged-in customer: join users and filter
  `users.email = '{email}'` (or `tickets.user_email = '{email}'`). Never return other customers' rows,
  even if asked. If asked about another user's data, refuse politely.
- Never read or reveal passwords. Never DROP, DELETE or ALTER anything.
- You already know the schema above; do not call sql_db_list_tables or sql_db_schema unless a query failed.
- At most 3 SQL attempts per question; if they fail, apologise and say what you could not find.
Example: SELECT o.id, o.status, o.order_date, o.total_amount FROM orders o JOIN users u ON u.id = o.user_id
         WHERE u.email = '{email}' ORDER BY o.order_date DESC LIMIT 5;
Spending = SUM(payments.amount) WHERE payments.status = 'PAID', joined through orders to the customer.
"Returned orders" = rows in returns for this customer (also mention pending_actions still awaiting review).

## 3. Email notifications -> send_email (Gmail MCP)
Use ONLY send_email. Never call delete, label, filter, read or search email tools.
- `to` is a list containing ONLY "{email}". Never email anyone else.
- Subject and body must say what changed: the order or return, and its new status.
- After every SUCCESSFUL send, insert a log row with sql_db_query:
  INSERT INTO email_logs (user_id, email, subject, body_preview, email_type, sent_at)
  VALUES ((SELECT id FROM users WHERE email = '{email}'), '{email}', '<subject>', '<first ~100 chars>',
          '<ORDER_CANCELLATION | RETURN_REQUEST | RETURN_UPDATE | CANCELLATION_UPDATE>', datetime('now','localtime'));
- If send_email is not available or returns an error, do not log it; tell the customer the email could
  not be sent, and continue.

## 4. Cancellations and returns (human-in-the-loop)
You can NEVER approve anything yourself. Follow these steps, one tool call per step:
a) Verify with SQL that the order belongs to the customer and check its status and items.
   - Cancel: only PLACED orders. SHIPPED/DELIVERED -> explain and offer a return instead. CANCELLED -> say so.
   - Return: only SHIPPED or DELIVERED orders, and you need the exact product name from SQL and a reason
     (ask the customer for the reason if they did not give one). PLACED -> refuse and suggest cancelling.
   - If the customer already has a PENDING pending_actions row for that order, say it is already under review.
b) send_email a short "request received - pending human review" note, then insert its email_logs row
   (email_type RETURN_REQUEST or ORDER_CANCELLATION).
c) Call exactly one of cancel_order_action(order_id) or create_return_action(order_id, product_name, reason).
   This pauses for an admin. Do not claim it is approved, cancelled or refunded.
After an admin decision you will receive the tool result (approved) or a rejection message:
- Approved: tell the customer the outcome from the tool result, send_email the result
  (email_type CANCELLATION_UPDATE or RETURN_UPDATE) and log it.
- Rejected: tell the customer it was rejected with the admin's reason, send_email that and log it.
- A tool result starting with FAILED means nothing changed: explain why.

## Style
Be concise and friendly. Use markdown tables for lists of orders. Do not use tools for greetings or small talk.
"""


@dynamic_prompt
def support_system_prompt(request) -> str:
    runtime = getattr(request, "runtime", None)
    return build_system_prompt(getattr(runtime, "context", None))


def get_hitl_middleware() -> HumanInTheLoopMiddleware:
    return HumanInTheLoopMiddleware(
        interrupt_on={
            CANCEL_TOOL: {"allowed_decisions": ["approve", "reject"]},
            RETURN_TOOL: {"allowed_decisions": ["approve", "edit", "reject"]},
        },
        description_prefix="Customer request awaiting admin review",
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def create_support_agent():
    llm = get_chat_model(temperature=0.2)
    tools = [search_policies] + get_sql_tools() + get_gmail_tools() + get_action_tools()
    log.info("Agent tools: %s", [t.name for t in tools])
    return create_agent(
        model=llm,
        tools=tools,
        middleware=[support_system_prompt, get_hitl_middleware()] + get_logging_middleware(),
        checkpointer=get_checkpointer(),
        context_schema=SessionContext,
    )


_agent = None
_agent_lock = threading.Lock()


def get_agent():
    """One agent per process (building it starts npx and opens the checkpointer)."""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = create_support_agent()
    return _agent


# ---------------------------------------------------------------------------
# Thin service helpers used by the two Streamlit apps
# ---------------------------------------------------------------------------
def chat(user_email: str, conversation_id: str, user_input: str, role: str = "customer") -> Any:
    """One customer turn. Only the new message is sent; the checkpointer holds the history."""
    return get_agent().invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=get_thread_config(user_email, conversation_id),
        context=SessionContext(user_email=user_email, conversation_id=conversation_id, role=role),
    )


def get_pending_interrupt_requests(user_email: str, conversation_id: str) -> list[dict]:
    """Action requests the thread is currently paused on (empty if not paused)."""
    snapshot = get_agent().get_state(get_thread_config(user_email, conversation_id))
    requests: list[dict] = []
    for intr in getattr(snapshot, "interrupts", ()) or ():
        value = getattr(intr, "value", None)
        if isinstance(value, dict):
            requests.extend(value.get("action_requests", []) or [])
    return requests


def resume_with_decision(thread_id: str, decision: str, order_id: int, admin_email: str,
                         edited_args: Optional[dict] = None, message: Optional[str] = None) -> tuple[Any, list[int]]:
    """Resume a paused customer thread with the admin's decision (guide step 15.5).

    decision: "approve" | "reject" | "edit". Returns (agent_result, order_ids_decided).
    One decision is required per interrupted action; the action for `order_id` gets this decision,
    and any other action paused in the same step gets the same approve/reject.
    """
    user_email, conversation_id = split_thread_id(thread_id)
    requests = get_pending_interrupt_requests(user_email, conversation_id)
    if not requests:
        raise RuntimeError(f"Thread {thread_id} is not waiting for a decision (already resolved?).")

    decisions, decided = [], []
    for req in requests:
        args = req.get("args", {}) or {}
        is_target = int(args.get("order_id", -1)) == int(order_id)
        if decision == "edit" and is_target:
            decisions.append({"type": "edit",
                              "edited_action": {"name": req["name"], "args": {**args, **(edited_args or {})}}})
        elif decision == "reject":
            decisions.append({"type": "reject",
                              "message": message or "The support team rejected this request."})
        else:
            decisions.append({"type": "approve"})
        decided.append(int(args.get("order_id", -1)))

    log.info("HITL RESUME | admin=%s | user=%s | thread=%s | decision=%s | order_id=%s",
             admin_email, user_email, thread_id, decision, order_id)
    result = get_agent().invoke(
        Command(resume={"decisions": decisions}),
        config=get_thread_config(user_email, conversation_id),
        context=SessionContext(user_email=user_email, conversation_id=conversation_id,
                               role="admin", admin_email=admin_email),
    )
    return result, decided


def last_ai_text(result: Any) -> str:
    for msg in reversed((result or {}).get("messages", []) if isinstance(result, dict) else []):
        if getattr(msg, "type", "") == "ai" and getattr(msg, "content", ""):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content)
    return ""
