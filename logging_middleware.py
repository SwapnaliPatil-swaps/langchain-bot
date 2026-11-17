"""Logging / observability via LangChain 1.x middleware (spec 7, guide step 12).

Every model call and every tool call (SQL toolkit, rag_policy_search, Gmail MCP tools,
action tools) passes through these hooks, which write to logs/app.log and the terminal:
timestamp | user_email | tool/action | input summary | SUCCESS/FAILED (+error).

Both hooks are plain `def` (synchronous). The app always calls agent.invoke, and the
decorators only register the sync hook; see the Gmail notes in agent.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from langchain_bot.config import get_logger

log = get_logger()

_PREVIEW = 160

# Destructive statements the support agent never needs. The toolkit itself would run
# anything, so this is a cheap safety net on top of the system-prompt rules.
_BLOCKED_SQL = re.compile(r"\b(DROP|DELETE|ALTER|ATTACH|DETACH|PRAGMA|VACUUM|REPLACE|TRUNCATE)\b", re.IGNORECASE)
_PASSWORD_SQL = re.compile(r"\bpassword\b", re.IGNORECASE)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _preview(value: Any, limit: int = _PREVIEW) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _user_email(runtime: Any) -> str:
    ctx = getattr(runtime, "context", None)
    return getattr(ctx, "user_email", None) or "-"


def _summarise_args(name: str, args: dict) -> str:
    """Short input summary: ids first, then a trimmed view of everything else."""
    if name == "sql_db_query":
        return "query=" + _preview(args.get("query", ""))
    if name == "send_email":
        return f"to={args.get('to')} subject={_preview(args.get('subject', ''), 80)}"
    keys = ("order_id", "return_id", "product_name", "reason", "query", "table_names")
    picked = {k: args[k] for k in keys if k in args}
    return _preview(picked or args)


@wrap_model_call
def model_logging_middleware(request, handler):
    user = _user_email(getattr(request, "runtime", None))
    messages = getattr(request, "messages", []) or []
    last = messages[-1] if messages else None
    log.info("MODEL REQUEST | user=%s | messages=%d | last=%s", user, len(messages),
             _preview(getattr(last, "content", "")) if last is not None else "-")
    try:
        response = handler(request)
    except GraphBubbleUp:
        raise
    except Exception as exc:  # noqa: BLE001 - log then re-raise
        log.error("MODEL ERROR | user=%s | %s: %s", user, type(exc).__name__, exc)
        raise

    for msg in getattr(response, "result", []) or []:
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            log.info("MODEL RESPONSE | user=%s | tool_calls=%s", user,
                     [f"{tc['name']}({_summarise_args(tc['name'], tc.get('args', {}))})" for tc in tool_calls])
        else:
            log.info("MODEL RESPONSE | user=%s | text=%s", user, _preview(getattr(msg, "content", "")))
    return response


@wrap_tool_call
def tool_logging_middleware(request, handler):
    call = request.tool_call
    name = call.get("name", "?")
    args = call.get("args", {}) or {}
    user = _user_email(getattr(request, "runtime", None))
    summary = _summarise_args(name, args)
    log.info("TOOL START | user=%s | tool=%s | input=%s", user, name, summary)

    if name == "sql_db_query":
        query = str(args.get("query", ""))
        if _BLOCKED_SQL.search(query) or _PASSWORD_SQL.search(query):
            log.warning("TOOL BLOCKED | user=%s | tool=%s | input=%s", user, name, summary)
            return ToolMessage(
                content="Error: this statement is not allowed (no DROP/DELETE/ALTER/PRAGMA and no password "
                        "columns). Use SELECT, or INSERT/UPDATE scoped to the logged-in user.",
                tool_call_id=call["id"], name=name, status="error",
            )

    try:
        result = handler(request)
    except GraphBubbleUp:  # interrupts must propagate untouched
        raise
    except Exception as exc:  # noqa: BLE001 - recorded, then surfaced to the agent
        log.error("TOOL FAILED | user=%s | tool=%s | input=%s | error=%s: %s",
                  user, name, summary, type(exc).__name__, exc)
        return ToolMessage(
            content=f"Error: tool '{name}' failed: {type(exc).__name__}: {exc}. "
                    "Tell the user this step could not be completed.",
            tool_call_id=call["id"], name=name, status="error",
        )

    content = getattr(result, "content", result)
    status = getattr(result, "status", "success")
    text = _preview(content)
    failed = status == "error" or text.startswith(("Error", "FAILED"))
    log.log(30 if failed else 20, "TOOL %s | user=%s | tool=%s | input=%s | output=%s",
            "FAILED" if failed else "SUCCESS", user, name, summary, text)
    return result


def get_logging_middleware():
    return [model_logging_middleware, tool_logging_middleware]
