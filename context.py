"""Per-invocation identity passed to the agent as `context` (guide step 15.1).

Tools read it through ToolRuntime, so the model never has to (and cannot) supply
the user's email itself; it only passes order ids, product names and reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionContext:
    user_email: str
    conversation_id: str
    role: str = "customer"
    admin_email: Optional[str] = None  # set when an admin resumes a customer's thread


def make_thread_id(user_email: str, conversation_id: str) -> str:
    """Logical checkpointer thread id = (user_email, conversation_id) (spec 5.2)."""
    return f"{user_email}:{conversation_id}"


def split_thread_id(thread_id: str) -> tuple[str, str]:
    email, _, conversation_id = thread_id.partition(":")
    return email, conversation_id
