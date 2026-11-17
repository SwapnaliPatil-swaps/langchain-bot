"""Per-user list of conversation ids for the sidebar (guide step 11.2).

Only ids and labels are stored here. Message bodies live in the LangGraph
checkpointer and are read back with agent.get_state(config).
"""

from __future__ import annotations

import json
from uuid import uuid4

from langchain_bot.config import THREADS_FILE


def _load_all() -> dict[str, list[dict]]:
    if not THREADS_FILE.exists():
        return {}
    try:
        return json.loads(THREADS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_threads(data: dict[str, list[dict]]) -> None:
    THREADS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_threads(user_email: str) -> list[dict]:
    return _load_all().get(user_email, [])


def add_thread(user_email: str, conversation_id: str) -> None:
    data = _load_all()
    threads = data.setdefault(user_email, [])
    if not any(t["id"] == conversation_id for t in threads):
        threads.insert(0, {"id": conversation_id, "label": f"Chat {conversation_id[:8]}"})
    save_threads(data)


def new_conversation(user_email: str) -> str:
    conversation_id = str(uuid4())
    add_thread(user_email, conversation_id)
    return conversation_id
