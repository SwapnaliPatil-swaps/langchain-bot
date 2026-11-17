"""Simple login helpers (spec 3.1, guide step 6.1).

Plain-text passwords are an accepted simplification for this lab (see README).
user_email is the only user identifier used by the rest of the application.
"""

from __future__ import annotations

import sqlite3
from typing import Optional, TypedDict

from langchain_bot.config import DB_PATH


class UserRecord(TypedDict):
    email: str
    full_name: str
    role: str


def authenticate_user(email: str, password: str, role: Optional[str] = None) -> Optional[UserRecord]:
    query = "SELECT email, full_name, role FROM users WHERE email = ? AND password = ?"
    params: list[str] = [email.strip().lower(), password]
    if role:
        query += " AND role = ?"
        params.append(role)

    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(query, params).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    return {"email": row[0], "full_name": row[1], "role": row[2]}
