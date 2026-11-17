"""Create and seed ecommerce.db from ecommerce_setup.sql (guide step 5).

Usage:  python -m uv run python -m langchain_bot.db_init
Warning: the seed script drops and recreates every table, so this wipes existing data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_bot.config import DB_PATH, SQL_SEED_PATH


def init_database() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sql = SQL_SEED_PATH.read_text(encoding="utf-8")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()
    return DB_PATH


def main() -> None:
    path = init_database()
    print(f"Database initialised at {path}")


if __name__ == "__main__":
    main()
