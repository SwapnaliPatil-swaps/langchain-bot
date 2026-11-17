"""Configuration and environment loading (spec section 8: src/config.py).

All paths are resolved from the project root so the customer app, the admin app and the
CLI helpers share the same ecommerce.db, checkpoints.sqlite, chroma_db/ and logs/.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# src/langchain_bot/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# `uv run` does not load .env by itself, so load it here at import time.
load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = PROJECT_ROOT / "ecommerce.db"
SQL_SEED_PATH = PROJECT_ROOT / "ecommerce_setup.sql"
POLICIES_DIR = PROJECT_ROOT / "policies"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
THREADS_FILE = PROJECT_ROOT / "chat_threads.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHECKPOINTS_DB_PATH = Path(os.getenv("CHECKPOINTS_DB_PATH", "checkpoints.sqlite"))
if not CHECKPOINTS_DB_PATH.is_absolute():
    CHECKPOINTS_DB_PATH = PROJECT_ROOT / CHECKPOINTS_DB_PATH

GMAIL_MCP_ENABLED = os.getenv("GMAIL_MCP_ENABLED", "true").strip().lower() not in {"0", "false", "no"}

_LOGGER_NAME = "support_bot"


def get_logger() -> logging.Logger:
    """Return the app logger that writes to logs/app.log (and the terminal).

    Configured once per process; Streamlit reruns the script many times, so the
    handler check prevents duplicate log lines.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
