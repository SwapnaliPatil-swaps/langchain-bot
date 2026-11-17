"""SQL Database Toolkit (spec 4.2, guide step 13).

The agent generates SQL itself through the toolkit; there are deliberately no
hand-written helpers like get_orders_for_user().
"""

from __future__ import annotations

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_bot.llm import get_chat_model

from langchain_bot.config import DB_PATH, OPENAI_MODEL

# Chat transcripts are not business data (spec 5.2), so the optional sample
# conversations/messages tables are hidden from the agent.
_IGNORED_TABLES = ["conversations", "messages"]

_db: SQLDatabase | None = None


def get_database() -> SQLDatabase:
    global _db
    if _db is None:
        _db = SQLDatabase.from_uri(
            f"sqlite:///{DB_PATH.as_posix()}",
            ignore_tables=_IGNORED_TABLES,
            sample_rows_in_table_info=0,  # keeps passwords and other users' rows out of the schema text
        )
    return _db


def get_sql_tools():
    toolkit = SQLDatabaseToolkit(db=get_database(), llm=get_chat_model(temperature=0))
    return toolkit.get_tools()
