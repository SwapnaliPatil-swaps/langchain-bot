# ShopBot – LangChain 1.x E-commerce Support Agent (Capstone)

One LangChain 1.x agent behind two Streamlit apps:

| Piece | Where |
|---|---|
| Customer chat (port 8501) | `app.py` |
| Admin dashboard (port 8502) | `app_admin.py` |
| Single main agent + checkpointer + HITL + Gmail MCP | `src/langchain_bot/agent.py` |
| RAG tool `rag_policy_search` (ChromaDB) | `src/langchain_bot/rag_tool.py` |
| SQL Database Toolkit | `src/langchain_bot/sql_tools.py` |
| Guarded actions `cancel_order_action`, `create_return_action` | `src/langchain_bot/action_tools.py` |
| Logging middleware → `logs/app.log` | `src/langchain_bot/logging_middleware.py` |
| Config / env / logger | `src/langchain_bot/config.py` |
| Login | `src/langchain_bot/auth.py` |
| Identity passed to tools (`ToolRuntime`) | `src/langchain_bot/context.py` |
| `pending_actions` queue helpers | `src/langchain_bot/hitl_utils.py` |
| Sidebar thread ids (no messages) | `src/langchain_bot/threads.py` → `chat_threads.json` |
| DB seed | `src/langchain_bot/db_init.py` + `ecommerce_setup.sql` |
| Policy knowledge base | `policies/*.txt` |

## Architecture

```
Streamlit (thin)  ──invoke(context=SessionContext, thread_id=email:conversation_id)──►  ONE agent (create_agent)
                                                                                        ├─ rag_policy_search ─► chroma_db/
                                                                                        ├─ SQL toolkit ───────► ecommerce.db
                                                                                        ├─ send_email ────────► Gmail MCP server (npx)
                                                                                        ├─ cancel/return action tools ◄─ HumanInTheLoopMiddleware (interrupt)
                                                                                        ├─ logging middleware ► logs/app.log
                                                                                        └─ SqliteSaver ───────► checkpoints.sqlite
```

Customer asks to cancel/return → agent verifies with SQL → `send_email` "request received" + `email_logs` row →
calls the action tool → **interrupt** → row in `pending_actions` → admin approves / edits / rejects on 8502 →
`resume_with_decision` resumes the **same thread** with `Command(resume=...)` → tool runs SQL updates →
`send_email` with the outcome + `email_logs` row → reply appears in the customer's conversation.

## Prerequisites

- Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/)
- Node.js 18+ (for `npx`, which starts the Gmail MCP server)
- An OpenAI API key
- A Google Cloud OAuth client (Desktop app) with the Gmail API enabled

## Setup

```bash
# 1. dependencies (creates .venv and installs this package in editable mode)
python -m uv sync

# 2. secrets
cp .env.example .env          # Windows: copy .env.example .env   -> then put your OPENAI_API_KEY in it

# 3. business database (drops and recreates all tables)
python -m uv run python -m langchain_bot.db_init
```

### ⚠️ Before testing email: use your own address

The seed customer `swapnali.p@gmail.com` is a real-looking address. The agent emails the logged-in
customer, so point that customer at an inbox you own first:

```bash
python -m uv run python -c "import sqlite3; c=sqlite3.connect('ecommerce.db'); c.execute(\"UPDATE users SET email='YOUR_GMAIL@gmail.com' WHERE id=1\"); c.execute(\"UPDATE tickets SET user_email='YOUR_GMAIL@gmail.com' WHERE user_id=1\"); c.commit()"
```
Then log in with `YOUR_GMAIL@gmail.com` / `swap@123`.

### Gmail MCP sign-in (once)

1. Put the OAuth client file at `%USERPROFILE%\.gmail-mcp\gcp-oauth.keys.json` (macOS/Linux: `~/.gmail-mcp/gcp-oauth.keys.json`).
2. Run and sign in with your test Gmail account:
   ```bash
   npx -y @gongrzhe/server-gmail-autoauth-mcp auth
   ```
   The token is saved in `.gmail-mcp/credentials.json` in your user folder, not in this project.

If Gmail is not set up, the agent still starts (SQL + RAG work), logs the error, and tells customers the
email could not be sent. Set `GMAIL_MCP_ENABLED=false` in `.env` to skip it entirely.

## Run (two terminals)

```bash
python -m uv run streamlit run app.py
python -m uv run streamlit run app_admin.py --server.port 8502
```

| Role | Login |
|---|---|
| Customer | `swapnali.p@gmail.com` / `swap@123` (or your remapped Gmail) |
| Customer | `bob@example.com` / `bob123`, `charlie@example.com` / `charlie123` |
| Admin | `admin@example.com` / `admin123` |

## Test script (guide checkpoints)

Log in as customer 1 and watch the terminal / `logs/app.log`.

| Check | Ask | Expected |
|---|---|---|
| RAG | "What is your returns policy?" / "How long do refunds take?" | answer from policy text; log shows `rag_policy_search` |
| SQL | "Show my last 5 orders" / "What items are in order #10?" | only your orders |
| Spend | "How much did I spend last month?" | sum of PAID payments for Aug 2026 (reference date 2026-09-30) |
| Isolation | "Show bob@example.com's orders" | refused / nothing returned |
| Memory | "My name is Alice and I love hiking" → "What did I tell you?" → new conversation → "What's my name?" | remembers only in the first thread; survives a restart |
| Gmail | "Send me a one-line email that the support bot is connected. Subject: MCP capstone test." | mail arrives; `send_email` in log; `email_logs` row |
| Cancel | "Cancel order 13" | pending message → admin **Approve** → order 13 CANCELLED, payment REFUNDED, email sent |
| Return | "Return the Bluetooth Speaker from order 10, it stopped charging" | pending → admin Approve (optionally edit reason) → return row APPROVED, ticket, refund row |
| Invalid | "Return something from order 13" (PLACED) | refused, suggests cancelling |
| Reject | Any request → admin **Reject** with a reason | customer gets the reason in chat + email |

After an admin decision, click **Refresh** in the customer sidebar to see the agent's reply.

## Design notes, simplifications and deviations

- **Plain-text passwords**, no sign-up, no password reset — accepted simplification for this lab.
- **Chat transcripts are never stored in SQL.** History comes from `agent.get_state(config)`;
  `chat_threads.json` stores only conversation ids per user for the sidebar.
- **Thread id** = `"{user_email}:{conversation_id}"`; tickets store `user_email` and the conversation UUID in `thread_id`.
- **Cancellation rule:** only `PLACED` orders can be cancelled (guide + FAQ); shipped/delivered orders go through returns.
- **Approval-time writes:** following the guide, return/ticket/refund rows are written by the action tool
  *after* the admin approves; while waiting, the request lives in `pending_actions` (the admin's queue).
  The dashboard shows that queue (PENDING / APPROVED / REJECTED) and the tickets/returns history, both with
  status, email and date filters.
- **Edit:** the admin can edit product name / reason of a return before approving (HITL `edit` decision).
  Cancellations allow approve/reject only.
- **Gmail tools:** the MCP server also exposes read/search/delete/label tools; only `send_email` is given to the agent.
- **Sync wrapper for MCP tools** (guide 14.2/14.3): kept as described, but the wrapper uses
  `response_format="content"` because `tool.ainvoke(kwargs)` returns content only, not a `(content, artifact)` tuple.
- **`langchain.mcp.MCPAdapter`** is used when present; otherwise `langchain-mcp-adapters`' `MultiServerMCPClient` is used with the same server config.
- **SQL safety net:** the logging middleware blocks `DROP/DELETE/ALTER/PRAGMA…` and any query touching `password`, on top of the per-user filter rules in the system prompt.
- **`recursion_limit` = 40** (guide says 20) because one HITL turn chains several SQL, email and action steps; set `RECURSION_LIMIT` in `.env` to change it.
- **Reference date** for "last month" is `2026-09-30` (from the seed file); change with `REFERENCE_DATE`.
- `.env`, `*.db`, `checkpoints.sqlite`, `chroma_db/`, `chat_threads.json`, `logs/*.log` are git-ignored.

## Deployment

Run locally. Vercel cannot host this app: Streamlit needs a long-running server with websockets, and
the Gmail MCP server runs as a local `npx` process using OAuth tokens from your user folder.
