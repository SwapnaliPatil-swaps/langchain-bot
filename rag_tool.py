"""RAG policy search tool over ChromaDB (spec 4.3 + 6, guide step 8).

The tool is exposed to the agent as `rag_policy_search` (the name the spec uses);
the Python function is `search_policies` (the name the guide uses).
"""

from __future__ import annotations

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_bot.llm import get_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:  # preferred integration package
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - fallback for older installs
    from langchain_community.vectorstores import Chroma

# Importing config runs load_dotenv(), so OpenAIEmbeddings finds OPENAI_API_KEY
# regardless of which module imported first.
from langchain_bot.config import CHROMA_DIR, EMBEDDING_MODEL, POLICIES_DIR

COLLECTION_NAME = "store_policies"
TOP_K = 4

_vector_store: Chroma | None = None


def load_policy_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(POLICIES_DIR.glob("*.txt")):
        for doc in TextLoader(str(path), encoding="utf-8").load():
            doc.metadata["source"] = path.name
            documents.append(doc)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = get_embeddings()

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("Loaded existing Chroma index from", CHROMA_DIR)
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        return _vector_store

    print("Loading policy documents and building Chroma index ...")
    chunks = split_documents(load_policy_documents())
    if not chunks:
        raise RuntimeError(f"No policy .txt files found in {POLICIES_DIR}")
    _vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Indexed {len(chunks)} chunks into {CHROMA_DIR}")
    return _vector_store


@tool("rag_policy_search")
def search_policies(query: str) -> str:
    """Search the store's policy knowledge base (returns policy, shipping policy, FAQ on
    returns/refunds/cancellations).

    Use ONLY for general policy or FAQ questions, e.g. return window, refund timelines,
    who pays return shipping, shipping times and fees, cancellation rules.
    Do NOT use for user-specific data (orders, order status, payments, returns, tickets);
    use the SQL tools for those.
    """
    results = get_vector_store().similarity_search(query, k=TOP_K)
    if not results:
        return "No relevant policy text found."
    parts = [f"[{i}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content.strip()}"
             for i, doc in enumerate(results, start=1)]
    return "\n\n---\n\n".join(parts)


def initialize_vector_store() -> None:
    """Build or load the index at app start-up so the first question is fast."""
    get_vector_store()


__all__ = ["search_policies", "initialize_vector_store", "load_policy_documents", "get_vector_store"]
