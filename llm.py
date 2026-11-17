"""Chat model + embeddings factory. Provider is chosen in .env:

    MODEL_PROVIDER=openai     MODEL_NAME=gpt-4.1-mini        (needs OPENAI_API_KEY)
    MODEL_PROVIDER=anthropic  MODEL_NAME=claude-sonnet-5-5   (needs ANTHROPIC_API_KEY)

Anthropic has no embeddings API, so with a non-OpenAI provider the RAG index uses
Chroma's built-in local model (all-MiniLM-L6-v2, ~80 MB download on first run, no key needed).
If you switch provider later, delete the chroma_db/ folder so the index is rebuilt.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings

from langchain_bot.config import EMBEDDING_MODEL  # importing config also loads .env

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"


def get_chat_model(temperature: float = 0.2):
    # Newer Claude models reject the `temperature` parameter, so only send it to OpenAI.
    if MODEL_PROVIDER == "openai":
        return init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=temperature)
    return init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)


class LocalEmbeddings(Embeddings):
    """Wraps Chroma's default ONNX embedding model as a LangChain Embeddings object."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> Embeddings:
    if MODEL_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return LocalEmbeddings()