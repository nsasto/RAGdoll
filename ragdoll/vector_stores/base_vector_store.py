from __future__ import annotations

from typing import Any, List, Sequence, Type, TypeVar

from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_core.vectorstores import VectorStore

VectorStoreT = TypeVar("VectorStoreT", bound=VectorStore)


class BaseVectorStore:
    """Thin wrapper that delegates to a LangChain VectorStore implementation."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    @property
    def store(self) -> VectorStore:
        """Expose the underlying LangChain VectorStore instance."""
        return self._store

    def add_documents(
        self, documents: Sequence[Document], batch_size: int | None = None
    ) -> List[str]:
        """Add documents, splitting into batches if the backend advertises a limit."""
        docs = list(documents)
        if not docs:
            return []

        limit = batch_size or self._detect_batch_limit()
        if not limit or limit <= 0 or len(docs) <= limit:
            return self._store.add_documents(docs)

        ids: List[str] = []
        for start in range(0, len(docs), limit):
            ids.extend(self._store.add_documents(docs[start : start + limit]))
        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return the top-k similar documents from the wrapped store."""
        return self._store.similarity_search(query, k=k)

    def delete(self, ids: Sequence[str]) -> Any:
        """Delete documents if the wrapped store supports deletion."""
        if not hasattr(self._store, "delete"):
            raise NotImplementedError(
                f"{self._store.__class__.__name__} does not implement delete()."
            )
        return self._store.delete(ids=list(ids))

    @classmethod
    def from_documents(
        cls,
        store_cls: Type[VectorStoreT],
        documents: Sequence[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "BaseVectorStore":
        """Build a wrapped store pre-populated with the provided documents."""
        store = store_cls.from_documents(documents=documents, embedding=embedding, **kwargs)
        return cls(store)

    def _detect_batch_limit(self) -> int | None:
        """Infer the max batch size supported by the underlying store, if exposed."""
        direct_limit = getattr(self._store, "max_batch_size", None)
        if isinstance(direct_limit, int) and direct_limit > 0:
            return direct_limit

        client = getattr(self._store, "_client", None)
        if client is None:
            return None

        getter = getattr(client, "get_max_batch_size", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, int) and value > 0:
                    return value
            except Exception:
                pass

        client_limit = getattr(client, "max_batch_size", None)
        if isinstance(client_limit, int) and client_limit > 0:
            return client_limit
        return None
