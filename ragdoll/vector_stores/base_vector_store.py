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

    def add_documents(self, documents: Sequence[Document]) -> List[str]:
        """Add documents by delegating to the wrapped VectorStore."""
        return self._store.add_documents(list(documents))

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
