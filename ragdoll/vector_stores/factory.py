"""Helpers for instantiating LangChain VectorStore implementations."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any, Dict, Sequence, Type

from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_core.vectorstores import VectorStore

from ragdoll.config.base_config import VectorStoreConfig

from .base_vector_store import BaseVectorStore

_VECTOR_STORE_REGISTRY: Dict[str, str] = {
    "chroma": "langchain_chroma.Chroma",
    "faiss": "langchain_community.vectorstores.FAISS",
    "docarrayinmemory": "langchain_community.vectorstores.DocArrayInMemorySearch",
}


def _resolve_store_class(store_type: str) -> Type[VectorStore]:
    """Resolve a VectorStore class from a registry key or dotted path."""
    key = store_type.lower()
    class_path = _VECTOR_STORE_REGISTRY.get(key, store_type)
    if "." not in class_path:
        raise ValueError(
            f"Unknown vector store '{store_type}'. "
            "Provide a fully-qualified class path or register it in the factory."
        )
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    store_cls = getattr(module, class_name)
    if not issubclass(store_cls, VectorStore):
        raise TypeError(f"{class_path} is not a langchain VectorStore subclass.")
    return store_cls


def _maybe_attach_embedding(
    store_cls: Type[VectorStore],
    kwargs: Dict[str, Any],
    embedding: Embeddings | None,
) -> None:
    if embedding is None:
        return
    signature = inspect.signature(store_cls.__init__)
    if "embedding_function" in signature.parameters:
        kwargs.setdefault("embedding_function", embedding)
    elif "embedding" in signature.parameters:
        kwargs.setdefault("embedding", embedding)


def create_vector_store(
    store_type: str,
    *,
    embedding: Embeddings | None = None,
    **store_kwargs: Any,
) -> BaseVectorStore:
    """Instantiate and wrap a LangChain VectorStore by type name or path."""
    store_cls = _resolve_store_class(store_type)
    kwargs = dict(store_kwargs)
    _maybe_attach_embedding(store_cls, kwargs, embedding)
    store = store_cls(**kwargs)
    return BaseVectorStore(store)


def create_vector_store_from_documents(
    store_type: str,
    documents: Sequence[Document],
    embedding: Embeddings,
    **store_kwargs: Any,
) -> BaseVectorStore:
    """Build a populated vector store instance from documents."""
    store_cls = _resolve_store_class(store_type)
    store = store_cls.from_documents(
        documents=documents, embedding=embedding, **store_kwargs
    )
    return BaseVectorStore(store)


def vector_store_from_config(
    config: VectorStoreConfig,
    *,
    embedding: Embeddings | None = None,
    documents: Sequence[Document] | None = None,
) -> BaseVectorStore:
    """Instantiate a vector store based on the vector_store config section."""
    if not config.enabled:
        raise ValueError("Vector store configuration is disabled.")
    params = dict(config.params or {})
    store_type = config.store_type
    if documents is not None:
        if embedding is None:
            raise ValueError("Embedding model is required when loading documents.")
        return create_vector_store_from_documents(
            store_type, documents, embedding, **params
        )
    return create_vector_store(store_type, embedding=embedding, **params)


__all__ = [
    "BaseVectorStore",
    "create_vector_store",
    "create_vector_store_from_documents",
    "vector_store_from_config",
]
