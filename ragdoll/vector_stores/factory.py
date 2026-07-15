"""Helpers for instantiating LangChain VectorStore implementations."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any, Dict, Sequence, Type

from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_core.vectorstores import VectorStore

from ragdoll.config.base_config import VectorStoreConfig
from ragdoll.utils.env import resolve_env_reference

from .base_vector_store import BaseVectorStore

_VECTOR_STORE_REGISTRY: Dict[str, str] = {
    "chroma": "langchain_chroma.Chroma",
    "faiss": "langchain_community.vectorstores.FAISS",
    "docarrayinmemory": "langchain_community.vectorstores.DocArrayInMemorySearch",
    "qdrant": "langchain_qdrant.QdrantVectorStore",
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

    # FAISS requires a special case for creating an empty index.
    if store_type.lower() == "qdrant":
        store = _create_qdrant_store(store_cls, embedding, kwargs)
    elif store_type.lower() == "faiss" and not kwargs.get("index"):
        if not embedding:
            raise ValueError(
                "FAISS requires an embedding model to create an empty index."
            )
        try:
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore

            dummy_vector = embedding.embed_query("dummy")
            dimension = len(dummy_vector)
            index = faiss.IndexFlatL2(dimension)
            docstore = InMemoryDocstore()
            index_to_docstore_id = {}
            store = store_cls(
                embedding_function=embedding,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
                **kwargs,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import faiss, please install it with `pip install faiss-cpu` or `pip install faiss-gpu`"
            ) from e
    else:
        _maybe_attach_embedding(store_cls, kwargs, embedding)
        store = store_cls(**kwargs)

    return BaseVectorStore(store)


def _create_qdrant_store(
    store_cls: Type[VectorStore],
    embedding: Embeddings | None,
    kwargs: Dict[str, Any],
) -> VectorStore:
    if embedding is None:
        raise ValueError("Qdrant requires an embedding model")
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Qdrant requires `pip install python-ragdoll[scaled]`"
        ) from exc

    collection_name = kwargs.pop("collection_name", None)
    if not collection_name:
        raise ValueError("Qdrant requires params.collection_name")
    client = kwargs.pop("client", None)
    client_keys = {
        "location",
        "url",
        "port",
        "grpc_port",
        "prefer_grpc",
        "https",
        "api_key",
        "prefix",
        "timeout",
        "host",
        "path",
    }
    client_kwargs = {
        key: resolve_env_reference(kwargs.pop(key))
        for key in tuple(kwargs)
        if key in client_keys
    }
    client = client or QdrantClient(**client_kwargs)
    if not client.collection_exists(collection_name):
        dimension = len(embedding.embed_query("ragdoll dimension probe"))
        distance_name = str(kwargs.pop("distance", "cosine")).upper()
        try:
            distance = getattr(models.Distance, distance_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported Qdrant distance: {distance_name}") from exc
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dimension, distance=distance),
        )
    return store_cls(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
        **kwargs,
    )


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
