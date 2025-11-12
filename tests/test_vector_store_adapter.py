import uuid

import pytest
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings

from ragdoll.config.base_config import VectorStoreConfig
from ragdoll.vector_stores import (
    BaseVectorStore,
    create_vector_store,
    create_vector_store_from_documents,
    vector_store_from_config,
)


def _unique_collection() -> str:
    return f"test-{uuid.uuid4().hex}"


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for offline tests."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_documents(self, texts):
        return [self._encode(text) for text in texts]

    def embed_query(self, text):
        return self._encode(text)

    def _encode(self, text: str):
        lower = text.lower()
        cherry = float(lower.count("cherry"))
        pie = float(lower.count("pie"))
        length = float(len(lower))
        unique_terms = float(len(set(lower.split())))
        return [
            cherry * 10.0,
            pie * 10.0,
            length * 0.01,
            unique_terms * 0.01,
        ]


def _sample_documents():
    return [
        Document(page_content="apples and oranges", metadata={"id": "fruit-1", "source": "alpha"}),
        Document(page_content="cherry pies and cakes", metadata={"id": "fruit-2", "source": "beta"}),
    ]


def test_create_vector_store_delegates_to_langchain():
    embeddings = FakeEmbeddings()
    store = create_vector_store(
        "chroma",
        embedding=embeddings,
        collection_name=_unique_collection(),
    )
    inserted_ids = store.add_documents(_sample_documents())
    assert len(inserted_ids) == 2

    results = store.similarity_search("cherry pie recipe", k=1)
    assert len(results) == 1
    assert results[0].metadata.get("source") == "beta"


def test_delete_propagates_to_underlying_store():
    embeddings = FakeEmbeddings()
    store = create_vector_store(
        "chroma",
        embedding=embeddings,
        collection_name=_unique_collection(),
    )
    docs = _sample_documents()
    ids = store.add_documents(docs)
    store.delete([ids[0]])

    results = store.similarity_search("apples", k=2)
    assert all(doc.metadata.get("id") != docs[0].metadata["id"] for doc in results)


def test_create_vector_store_from_documents_bootstraps_state():
    embeddings = FakeEmbeddings()
    store = create_vector_store_from_documents(
        "chroma",
        documents=_sample_documents(),
        embedding=embeddings,
        collection_name=_unique_collection(),
    )
    assert isinstance(store, BaseVectorStore)
    results = store.similarity_search("oranges", k=1)
    assert results[0].metadata.get("source") == "alpha"


def test_vector_store_from_config_respects_params():
    config = VectorStoreConfig(
        enabled=True,
        store_type="chroma",
        params={"collection_name": _unique_collection()},
    )
    embeddings = FakeEmbeddings()
    store = vector_store_from_config(config, embedding=embeddings)
    store.add_documents(_sample_documents())
    top = store.similarity_search("cherry pie", k=1)
    assert top[0].metadata.get("source") == "beta"


def test_vector_store_from_config_requires_embedding_when_loading_documents():
    config = VectorStoreConfig(
        enabled=True,
        store_type="chroma",
        params={"collection_name": _unique_collection()},
    )
    docs = _sample_documents()
    with pytest.raises(ValueError):
        vector_store_from_config(config, documents=docs)
