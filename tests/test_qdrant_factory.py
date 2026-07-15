import sys
from types import SimpleNamespace

from ragdoll.vector_stores.factory import _create_qdrant_store


class Embeddings:
    def embed_query(self, text):
        return [0.0, 0.0, 0.0]


def test_qdrant_factory_creates_missing_collection_and_attaches_embedding(monkeypatch):
    created = []

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def collection_exists(self, name):
            return False

        def create_collection(self, **kwargs):
            created.append(kwargs)

    class VectorParams:
        def __init__(self, **kwargs):
            self.size = kwargs["size"]
            self.distance = kwargs["distance"]

    qdrant_module = SimpleNamespace(
        QdrantClient=Client,
        models=SimpleNamespace(
            Distance=SimpleNamespace(COSINE="cosine"),
            VectorParams=VectorParams,
        ),
    )
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_module)

    class Store:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    store = _create_qdrant_store(
        Store,
        Embeddings(),
        {"url": "http://qdrant:6333", "collection_name": "docs"},
    )

    assert created[0]["vectors_config"].size == 3
    assert store.kwargs["collection_name"] == "docs"
    assert isinstance(store.kwargs["embedding"], Embeddings)
