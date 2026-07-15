import uuid

import pytest
from langchain_core.documents import Document

from ragdoll.corpus import InMemoryCorpusIndex, VectorCorpusIndex
from ragdoll.errors import GenerationNotFoundError, TenantIsolationError


@pytest.mark.asyncio
async def test_queries_only_see_promoted_generation():
    index = InMemoryCorpusIndex()
    first = await index.stage(
        tenant="acme",
        corpus="manuals",
        documents=[Document(page_content="old authentication guide")],
    )

    assert (
        await index.query(tenant="acme", corpus="manuals", text="authentication") == []
    )

    await index.promote(first)
    assert [
        doc.page_content
        for doc in await index.query(
            tenant="acme", corpus="manuals", text="authentication"
        )
    ] == ["old authentication guide"]

    second = await index.stage(
        tenant="acme",
        corpus="manuals",
        documents=[Document(page_content="new authentication guide")],
    )
    assert [
        doc.page_content
        for doc in await index.query(
            tenant="acme", corpus="manuals", text="authentication"
        )
    ] == ["old authentication guide"]

    await index.promote(second)
    assert [
        doc.page_content
        for doc in await index.query(
            tenant="acme", corpus="manuals", text="authentication"
        )
    ] == ["new authentication guide"]


@pytest.mark.asyncio
async def test_rollback_restores_previous_generation():
    index = InMemoryCorpusIndex()
    first = await index.stage("acme", "docs", [Document(page_content="version one")])
    second = await index.stage("acme", "docs", [Document(page_content="version two")])
    await index.promote(first)
    await index.promote(second)

    restored = await index.rollback(tenant="acme", corpus="docs")

    assert restored == first.id
    assert [
        doc.page_content for doc in await index.query("acme", "docs", "version")
    ] == ["version one"]


@pytest.mark.asyncio
async def test_generation_cannot_be_promoted_by_another_tenant():
    index = InMemoryCorpusIndex()
    generation = await index.stage("acme", "docs", [Document(page_content="private")])

    with pytest.raises(TenantIsolationError):
        await index.promote(generation.id, tenant="other", corpus="docs")


@pytest.mark.asyncio
async def test_unknown_generation_fails_explicitly():
    index = InMemoryCorpusIndex()

    with pytest.raises(GenerationNotFoundError):
        await index.promote("missing", tenant="acme", corpus="docs")


class RecordingVectorStore:
    def __init__(self):
        self.documents = []
        self.filters = []
        self.deleted = []
        self.add_calls = 0

    async def aadd_documents(self, documents, **kwargs):
        self.add_calls += 1
        start = len(self.documents)
        self.documents.extend(documents)
        return kwargs.get("ids") or [
            f"vector-{start + offset}" for offset in range(len(documents))
        ]

    async def asimilarity_search(self, text, k=4, **kwargs):
        filters = kwargs.get("filter", {})
        self.filters.append(filters)
        matches = [
            doc
            for doc in self.documents
            if all(doc.metadata.get(key) == value for key, value in filters.items())
        ]
        return matches[:k]

    def delete(self, ids):
        self.deleted.extend(ids)


class UpsertingVectorStore(RecordingVectorStore):
    """Models Qdrant/Chroma semantics where an existing point ID is replaced."""

    def __init__(self):
        super().__init__()
        self.points = {}

    async def aadd_documents(self, documents, **kwargs):
        ids = kwargs["ids"]
        for item_id, document in zip(ids, documents):
            self.points[item_id] = document
        self.documents = list(self.points.values())
        self.add_calls += 1
        return ids

    def delete(self, ids):
        super().delete(ids)
        for item_id in ids:
            self.points.pop(item_id, None)
        self.documents = list(self.points.values())


@pytest.mark.asyncio
async def test_vector_index_enforces_active_generation_in_backend_filter(tmp_path):
    store = RecordingVectorStore()
    index = VectorCorpusIndex(store, state_path=tmp_path / "corpora.json")
    generation = await index.stage(
        "acme", "docs", [Document(page_content="private content")]
    )

    assert await index.query("acme", "docs", "private") == []
    await index.promote(generation)
    results = await index.query("acme", "docs", "private")

    assert [doc.page_content for doc in results] == ["private content"]
    assert store.filters[-1] == {
        "tenant_id": "acme",
        "corpus_id": "docs",
        "generation_id": str(generation.id),
    }


@pytest.mark.asyncio
async def test_vector_index_state_survives_adapter_recreation(tmp_path):
    store = RecordingVectorStore()
    state_path = tmp_path / "corpora.json"
    original = VectorCorpusIndex(store, state_path=state_path)
    generation = await original.stage(
        "acme", "docs", [Document(page_content="persisted generation")]
    )
    await original.promote(generation)

    recreated = VectorCorpusIndex(store, state_path=state_path)
    results = await recreated.query("acme", "docs", "persisted")

    assert [doc.page_content for doc in results] == ["persisted generation"]


@pytest.mark.asyncio
async def test_vector_index_rejects_reserved_filter_override():
    store = RecordingVectorStore()
    index = VectorCorpusIndex(store)
    generation = await index.stage("acme", "docs", [Document(page_content="private")])
    await index.promote(generation)

    with pytest.raises(TenantIsolationError):
        await index.query("acme", "docs", "private", filters={"tenant_id": "other"})


@pytest.mark.asyncio
async def test_vector_index_uses_generation_scoped_uuid_backend_ids():
    store = RecordingVectorStore()
    index = VectorCorpusIndex(store)

    generation = await index.stage("acme", "docs", [Document(page_content="stable")])
    record = await index.state.get(generation.id)

    vector_id = store.documents[0].metadata["vector_id"]
    assert record.vector_ids == (vector_id,)
    assert vector_id != store.documents[0].metadata["chunk_id"]
    assert str(uuid.UUID(vector_id)) == vector_id


@pytest.mark.asyncio
async def test_staging_cannot_overwrite_or_discard_active_backend_points():
    store = UpsertingVectorStore()
    index = VectorCorpusIndex(store)
    first = await index.stage(
        "acme", "docs", [Document(page_content="old", metadata={"source": "a.md"})]
    )
    await index.promote(first)
    active_ids = set(store.points)

    staged = await index.stage(
        "acme", "docs", [Document(page_content="new", metadata={"source": "a.md"})]
    )

    assert active_ids.issubset(store.points)
    assert set(store.points) - active_ids
    await index.discard(staged)
    assert set(store.points) == active_ids


@pytest.mark.asyncio
async def test_unchanged_content_does_not_rewrite_vectors():
    store = RecordingVectorStore()
    index = VectorCorpusIndex(store)
    documents = [Document(page_content="same", metadata={"source": "manual.md"})]
    first = await index.stage("acme", "docs", documents)
    await index.promote(first)

    second = await index.stage("acme", "docs", documents)

    assert second.unchanged is True
    assert second.id == first.id
    assert store.add_calls == 1


@pytest.mark.asyncio
async def test_discard_removes_an_unpublished_vector_generation():
    store = RecordingVectorStore()
    index = VectorCorpusIndex(store)
    generation = await index.stage(
        "acme", "docs", [Document(page_content="staged only")]
    )

    await index.discard(generation)

    assert store.deleted
    with pytest.raises(GenerationNotFoundError):
        await index.state.get(generation.id)


@pytest.mark.asyncio
async def test_document_id_is_stable_but_chunk_id_changes_with_source_content():
    store = RecordingVectorStore()
    index = VectorCorpusIndex(store)
    first = await index.stage(
        "acme", "docs", [Document(page_content="old", metadata={"source": "manual.md"})]
    )
    await index.promote(first)
    old = store.documents[-1].metadata

    await index.stage(
        "acme", "docs", [Document(page_content="new", metadata={"source": "manual.md"})]
    )
    new = store.documents[-1].metadata

    assert new["document_id"] == old["document_id"]
    assert new["chunk_id"] != old["chunk_id"]
