import pytest
from langchain_core.documents import Document

from ragdoll.contracts import IngestionSpec
from ragdoll.corpus import InMemoryCorpusIndex, StagedGeneration
from ragdoll.entity_extraction.models import Graph, GraphNode
from ragdoll.graph_index import InMemoryGraphBackend, VersionedGraphIndex
from ragdoll.ingestion.jobs import DurableIngestion, JobStatus
from ragdoll.generation_state import GenerationRecord, MemoryGenerationStateStore
from ragdoll.contracts import CorpusId, GenerationId, TenantId


@pytest.mark.asyncio
async def test_graph_generation_is_invisible_until_corpus_promotion():
    state = MemoryGenerationStateStore()
    generation = StagedGeneration(
        id=GenerationId("generation-one"),
        tenant=TenantId("acme"),
        corpus=CorpusId("docs"),
        document_count=1,
        checksum="checksum",
    )
    await state.put(
        GenerationRecord(
            id=generation.id,
            tenant=generation.tenant,
            corpus=generation.corpus,
            vector_ids=("chunk-one",),
            document_count=1,
            checksum="checksum",
        )
    )
    graph_index = VersionedGraphIndex(InMemoryGraphBackend(), state)
    await graph_index.stage(
        generation,
        Graph(nodes=[GraphNode(id="auth", type="concept", name="Authentication")]),
    )

    assert (
        await graph_index.query(tenant="acme", corpus="docs", question="Authentication")
        == []
    )

    await state.promote(generation.id)
    results = await graph_index.query(
        tenant="acme", corpus="docs", question="Authentication"
    )
    assert results[0].metadata["generation_id"] == generation.id


@pytest.mark.asyncio
async def test_graph_stage_failure_does_not_publish_vector_generation():
    index = InMemoryCorpusIndex()

    class FailingGraphIndex:
        async def stage(self, generation, graph):
            raise OSError("neo4j unavailable")

    async def prepare(spec):
        return [Document(page_content="new", metadata={"source": "manual.md"})]

    async def graph_builder(documents):
        assert documents[0].metadata["vector_id"] == documents[0].metadata["chunk_id"]
        return Graph()

    ingestion = DurableIngestion(
        index=index,
        prepare=prepare,
        graph_builder=graph_builder,
        graph_index=FailingGraphIndex(),
    )
    job = await ingestion.submit(IngestionSpec(corpus="docs", sources=["manual.md"]))
    result = await job.wait()

    assert result.status is JobStatus.FAILED
    assert await index.query("default", "docs", "new") == []
    assert index._generations == {}


@pytest.mark.asyncio
async def test_graph_discard_removes_only_the_staged_generation():
    state = MemoryGenerationStateStore()
    backend = InMemoryGraphBackend()
    graph_index = VersionedGraphIndex(backend, state)
    first = StagedGeneration(
        GenerationId("one"), TenantId("acme"), CorpusId("docs"), 1, "one"
    )
    second = StagedGeneration(
        GenerationId("two"), TenantId("acme"), CorpusId("docs"), 1, "two"
    )
    graph = Graph(nodes=[GraphNode(id="auth", type="concept", name="Authentication")])
    await graph_index.stage(first, graph)
    await graph_index.stage(second, graph)

    await graph_index.discard(second)

    assert ("acme", "docs", "one") in backend._graphs
    assert ("acme", "docs", "two") not in backend._graphs
