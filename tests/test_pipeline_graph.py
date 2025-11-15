from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ragdoll.pipeline import IngestionOptions, IngestionPipeline
from ragdoll.entity_extraction.models import Graph, GraphNode


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeEntityExtractor:
    graph_retriever_enabled = True

    def __init__(self):
        self.graph = Graph(nodes=[GraphNode(id="1", type="Person", name="Alice")], edges=[])
        self.retriever_created_with = None

    async def extract(self, documents):
        self.extracted = documents
        return self.graph

    def create_graph_retriever(self, graph=None, **kwargs):
        self.retriever_created_with = graph
        return "retriever"


class ParallelEntityExtractor:
    graph_retriever_enabled = False

    def __init__(self):
        self.calls = 0

    async def extract(self, documents):
        self.calls += 1
        node = GraphNode(id=str(self.calls), type="Chunk", name=f"chunk-{self.calls}")
        return Graph(nodes=[node], edges=[])

    def create_graph_retriever(self, graph=None, **kwargs):
        raise AssertionError("Should not create retriever in this test")


class StubLoader:
    def ingest_documents(self, sources):
        return [{"page_content": source, "metadata": {}} for source in sources]


@pytest.mark.anyio
async def test_pipeline_exposes_graph_retriever():
    options = IngestionOptions(
        skip_vector_store=True,
        skip_graph_store=True,
        extract_entities=True,
        collect_metrics=False,
    )
    fake_extractor = FakeEntityExtractor()
    pipeline = IngestionPipeline(
        content_extraction_service=StubLoader(),
        text_splitter=None,
        embedding_model=object(),
        entity_extractor=fake_extractor,
        options=options,
    )

    doc = Document(page_content="hello world", metadata={})
    await pipeline._extract_entities([doc])

    assert pipeline.graph_retriever == "retriever"
    assert pipeline.stats["graph_retriever_available"] is True
    assert pipeline.last_graph == fake_extractor.graph


@pytest.mark.anyio
async def test_parallel_extraction_merges_graphs():
    options = IngestionOptions(
        skip_vector_store=True,
        skip_graph_store=True,
        extract_entities=True,
        collect_metrics=False,
        parallel_extraction=True,
        max_workers=2,
    )
    extractor = ParallelEntityExtractor()
    pipeline = IngestionPipeline(
        content_extraction_service=StubLoader(),
        text_splitter=None,
        embedding_model=object(),
        entity_extractor=extractor,
        options=options,
    )

    docs = [
        Document(page_content=f"doc {idx}", metadata={"idx": idx}) for idx in range(4)
    ]
    await pipeline._extract_entities(docs)

    assert extractor.calls >= 1
    assert pipeline.last_graph is not None
    assert len(pipeline.last_graph.nodes) == extractor.calls
