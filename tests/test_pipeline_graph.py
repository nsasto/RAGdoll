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
        self.graph = Graph(
            nodes=[GraphNode(id="1", type="Person", name="Alice")], edges=[]
        )
        self.retriever_created_with = None

    async def extract(self, documents):
        self.extracted = documents
        return self.graph

    def create_graph_retriever(self, graph=None, **kwargs):
        self.retriever_created_with = graph
        return "retriever"


class StubLoader:
    def ingest_documents(self, sources):
        return [{"page_content": source, "metadata": {}} for source in sources]


class RecordingGraphStore:
    def __init__(self):
        self.saved = []

    def save_graph(self, graph, output_file=None):
        self.saved.append(graph)
        return True


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
    assert pipeline.stats["relationships_extracted"] == len(fake_extractor.graph.edges)


@pytest.mark.anyio
async def test_extraction_updates_stats_and_persists_graph():
    options = IngestionOptions(
        skip_vector_store=True,
        skip_graph_store=False,
        extract_entities=True,
        collect_metrics=False,
    )

    class SingleGraphExtractor:
        graph_retriever_enabled = False

        def __init__(self):
            self.graph = Graph(
                nodes=[
                    GraphNode(id="a", type="Entity", name="Alice"),
                    GraphNode(id="b", type="Entity", name="Bob"),
                ],
                edges=[],
            )

        async def extract(self, documents):
            return self.graph

        def create_graph_retriever(self, graph=None, **kwargs):
            raise AssertionError("Retriever creation should not run in this test.")

    graph_store = RecordingGraphStore()
    extractor = SingleGraphExtractor()
    pipeline = IngestionPipeline(
        content_extraction_service=StubLoader(),
        text_splitter=None,
        embedding_model=object(),
        entity_extractor=extractor,
        graph_store=graph_store,
        options=options,
    )

    docs = [Document(page_content="doc", metadata={})]
    await pipeline._extract_entities(docs)

    assert pipeline.stats["entities_extracted"] == len(extractor.graph.nodes)
    assert graph_store.saved and graph_store.saved[0] is extractor.graph
