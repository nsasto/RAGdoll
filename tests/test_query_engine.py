import asyncio

import pytest
from langchain_core.documents import Document

from ragdoll.corpus import InMemoryCorpusIndex
from ragdoll.errors import QueryTimeoutError
from ragdoll.query import QueryEngine, QueryOptions


class RecordingLLM:
    def __init__(self, answer="answer"):
        self.answer = answer
        self.prompts = []

    async def call(self, prompt):
        self.prompts.append(prompt)
        return self.answer


@pytest.mark.asyncio
async def test_query_enforces_scope_filters_and_returns_citations():
    index = InMemoryCorpusIndex()
    generation = await index.stage(
        "acme",
        "docs",
        [
            Document(
                page_content="Authentication uses signed tokens.",
                metadata={"source": "auth.md", "audience": "public"},
            ),
            Document(
                page_content="Internal draft.",
                metadata={"source": "draft.md", "audience": "internal"},
            ),
        ],
    )
    await index.promote(generation)
    llm = RecordingLLM()
    engine = QueryEngine(index=index, llm_caller=llm)

    result = await engine.query(
        tenant="acme",
        corpus="docs",
        question="How does authentication work?",
        options=QueryOptions(filters={"audience": "public"}),
    )

    assert result.answer == "answer"
    assert [citation.source for citation in result.citations] == ["auth.md"]
    assert "Internal draft" not in llm.prompts[0]
    assert result.trace.strategy == "vector"


@pytest.mark.asyncio
async def test_context_is_packed_within_token_budget():
    index = InMemoryCorpusIndex()
    generation = await index.stage(
        "default",
        "docs",
        [Document(page_content="word " * 200, metadata={"source": "large.md"})],
    )
    await index.promote(generation)
    llm = RecordingLLM()
    engine = QueryEngine(index=index, llm_caller=llm)

    result = await engine.query(
        tenant="default",
        corpus="docs",
        question="Summarise",
        options=QueryOptions(max_context_tokens=20),
    )

    assert result.trace.context_tokens <= 20
    assert len(llm.prompts[0]) < 500


@pytest.mark.asyncio
async def test_deadline_cancels_slow_retrieval():
    cancelled = asyncio.Event()

    class SlowIndex:
        async def query(self, *args, **kwargs):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    engine = QueryEngine(index=SlowIndex())

    with pytest.raises(QueryTimeoutError):
        await engine.query(
            tenant="default",
            corpus="docs",
            question="slow",
            options=QueryOptions(timeout_seconds=0.01),
        )

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_hybrid_strategy_deduplicates_vector_and_graph_results():
    shared = Document(page_content="shared", metadata={"chunk_id": "same"})

    class Index:
        async def query(self, *args, **kwargs):
            return [shared, Document(page_content="vector", metadata={"chunk_id": "v"})]

    async def graph_retrieve(**kwargs):
        return [shared, Document(page_content="graph", metadata={"chunk_id": "g"})]

    engine = QueryEngine(index=Index(), graph_retriever=graph_retrieve)
    result = await engine.query(
        tenant="default",
        corpus="docs",
        question="hybrid",
        options=QueryOptions(strategy="hybrid", k=5),
    )

    assert [doc.page_content for doc in result.documents] == [
        "shared",
        "vector",
        "graph",
    ]


@pytest.mark.asyncio
async def test_hybrid_cutoff_preserves_graph_channel_representation():
    class Index:
        async def query(self, *args, **kwargs):
            return [Document(page_content=f"vector-{item}") for item in range(4)]

    async def graph_retrieve(**kwargs):
        return [Document(page_content=f"graph-{item}") for item in range(4)]

    result = await QueryEngine(index=Index(), graph_retriever=graph_retrieve).query(
        tenant="default",
        corpus="docs",
        question="hybrid",
        options=QueryOptions(strategy="hybrid", k=2),
    )

    assert [doc.page_content for doc in result.documents] == [
        "vector-0",
        "graph-0",
    ]


@pytest.mark.asyncio
async def test_missing_graph_retriever_reports_vector_fallback():
    class Index:
        async def query(self, *args, **kwargs):
            return [Document(page_content="vector")]

    result = await QueryEngine(index=Index()).query(
        tenant="default",
        corpus="docs",
        question="fallback",
        options=QueryOptions(strategy="graph"),
    )

    assert result.trace.strategy == "vector"
