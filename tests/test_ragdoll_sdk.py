import pytest
from langchain_core.documents import Document

from ragdoll.corpus import InMemoryCorpusIndex
from ragdoll.ragdoll import Ragdoll
from ragdoll.ingestion.jobs import MemoryJobStore


class FakeLLM:
    async def call(self, prompt):
        return "grounded answer"


class UnusedLoader:
    def ingest_documents(self, sources):
        raise AssertionError("injected document preparer should be used")


class UnusedVectorStore:
    def similarity_search(self, query, k=4, **kwargs):
        return []


@pytest.mark.asyncio
async def test_north_star_ingest_and_query_interface():
    async def prepare(spec):
        return [
            Document(
                page_content="Authentication uses signed tokens.",
                metadata={"source": str(spec.sources[0])},
            )
        ]

    rag = Ragdoll(
        ingestion_service=UnusedLoader(),
        vector_store=UnusedVectorStore(),
        embedding_model=object(),
        llm_caller=FakeLLM(),
        corpus_index=InMemoryCorpusIndex(),
        document_preparer=prepare,
        job_store=MemoryJobStore(),
    )

    job = await rag.ingest(
        tenant="acme",
        corpus="product-docs",
        sources=["manual.md"],
        idempotency_key="manual-v1",
    )
    ingestion = await job.wait()
    answer = await rag.query(
        tenant="acme",
        corpus="product-docs",
        question="How does authentication work?",
    )

    assert ingestion.status.value == "completed"
    assert answer["answer"] == "grounded answer"
    assert answer["citations"][0].source == "manual.md"


@pytest.mark.asyncio
async def test_high_level_corpus_rollback_and_delete():
    versions = iter(["version one", "version two"])

    async def prepare(spec):
        return [Document(page_content=next(versions), metadata={"source": "manual.md"})]

    index = InMemoryCorpusIndex()
    rag = Ragdoll(
        ingestion_service=UnusedLoader(),
        vector_store=UnusedVectorStore(),
        embedding_model=object(),
        llm_caller=FakeLLM(),
        corpus_index=index,
        document_preparer=prepare,
        job_store=MemoryJobStore(),
    )
    first = await rag.ingest(corpus="docs", sources=["manual.md"])
    await first.wait()
    second = await rag.ingest(corpus="docs", sources=["manual.md"])
    await second.wait()

    await rag.rollback_corpus(corpus="docs")
    assert (await index.query("default", "docs", "version"))[
        0
    ].page_content == "version one"

    await rag.delete_corpus(corpus="docs")
    assert await index.query("default", "docs", "version") == []


def test_query_sync_preserves_local_script_usage():
    documents = [Document(page_content="legacy", metadata={"source": "legacy.md"})]

    class LegacyStore:
        def similarity_search(self, query, k=4, **kwargs):
            return documents

    rag = Ragdoll(
        ingestion_service=UnusedLoader(),
        vector_store=LegacyStore(),
        embedding_model=object(),
        llm_caller=FakeLLM(),
        corpus_index=InMemoryCorpusIndex(),
        document_preparer=lambda spec: [],
        job_store=MemoryJobStore(),
    )

    result = rag.query_sync("legacy question")

    assert result["answer"] == "grounded answer"
    assert result["documents"] == documents
