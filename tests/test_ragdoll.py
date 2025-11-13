from __future__ import annotations

import asyncio
from typing import List, Sequence

import pytest
from langchain_core.documents import Document

from ragdoll.llms.callers import BaseLLMCaller, call_llm_sync
from ragdoll.ragdoll import Ragdoll


class FakeLLMCaller(BaseLLMCaller):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: List[str] = []

    async def call(self, prompt: str) -> str:  # pragma: no cover - exercised via tests
        self.prompts.append(prompt)
        return self.response


class FakeVectorStore:
    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        self.added: List[Document] = []
        self.last_query: tuple[str, int] | None = None

    def add_documents(self, documents: Sequence[Document]) -> List[str]:
        self.added.extend(documents)
        return []

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        self.last_query = (query, k)
        return self._documents[:k]


class FakeIngestionService:
    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)

    def ingest_documents(self, sources):
        return self.documents


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_ragdoll_query_uses_llm_caller():
    documents = [
        Document(page_content="Alpha content", metadata={"source": "alpha"}),
        Document(page_content="Beta content", metadata={"source": "beta"}),
    ]
    fake_llm = FakeLLMCaller(" final answer ")
    ragdoll = Ragdoll(
        ingestion_service=FakeIngestionService(documents),
        vector_store=FakeVectorStore(documents),
        embedding_model=object(),  # Prevent default embedding lookup.
        llm_caller=fake_llm,
    )

    result = ragdoll.query("What is Alpha?")

    assert result["answer"] == "final answer"
    assert result["documents"] == documents
    assert fake_llm.prompts, "Caller should be invoked with the aggregated prompt"


def test_call_llm_sync_inside_running_loop():
    fake_llm = FakeLLMCaller("loop-safe response")

    async def invoke():
        return call_llm_sync(fake_llm, "Prompt while loop is running.")

    assert asyncio.run(invoke()) == "loop-safe response"


@pytest.mark.anyio("asyncio")
async def test_ragdoll_ingest_with_graph_exposes_retriever(monkeypatch):
    documents = [Document(page_content="Doc", metadata={})]
    fake_llm = FakeLLMCaller("ok")
    ragdoll = Ragdoll(
        ingestion_service=FakeIngestionService(documents),
        vector_store=FakeVectorStore(documents),
        embedding_model=object(),
        llm_caller=fake_llm,
    )

    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            self.last_graph = "graph-object"
            self._retriever = "retriever-obj"

        async def ingest(self, sources):
            self.sources = sources
            return {"documents_processed": len(sources)}

        def get_graph_retriever(self):
            return self._retriever

    monkeypatch.setattr("ragdoll.ragdoll.IngestionPipeline", DummyPipeline)

    result = await ragdoll.ingest_with_graph(["foo.txt"])

    assert result["graph_retriever"] == "retriever-obj"
    assert ragdoll.graph_retriever == "retriever-obj"
    assert ragdoll.last_graph == "graph-object"
