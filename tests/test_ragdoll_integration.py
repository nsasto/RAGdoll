from __future__ import annotations

import asyncio
from typing import Any, List, Sequence

import pytest
from langchain_core.documents import Document

from ragdoll.ragdoll import Ragdoll
from ragdoll.llms.callers import BaseLLMCaller


class RecordingLLMCaller(BaseLLMCaller):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class RecordingVectorStore:
    def __init__(self) -> None:
        self.added: List[Document] = []
        self.last_query: tuple[str, int] | None = None
        self.results: List[Document] = [
            Document(page_content="Vector hit", metadata={"source": "vector"})
        ]

    def add_documents(self, documents: Sequence[Document]) -> List[str]:
        self.added.extend(documents)
        return []

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        self.last_query = (query, k)
        return self.results[:k]


class RecordingLoader:
    def __init__(self, docs: Sequence[Document]) -> None:
        self.docs = list(docs)
        self.calls: list[Sequence[str]] = []

    def ingest_documents(self, sources: Sequence[str]):
        self.calls.append(tuple(sources))
        return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in self.docs
        ]


@pytest.fixture
def base_documents() -> list[Document]:
    return [
        Document(page_content="Alpha document", metadata={"source": "alpha"}),
        Document(page_content="Beta document", metadata={"source": "beta"}),
    ]


def test_ragdoll_end_to_end_ingest_and_query(base_documents):
    loader = RecordingLoader(base_documents)
    vector_store = RecordingVectorStore()
    llm_caller = RecordingLLMCaller("final answer")

    ragdoll = Ragdoll(
        ingestion_service=loader,
        vector_store=vector_store,
        embedding_model=object(),
        llm_caller=llm_caller,
    )

    ingested = ragdoll.ingest_data(["alpha.txt", "beta.txt"])
    assert len(ingested) == len(base_documents)
    assert vector_store.added == base_documents
    assert loader.calls == [("alpha.txt", "beta.txt")]

    result = ragdoll.query("What is Alpha?")
    assert result["answer"] == "final answer"
    assert vector_store.last_query == ("What is Alpha?", 4)
    assert llm_caller.prompts, "LLM caller should receive the composed prompt"


class FakePipeline:
    def __init__(self, *args, **kwargs):
        self.last_graph = "graph-object"
        self._retriever = "retriever"

    async def ingest(self, sources):
        self.sources = sources
        return {"documents_processed": len(sources)}

    def get_graph_retriever(self):
        return self._retriever


def test_ragdoll_ingest_with_graph_sync(monkeypatch, base_documents):
    loader = RecordingLoader(base_documents)
    vector_store = RecordingVectorStore()
    llm_caller = RecordingLLMCaller("ok")

    ragdoll = Ragdoll(
        ingestion_service=loader,
        vector_store=vector_store,
        embedding_model=object(),
        llm_caller=llm_caller,
    )

    monkeypatch.setattr("ragdoll.ragdoll.IngestionPipeline", FakePipeline)

    result = ragdoll.ingest_with_graph_sync(["file.pdf"])
    assert result["stats"]["documents_processed"] == 1
    assert result["graph_retriever"] == "retriever"
    assert ragdoll.graph_retriever == "retriever"
    assert ragdoll.last_graph == "graph-object"
