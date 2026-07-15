"""Predictable retrieval, context packing, citation, and generation."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

from langchain_core.documents import Document

from ragdoll.corpus import CorpusIndex
from ragdoll.errors import QueryTimeoutError
from ragdoll.llms.callers import BaseLLMCaller
from ragdoll.observability import EventSink, NullEventSink

Strategy = Literal["vector", "graph", "hybrid"]
GraphRetriever = Callable[..., Sequence[Document] | Awaitable[Sequence[Document]]]
Reranker = Callable[
    [str, Sequence[Document]], Sequence[Document] | Awaitable[Sequence[Document]]
]


@dataclass(frozen=True, slots=True)
class QueryOptions:
    k: int = 4
    strategy: Strategy = "vector"
    filters: Mapping[str, object] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    max_context_tokens: int = 2_000
    include_citations: bool = True
    fallback_on_error: bool = True

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        object.__setattr__(self, "filters", dict(self.filters))


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    source: str
    document_id: str | None
    chunk_id: str | None
    excerpt: str


@dataclass(frozen=True, slots=True)
class QueryTrace:
    strategy: Strategy
    elapsed_ms: float
    retrieval_ms: float
    generation_ms: float
    retrieved_count: int
    context_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float | None


@dataclass(frozen=True, slots=True)
class QueryResult:
    answer: str | None
    documents: tuple[Document, ...]
    citations: tuple[Citation, ...]
    trace: QueryTrace

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "documents": list(self.documents),
            "citations": list(self.citations),
            "trace": self.trace,
            "retriever_used": self.trace.strategy,
            "num_documents": len(self.documents),
        }


class QueryEngine:
    """One async interface for scoped retrieval through answer generation."""

    def __init__(
        self,
        *,
        index: CorpusIndex,
        llm_caller: BaseLLMCaller | None = None,
        graph_retriever: GraphRetriever | None = None,
        reranker: Reranker | None = None,
        events: EventSink | None = None,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
    ) -> None:
        self.index = index
        self.llm_caller = llm_caller
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.events = events or NullEventSink()
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    async def query(
        self,
        *,
        tenant: str,
        corpus: str,
        question: str,
        options: QueryOptions | None = None,
    ) -> QueryResult:
        selected = options or QueryOptions()
        started = time.perf_counter()
        self.events.emit(
            "query.started",
            {
                "tenant": tenant,
                "corpus": corpus,
                "strategy": selected.strategy,
            },
        )
        try:
            return await asyncio.wait_for(
                self._execute(tenant, corpus, question, selected, started),
                timeout=selected.timeout_seconds,
            )
        except TimeoutError as exc:
            self.events.emit(
                "query.timed_out",
                {
                    "tenant": tenant,
                    "corpus": corpus,
                    "timeout": selected.timeout_seconds,
                },
            )
            raise QueryTimeoutError(
                f"Query exceeded {selected.timeout_seconds:.3f}s deadline"
            ) from exc

    async def _execute(
        self,
        tenant: str,
        corpus: str,
        question: str,
        options: QueryOptions,
        started: float,
    ) -> QueryResult:
        retrieval_started = time.perf_counter()
        documents, actual_strategy = await self._retrieve(
            tenant, corpus, question, options
        )
        if self.reranker is not None and documents:
            ranked = self.reranker(question, documents)
            documents = list(await ranked if inspect.isawaitable(ranked) else ranked)
        documents = _deduplicate(documents)[: options.k]
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        self.events.emit(
            "query.retrieved",
            {
                "tenant": tenant,
                "corpus": corpus,
                "documents": len(documents),
                "latency_ms": retrieval_ms,
            },
        )
        packed, context_tokens = _pack_context(documents, options.max_context_tokens)
        citations = _citations(packed) if options.include_citations else ()
        prompt = _prompt(question, packed)
        generation_started = time.perf_counter()
        answer = None
        if self.llm_caller is not None and packed:
            answer = (await self.llm_caller.call(prompt)).strip() or None
        generation_ms = (time.perf_counter() - generation_started) * 1000
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(answer or "")
        estimated_cost = self._cost(prompt_tokens, completion_tokens)
        result = QueryResult(
            answer=answer,
            documents=tuple(packed),
            citations=citations,
            trace=QueryTrace(
                strategy=actual_strategy,
                elapsed_ms=elapsed_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                retrieved_count=len(documents),
                context_tokens=context_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost,
            ),
        )
        self.events.emit(
            "query.completed",
            {
                "tenant": tenant,
                "corpus": corpus,
                "latency_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": estimated_cost,
            },
        )
        return result

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            prompt_tokens * self.input_cost_per_million
            + completion_tokens * self.output_cost_per_million
        ) / 1_000_000

    async def _retrieve(
        self,
        tenant: str,
        corpus: str,
        question: str,
        options: QueryOptions,
    ) -> tuple[list[Document], Strategy]:
        async def vector() -> list[Document]:
            return list(
                await self.index.query(
                    tenant,
                    corpus,
                    question,
                    k=options.k,
                    filters=options.filters,
                )
            )

        if options.strategy == "vector":
            return await vector(), "vector"
        if self.graph_retriever is None:
            self.events.emit(
                "query.fallback",
                {
                    "requested": options.strategy,
                    "actual": "vector",
                    "reason": "graph_unavailable",
                },
            )
            return await vector(), "vector"
        try:
            graph_value = self.graph_retriever(
                tenant=tenant,
                corpus=corpus,
                question=question,
                k=options.k,
                filters=options.filters,
            )
            graph_call = (
                graph_value
                if inspect.isawaitable(graph_value)
                else _immediate(graph_value)
            )
            if options.strategy == "graph":
                return list(await graph_call), "graph"
            vector_docs, graph_docs = await asyncio.gather(vector(), graph_call)
            return _interleave(vector_docs, graph_docs), "hybrid"
        except Exception as exc:
            if not options.fallback_on_error:
                raise
            self.events.emit(
                "query.fallback",
                {
                    "requested": options.strategy,
                    "actual": "vector",
                    "reason": type(exc).__name__,
                },
            )
            return await vector(), "vector"


async def _immediate(value: Sequence[Document]) -> Sequence[Document]:
    return value


def _interleave(*groups: Sequence[Document]) -> list[Document]:
    """Give each retrieval channel representation before the final k cutoff."""
    result: list[Document] = []
    for position in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if position < len(group):
                result.append(group[position])
    return result


def _deduplicate(documents: Sequence[Document]) -> list[Document]:
    seen: set[str] = set()
    result = []
    for document in documents:
        key = str(
            document.metadata.get("chunk_id")
            or hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
        )
        if key not in seen:
            seen.add(key)
            result.append(document)
    return result


def _pack_context(
    documents: Sequence[Document], budget: int
) -> tuple[list[Document], int]:
    packed = []
    used = 0
    for document in documents:
        remaining = budget - used
        if remaining <= 0:
            break
        tokens = _estimate_tokens(document.page_content)
        if tokens <= remaining:
            packed.append(document)
            used += tokens
            continue
        character_limit = remaining * 4
        packed.append(
            Document(
                page_content=document.page_content[:character_limit],
                metadata=dict(document.metadata),
            )
        )
        used += _estimate_tokens(packed[-1].page_content)
        break
    return packed, min(used, budget)


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _citations(documents: Sequence[Document]) -> tuple[Citation, ...]:
    return tuple(
        Citation(
            number=index,
            source=str(
                document.metadata.get("source")
                or document.metadata.get("path")
                or "unknown"
            ),
            document_id=(
                str(document.metadata["document_id"])
                if document.metadata.get("document_id")
                else None
            ),
            chunk_id=(
                str(document.metadata["chunk_id"])
                if document.metadata.get("chunk_id")
                else None
            ),
            excerpt=document.page_content[:240],
        )
        for index, document in enumerate(documents, start=1)
    )


def _prompt(question: str, documents: Sequence[Document]) -> str:
    context = "\n\n".join(
        f"[{index}] {document.page_content}"
        for index, document in enumerate(documents, start=1)
    )
    return (
        "Answer strictly from the supplied context. Cite supporting passages using "
        "square brackets such as [1].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
