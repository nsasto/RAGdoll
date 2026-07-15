from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections.abc import AsyncGenerator

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, BaseLanguageModel

from ragdoll import settings
from ragdoll.app_config import AppConfig, bootstrap_app
from ragdoll.embeddings import get_embedding_model
from ragdoll.chunkers import get_text_splitter
from ragdoll.contracts import IngestionSpec
from ragdoll.corpus import CorpusIndex, VectorCorpusIndex
from ragdoll.corpus import InMemoryCorpusIndex
from ragdoll.generation_state import (
    FileGenerationStateStore,
    MemoryGenerationStateStore,
    PostgresGenerationStateStore,
)
from ragdoll.graph_index import VersionedGraphIndex
from ragdoll.entity_extraction.models import Graph
from ragdoll.ingestion import DocumentLoaderService
from ragdoll.ingestion.jobs import (
    DocumentPreparer,
    DurableIngestion,
    CeleryExecutionAdapter,
    ExecutionAdapter,
    FileJobStore,
    IngestionJob,
    JobStore,
    MemoryJobStore,
    PostgresJobStore,
    PrepareDocuments,
)
from ragdoll.llms import get_llm_caller
import threading

from ragdoll.llms.callers import BaseLLMCaller, call_llm_sync
from ragdoll.pipeline import IngestionOptions, IngestionPipeline
from ragdoll.query import QueryEngine, QueryOptions
from ragdoll.observability import EventSink, NullEventSink
from ragdoll.quarantine import (
    FileQuarantineStore,
    MemoryQuarantineStore,
    PostgresQuarantineStore,
    QuarantineStore,
)
from ragdoll.retrieval import (
    VectorRetriever,
    GraphRetriever,
    HybridRetriever,
    PageRankGraphRetriever,
)
from ragdoll.vector_stores import BaseVectorStore, vector_store_from_config
from ragdoll.utils.env import resolve_env_reference

logger = logging.getLogger(__name__)


class Ragdoll:
    """
    Stable SDK entry point for durable ingestion and scoped RAG queries.

    Local and scaled deployments share this interface; execution, state,
    vector, graph, and observability infrastructure are replaceable adapters.
    """

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        app_config: Optional[AppConfig] = None,
        ingestion_service: Optional[DocumentLoaderService] = None,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_model: Optional[Embeddings] = None,
        llm: Optional[Any] = None,
        llm_caller: Optional[BaseLLMCaller] = None,
        corpus_index: Optional[CorpusIndex] = None,
        job_store: Optional[JobStore] = None,
        execution_adapter: Optional[ExecutionAdapter] = None,
        document_preparer: Optional[PrepareDocuments] = None,
        event_sink: Optional[EventSink] = None,
        quarantine_store: Optional[QuarantineStore] = None,
        graph_builder: Optional[Any] = None,
        graph_index: Optional[VersionedGraphIndex] = None,
    ) -> None:
        if config_path and app_config:
            raise ValueError("Provide either config_path or app_config, not both.")

        if app_config is not None:
            self.app_config = app_config
        elif config_path:
            self.app_config = bootstrap_app(config_path)
        else:
            self.app_config = settings.get_app()

        self.config_manager = self.app_config.config

        self.ingestion_service = ingestion_service or DocumentLoaderService(
            app_config=self.app_config
        )

        self.embedding_model = embedding_model or get_embedding_model(
            config_manager=self.config_manager, app_config=self.app_config
        )

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            vector_config = self.config_manager.vector_store_config
            if self.embedding_model is None:
                raise ValueError(
                    "An embedding model is required to build the default vector store."
                )
            self.vector_store = vector_store_from_config(
                vector_config, embedding=self.embedding_model
            )

        self.llm_caller = self._resolve_llm_caller(llm=llm, llm_caller=llm_caller)
        self.llm = (
            llm
            if llm is not None and not isinstance(llm, BaseLLMCaller)
            else getattr(self.llm_caller, "llm", None)
        )
        index_store = self.vector_store
        if not hasattr(index_store, "aadd_documents"):
            index_store = BaseVectorStore(index_store)
        corpus_runtime = self.config_manager.corpus_index_runtime_config
        self.corpus_index = corpus_index or (
            InMemoryCorpusIndex()
            if corpus_runtime.adapter == "memory"
            else VectorCorpusIndex(
                index_store,
                state_store=self._build_generation_state_store(corpus_runtime),
            )
        )
        if document_preparer is None:
            splitter = get_text_splitter(
                config_manager=self.config_manager, app_config=self.app_config
            )
            document_preparer = DocumentPreparer(
                self.ingestion_service,
                splitter,
                batch_size=self.config_manager.ingestion_config.batch_size,
            )
        runtime_job_store = job_store or self._build_job_store()
        runtime_execution = execution_adapter or self._build_execution_adapter()
        self.event_sink = event_sink or NullEventSink()
        self.durable_ingestion = DurableIngestion(
            index=self.corpus_index,
            prepare=document_preparer,
            store=runtime_job_store,
            execution=runtime_execution,
            events=self.event_sink,
            quarantine=quarantine_store or self._build_quarantine_store(),
            graph_builder=graph_builder,
            graph_index=graph_index,
            max_concurrent_jobs=(
                self.config_manager.execution_config.max_concurrent_jobs
            ),
        )
        self.query_engine = QueryEngine(
            index=self.corpus_index,
            llm_caller=self.llm_caller,
            events=self.event_sink,
            input_cost_per_million=(
                self.config_manager.query_runtime_config.input_cost_per_million
            ),
            output_cost_per_million=(
                self.config_manager.query_runtime_config.output_cost_per_million
            ),
            graph_retriever=(graph_index.query if graph_index else None),
        )
        self.graph_retriever: Optional[GraphRetriever] = None
        self.pagerank_retriever: Optional[PageRankGraphRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.last_graph: Optional[Graph] = None
        self.graph_ingestion_stats: Optional[Dict[str, Any]] = None
        self.graph_store: Optional[Any] = None

    def _build_job_store(self) -> JobStore:
        config = self.config_manager.job_store_runtime_config
        if config.adapter == "memory":
            return MemoryJobStore()
        if config.adapter == "file":
            return FileJobStore(config.path)
        dsn = resolve_env_reference(config.dsn, label="job_store.dsn")
        if not dsn:
            raise ValueError("Postgres job store requires job_store.dsn")
        return PostgresJobStore(str(dsn))

    @staticmethod
    def _build_generation_state_store(config: Any):
        if config.state_adapter == "memory":
            return MemoryGenerationStateStore()
        if config.state_adapter == "file":
            return FileGenerationStateStore(config.state_path)
        dsn = resolve_env_reference(config.dsn, label="corpus_index.dsn")
        if not dsn:
            raise ValueError(
                "Postgres corpus generation state requires corpus_index.dsn"
            )
        return PostgresGenerationStateStore(str(dsn))

    def _build_execution_adapter(self) -> ExecutionAdapter:
        config = self.config_manager.execution_config
        if config.adapter == "inline":
            from ragdoll.ingestion.jobs import InlineExecutionAdapter

            return InlineExecutionAdapter()
        try:
            from celery import Celery
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Celery execution requires `pip install python-ragdoll[scaled]`"
            ) from exc
        broker = resolve_env_reference(config.broker_url, label="execution.broker_url")
        backend = resolve_env_reference(
            config.result_backend, label="execution.result_backend"
        )
        if not broker:
            raise ValueError("Celery execution requires execution.broker_url")
        app = Celery("ragdoll", broker=broker, backend=backend)
        return CeleryExecutionAdapter(app, task_name=config.task_name)

    def _build_quarantine_store(self) -> QuarantineStore:
        config = self.config_manager.quarantine_runtime_config
        if config.adapter == "memory":
            return MemoryQuarantineStore()
        if config.adapter == "file":
            return FileQuarantineStore(config.path)
        dsn = resolve_env_reference(config.dsn, label="quarantine.dsn")
        if not dsn:
            raise ValueError("Postgres quarantine requires quarantine.dsn")
        return PostgresQuarantineStore(str(dsn))

    def ingest_data(self, sources: Sequence[str]) -> List[Document]:
        """
        Load documents from the provided sources and index them in the vector store.
        """
        raw_documents = self.ingestion_service.ingest_documents(list(sources))
        documents = self._to_documents(raw_documents)
        if documents:
            self.vector_store.add_documents(documents)
        return documents

    @classmethod
    def from_config(cls, config_path: str, **overrides: Any) -> "Ragdoll":
        """Build the same SDK interface from a deployment configuration file."""
        return cls(config_path=config_path, **overrides)

    async def ingest(
        self,
        *,
        corpus: str,
        sources: Sequence[Union[str, Document]],
        tenant: str = "default",
        idempotency_key: str | None = None,
    ) -> IngestionJob:
        """Submit durable ingestion without exposing execution infrastructure."""
        return await self.durable_ingestion.submit(
            IngestionSpec(
                tenant=tenant,
                corpus=corpus,
                sources=sources,
                idempotency_key=idempotency_key,
            )
        )

    async def query(
        self,
        question: str,
        *,
        corpus: str,
        tenant: str = "default",
        options: QueryOptions | None = None,
        k: int | None = None,
        filters: Optional[Dict[str, object]] = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        """Run scoped retrieval and generation through the async query engine."""
        selected = options or QueryOptions(
            k=k or 4,
            filters=filters or {},
            timeout_seconds=(
                timeout_seconds
                or self.config_manager.query_runtime_config.timeout_seconds
            ),
            max_context_tokens=(
                self.config_manager.query_runtime_config.max_context_tokens
            ),
        )
        result = await self.query_engine.query(
            tenant=tenant,
            corpus=corpus,
            question=question,
            options=selected,
        )
        return result.as_dict()

    async def rollback_corpus(self, *, corpus: str, tenant: str = "default") -> str:
        """Atomically restore the previously active corpus generation."""
        return str(await self.corpus_index.rollback(tenant, corpus))

    async def delete_corpus(self, *, corpus: str, tenant: str = "default") -> None:
        """Delete every generation belonging to exactly one tenant corpus."""
        await self.corpus_index.delete(tenant, corpus)
        if self.durable_ingestion.graph_index is not None:
            await self.durable_ingestion.graph_index.delete(tenant, corpus)

    def query_sync(
        self,
        question: str,
        *,
        corpus: str | None = None,
        tenant: str = "default",
        options: QueryOptions | None = None,
        k: int = 4,
        filters: Optional[Dict[str, object]] = None,
        timeout_seconds: float | None = None,
        use_hybrid: bool = False,
        retriever_mode: str = "vector",
    ) -> dict:
        """Notebook/script helper; omit corpus to use the legacy vector path."""
        if corpus is None:
            return self._query_legacy(
                question,
                k=k,
                use_hybrid=use_hybrid,
                retriever_mode=retriever_mode,
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.query(
                    question,
                    corpus=corpus,
                    tenant=tenant,
                    options=options,
                    k=k,
                    filters=filters,
                    timeout_seconds=timeout_seconds,
                )
            )
        raise RuntimeError("An event loop is running; await `query` instead")

    def _query_legacy(
        self,
        question: str,
        *,
        k: int = 4,
        use_hybrid: bool = False,
        retriever_mode: str = "vector",
    ) -> dict:
        """
        Retrieve context from the vector store, optionally call the configured LLM,
        and return both the answer (if available) and the supporting documents.

        Args:
            question: The question to answer
            k: Number of documents to retrieve
            use_hybrid: Legacy flag for hybrid retrieval (deprecated, use retriever_mode)
            retriever_mode: One of "vector", "graph", "pagerank", or "hybrid"
        """
        # Select retriever based on mode
        if retriever_mode == "pagerank" and self.pagerank_retriever:
            hits = self.pagerank_retriever.get_relevant_documents(question, top_k=k)
            retriever_used = "pagerank"
        elif retriever_mode == "graph" and self.graph_retriever:
            hits = self.graph_retriever.get_relevant_documents(question, top_k=k)
            retriever_used = "graph"
        elif retriever_mode == "hybrid" and self.hybrid_retriever:
            hits = self.hybrid_retriever.get_relevant_documents(question, top_k=k)
            retriever_used = "hybrid"
        elif use_hybrid and self.hybrid_retriever:
            # Legacy support for use_hybrid flag
            hits = self.hybrid_retriever.get_relevant_documents(question, top_k=k)
            retriever_used = "hybrid"
        else:
            # Default to vector retrieval
            hits = self.vector_store.similarity_search(question, k=k)
            retriever_used = "vector"

        answer = None
        if self.llm_caller and hits:
            prompt = self._build_prompt(question, hits)
            answer = self._call_llm(prompt)

        return {
            "answer": answer,
            "documents": hits,
            "retriever_used": retriever_used,
            "num_documents": len(hits),
        }

    def query_hybrid(self, question: str, *, k: int = 10) -> dict:
        """
        Retrieve context using the hybrid retriever (vector + graph) when available.
        """
        if not self.hybrid_retriever:
            # Fallback to vector-only path if hybrid retriever is unavailable.
            return self._query_legacy(question, k=k, use_hybrid=False)

        logger.info("query_hybrid:start question=%s k=%s", question, k)
        hits = self.hybrid_retriever.get_relevant_documents(question, top_k=k)
        logger.info("query_hybrid:retrieved documents=%s", len(hits))

        answer = None
        if self.llm_caller and hits:
            prompt = self._build_prompt(question, hits)
            answer = self._call_llm(prompt)
            logger.info("query_hybrid:llm_answer_present=%s", bool(answer))
        else:
            logger.info(
                "query_hybrid:skipping_llm llm_caller=%s hits=%s",
                bool(self.llm_caller),
                len(hits),
            )

        return {"answer": answer, "documents": hits}

    def query_pagerank(self, question: str, *, k: int = 5) -> dict:
        """
        Retrieve context using PageRank-based graph retrieval when available.

        Falls back to vector retrieval if PageRank retriever is unavailable.

        Args:
            question: The question to answer
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer, documents, and retriever metadata
        """
        if not self.pagerank_retriever:
            # Fallback to vector-only retrieval
            return self._query_legacy(question, k=k, retriever_mode="vector")

        return self._query_legacy(question, k=k, retriever_mode="pagerank")

    @staticmethod
    def _to_documents(documents: Iterable[Any]) -> List[Document]:
        """Normalize loader output into LangChain Document objects."""
        normalized: List[Document] = []
        for doc in documents:
            if isinstance(doc, Document):
                normalized.append(doc)
                continue

            if isinstance(doc, dict):
                page_content = doc.get("page_content", "")
                metadata = doc.get("metadata", {}) or {}
            else:
                page_content = str(doc)
                metadata = {}

            normalized.append(Document(page_content=page_content, metadata=metadata))
        return normalized

    @staticmethod
    def _build_prompt(question: str, documents: Sequence[Document]) -> str:
        """Create a lightweight prompt that includes retrieved context."""
        context_sections = []
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            source = metadata.get("source") or metadata.get("path") or "unknown"
            context_sections.append(
                f"Document {idx} (source: {source}):\n{doc.page_content}"
            )

        context_blob = "\n\n".join(context_sections)
        return (
            "You are a concise assistant that answers questions strictly using the "
            "provided context.\n\n"
            f"Context:\n{context_blob}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _resolve_llm_caller(
        self,
        *,
        llm: Optional[Any],
        llm_caller: Optional[BaseLLMCaller],
    ) -> Optional[BaseLLMCaller]:
        if llm_caller is not None:
            return llm_caller

        if isinstance(llm, BaseLLMCaller):
            return llm

        if isinstance(llm, (BaseChatModel, BaseLanguageModel)):
            return get_llm_caller(
                config_manager=self.config_manager,
                app_config=self.app_config,
                llm=llm,
            )

        if isinstance(llm, (str, dict)):
            return get_llm_caller(
                model_name_or_config=llm,
                config_manager=self.config_manager,
                app_config=self.app_config,
            )

        return get_llm_caller(
            config_manager=self.config_manager, app_config=self.app_config
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        if not self.llm_caller:
            return None

        import time

        # Allow a generous timeout so we don't hang the whole example if the provider stalls
        llm_timeout = 60  # seconds

        start = time.perf_counter()
        logger.info(
            "query_hybrid:llm_call_start prompt_chars=%s prompt_preview=%s",
            len(prompt),
            prompt[:500],
        )

        response_box: dict[str, Optional[str]] = {}
        error_box: dict[str, Exception] = {}

        def _run() -> None:
            try:
                response_box["value"] = call_llm_sync(self.llm_caller, prompt)
            except Exception as exc:  # pragma: no cover - defensive
                error_box["error"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(llm_timeout)

        if worker.is_alive():
            logger.error(
                "LLM call timed out after %ss (check provider/network)", llm_timeout
            )
            return None

        if error_box:
            logger.error("LLM call failed: %s", error_box["error"])
            return None

        response = response_box.get("value", "")
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "query_hybrid:llm_call_done elapsed_ms=%.2f has_result=%s",
            elapsed_ms,
            bool(response),
        )

        cleaned = (response or "").strip()
        return cleaned or None

    async def _acall_llm(self, prompt: str) -> AsyncGenerator[str, None]:
        if not self.llm_caller:
            return

        try:
            async for token in self.llm_caller.astream(prompt):
                yield token
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("LLM stream call failed: %s", exc)
            return

    async def ingest_with_graph(
        self,
        sources: Sequence[Union[str, Document]],
        *,
        options: Optional[IngestionOptions] = None,
    ) -> Dict[str, Any]:
        """
        Run the ingestion pipeline (chunking, embeddings, entity extraction,
        persistence) and expose the resulting graph retriever.

        Args:
            sources: File paths, URLs, or LangChain Documents to ingest.
            options: Optional :class:`IngestionOptions` overrides.

        Returns:
            Dictionary containing pipeline stats, the generated graph (if any),
            and the retriever object.
        """

        pipeline = IngestionPipeline(
            config_manager=self.config_manager,
            content_extraction_service=self.ingestion_service,
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            options=options or IngestionOptions(),
        )
        stats = await pipeline.ingest(list(sources))
        retriever = pipeline.get_graph_retriever()
        graph = pipeline.last_graph
        graph_store = pipeline.get_graph_store()

        self.graph_ingestion_stats = stats
        self.last_graph = graph
        self.graph_store = graph_store

        # Build new-style retrievers
        self.graph_retriever = self._build_graph_retriever(graph_store)
        self.pagerank_retriever = self._build_pagerank_retriever(graph_store)
        self.hybrid_retriever = self._build_retriever()

        return {
            "stats": stats,
            "graph": graph,
            "graph_retriever": self.graph_retriever,
            "pagerank_retriever": self.pagerank_retriever,
            "graph_store": graph_store,
        }

    def ingest_with_graph_sync(
        self,
        sources: Sequence[Union[str, Document]],
        *,
        options: Optional[IngestionOptions] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper around :meth:`ingest_with_graph` for synchronous code.

        Raises:
            RuntimeError: if called while an event loop is already running.
        """

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "An event loop is running. Await `ingest_with_graph` instead of "
                "calling the synchronous helper."
            )

        return asyncio.run(self.ingest_with_graph(sources, options=options))

    def _build_graph_retriever(
        self, graph_store: Optional[Any]
    ) -> Optional[GraphRetriever]:
        """
        Build a GraphRetriever from the graph store.

        Args:
            graph_store: Graph persistence service or graph structure

        Returns:
            Configured GraphRetriever or None if no graph store
        """
        if not graph_store:
            return None

        # Get graph retriever config
        raw_config = getattr(self.config_manager, "_config", None)
        graph_cfg = {}
        if isinstance(raw_config, dict):
            graph_cfg = raw_config.get("retriever", {}).get("graph", {})

        # Only build if enabled
        if not graph_cfg.get("enabled", True):
            return None

        return GraphRetriever(
            graph_store=graph_store,
            top_k=graph_cfg.get("top_k", 5),
            max_hops=graph_cfg.get("max_hops", 2),
            traversal_strategy=graph_cfg.get("traversal_strategy", "bfs"),
            include_edges=graph_cfg.get("include_edges", True),
            min_score=graph_cfg.get("min_score", 0.0),
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            prebuild_index=graph_cfg.get("prebuild_index", False),
            hybrid_alpha=graph_cfg.get("hybrid_alpha", 1.0),
            enable_fallback=graph_cfg.get("enable_fallback", True),
            log_fallback_warnings=graph_cfg.get("log_fallback_warnings", True),
        )

    def _maybe_wrap_with_reranker(
        self, base_retriever: Optional[BaseRetriever]
    ) -> Optional[BaseRetriever]:
        """
        Wrap a retriever with reranking if enabled in config.

        Args:
            base_retriever: The retriever to potentially wrap

        Returns:
            RerankerRetriever wrapping base_retriever, or base_retriever unchanged
        """
        if base_retriever is None:
            return None

        # Get reranker config
        raw_config = getattr(self.config_manager, "_config", None)
        reranker_cfg = {}
        if isinstance(raw_config, dict):
            reranker_cfg = raw_config.get("retriever", {}).get("reranker", {})

        # Only wrap if enabled
        if not reranker_cfg.get("enabled", False):
            return base_retriever

        try:
            from ragdoll.retrieval.reranker import RerankerRetriever

            return RerankerRetriever(
                base_retriever=base_retriever,
                app_config=self.app_config,
                config_manager=self.config_manager,
                provider=reranker_cfg.get("provider", "llm"),
                top_k=reranker_cfg.get("top_k", 5),
                over_retrieve_multiplier=reranker_cfg.get(
                    "over_retrieve_multiplier", 2
                ),
                score_threshold=reranker_cfg.get("score_threshold", 0.0),
                batch_size=reranker_cfg.get("batch_size", 10),
                log_scores=reranker_cfg.get("log_scores", False),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize reranker, using base retriever: {e}")
            return base_retriever

    def _build_pagerank_retriever(
        self, graph_store: Optional[Any]
    ) -> Optional[PageRankGraphRetriever]:
        """
        Build a PageRankGraphRetriever for personalized PageRank retrieval.

        Args:
            graph_store: Graph persistence service or graph structure

        Returns:
            Configured PageRankGraphRetriever or None if no graph store
        """
        if not graph_store:
            return None

        # Get pagerank retriever config
        raw_config = getattr(self.config_manager, "_config", None)
        pr_cfg = {}
        if isinstance(raw_config, dict):
            pr_cfg = raw_config.get("retriever", {}).get("pagerank", {})

        # Only build if enabled
        if not pr_cfg.get("enabled", False):
            return None

        base_retriever = PageRankGraphRetriever(
            graph_store=graph_store,
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            top_k=pr_cfg.get("top_k", 5),
            max_nodes=pr_cfg.get("max_nodes", 200),
            max_hops=pr_cfg.get("max_hops", 3),
            seed_strategy=pr_cfg.get("seed_strategy", "embedding"),
            num_seed_chunks=pr_cfg.get("num_seed_chunks", 5),
            damping_factor=pr_cfg.get("damping_factor", 0.15),
            max_iter=pr_cfg.get("max_iter", 50),
            tol=pr_cfg.get("tol", 1e-6),
            allowed_node_types=pr_cfg.get(
                "allowed_node_types", ["entity", "event", "document"]
            ),
            min_score=pr_cfg.get("min_score", 0.0),
            dedup_on_vector_id=pr_cfg.get("dedup_on_vector_id", True),
            include_edges=pr_cfg.get("include_edges", True),
            enable_fallback=pr_cfg.get("enable_fallback", True),
            log_fallback_warnings=pr_cfg.get("log_fallback_warnings", True),
            edge_weight_field=pr_cfg.get("edge_weight_field", "weight"),
        )

        # Wrap with reranker if enabled
        return self._maybe_wrap_with_reranker(base_retriever)

    def _build_retriever(self) -> Optional[HybridRetriever]:
        """
        Build a HybridRetriever combining vector and graph retrieval.

        Returns:
            Configured HybridRetriever or None if vector store unavailable
        """
        if not self.vector_store:
            return None

        # Get retriever config
        raw_config = getattr(self.config_manager, "_config", None)
        vector_cfg = {}
        hybrid_cfg = {}

        if isinstance(raw_config, dict):
            retriever_config = raw_config.get("retriever", {})
            vector_cfg = retriever_config.get("vector", {})
            hybrid_cfg = retriever_config.get("hybrid", {})

        # Build vector retriever
        vector_retriever = VectorRetriever(
            vector_store=self.vector_store,
            top_k=vector_cfg.get("top_k", 3),
            search_type=vector_cfg.get("search_type", "similarity"),
            search_kwargs=vector_cfg.get("search_kwargs", {}),
        )

        # Build hybrid retriever (graph retriever may be None)
        base_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            graph_retriever=self.graph_retriever,
            mode=hybrid_cfg.get("mode", "concat"),
            vector_weight=hybrid_cfg.get("vector_weight", 0.6),
            graph_weight=hybrid_cfg.get("graph_weight", 0.4),
            deduplicate=hybrid_cfg.get("deduplicate", True),
        )

        # Wrap with reranker if enabled
        return self._maybe_wrap_with_reranker(base_retriever)

    async def query_stream(
        self, question: str, *, k: int = 4, retriever_mode: str = "hybrid"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Retrieve context and stream the response from the LLM.
        """
        hits = []
        if retriever_mode == "pagerank" and self.pagerank_retriever:
            hits = self.pagerank_retriever.get_relevant_documents(question, top_k=k)
        elif retriever_mode == "hybrid" and self.hybrid_retriever:
            hits = self.hybrid_retriever.get_relevant_documents(question, top_k=k)
        elif retriever_mode == "graph" and self.graph_retriever:
            hits = self.graph_retriever.get_relevant_documents(question, top_k=k)
        elif self.vector_store:
            vector_retriever = VectorRetriever(vector_store=self.vector_store, top_k=k)
            hits = vector_retriever.get_relevant_documents(question)

        # First, yield the retrieved documents
        yield {
            "type": "documents",
            "data": [doc.dict() for doc in hits],
        }

        if self.llm_caller and hits:
            prompt = self._build_prompt(question, hits)
            async for token in self._acall_llm(prompt):
                yield {"type": "token", "data": token}
