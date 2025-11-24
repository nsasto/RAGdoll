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
from ragdoll.entity_extraction.models import Graph
from ragdoll.ingestion import DocumentLoaderService
from ragdoll.llms import get_llm_caller
from ragdoll.llms.callers import BaseLLMCaller, call_llm_sync
from ragdoll.pipeline import IngestionOptions, IngestionPipeline
from ragdoll.retrieval import VectorRetriever, GraphRetriever, HybridRetriever
from ragdoll.vector_stores import BaseVectorStore, vector_store_from_config

logger = logging.getLogger(__name__)


class Ragdoll:
    """
    Thin orchestration layer that wires together ingestion, embeddings,
    vector storage, and optional LLM answering.

    The goal is to provide a stable public entry point that relies only on the
    modules that actually exist in RAGdoll 2.x.
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
        self.graph_retriever: Optional[GraphRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.last_graph: Optional[Graph] = None
        self.graph_ingestion_stats: Optional[Dict[str, Any]] = None
        self.graph_store: Optional[Any] = None

    def ingest_data(self, sources: Sequence[str]) -> List[Document]:
        """
        Load documents from the provided sources and index them in the vector store.
        """
        raw_documents = self.ingestion_service.ingest_documents(list(sources))
        documents = self._to_documents(raw_documents)
        if documents:
            self.vector_store.add_documents(documents)
        return documents

    def query(
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
            retriever_mode: One of "vector", "graph", or "hybrid"
        """
        # Select retriever based on mode
        if retriever_mode == "graph" and self.graph_retriever:
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
            return self.query(question, k=k, use_hybrid=False)

        hits = self.hybrid_retriever.get_relevant_documents(question, top_k=k)

        answer = None
        if self.llm_caller and hits:
            prompt = self._build_prompt(question, hits)
            answer = self._call_llm(prompt)

        return {"answer": answer, "documents": hits}

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

        try:
            response = call_llm_sync(self.llm_caller, prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("LLM call failed: %s", exc)
            return None

        cleaned = response.strip()
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
        self.hybrid_retriever = self._build_retriever()

        return {
            "stats": stats,
            "graph": graph,
            "graph_retriever": self.graph_retriever,
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
        return HybridRetriever(
            vector_retriever=vector_retriever,
            graph_retriever=self.graph_retriever,
            mode=hybrid_cfg.get("mode", "concat"),
            vector_weight=hybrid_cfg.get("vector_weight", 0.6),
            graph_weight=hybrid_cfg.get("graph_weight", 0.4),
            deduplicate=hybrid_cfg.get("deduplicate", True),
        )

    async def query_stream(
        self, question: str, *, k: int = 4, retriever_mode: str = "hybrid"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Retrieve context and stream the response from the LLM.
        """
        hits = []
        if retriever_mode == "hybrid" and self.hybrid_retriever:
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
