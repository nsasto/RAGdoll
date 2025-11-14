from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

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
        self.graph_retriever: Optional[Any] = None
        self.last_graph: Optional[Graph] = None
        self.graph_ingestion_stats: Optional[Dict[str, Any]] = None

    def ingest_data(self, sources: Sequence[str]) -> List[Document]:
        """
        Load documents from the provided sources and index them in the vector store.
        """
        raw_documents = self.ingestion_service.ingest_documents(list(sources))
        documents = self._to_documents(raw_documents)
        if documents:
            self.vector_store.add_documents(documents)
        return documents

    def query(self, question: str, *, k: int = 4) -> dict:
        """
        Retrieve context from the vector store, optionally call the configured LLM,
        and return both the answer (if available) and the supporting documents.
        """
        hits = self.vector_store.similarity_search(question, k=k)

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

        self.graph_ingestion_stats = stats
        self.graph_retriever = retriever
        self.last_graph = graph

        return {"stats": stats, "graph": graph, "graph_retriever": retriever}

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

