from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, BaseLanguageModel

from ragdoll import settings
from ragdoll.config import Config
from ragdoll.ingestion import DocumentLoaderService
from ragdoll.vector_stores import BaseVectorStore, vector_store_from_config
from ragdoll.embeddings import get_embedding_model
from ragdoll.llms import get_llm_caller
from ragdoll.llms.callers import BaseLLMCaller, call_llm_sync

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
        ingestion_service: Optional[DocumentLoaderService] = None,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_model: Optional[Embeddings] = None,
        llm: Optional[Any] = None,
        llm_caller: Optional[BaseLLMCaller] = None,
    ) -> None:
        self.config_manager = (
            Config(config_path) if config_path else settings.get_config_manager()
        )

        self.ingestion_service = ingestion_service or DocumentLoaderService(
            config_manager=self.config_manager
        )

        self.embedding_model = embedding_model or get_embedding_model(
            config_manager=self.config_manager
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
            return get_llm_caller(config_manager=self.config_manager, llm=llm)

        if isinstance(llm, (str, dict)):
            return get_llm_caller(
                model_name_or_config=llm, config_manager=self.config_manager
            )

        return get_llm_caller(config_manager=self.config_manager)

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

