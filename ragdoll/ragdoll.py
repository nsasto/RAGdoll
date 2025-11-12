from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragdoll import settings
from ragdoll.config import Config
from ragdoll.ingestion import DocumentLoaderService
from ragdoll.vector_stores import BaseVectorStore, vector_store_from_config
from ragdoll.embeddings import get_embedding_model
from ragdoll.llms import get_llm


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

        self.llm = llm or get_llm(config_manager=self.config_manager)

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

        answer: Optional[str] = None
        if self.llm and hasattr(self.llm, "invoke") and hits:
            prompt = self._build_prompt(question, hits)
            response = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            answer = getattr(response, "content", response)
            if isinstance(answer, str):
                answer = answer.strip()
            else:
                answer = str(answer)

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

