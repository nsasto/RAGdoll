from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document


from ragdoll import settings
from ragdoll.app_config import AppConfig, bootstrap_app
from ragdoll.config import Config
from ragdoll.chunkers import get_text_splitter, split_documents
from ragdoll.embeddings import get_embedding_model
from ragdoll.entity_extraction import EntityExtractionService
from ragdoll.vector_stores import vector_store_from_config
from ragdoll.graph_stores import get_graph_store
from ragdoll.ingestion import DocumentLoaderService
from ragdoll.llms import get_llm_caller
from ragdoll.llms.callers import BaseLLMCaller
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger("ragdoll.pipeline")


@dataclass
class IngestionOptions:
    """Options for the ingestion pipeline."""

    batch_size: int = 10
    parallel_extraction: bool = False
    max_workers: int = 4
    skip_vector_store: bool = False
    skip_graph_store: bool = False
    extract_entities: bool = True
    collect_metrics: bool = True

    # Additional options for sub-components
    chunking_options: Dict[str, Any] = None
    embedding_options: Dict[str, Any] = None
    vector_store_options: Dict[str, Any] = None
    graph_store_options: Dict[str, Any] = None
    entity_extraction_options: Dict[str, Any] = None
    llm: Any = None  # Optional legacy LLM override (string/config/object)
    llm_caller: Optional[BaseLLMCaller] = None  # Preferred injection point


class IngestionPipeline:
    """
    Flexible ingestion pipeline for Graph RAG.

    Coordinates document extraction, chunking, embedding, entity extraction,
    and storage in both vector and graph stores.
    """

    def __init__(
        self,
        config_manager: Optional[Config] = None,
        app_config: Optional[AppConfig] = None,
        content_extraction_service: Optional[DocumentLoaderService] = None,
        text_splitter=None,
        embedding_model=None,
        entity_extractor=None,
        vector_store=None,
        graph_store=None,
        options: Optional[IngestionOptions] = None,
    ):
        if (
            app_config is not None
            and config_manager is not None
            and app_config.config is not config_manager
        ):
            raise ValueError(
                "Provide app_config and config_manager pointing to the same Config "
                "instance or pass only one reference."
            )

        if app_config is not None:
            self.app_config = app_config
        elif config_manager is not None:
            self.app_config = AppConfig(config=config_manager)
        else:
            self.app_config = settings.get_app()

        self.config_manager = config_manager or self.app_config.config
        self.options = options or IngestionOptions()

        self.content_extraction_service = (
            content_extraction_service
            or DocumentLoaderService(
                app_config=self.app_config,
                collect_metrics=self.options.collect_metrics,
            )
        )

        self.text_splitter = text_splitter or get_text_splitter(
            config_manager=self.config_manager,
            app_config=self.app_config,
            **(self.options.chunking_options or {}),
        )

        self.embedding_model = embedding_model or get_embedding_model(
            config_manager=self.config_manager,
            app_config=self.app_config,
            **(self.options.embedding_options or {}),
        )

        if self.options.extract_entities:
            extraction_options = self.options.entity_extraction_options or {}
            config_overrides = extraction_options.get("config", {})
            entity_config = self.config_manager.entity_extraction_config.model_dump()
            entity_config.update(config_overrides)

            llm_override = extraction_options.get("llm") or self.options.llm
            llm_caller_override = (
                extraction_options.get("llm_caller") or self.options.llm_caller
            )
            resolved_llm_caller = self._resolve_llm_caller(
                llm_override, llm_caller_override
            )

            self.entity_extractor = entity_extractor or EntityExtractionService(
                config=entity_config,
                llm_caller=resolved_llm_caller,
                text_splitter=self.text_splitter,
                chunk_documents=False,
                app_config=self.app_config,
            )
        else:
            self.entity_extractor = None

        if not self.options.skip_vector_store:
            if self.embedding_model is None:
                raise ValueError(
                    "Embedding model is required for vector store indexing."
                )

            vector_config = self.config_manager.vector_store_config.model_copy(
                deep=True
            )
            vector_overrides = self.options.vector_store_options or {}
            for key, value in vector_overrides.items():
                if key == "params" and isinstance(value, dict):
                    vector_config.params.update(value)
                elif hasattr(vector_config, key):
                    setattr(vector_config, key, value)

            self.vector_store = vector_store or vector_store_from_config(
                vector_config,
                embedding=self.embedding_model,
            )
        else:
            self.vector_store = None

        if not self.options.skip_graph_store:
            graph_config = self.config_manager.entity_extraction_config.graph_database_config.model_copy(
                deep=True
            )
            graph_overrides = self.options.graph_store_options or {}
            for key, value in graph_overrides.items():
                if hasattr(graph_config, key):
                    setattr(graph_config, key, value)
                else:
                    graph_config.extra_config[key] = value

            self.graph_store = graph_store or get_graph_store(
                graph_config=graph_config,
                app_config=self.app_config,
            )
        else:
            self.graph_store = None

        self.stats = self._build_initial_stats()
        self.graph_retriever = None
        self.last_graph = None

    @staticmethod
    def _build_initial_stats() -> Dict[str, Any]:
        return {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "vector_entries_added": 0,
            "graph_entries_added": 0,
            "errors": [],
            "graph_retriever_available": False,
        }

    async def ingest(self, sources: List[Union[str, Path, Document]]) -> Dict[str, Any]:
        logger.info("Starting ingestion pipeline for %s sources", len(sources))
        self.stats = self._build_initial_stats()

        documents = await self._extract_documents(sources)
        if not documents:
            logger.warning("No documents extracted. Aborting pipeline.")
            return self.stats

        chunks = self._chunk_documents(documents)
        self.stats["chunks_created"] = len(chunks)

        if self.vector_store and chunks:
            self.vector_store.add_documents(chunks)
            self.stats["vector_entries_added"] = len(chunks)

        if self.entity_extractor and chunks:
            await self._extract_entities(chunks)

        logger.info("Ingestion pipeline complete.")
        return self.stats

    async def _extract_documents(
        self, sources: List[Union[str, Path, Document]]
    ) -> List[Document]:
        raw_docs = []
        string_sources = [
            str(source) for source in sources if not isinstance(source, Document)
        ]
        document_sources = [
            source for source in sources if isinstance(source, Document)
        ]

        if string_sources:
            extracted = self.content_extraction_service.ingest_documents(string_sources)
            raw_docs.extend(
                Document(page_content=doc["page_content"], metadata=doc["metadata"])
                for doc in extracted
            )

        raw_docs.extend(document_sources)
        self.stats["documents_processed"] = len(raw_docs)
        return raw_docs

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        logger.info(
            "Chunking %s documents using %s", len(documents), self.text_splitter
        )
        return split_documents(
            documents=documents,
            splitter=self.text_splitter,
            batch_size=self.options.batch_size,
        )

    async def _extract_entities(self, chunks: List[Document]) -> None:
        if not self.entity_extractor:
            return

        logger.info("Extracting entities from %s chunks", len(chunks))

        if self.options.parallel_extraction:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.options.max_workers) as executor:
                tasks = [
                    loop.run_in_executor(
                        executor, self.entity_extractor.extract, [chunk]
                    )
                    for chunk in chunks
                ]
                await asyncio.gather(*tasks)
        else:
            graph = await self.entity_extractor.extract(chunks)
            self.last_graph = graph
            self.stats["graph_entries_added"] = len(graph.edges)
            if getattr(self.entity_extractor, "graph_retriever_enabled", False):
                try:
                    self.graph_retriever = self.entity_extractor.create_graph_retriever(
                        graph=graph
                    )
                    self.stats["graph_retriever_available"] = True
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Unable to create graph retriever: %s", exc)

        self.stats["entities_extracted"] = len(chunks)

    def get_graph_retriever(self):
        return self.graph_retriever

    def _resolve_llm_caller(
        self,
        llm_value: Any,
        llm_caller_value: Optional[BaseLLMCaller],
    ) -> Optional[BaseLLMCaller]:
        if llm_caller_value is not None:
            return llm_caller_value

        if isinstance(llm_value, BaseLLMCaller):
            return llm_value

        if llm_value is None:
            return None

        if isinstance(llm_value, BaseLanguageModel):
            return get_llm_caller(
                config_manager=self.config_manager,
                app_config=self.app_config,
                llm=llm_value,
            )

        return get_llm_caller(
            model_name_or_config=llm_value,
            config_manager=self.config_manager,
            app_config=self.app_config,
        )


async def ingest_documents(
    sources: List[Union[str, Path, Document]],
    config: Optional[Dict[str, Any]] = None,
    options: Optional[IngestionOptions] = None,
) -> Dict[str, Any]:
    app_config = None
    config_manager = None
    if config:
        app_config = bootstrap_app(overrides=config)
        config_manager = app_config.config

    pipeline = IngestionPipeline(
        config_manager=config_manager,
        app_config=app_config,
        options=options or IngestionOptions(),
    )
    return await pipeline.ingest(sources)


__all__ = ["IngestionPipeline", "IngestionOptions", "ingest_documents"]
