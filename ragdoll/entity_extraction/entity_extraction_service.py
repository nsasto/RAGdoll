from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import spacy
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from ragdoll.chunkers import split_documents, get_text_splitter
from ragdoll import settings
from ragdoll.llms import call_llm, get_llm
from .base import BaseEntityExtractor
from .models import Graph, GraphNode, GraphEdge
from .graph_persistence import GraphPersistenceService

logger = logging.getLogger(__name__)


class EntityExtractionService(BaseEntityExtractor):
    """
    Lightweight entity/relationship extractor that returns a Graph.

    The service handles:
      • Optional chunking via ragdoll.chunkers
      • spaCy NER
      • Optional LLM-based extraction via injected LangChain models
      • Delegates graph persistence to GraphPersistenceService
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        chunk_documents: bool = True,
    ) -> None:
        config_manager = settings.get_config_manager()
        base_config = config_manager.entity_extraction_config.model_dump()
        prompt_mapping = config_manager.get_default_prompt_templates()
        merged = {**base_config, **(config or {})}
        merged["prompts"] = prompt_mapping

        self.config = merged
        self.chunk_documents = chunk_documents
        self.llm = llm or get_llm(config_manager=config_manager)

        graph_db_config = merged.get("graph_database_config", {}) or {}
        self.graph_persistence = GraphPersistenceService(
            output_format=graph_db_config.get("output_format", "custom_graph_object"),
            output_path=graph_db_config.get("output_file"),
            neo4j_config={
                key: graph_db_config.get(key)
                for key in ("uri", "user", "password")
                if graph_db_config.get(key)
            }
            or None,
        )

        spacy_model = merged.get("spacy_model", "en_core_web_sm")
        self.nlp = self._load_spacy(spacy_model)

    def _load_spacy(self, model_name: str):
        try:
            return spacy.load(model_name)
        except OSError:
            logger.info("spaCy model '%s' not found. Attempting download...", model_name)
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    async def extract(
        self,
        documents: Sequence[Document],
    ) -> Graph:
        logger.info("Extracting entities from %s documents", len(documents))
        processed_docs = await self._maybe_chunk_documents(documents)
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        for doc in processed_docs:
            nodes.extend(self._run_spacy(doc))
            extracted_edges = await self._run_relationship_llm(doc)
            edges.extend(extracted_edges)

        graph = Graph(nodes=nodes, edges=edges)
        await self._store_graph(graph)
        return graph

    async def _maybe_chunk_documents(
        self, documents: Sequence[Document]
    ) -> List[Document]:
        if not self.chunk_documents:
            return list(documents)

        splitter = get_text_splitter(config=self.config)
        return split_documents(
            list(documents),
            text_splitter=splitter,
        )

    def _run_spacy(self, document: Document) -> List[GraphNode]:
        doc = self.nlp(document.page_content)
        return [
            GraphNode(
                id=f"spacy-{ent.start_char}-{ent.end_char}",
                type=ent.label_,
                name=ent.text,
                metadata=document.metadata or {},
            )
            for ent in doc.ents
        ]

    async def _run_relationship_llm(self, document: Document) -> List[GraphEdge]:
        if not self.llm:
            return []
        prompt = self._build_relationship_prompt(document)
        response = await call_llm(self.llm, prompt, return_raw_response=False)
        return self._parse_relationships(response, document)

    def _build_relationship_prompt(self, document: Document) -> str:
        template = self.config["prompts"].get("relationship_extraction")
        return template.format(document=document.page_content) if template else document.page_content

    def _parse_relationships(self, response: str, document: Document) -> List[GraphEdge]:
        # Placeholder parsing logic; implement JSON parsing as needed.
        return []

    async def _store_graph(self, graph: Graph) -> None:
        if not self.graph_persistence:
            return
        try:
            self.graph_persistence.save(graph)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to persist graph: %s", exc)
