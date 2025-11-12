from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import spacy
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from ragdoll import settings
from ragdoll.chunkers import get_text_splitter, split_documents
from ragdoll.llms import call_llm, get_llm
from ragdoll.utils import json_parse
from .base import BaseEntityExtractor
from .graph_persistence import GraphPersistenceService
from .models import (
    EntityList,
    Graph,
    GraphEdge,
    GraphNode,
    RelationshipList,
)

logger = logging.getLogger(__name__)


class EntityExtractionService(BaseEntityExtractor):
    """
    Extract entities/relationships from documents and return a Graph.

    Responsibilities:
      • Optional chunking via langchain splitters
      • spaCy NER for entity detection
      • Optional LLM-based relationship extraction
      • Delegated graph persistence
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        text_splitter=None,
        chunk_documents: bool = True,
    ) -> None:
        config_manager = settings.get_config_manager()
        base_config = config_manager.entity_extraction_config.model_dump()
        merged_config = {**base_config, **(config or {})}
        merged_config["prompts"] = config_manager.get_default_prompt_templates()

        self.config = merged_config
        self.chunk_documents = chunk_documents
        self.text_splitter = text_splitter
        self.llm = llm or get_llm(config_manager=config_manager)

        graph_db_config = merged_config.get("graph_database_config", {}) or {}
        self.graph_persistence = GraphPersistenceService(
            output_format=graph_db_config.get(
                "output_format", merged_config.get("output_format", "custom_graph_object")
            ),
            output_path=graph_db_config.get("output_file"),
            neo4j_config={
                key: graph_db_config.get(key)
                for key in ("uri", "user", "password")
                if graph_db_config.get(key)
            }
            or None,
        )

        spacy_model = merged_config.get("spacy_model", "en_core_web_sm")
        self.nlp = self._load_spacy(spacy_model)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def extract(
        self,
        documents: Sequence[Document],
        llm_override: Optional[BaseLanguageModel] = None,
    ) -> Graph:
        logger.info("Extracting entities from %s documents", len(documents))
        processed_docs = await self._maybe_chunk_documents(documents)
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        llm_runner = llm_override or self.llm

        for doc in processed_docs:
            nodes.extend(self._run_spacy(doc))
            edges.extend(await self._run_relationship_llm(doc, nodes, llm_runner))

        graph = Graph(nodes=nodes, edges=edges)
        await self._store_graph(graph)
        return graph

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load_spacy(self, model_name: str):
        if spacy is None:
            raise ImportError(
                "spaCy is required for EntityExtractionService. Install with `pip install spacy`."
            )
        try:
            return spacy.load(model_name)
        except OSError:
            logger.info("spaCy model '%s' not found. Downloading...", model_name)
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    async def _maybe_chunk_documents(
        self, documents: Sequence[Document]
    ) -> List[Document]:
        if not self.chunk_documents:
            return list(documents)

        splitter = self.text_splitter or get_text_splitter(config=self.config)
        return split_documents(
            list(documents),
            text_splitter=splitter,
        )

    def _run_spacy(self, document: Document) -> List[GraphNode]:
        doc = self.nlp(document.page_content)
        nodes: List[GraphNode] = []
        for ent in doc.ents:
            node = GraphNode(
                id=f"spacy-{uuid.uuid4().hex}",
                type=ent.label_,
                name=ent.text,
                metadata=document.metadata or {},
            )
            nodes.append(node)
        return nodes

    async def _run_relationship_llm(
        self,
        document: Document,
        nodes: List[GraphNode],
        llm_runner: Optional[BaseLanguageModel],
    ) -> List[GraphEdge]:
        if not llm_runner:
            return []

        prompt = self._build_relationship_prompt(document)
        response = await call_llm(llm_runner, prompt, return_raw_response=False)
        return self._parse_relationships(response, document, nodes)

    def _build_relationship_prompt(self, document: Document) -> str:
        template = self.config["prompts"].get("relationship_extraction")
        if template:
            return template.format(document=document.page_content)
        return (
            f"Extract relationships from the following text:\n\n{document.page_content}\n"
        )

    def _parse_relationships(
        self,
        response: str,
        document: Document,
        nodes: List[GraphNode],
    ) -> List[GraphEdge]:
        parsed = json_parse(response, RelationshipList)
        if not parsed:
            return []

        edges: List[GraphEdge] = []
        for rel in parsed.relationships:
            source_id = self._ensure_node(nodes, rel.subject, document.metadata)
            target_id = self._ensure_node(nodes, rel.object, document.metadata)
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    type=rel.relationship,
                    metadata=document.metadata or {},
                    source_document_id=document.metadata.get("id")
                    if document.metadata
                    else None,
                )
            )
        return edges

    def _ensure_node(
        self,
        nodes: List[GraphNode],
        name: str,
        metadata: Optional[dict],
    ) -> str:
        for node in nodes:
            if node.name == name:
                return node.id
        node = GraphNode(
            id=f"llm-{uuid.uuid4().hex}",
            type="ENTITY",
            name=name,
            metadata=metadata or {},
        )
        nodes.append(node)
        return node.id

    async def _store_graph(self, graph: Graph) -> None:
        if not self.graph_persistence:
            return
        try:
            self.graph_persistence.save(graph)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to persist graph: %s", exc)


# Backward compatibility
GraphCreationService = EntityExtractionService
