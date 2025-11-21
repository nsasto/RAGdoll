from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence, Type, Union

try:  # pragma: no cover - optional dependency
    import spacy
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from ragdoll import settings
from ragdoll.app_config import AppConfig
from ragdoll.chunkers import get_text_splitter, split_documents
from ragdoll.llms import get_llm
from ragdoll.llms.callers import BaseLLMCaller, LangChainLLMCaller
from ragdoll.prompts import get_prompt
from .models import Graph, GraphEdge, GraphNode, RelationshipList
from .base import BaseEntityExtractor
from .graph_persistence import GraphPersistenceService
from .models import Graph, GraphEdge, GraphNode, RelationshipList
from .relationship_parser import RelationshipOutputParser

logger = logging.getLogger(__name__)


class _SafeFormatDict(dict):
    """Format dictionary that leaves unknown keys untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


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
        llm_caller: Optional[BaseLLMCaller] = None,
        app_config: Optional[AppConfig] = None,
    ) -> None:
        self.app_config = app_config or settings.get_app()
        config_manager = self.app_config.config
        self.config_manager = config_manager
        base_config = config_manager.entity_extraction_config.model_dump()
        merged_config = {**base_config, **(config or {})}
        merged_config["prompts"] = self.app_config.get_prompt_templates()

        self.config = merged_config
        self.prompt_templates = merged_config.get("prompts", {}) or {}
        self.chunk_documents = chunk_documents
        self.text_splitter = text_splitter
        llm_instance = llm or get_llm(
            config_manager=config_manager, app_config=self.app_config
        )
        if llm_caller is not None:
            self.llm_caller = llm_caller
        elif llm_instance is not None:
            self.llm_caller = LangChainLLMCaller(llm_instance)
        else:
            self.llm_caller = None
        self._active_llm_provider = merged_config.get(
            "llm_provider_hint"
        ) or self._infer_llm_provider(llm_instance)
        if self._active_llm_provider:
            self._active_llm_provider = self._active_llm_provider.lower()
        elif getattr(self.llm_caller, "provider", None):
            self._active_llm_provider = str(self.llm_caller.provider).lower()

        graph_db_config = merged_config.get("graph_database_config", {}) or {}
        graph_retriever_config = merged_config.get("graph_retriever", {}) or {}
        self.graph_persistence = GraphPersistenceService(
            output_format=graph_db_config.get(
                "output_format",
                merged_config.get("output_format", "custom_graph_object"),
            ),
            output_path=graph_db_config.get("output_file"),
            neo4j_config={
                key: graph_db_config.get(key)
                for key in ("uri", "user", "password")
                if graph_db_config.get(key)
            }
            or None,
            retriever_backend=graph_retriever_config.get("backend"),
            retriever_config={
                key: value
                for key, value in graph_retriever_config.items()
                if key not in {"enabled", "backend"}
            },
        )
        self.graph_retriever_enabled = graph_retriever_config.get("enabled", False)
        self.graph_retriever_config = graph_retriever_config
        self._last_graph: Optional[Graph] = None

        spacy_model = merged_config.get("spacy_model", "en_core_web_sm")
        self.nlp = self._load_spacy(spacy_model)
        self.relationship_parser = self._build_relationship_parser()
        prompts_cfg = merged_config.get("relationship_prompts", {}) or {}
        self.relationship_prompt_default = prompts_cfg.get(
            "default", "relationship_extraction"
        )
        providers_map = prompts_cfg.get("providers", {}) or {}
        self.relationship_prompt_overrides = {
            key.lower(): value for key, value in providers_map.items()
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def extract(
        self,
        documents: Sequence[Document],
        llm_override: Optional[Union[BaseLanguageModel, BaseLLMCaller]] = None,
    ) -> Graph:
        logger.info("Extracting entities from %s documents", len(documents))
        processed_docs = await self._maybe_chunk_documents(documents)
        logger.info("After chunking, processing %s documents", len(processed_docs))
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        llm_runner = self._resolve_llm_caller(llm_override)

        for idx, doc in enumerate(processed_docs, 1):
            logger.info(
                "EntityExtraction: processing doc %d/%d (chars=%d)",
                idx,
                len(processed_docs),
                len(doc.page_content),
            )
            before_node_count = len(nodes)
            before_edge_count = len(edges)
            nodes.extend(self._run_spacy(doc))
            edges.extend(await self._run_relationship_llm(doc, nodes, llm_runner))
            logger.info(
                "EntityExtraction: doc %d complete (+%d nodes, +%d edges, totals: nodes=%d, edges=%d)",
                idx,
                len(nodes) - before_node_count,
                len(edges) - before_edge_count,
                len(nodes),
                len(edges),
            )

        graph = Graph(nodes=nodes, edges=edges)
        await self._store_graph(graph)
        return graph

    async def extract_from_vector_store(
        self,
        vector_store,
        batch_size: int = 10,
        include_metadata: bool = True,
        llm_override: Optional[Union[BaseLanguageModel, BaseLLMCaller]] = None,
    ) -> Graph:
        """
        Extract entities directly from documents in a vector store.

        This ensures graph nodes have valid vector_id references that match
        the vector store's document IDs, eliminating ID mismatch issues.

        Args:
            vector_store: Vector store containing documents to process
            batch_size: Number of documents to process at once
            include_metadata: Whether to include document metadata in node properties
            llm_override: Optional LLM override for relationship extraction

        Returns:
            Graph with entities and relationships, nodes contain vector_id references
        """
        if not vector_store:
            raise ValueError("Vector store is required for this extraction method")

        logger.info("Extracting entities from vector store")

        # Get all documents from the vector store with their IDs
        all_docs = []

        # Unwrap if using BaseVectorStore wrapper
        actual_store = getattr(vector_store, "_store", vector_store)
        if hasattr(vector_store, "store"):
            actual_store = vector_store.store

        try:
            # For Chroma (most common)
            if hasattr(actual_store, "_collection"):
                results = actual_store._collection.get()
                doc_ids = results.get("ids", [])
                doc_texts = results.get("documents", [])
                doc_metadatas = results.get("metadatas", [])

                for doc_id, text, metadata in zip(
                    doc_ids, doc_texts, doc_metadatas or [{}] * len(doc_ids)
                ):
                    if text:  # Only process non-empty documents
                        full_metadata = {**(metadata or {}), "vector_id": doc_id}
                        doc = Document(
                            page_content=text,
                            metadata=(
                                full_metadata
                                if include_metadata
                                else {"vector_id": doc_id}
                            ),
                        )
                        all_docs.append(doc)

                logger.info(
                    f"Retrieved {len(all_docs)} documents from Chroma vector store"
                )

            # For FAISS
            elif hasattr(actual_store, "docstore") and hasattr(
                actual_store, "index_to_docstore_id"
            ):
                for idx, doc_id in actual_store.index_to_docstore_id.items():
                    doc = actual_store.docstore.search(doc_id)
                    if doc and hasattr(doc, "page_content"):
                        doc.metadata = {**(doc.metadata or {}), "vector_id": doc_id}
                        if not include_metadata:
                            doc.metadata = {"vector_id": doc_id}
                        all_docs.append(doc)

                logger.info(
                    f"Retrieved {len(all_docs)} documents from FAISS vector store"
                )

            # Generic fallback
            else:
                logger.warning(
                    f"Vector store type not recognized ({type(actual_store).__name__}), attempting generic retrieval"
                )
                try:
                    all_docs = vector_store.similarity_search("", k=10000)
                    logger.warning(
                        f"Retrieved {len(all_docs)} via similarity search (vector_ids may not be set)"
                    )
                except Exception as e:
                    logger.error(f"Failed to retrieve documents: {e}")
                    return Graph(nodes=[], edges=[])

        except Exception as e:
            logger.error(f"Error retrieving documents from vector store: {e}")
            return Graph(nodes=[], edges=[])

        if not all_docs:
            logger.warning("No documents found in vector store")
            return Graph(nodes=[], edges=[])

        logger.info(f"Processing {len(all_docs)} documents for entity extraction")

        # Process in batches
        all_nodes = []
        all_edges = []

        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(all_docs) + batch_size - 1)//batch_size}"
            )

            # Extract entities from this batch (reuse existing logic)
            batch_graph = await self.extract(batch, llm_override=llm_override)
            all_nodes.extend(batch_graph.nodes)
            all_edges.extend(batch_graph.edges)

        # Deduplicate
        unique_nodes = self._deduplicate_nodes(all_nodes)
        unique_edges = self._deduplicate_edges(all_edges)

        final_graph = Graph(nodes=unique_nodes, edges=unique_edges)
        logger.info(
            f"Extraction complete: {len(unique_nodes)} nodes, {len(unique_edges)} edges"
        )

        return final_graph

    def _deduplicate_nodes(self, nodes: List[GraphNode]) -> List[GraphNode]:
        """Deduplicate nodes by ID, keeping the first occurrence."""
        seen = set()
        unique = []
        for node in nodes:
            if node.id not in seen:
                seen.add(node.id)
                unique.append(node)
        return unique

    def _deduplicate_edges(self, edges: List[GraphEdge]) -> List[GraphEdge]:
        """Deduplicate edges by (source, target, type) tuple."""
        seen = set()
        unique = []
        for edge in edges:
            key = (edge.source, edge.target, edge.type)
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        return unique

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

    def _build_relationship_parser(self) -> RelationshipOutputParser:
        parsing_config: Dict[str, Any] = (
            self.config.get("relationship_parsing", {}) or {}
        )
        parser_class_path = parsing_config.get("parser_class")
        parser_kwargs = parsing_config.get("parser_kwargs", {}) or {}
        schema_path = parsing_config.get("schema")
        schema_model: Type[RelationshipList] = RelationshipList
        if schema_path:
            schema_model = self._import_from_string(schema_path)

        preferred_format = parsing_config.get("preferred_format", "auto")
        if parser_class_path:
            parser_cls = self._import_from_string(parser_class_path)
            if callable(parser_cls):
                return parser_cls(**parser_kwargs)
            raise TypeError(f"Custom parser '{parser_class_path}' is not callable.")

        return RelationshipOutputParser(
            preferred_format=preferred_format,
            schema_model=schema_model,
        )

    def _import_from_string(self, dotted_path: str):
        if not dotted_path:
            raise ValueError("A dotted path is required to import a symbol.")
        module_path, _, attr = dotted_path.rpartition(".")
        if not module_path or not attr:
            raise ImportError(f"Invalid dotted path '{dotted_path}'.")
        module = import_module(module_path)
        return getattr(module, attr)

    def _infer_llm_provider(self, llm_instance: Optional[Any]) -> Optional[str]:
        if llm_instance is None:
            return None

        namespace = ""
        if hasattr(llm_instance, "lc_namespace"):
            try:
                namespace = ".".join(llm_instance.lc_namespace)
            except Exception:  # pragma: no cover - defensive
                namespace = ""
        if not namespace:
            namespace = getattr(llm_instance, "__module__", "") or ""
        namespace = namespace.lower()

        for provider in ("openai", "anthropic", "google", "azure", "bedrock", "cohere"):
            if provider in namespace:
                return provider
        return None

    def _select_relationship_prompt_key(self) -> str:
        provider = (self._active_llm_provider or "").lower()
        if provider and provider in self.relationship_prompt_overrides:
            return self.relationship_prompt_overrides[provider]
        return self.relationship_prompt_default

    def _lookup_prompt_template(self, prompt_key: Optional[str]) -> Optional[str]:
        if not prompt_key:
            return None
        template = self.prompt_templates.get(prompt_key)
        if template:
            return template
        try:
            return get_prompt(prompt_key)
        except ValueError:
            return None

    def _resolve_llm_caller(
        self, override: Optional[Union[BaseLanguageModel, BaseLLMCaller]]
    ) -> Optional[BaseLLMCaller]:
        if override is None:
            return self.llm_caller
        if isinstance(override, BaseLLMCaller):
            provider = getattr(override, "provider", None)
            if provider:
                self._active_llm_provider = provider.lower()
            return override
        inferred = self._infer_llm_provider(override)
        if inferred:
            self._active_llm_provider = inferred
        return LangChainLLMCaller(override)

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
            # Build properties dict with entity text, context, and vector references
            properties = {
                "name": ent.text,
                "text": ent.text,
                "context": document.page_content,
                "embedding_source": "chunk",
            }

            # Copy all document metadata into properties
            properties.update(document.metadata or {})

            # Extract vector references if available
            if "chunk_id" in document.metadata:
                properties["chunk_id"] = document.metadata["chunk_id"]
            if "vector_id" in document.metadata:
                properties["vector_id"] = document.metadata["vector_id"]
            if "embedding_timestamp" in document.metadata:
                properties["timestamp"] = document.metadata["embedding_timestamp"]

            node = GraphNode(
                id=f"spacy-{uuid.uuid4().hex}",
                type=ent.label_,
                name=ent.text,  # Required field for backward compatibility
                label=ent.text,  # Store entity text as label
                properties=properties,
            )
            nodes.append(node)
        return nodes

    async def _run_relationship_llm(
        self,
        document: Document,
        nodes: List[GraphNode],
        llm_runner: Optional[BaseLLMCaller],
    ) -> List[GraphEdge]:
        if not llm_runner:
            return []

        prompt = self._build_relationship_prompt(document)
        response = await llm_runner.call(prompt)
        return self._parse_relationships(response, document, nodes)

    def _build_relationship_prompt(self, document: Document) -> str:
        prompt_key = self._select_relationship_prompt_key()
        template = self._lookup_prompt_template(prompt_key)
        if template:
            context = self._build_prompt_context(document)
            return template.format_map(_SafeFormatDict(context))
        return f"Extract relationships from the following text:\n\n{document.page_content}\n"

    def _build_prompt_context(self, document: Document) -> Dict[str, str]:
        metadata = document.metadata or {}
        relationship_types = self.config.get("relationship_types") or []
        entities = metadata.get("entities") or metadata.get("entity_list") or ""
        if isinstance(entities, (list, tuple)):
            entities_value = "\n".join(str(item) for item in entities)
        else:
            entities_value = str(entities)

        return {
            "document": document.page_content,
            "text": document.page_content,
            "source_text": document.page_content,
            "entities": entities_value,
            "relationship_types": ", ".join(relationship_types),
        }

    def _parse_relationships(
        self,
        response: str,
        document: Document,
        nodes: List[GraphNode],
    ) -> List[GraphEdge]:
        relationship_list = self.relationship_parser.parse(response)
        if not relationship_list.relationships:
            return []

        # Enrich metadata with document context
        enriched_metadata = dict(document.metadata or {})
        enriched_metadata["context"] = document.page_content

        edges: List[GraphEdge] = []
        for rel in relationship_list.relationships:
            source_id = self._ensure_node(nodes, rel.subject, enriched_metadata)
            target_id = self._ensure_node(nodes, rel.object, enriched_metadata)
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    type=rel.relationship,
                    metadata=document.metadata or {},
                    source_document_id=(
                        document.metadata.get("id") if document.metadata else None
                    ),
                )
            )
        return edges

    def _ensure_node(
        self,
        nodes: List[GraphNode],
        name: str,
        metadata: Optional[dict],
    ) -> str:
        mention_payload = self._build_mention_payload(metadata)
        for node in nodes:
            # Match on label (preferred) or name for backwards compatibility
            node_name = node.label or node.name
            if node_name == name:
                if mention_payload:
                    # Store mentions in both metadata (for backward compatibility) and properties
                    node.metadata.setdefault("mentions", []).append(mention_payload)
                    if node.properties:
                        node.properties.setdefault("mentions", []).append(
                            mention_payload
                        )
                    else:
                        node.properties = {"mentions": [mention_payload]}
                return node.id

        # Build properties dict with entity text, context, and vector references
        properties: Dict[str, Any] = {
            "name": name,
            "text": name,
            "embedding_source": "chunk",
        }

        # Copy all document metadata into properties
        if metadata:
            properties.update(metadata)

            # Add context from page_content if available (not in metadata typically)
            # Note: context will be added by caller via document.page_content reference

            # Extract vector references if available
            if "chunk_id" in metadata:
                properties["chunk_id"] = metadata["chunk_id"]
            if "vector_id" in metadata:
                properties["vector_id"] = metadata["vector_id"]
            if "embedding_timestamp" in metadata:
                properties["timestamp"] = metadata["embedding_timestamp"]

        # Build metadata dict with mentions for backward compatibility
        node_metadata = {}
        if mention_payload:
            properties["mentions"] = [mention_payload]
            node_metadata["mentions"] = [mention_payload]

        node = GraphNode(
            id=f"llm-{uuid.uuid4().hex}",
            type="ENTITY",
            name=name,
            metadata=node_metadata,
            label=name,  # Store entity text as label
            properties=properties,
        )
        nodes.append(node)
        return node.id

    @staticmethod
    def _build_mention_payload(metadata: Optional[dict]) -> Optional[Dict[str, Any]]:
        if not metadata:
            return None
        try:
            return dict(metadata)
        except Exception:  # pragma: no cover - defensive
            return None

    async def _store_graph(self, graph: Graph) -> None:
        if not self.graph_persistence:
            return
        self._last_graph = graph
        try:
            self.graph_persistence.save(graph)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to persist graph: %s", exc)

    def create_graph_retriever(self, **kwargs: Any):
        if not self.graph_retriever_enabled:
            raise RuntimeError(
                "Graph retriever creation is disabled. Enable "
                "`entity_extraction.graph_retriever.enabled` in config."
            )
        return self.graph_persistence.create_retriever(
            graph=kwargs.pop("graph", self._last_graph),
            **kwargs,
        )

    def get_last_graph(self) -> Optional[Graph]:
        return self._last_graph
