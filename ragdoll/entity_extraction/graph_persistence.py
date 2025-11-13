"""
Graph persistence utilities for entity extraction.

Handles saving graph outputs to different targets (in-memory, JSON files,
networkx graphs, or Neo4j) with lazy imports so heavy dependencies are only
loaded when needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, ClassVar

from langchain_core.documents import Document

try:  # pragma: no cover - allow compatibility across Pydantic versions
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 fallback
    ConfigDict = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from langchain_core.retrievers import BaseRetriever as LangChainRetrieverBase
except ImportError:  # pragma: no cover - fallback when LangChain isn't installed
    class _RetrieverBase:  # type: ignore[override]
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def _get_relevant_documents(self, query: str):
            raise NotImplementedError(
                "LangChain is not installed, so no retriever is available."
            )

        def get_relevant_documents(self, query: str):
            return self._get_relevant_documents(query)

        def invoke(self, query: str, config: Optional[dict] = None, **kwargs):
            return self._get_relevant_documents(query)

else:  # pragma: no cover - exercised indirectly in tests
    class _RetrieverBase(LangChainRetrieverBase):
        if ConfigDict is not None:
            model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[assignment]
        else:  # pragma: no cover - only needed for legacy Pydantic versions
            class Config:
                arbitrary_types_allowed = True

from .models import Graph, GraphNode

logger = logging.getLogger(__name__)


class GraphPersistenceService:
    """
    Persist graphs produced by the entity extraction pipeline.

    Args:
        output_format: One of ``custom_graph_object``, ``json``, ``networkx``, ``neo4j``.
        output_path: Destination for file-based formats (JSON/networkx pickle).
        neo4j_config: Optional dictionary containing ``uri``, ``user``, ``password``.
    """

    def __init__(
        self,
        output_format: str = "custom_graph_object",
        output_path: Optional[str] = None,
        neo4j_config: Optional[dict[str, Any]] = None,
        *,
        retriever_backend: Optional[str] = None,
        retriever_config: Optional[dict[str, Any]] = None,
        retriever_factory: Optional[Callable[[Graph, dict[str, Any]], Any]] = None,
    ) -> None:
        self.output_format = (output_format or "custom_graph_object").lower()
        self.output_path = output_path
        self.neo4j_config = neo4j_config or {}
        self.retriever_backend = retriever_backend
        self.retriever_config = retriever_config or {}
        self.retriever_factory = retriever_factory
        self._last_graph: Optional[Graph] = None

    def save(self, graph: Graph) -> Graph:
        self._last_graph = graph
        logger.info(
            "Storing graph with %s nodes and %s edges in format %s",
            len(graph.nodes),
            len(graph.edges),
            self.output_format,
        )

        if self.output_format == "custom_graph_object":
            return graph
        if self.output_format == "json":
            self._save_json(graph)
            return graph
        if self.output_format == "networkx":
            self._save_networkx(graph)
            return graph
        if self.output_format == "neo4j":
            self._save_neo4j(graph)
            return graph

        logger.warning(
            "Unsupported output format %s; returning graph unchanged", self.output_format
        )
        return graph

    def create_retriever(
        self,
        *,
        graph: Optional[Graph] = None,
        backend: Optional[str] = None,
        **config: Any,
    ) -> Any:
        """
        Build a retriever over the most recently persisted graph.

        Args:
            graph: Explicit graph instance to index (defaults to the last saved graph).
            backend: Optional backend identifier (``"simple"`` is built-in).
            **config: Additional backend-specific settings.

        Returns:
            An object that implements ``get_relevant_documents(query: str)``.
        """

        graph_obj = graph or self._last_graph
        if graph_obj is None:
            raise ValueError(
                "A graph instance is required to create a retriever. "
                "Call `save(graph)` first or pass `graph=` explicitly."
            )

        merged_config = {**self.retriever_config, **config}

        if self.retriever_factory:
            return self.retriever_factory(graph_obj, merged_config)

        backend_name = (backend or self.retriever_backend or "simple").lower()

        if backend_name == "simple":
            return SimpleGraphRetriever(graph_obj, **merged_config)
        if backend_name in {"neo4j", "langchain_neo4j"}:
            config_dict = {
                "uri": self.neo4j_config.get("uri"),
                "user": self.neo4j_config.get("user"),
                "password": self.neo4j_config.get("password"),
                **merged_config,
            }
            missing = [key for key in ("uri", "user", "password") if not config_dict.get(key)]
            if missing:
                raise ValueError(
                    "Neo4j retriever requires connection details. Missing: "
                    + ", ".join(missing)
                )
            return Neo4jCypherRetriever(
                uri=config_dict["uri"],
                user=config_dict["user"],
                password=config_dict["password"],
                top_k=int(config_dict.get("top_k", 5)),
                cypher_template=config_dict.get("cypher_template"),
                include_relationships=config_dict.get("include_relationships", True),
            )

        raise ValueError(
            f"Unsupported graph retriever backend '{backend_name}'. "
            "Provide `retriever_factory` or use the built-in 'simple' backend."
        )

    # ------------------------------------------------------------------ #
    # JSON persistence
    # ------------------------------------------------------------------ #
    def _save_json(self, graph: Graph) -> None:
        output_path = Path(self.output_path or "graph_output.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(graph.model_dump(), f, indent=2)

        logger.info("Graph saved to JSON file at %s", output_path)

    # ------------------------------------------------------------------ #
    # NetworkX persistence
    # ------------------------------------------------------------------ #
    def _save_networkx(self, graph: Graph) -> None:
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "networkx is required for networkx graph persistence. Install with `pip install networkx`."
            ) from exc

        nx_graph = nx.DiGraph()
        for node in graph.nodes:
            nx_graph.add_node(
                node.id,
                type=node.type,
                name=node.name,
                **(node.metadata or {}),
            )
        for edge in graph.edges:
            nx_graph.add_edge(
                edge.source,
                edge.target,
                type=edge.type,
                id=edge.id,
                metadata=edge.metadata or {},
                source_document_id=edge.source_document_id,
            )

        output_path = Path(self.output_path or "graph_output.gpickle")
        nx.write_gpickle(nx_graph, output_path)
        logger.info("Graph saved to NetworkX pickle at %s", output_path)

    # ------------------------------------------------------------------ #
    # Neo4j persistence
    # ------------------------------------------------------------------ #
    def _save_neo4j(self, graph: Graph) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "neo4j Python driver is required for Neo4j persistence. Install with `pip install neo4j`."
            ) from exc

        uri = self.neo4j_config.get("uri", "bolt://localhost:7687")
        user = self.neo4j_config.get("user", "neo4j")
        password = self.neo4j_config.get("password", "password")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Writing graph to Neo4j at %s", uri)

        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

            for node in graph.nodes:
                session.run(
                    """
                    MERGE (n:Entity {id: $id})
                    SET n += $props
                    """,
                    id=node.id,
                    props={"type": node.type, "name": node.name, **(node.metadata or {})},
                )

            for edge in graph.edges:
                session.run(
                    """
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    MERGE (source)-[r:RELATIONSHIP {id: $edge_id}]->(target)
                    SET r.type = $type,
                        r.metadata = $metadata,
                        r.source_document_id = $source_document_id
                    """,
                    source_id=edge.source,
                    target_id=edge.target,
                    edge_id=edge.id,
                    type=edge.type,
                    metadata=edge.metadata or {},
                    source_document_id=edge.source_document_id,
                )

        driver.close()
        logger.info("Graph successfully persisted to Neo4j.")


class SimpleGraphRetriever(_RetrieverBase):
    """Lightweight retriever that surfaces graph nodes as LangChain Documents."""

    graph: Graph
    top_k: int = 5
    include_edges: bool = True

    def __init__(
        self,
        graph: Graph,
        *,
        top_k: int = 5,
        include_edges: bool = True,
    ) -> None:
        top_k = max(1, top_k)
        super().__init__(graph=graph, top_k=top_k, include_edges=include_edges)

    def _get_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        if not query:
            return []

        query_lower = query.lower()
        scored_nodes: list[tuple[int, GraphNode]] = []

        for node in self.graph.nodes:
            score = 0
            name = node.name or ""
            if query_lower in name.lower():
                score += 2
            for value in (node.metadata or {}).values():
                if isinstance(value, str) and query_lower in value.lower():
                    score += 1

            if score > 0:
                scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda pair: pair[0], reverse=True)

        documents: list[Document] = []
        for score, node in scored_nodes[: self.top_k]:
            metadata = dict(node.metadata or {})
            metadata.setdefault("node_id", node.id)
            metadata.setdefault("node_type", node.type)
            metadata["score"] = score

            if self.include_edges:
                metadata["connected_to"] = self._connected_node_ids(node.id)

            documents.append(
                Document(page_content=node.name or "", metadata=metadata)
            )

        return documents

    def _connected_node_ids(self, node_id: str) -> list[str]:
        neighbors: list[str] = []
        for edge in self.graph.edges:
            if edge.source == node_id:
                neighbors.append(edge.target)
            elif edge.target == node_id:
                neighbors.append(edge.source)
        return neighbors

    def invoke(
        self, query: str, config: Optional[dict] = None, **kwargs
    ) -> list[Document]:  # pragma: no cover - compatibility
        return self._get_relevant_documents(query)

class Neo4jCypherRetriever(_RetrieverBase):
    """Retriever that runs lightweight Cypher against Neo4j."""

    DEFAULT_QUERY: ClassVar[str] = """
    MATCH (n)
    WHERE toLower(n.name) CONTAINS toLower($query)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN n, collect({type: type(r), target: coalesce(m.name, ""), target_id: m.id}) AS connections
    LIMIT $top_k
    """

    uri: str
    user: str
    password: str
    top_k: int = 5
    cypher: str = DEFAULT_QUERY
    include_relationships: bool = True

    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        top_k: int = 5,
        cypher_template: Optional[str] = None,
        include_relationships: bool = True,
    ) -> None:
        top_k = max(1, top_k)
        super().__init__(
            uri=uri,
            user=user,
            password=password,
            top_k=top_k,
            cypher=cypher_template or self.DEFAULT_QUERY,
            include_relationships=include_relationships,
        )

    def _get_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        if not query:
            return []

        records = self._run_query(query)
        documents: list[Document] = []

        for record in records:
            node = record.get("n")
            if node is None:
                continue
            node_dict = dict(node)
            metadata: Dict[str, Any] = {
                "node_id": node_dict.get("id", node.identity if hasattr(node, "identity") else None),
                "node_labels": list(node.labels) if hasattr(node, "labels") else [],
                **node_dict,
            }
            metadata.setdefault("name", node_dict.get("name"))
            metadata.setdefault("type", node_dict.get("type"))

            if self.include_relationships:
                metadata["connections"] = record.get("connections", [])

            documents.append(
                Document(page_content=node_dict.get("name", ""), metadata=metadata)
            )

        return documents

    def _run_query(self, query: str):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "neo4j Python driver is required for Neo4j retriever support. "
                "Install with `pip install neo4j`."
            ) from exc

        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            with driver.session() as session:
                result = session.run(
                    self.cypher,
                    query=query,
                    top_k=self.top_k,
                )
                return list(result)
        finally:  # pragma: no cover - defensive cleanup
            driver.close()

    def invoke(
        self, query: str, config: Optional[dict] = None, **kwargs
    ) -> list[Document]:  # pragma: no cover - compatibility
        return self._get_relevant_documents(query)
