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
from typing import Any, Dict, Optional

try:  # pragma: no cover - allow compatibility across Pydantic versions
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 fallback
    ConfigDict = None  # type: ignore

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
    ) -> None:
        self.output_format = (output_format or "custom_graph_object").lower()
        self.output_path = output_path
        self.neo4j_config = neo4j_config or {}
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
            "Unsupported output format %s; returning graph unchanged",
            self.output_format,
        )
        return graph

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
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning(
                "networkx is required for networkx graph persistence. Install with `pip install networkx`. Skipping save."
            )
            return

        nx_graph = nx.DiGraph()
        for node in graph.nodes:
            node_attrs = {
                "type": node.type,
                "name": node.name,
            }
            # Add label if present
            if node.label:
                node_attrs["label"] = node.label
            # Merge properties if present (contains vector_id and other metadata)
            if node.properties:
                node_attrs.update(node.properties)
            # Fall back to metadata for backwards compatibility
            elif node.metadata:
                node_attrs.update(node.metadata)

            nx_graph.add_node(node.id, **node_attrs)
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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # networkx 3.x removed the top-level write_gpickle; pickle directly instead.
            import pickle

            with output_path.open("wb") as f:
                pickle.dump(nx_graph, f)
            logger.info("Graph saved to NetworkX pickle at %s", output_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to pickle NetworkX graph: %s", exc)

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

        clear_before_save = self.neo4j_config.get("clear_before_save", False)
        with driver.session() as session:
            if clear_before_save:
                session.run("MATCH (n) DETACH DELETE n")

            for node in graph.nodes:
                session.run(
                    """
                    MERGE (n:Entity {id: $id})
                    SET n += $props
                    """,
                    id=node.id,
                    props={
                        "type": node.type,
                        "name": node.name,
                        **(node.metadata or {}),
                    },
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
