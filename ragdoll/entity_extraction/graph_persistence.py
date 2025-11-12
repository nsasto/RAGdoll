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
from typing import Any, Optional

from .models import Graph

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

    def save(self, graph: Graph) -> Graph:
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

        logger.warning("Unsupported output format %s; returning graph unchanged", self.output_format)
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
