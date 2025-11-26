"""
Graph store utilities for RAGdoll.

This module provides utilities for working with graph stores through different implementations.
It supports loading and storing graphs in various formats including NetworkX, Neo4j, JSON.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
import uuid

from ragdoll import settings
from ragdoll.app_config import AppConfig
from ragdoll.config.base_config import GraphDatabaseConfig
from ragdoll.entity_extraction.models import Graph, GraphNode, GraphEdge
from ragdoll.entity_extraction.graph_persistence import GraphPersistenceService

logger = logging.getLogger("ragdoll.graph_stores")


def get_graph_store(
    store_type: str = None,
    graph: Optional[Graph] = None,
    config_manager=None,
    app_config: Optional[AppConfig] = None,
    graph_config: Optional[Union[Dict[str, Any], GraphDatabaseConfig]] = None,
    **kwargs,
) -> Optional[Any]:
    """
    Get a graph store based on configuration.

    Args:
        store_type: Type of graph store ('neo4j', 'networkx', 'json')
        graph: Optional Graph object to be stored
        config_manager: Optional ConfigManager instance
        graph_config: Optional GraphDatabaseConfig or dict with configuration
        **kwargs: Additional arguments to pass to the graph store

    Returns:
        An initialized graph store, or None if an error occurs
    """
    # Initialize the config
    db_config = None

    # First prioritize explicitly provided config
    if isinstance(graph_config, GraphDatabaseConfig):
        db_config = graph_config
    elif isinstance(graph_config, dict):
        db_config = GraphDatabaseConfig(**graph_config)

    # If no explicit config provided, try to load from config_manager
    if db_config is None:
        if app_config is not None:
            config_manager = app_config.config
        else:
            config_manager = config_manager or settings.get_app().config

        entity_extraction_config = config_manager._config.get("entity_extraction", {})
        graph_db_dict = entity_extraction_config.get("graph_database_config", {})
        db_config = GraphDatabaseConfig(**graph_db_dict)

    # If still no config, create a default one
    if db_config is None:
        db_config = GraphDatabaseConfig()

    # Apply store_type if provided, otherwise use default from config
    if store_type is not None:
        actual_store_type = store_type
    else:
        actual_store_type = db_config.default_store
        logger.info(f"Using default graph store: {actual_store_type}")

    if actual_store_type is None:
        logger.error("No graph store type specified and no default found in config.")
        return None

    # Override config with any provided kwargs
    if hasattr(db_config, "model_dump"):
        config_dict = db_config.model_dump()  # For Pydantic v2
    elif hasattr(db_config, "dict"):
        config_dict = db_config.dict()  # For Pydantic v1
    elif hasattr(db_config, "to_dict"):
        config_dict = db_config.to_dict()  # For custom classes
    else:
        config_dict = dict(db_config)  # Fallback
    for key, value in kwargs.items():
        config_dict[key] = value

    actual_store_type = actual_store_type.lower()

    # Initialize the appropriate graph store
    try:
        if actual_store_type == "neo4j":
            return _create_neo4j_graph_store(config_dict, graph)
        elif actual_store_type == "networkx":
            return _create_networkx_graph_store(config_dict, graph)
        elif actual_store_type == "json":
            return _create_json_graph_store(config_dict, graph)
        else:
            logger.error(f"Unsupported graph store type: {actual_store_type}")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize graph store '{actual_store_type}': {e}")
        return None


def _create_neo4j_graph_store(
    config: Dict[str, Any], graph: Optional[Graph] = None
) -> Any:
    """Create a Neo4j graph store."""
    try:
        from py2neo import Graph as Neo4jGraph

        uri = config.get("uri", "bolt://localhost:7687")
        user = config.get("user", "neo4j")
        password = config.get("password", "password")

        neo4j_graph = Neo4jGraph(uri, auth=(user, password))

        if graph:
            persistence = GraphPersistenceService(
                output_format="neo4j",
                neo4j_config=config,
            )
            persistence.save(graph)

        return GraphStoreWrapper("neo4j", neo4j_graph, config)

    except ImportError:
        logger.error("py2neo not installed. Install with: pip install py2neo")
        return None
    except Exception as e:
        logger.error(f"Neo4j error: {e}")
        return None


def _create_networkx_graph_store(
    config: Dict[str, Any], graph: Optional[Graph] = None
) -> Any:
    """Create a NetworkX graph store."""
    try:
        import networkx as nx

        # Create NetworkX graph
        nx_graph = nx.DiGraph()

        # If graph is provided, convert it
        if graph:
            # Add nodes
            for node in graph.nodes:
                node_attrs = {
                    "type": node.type,
                    "name": node.name,
                }
                # Add label if present
                if node.label:
                    node_attrs["label"] = node.label
                # Merge properties if present
                if node.properties:
                    node_attrs.update(node.properties)
                # Fall back to metadata for backwards compatibility
                elif node.metadata:
                    node_attrs.update(node.metadata)

                nx_graph.add_node(node.id, **node_attrs)

            # Add edges
            for edge in graph.edges:
                nx_graph.add_edge(
                    edge.source,
                    edge.target,
                    type=edge.type,
                    source_document_id=edge.source_document_id,
                    **edge.metadata,
                )

            # Save if output file is specified
            if "output_file" in config:
                import pickle

                output_path = config["output_file"]
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_path, "wb") as f:
                    pickle.dump(nx_graph, f)
                logger.info(f"NetworkX graph saved to {output_path}")

        return GraphStoreWrapper("networkx", nx_graph, config)

    except ImportError:
        logger.error("networkx not installed. Install with: pip install networkx")
        return None
    except Exception as e:
        logger.error(f"NetworkX error: {e}")
        return None


def _create_json_graph_store(
    config: Dict[str, Any], graph: Optional[Graph] = None
) -> Any:
    """Create a JSON graph store."""
    try:
        import json

        if graph:
            # Convert to JSON
            graph_json = graph.model_dump_json(indent=2)  # For newer pydantic versions

            # Save if output file is specified
            if "output_file" in config:
                output_path = config["output_file"]
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(graph_json)
                logger.info(f"Graph saved as JSON to {output_path}")

            # Create a store that holds the JSON data
            json_store = {"data": graph_json, "graph": graph}
        else:
            json_store = {"data": None, "graph": None}

            # Load from file if specified
            input_file = config.get("input_file")
            if input_file and os.path.exists(input_file):
                with open(input_file, "r") as f:
                    json_data = f.read()
                    json_store["data"] = json_data

                    # Parse JSON to Graph
                    try:
                        graph_dict = json.loads(json_data)
                        nodes = []
                        edges = []

                        if "nodes" in graph_dict:
                            for node_data in graph_dict["nodes"]:
                                node = GraphNode(
                                    id=node_data.get("id", str(uuid.uuid4())),
                                    name=node_data.get("name", ""),
                                    type=node_data.get("type", ""),
                                    metadata=node_data.get("metadata", {}),
                                )
                                nodes.append(node)

                        if "edges" in graph_dict:
                            for edge_data in graph_dict["edges"]:
                                edge = GraphEdge(
                                    id=edge_data.get("id", str(uuid.uuid4())),
                                    source=edge_data.get("source", ""),
                                    target=edge_data.get("target", ""),
                                    type=edge_data.get("type", ""),
                                    source_document_id=edge_data.get(
                                        "source_document_id", ""
                                    ),
                                    metadata=edge_data.get("metadata", {}),
                                )
                                edges.append(edge)

                        json_store["graph"] = Graph(nodes=nodes, edges=edges)
                        logger.info(
                            f"Loaded graph from JSON file {config['input_file']}"
                        )
                    except Exception as parse_error:
                        logger.error(f"Error parsing graph JSON: {parse_error}")

        return GraphStoreWrapper("json", json_store, config)

    except Exception as e:
        logger.error(f"JSON graph store error: {e}")
        return None


class GraphStoreWrapper:
    """Wrapper for graph stores with common interface."""

    def __init__(self, store_type: str, store_impl: Any, config: Dict[str, Any]):
        """
        Initialize the graph store wrapper.

        Args:
            store_type: Type of graph store
            store_impl: The actual graph store implementation
            config: Configuration for the graph store
        """
        self.store_type = store_type
        self.store = store_impl
        self.config = config

    def save_graph(self, graph: Graph, output_file: Optional[str] = None) -> bool:
        """
        Save a graph to the store.

        Args:
            graph: The graph to save
            output_file: Optional file path to save to

        Returns:
            True if successful, False otherwise
        """
        try:
            if output_file:
                self.config["output_file"] = output_file

            if self.store_type == "neo4j":
                return self._save_to_neo4j(graph)
            elif self.store_type == "networkx":
                return self._save_to_networkx(graph)
            elif self.store_type == "json":
                return self._save_to_json(graph)
            else:
                logger.error(f"Unsupported graph store type: {self.store_type}")
                return False

        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            return False

    def _save_to_neo4j(self, graph: Graph) -> bool:
        """Save graph to Neo4j."""
        try:
            persistence = GraphPersistenceService(
                output_format="neo4j",
                neo4j_config=self.config,
            )
            persistence.save(graph)
            return True

        except Exception as e:
            logger.error(f"Error saving to Neo4j: {e}")
            return False

    def _save_to_networkx(self, graph: Graph) -> bool:
        """Save graph to NetworkX."""
        try:
            nx_graph = self.store

            # Clear existing graph
            nx_graph.clear()

            # Add nodes
            for node in graph.nodes:
                node_attrs = {
                    "type": node.type,
                    "name": node.name,
                }
                # Add label if present
                if node.label:
                    node_attrs["label"] = node.label
                # Merge properties if present
                if node.properties:
                    node_attrs.update(node.properties)
                # Fall back to metadata for backwards compatibility
                elif node.metadata:
                    node_attrs.update(node.metadata)

                nx_graph.add_node(node.id, **node_attrs)

            # Add edges
            for edge in graph.edges:
                nx_graph.add_edge(
                    edge.source,
                    edge.target,
                    type=edge.type,
                    source_document_id=edge.source_document_id,
                    **edge.metadata,
                )

            # Save to file if specified
            if "output_file" in self.config:
                import pickle

                output_path = self.config["output_file"]
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_path, "wb") as f:
                    pickle.dump(nx_graph, f)
                logger.info(f"NetworkX graph saved to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving to NetworkX: {e}")
            return False

    def _save_to_json(self, graph: Graph) -> bool:
        """Save graph to JSON."""
        try:
            # Convert to JSON
            graph_json = graph.model_dump_json(indent=2)

            # Update store
            self.store["data"] = graph_json
            self.store["graph"] = graph

            # Save to file if specified
            if "output_file" in self.config:
                output_path = self.config["output_file"]
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(graph_json)
                logger.info(f"Graph saved as JSON to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            return False

    def query(self, query_string: str, params: Dict = None) -> Any:
        """
        Execute a query against the graph store.

        Args:
            query_string: The query to execute
            params: Optional parameters for the query

        Returns:
            Query results, format depends on the graph store
        """
        try:
            if params is None:
                params = {}

            if self.store_type == "neo4j":
                return self.store.run(query_string, **params)
            elif self.store_type == "networkx":
                logger.warning(
                    "Direct queries not supported for NetworkX, use native NetworkX APIs instead"
                )
                return None
            elif self.store_type == "json":
                logger.warning(
                    "Direct queries not supported for JSON, use the graph property"
                )
                return None
            else:
                logger.error(f"Unsupported graph store type: {self.store_type}")
                return None

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None

    def load_graph(self) -> Optional[Graph]:
        """
        Load a graph from the store.

        Returns:
            Loaded Graph object, or None if not available
        """
        try:
            if self.store_type == "json" and "graph" in self.store:
                return self.store["graph"]
            elif self.store_type == "networkx":
                return self._load_from_networkx()
            else:
                logger.warning(
                    f"Direct graph loading not implemented for {self.store_type}"
                )
                return None

        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return None

    def _load_from_networkx(self) -> Optional[Graph]:
        """Load graph from NetworkX."""
        try:
            nx_graph = self.store

            nodes = []
            for node_id, attrs in nx_graph.nodes(data=True):
                # Extract label and properties from attributes
                label = attrs.get("label")
                name = attrs.get("name", str(node_id))
                node_type = attrs.get("type", "Entity")

                # Collect remaining attributes as properties
                properties = {
                    k: v for k, v in attrs.items() if k not in ["name", "type", "label"]
                }

                node = GraphNode(
                    id=node_id,
                    name=name,
                    type=node_type,
                    label=label,
                    properties=properties if properties else None,
                )
                nodes.append(node)

            edges = []
            for source, target, attrs in nx_graph.edges(data=True):
                edge = GraphEdge(
                    id=attrs.get("id", str(uuid.uuid4())),
                    source=source,
                    target=target,
                    type=attrs.get("type", "RELATED_TO"),
                    source_document_id=attrs.get("source_document_id", ""),
                    metadata={
                        k: v
                        for k, v in attrs.items()
                        if k not in ["type", "source_document_id"]
                    },
                )
                edges.append(edge)

            return Graph(nodes=nodes, edges=edges)

        except Exception as e:
            logger.error(f"Error loading from NetworkX: {e}")
            return None

    def nodes(self, data=False):
        """
        Delegate to underlying NetworkX graph's nodes() method.

        Args:
            data: If True, return node attributes as well

        Returns:
            NodeView or NodeDataView from NetworkX
        """
        if self.store_type == "networkx" and hasattr(self.store, "nodes"):
            return self.store.nodes(data=data)
        else:
            raise NotImplementedError(f"nodes() not available for {self.store_type}")

    def neighbors(self, node_id):
        """
        Delegate to underlying NetworkX graph's neighbors() method.

        Args:
            node_id: The node ID to get neighbors for

        Returns:
            Iterator of neighbor nodes
        """
        if self.store_type == "networkx" and hasattr(self.store, "neighbors"):
            return self.store.neighbors(node_id)
        else:
            raise NotImplementedError(
                f"neighbors() not available for {self.store_type}"
            )

    def get_edge_data(self, source, target):
        """
        Delegate to underlying NetworkX graph's get_edge_data() method.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Edge data dictionary or None
        """
        if self.store_type == "networkx" and hasattr(self.store, "get_edge_data"):
            return self.store.get_edge_data(source, target)
        else:
            raise NotImplementedError(
                f"get_edge_data() not available for {self.store_type}"
            )

    def number_of_edges(self):
        """
        Delegate to underlying NetworkX graph's number_of_edges() method.

        Returns:
            Number of edges in the graph
        """
        if self.store_type == "networkx" and hasattr(self.store, "number_of_edges"):
            return self.store.number_of_edges()
        else:
            raise NotImplementedError(
                f"number_of_edges() not available for {self.store_type}"
            )

    def __contains__(self, node_id):
        """
        Check if a node exists in the graph (for NetworkX compatibility).

        Args:
            node_id: The node ID to check

        Returns:
            True if node exists, False otherwise
        """
        if self.store_type == "networkx":
            return node_id in self.store
        else:
            raise NotImplementedError(
                f"__contains__ not available for {self.store_type}"
            )
