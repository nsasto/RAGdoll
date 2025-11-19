"""
Graph Retriever

Graph traversal-based retrieval with multi-hop reasoning.
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from collections import deque
from langchain_core.documents import Document
from ragdoll.retrieval.base import BaseRetriever


class GraphRetriever(BaseRetriever):
    """
    Graph-based retriever with traversal strategies.

    Performs multi-hop graph traversal to find contextually relevant
    entities and relationships. Supports BFS and DFS traversal strategies.

    Args:
        graph_store: Graph persistence service instance
        top_k: Maximum number of seed nodes to start from
        max_hops: Maximum traversal depth from seed nodes
        traversal_strategy: "bfs" (breadth-first) or "dfs" (depth-first)
        include_edges: Whether to include relationship information
        min_score: Minimum relevance score for seed nodes (0-1)
    """

    def __init__(
        self,
        graph_store,
        top_k: int = 5,
        max_hops: int = 2,
        traversal_strategy: str = "bfs",
        include_edges: bool = True,
        min_score: float = 0.0,
    ):
        self.graph_store = graph_store
        self.top_k = top_k
        self.max_hops = max_hops
        self.traversal_strategy = traversal_strategy.lower()
        self.include_edges = include_edges
        self.min_score = min_score

        if self.traversal_strategy not in ["bfs", "dfs"]:
            raise ValueError(f"Unknown traversal strategy: {traversal_strategy}")

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve documents via graph traversal.

        Strategy:
        1. Find seed nodes matching the query
        2. Traverse graph from seeds using BFS/DFS
        3. Extract subgraph with entities and relationships
        4. Convert to Document objects

        Args:
            query: Query string to search for
            **kwargs: Override retrieval parameters

        Returns:
            List of Document objects containing graph context
        """
        if not self.graph_store:
            return []

        # Allow runtime parameter overrides
        top_k = kwargs.get("top_k", self.top_k)
        max_hops = kwargs.get("max_hops", self.max_hops)
        include_edges = kwargs.get("include_edges", self.include_edges)

        try:
            # Step 1: Find seed nodes
            seed_nodes = self._find_seed_nodes(query, top_k)

            if not seed_nodes:
                return []

            # Step 2: Traverse graph from seeds
            subgraph = self._traverse_from_seeds(seed_nodes, max_hops=max_hops)

            # Step 3: Convert subgraph to documents
            documents = self._subgraph_to_documents(
                subgraph, include_edges=include_edges
            )

            return documents

        except Exception as e:
            print(f"Graph retrieval error: {e}")
            return []

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Async version of get_relevant_documents.

        Note: Currently delegates to sync version as most graph stores
        don't have native async support. Override in subclasses if needed.
        """
        return self.get_relevant_documents(query, **kwargs)

    def _find_seed_nodes(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Find initial seed nodes matching the query.

        Uses keyword matching and scoring against node names/metadata.

        Args:
            query: Query string
            top_k: Number of seeds to return

        Returns:
            List of (node_id, score) tuples
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Get all nodes from graph store
        nodes = self._get_all_nodes()

        scored_nodes = []
        for node_id, node_data in nodes.items():
            score = self._score_node(node_data, query_terms, query_lower)

            if score >= self.min_score:
                scored_nodes.append((node_id, score))

        # Sort by score and take top_k
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return scored_nodes[:top_k]

    def _score_node(
        self, node_data: Dict[str, Any], query_terms: Set[str], query_lower: str
    ) -> float:
        """
        Score a node's relevance to the query.

        Args:
            node_data: Node attributes/metadata
            query_terms: Set of query terms
            query_lower: Lowercased query string

        Returns:
            Relevance score (higher is better)
        """
        score = 0.0

        # Check node name/label
        name = node_data.get("name", "").lower()
        if name and query_lower in name:
            score += 3.0
        elif name:
            # Check for term overlap
            name_terms = set(name.split())
            overlap = len(query_terms & name_terms)
            score += overlap * 1.5

        # Check node type/label
        node_type = node_data.get("type", "").lower()
        if node_type and query_lower in node_type:
            score += 1.0

        # Check metadata/properties
        for key, value in node_data.items():
            if key in ["name", "type", "id"]:
                continue
            if isinstance(value, str):
                value_lower = value.lower()
                if query_lower in value_lower:
                    score += 0.5
                else:
                    value_terms = set(value_lower.split())
                    overlap = len(query_terms & value_terms)
                    score += overlap * 0.3

        return score

    def _traverse_from_seeds(
        self, seed_nodes: List[Tuple[str, float]], max_hops: int
    ) -> Dict[str, Any]:
        """
        Traverse graph from seed nodes.

        Args:
            seed_nodes: List of (node_id, score) tuples
            max_hops: Maximum traversal depth

        Returns:
            Subgraph dictionary with nodes and edges
        """
        if self.traversal_strategy == "bfs":
            return self._bfs_traverse(seed_nodes, max_hops)
        else:
            return self._dfs_traverse(seed_nodes, max_hops)

    def _bfs_traverse(
        self, seed_nodes: List[Tuple[str, float]], max_hops: int
    ) -> Dict[str, Any]:
        """
        Breadth-first traversal from seed nodes.

        Args:
            seed_nodes: Starting nodes with scores
            max_hops: Maximum depth

        Returns:
            Subgraph with visited nodes and edges
        """
        visited_nodes = {}
        visited_edges = []
        queue = deque()

        # Initialize queue with seed nodes at depth 0
        for node_id, score in seed_nodes:
            queue.append((node_id, 0, score))
            node_data = self._get_node_data(node_id)
            if node_data:
                visited_nodes[node_id] = {
                    **node_data,
                    "relevance_score": score,
                    "hop_distance": 0,
                }

        # BFS traversal
        while queue:
            current_id, depth, parent_score = queue.popleft()

            if depth >= max_hops:
                continue

            # Get neighbors
            neighbors = self._get_neighbors(current_id)

            for neighbor_id, edge_data in neighbors:
                # Add edge
                if self.include_edges:
                    visited_edges.append(
                        {
                            "source": current_id,
                            "target": neighbor_id,
                            "relationship": edge_data.get("type", "RELATED_TO"),
                            **edge_data,
                        }
                    )

                # Visit neighbor if not already visited
                if neighbor_id not in visited_nodes:
                    neighbor_data = self._get_node_data(neighbor_id)
                    if neighbor_data:
                        # Decay score by hop distance
                        decayed_score = parent_score * (0.7 ** (depth + 1))
                        visited_nodes[neighbor_id] = {
                            **neighbor_data,
                            "relevance_score": decayed_score,
                            "hop_distance": depth + 1,
                        }
                        queue.append((neighbor_id, depth + 1, decayed_score))

        return {"nodes": visited_nodes, "edges": visited_edges}

    def _dfs_traverse(
        self, seed_nodes: List[Tuple[str, float]], max_hops: int
    ) -> Dict[str, Any]:
        """
        Depth-first traversal from seed nodes.

        Args:
            seed_nodes: Starting nodes with scores
            max_hops: Maximum depth

        Returns:
            Subgraph with visited nodes and edges
        """
        visited_nodes = {}
        visited_edges = []

        def dfs_helper(node_id: str, depth: int, score: float):
            if depth > max_hops or node_id in visited_nodes:
                return

            # Visit current node
            node_data = self._get_node_data(node_id)
            if node_data:
                visited_nodes[node_id] = {
                    **node_data,
                    "relevance_score": score,
                    "hop_distance": depth,
                }

            if depth >= max_hops:
                return

            # Visit neighbors
            neighbors = self._get_neighbors(node_id)
            for neighbor_id, edge_data in neighbors:
                if self.include_edges and neighbor_id not in visited_nodes:
                    visited_edges.append(
                        {
                            "source": node_id,
                            "target": neighbor_id,
                            "relationship": edge_data.get("type", "RELATED_TO"),
                            **edge_data,
                        }
                    )

                # Recurse with decayed score
                decayed_score = score * 0.7
                dfs_helper(neighbor_id, depth + 1, decayed_score)

        # Start DFS from each seed
        for node_id, score in seed_nodes:
            dfs_helper(node_id, 0, score)

        return {"nodes": visited_nodes, "edges": visited_edges}

    def _subgraph_to_documents(
        self, subgraph: Dict[str, Any], include_edges: bool
    ) -> List[Document]:
        """
        Convert subgraph to Document objects.

        Args:
            subgraph: Dictionary with nodes and edges
            include_edges: Whether to include edge information

        Returns:
            List of Document objects
        """
        documents = []
        nodes = subgraph.get("nodes", {})
        edges = subgraph.get("edges", [])

        # Sort nodes by relevance score
        sorted_nodes = sorted(
            nodes.items(), key=lambda x: x[1].get("relevance_score", 0), reverse=True
        )

        # Create document for each node
        for node_id, node_data in sorted_nodes:
            content_parts = []

            # Node information
            name = node_data.get("name", node_id)
            node_type = node_data.get("type", "Entity")
            content_parts.append(f"{node_type}: {name}")

            # Add node properties
            for key, value in node_data.items():
                if key not in ["name", "type", "id", "relevance_score", "hop_distance"]:
                    content_parts.append(f"  {key}: {value}")

            # Add relationships if requested
            if include_edges:
                node_edges = [
                    e for e in edges if e["source"] == node_id or e["target"] == node_id
                ]
                if node_edges:
                    content_parts.append("  Relationships:")
                    for edge in node_edges[:5]:  # Limit to 5 relationships per node
                        rel_type = edge.get("relationship", "RELATED_TO")
                        if edge["source"] == node_id:
                            other = nodes.get(edge["target"], {}).get(
                                "name", edge["target"]
                            )
                            content_parts.append(f"    -> {rel_type} -> {other}")
                        else:
                            other = nodes.get(edge["source"], {}).get(
                                "name", edge["source"]
                            )
                            content_parts.append(f"    <- {rel_type} <- {other}")

            # Create document
            doc = Document(
                page_content="\n".join(content_parts),
                metadata={
                    "source": "graph_retrieval",
                    "node_id": node_id,
                    "node_type": node_type,
                    "relevance_score": node_data.get("relevance_score", 0),
                    "hop_distance": node_data.get("hop_distance", 0),
                },
            )
            documents.append(doc)

        return documents

    def _get_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get all nodes from the graph store."""
        if hasattr(self.graph_store, "get_all_nodes"):
            return self.graph_store.get_all_nodes()
        elif hasattr(self.graph_store, "nodes"):
            # NetworkX-style
            return {
                node_id: dict(data)
                for node_id, data in self.graph_store.nodes(data=True)
            }
        else:
            return {}

    def _get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific node."""
        if hasattr(self.graph_store, "get_node"):
            return self.graph_store.get_node(node_id)
        elif hasattr(self.graph_store, "nodes") and node_id in self.graph_store:
            return dict(self.graph_store.nodes[node_id])
        else:
            return None

    def _get_neighbors(self, node_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get neighbors of a node with edge data."""
        neighbors = []

        if hasattr(self.graph_store, "get_neighbors"):
            neighbors = self.graph_store.get_neighbors(node_id)
        elif hasattr(self.graph_store, "neighbors"):
            # NetworkX-style
            for neighbor_id in self.graph_store.neighbors(node_id):
                edge_data = self.graph_store.get_edge_data(node_id, neighbor_id) or {}
                neighbors.append((neighbor_id, edge_data))

        return neighbors

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.

        Returns:
            Dictionary of graph statistics
        """
        stats = {
            "top_k": self.top_k,
            "max_hops": self.max_hops,
            "traversal_strategy": self.traversal_strategy,
            "include_edges": self.include_edges,
        }

        # Try to get node/edge counts
        try:
            nodes = self._get_all_nodes()
            stats["node_count"] = len(nodes)

            if hasattr(self.graph_store, "number_of_edges"):
                stats["edge_count"] = self.graph_store.number_of_edges()
        except:
            pass

        return stats
