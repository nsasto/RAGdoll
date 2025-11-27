"""
Graph Retriever

Graph traversal-based retrieval with multi-hop reasoning using embedding-based similarity.
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from collections import deque
import logging
import numpy as np
from langchain_core.documents import Document
from ragdoll.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class GraphRetriever(BaseRetriever):
    """
    Graph-based retriever with traversal strategies.

    Performs multi-hop graph traversal to find contextually relevant
    entities and relationships. Supports BFS and DFS traversal strategies
    with embedding-based seed node selection.

    Args:
        graph_store: Graph persistence service instance
        vector_store: Optional vector store for embedding retrieval
        embedding_model: Optional embedding model for query encoding
        top_k: Maximum number of seed nodes to start from
        max_hops: Maximum traversal depth from seed nodes
        traversal_strategy: "bfs" (breadth-first) or "dfs" (depth-first)
        include_edges: Whether to include relationship information
        min_score: Minimum relevance score for seed nodes (0-1)
        prebuild_index: Whether to build FAISS index during initialization
        hybrid_alpha: Weight for embedding similarity (1.0 = embedding only)
        enable_fallback: If True, fall back to fuzzy matching when embeddings unavailable
        log_fallback_warnings: If True, log warnings when fallback mechanisms are used
    """

    def __init__(
        self,
        graph_store,
        vector_store=None,
        embedding_model=None,
        top_k: int = 5,
        max_hops: int = 2,
        traversal_strategy: str = "bfs",
        include_edges: bool = True,
        min_score: float = 0.0,
        prebuild_index: bool = False,
        hybrid_alpha: float = 1.0,
        enable_fallback: bool = True,
        log_fallback_warnings: bool = True,
    ):
        self.graph_store = graph_store
        self.top_k = top_k
        self.max_hops = max_hops
        self.traversal_strategy = traversal_strategy.lower()
        self.include_edges = include_edges
        self.min_score = min_score
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.prebuild_index = prebuild_index
        self.hybrid_alpha = hybrid_alpha
        self.enable_fallback = enable_fallback
        self.log_fallback_warnings = log_fallback_warnings

        # Lazy-loaded components for embedding-based search
        self._embedding_index = None
        self._node_id_to_index = None
        self._index_to_node_id = None
        self._embedding_dimension = None
        self._orphaned_nodes_count = 0

        if self.traversal_strategy not in ["bfs", "dfs"]:
            raise ValueError(f"Unknown traversal strategy: {traversal_strategy}")

        # Precompute embeddings index if requested
        if self.prebuild_index and self.vector_store and self.embedding_model:
            try:
                self._build_embedding_index()
            except Exception as e:
                logger.warning(f"Failed to prebuild embedding index: {e}")

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
        Find initial seed nodes matching the query using embedding similarity.

        Falls back to fuzzy string matching if embeddings are unavailable
        and enable_fallback is True.

        Args:
            query: Query string
            top_k: Number of seeds to return

        Returns:
            List of (node_id, score) tuples
        """
        # Check if embedding-based search is possible
        if not self.vector_store or not self.embedding_model:
            if not self.enable_fallback:
                raise ValueError(
                    "GraphRetriever requires vector_store and embedding_model for embedding-based search. "
                    "Set enable_fallback=True to use fuzzy matching fallback, or provide vector_store and embedding_model."
                )
            if self.log_fallback_warnings:
                logger.warning(
                    "vector_store or embedding_model not available. Falling back to fuzzy matching for seed node selection."
                )
            return self._find_seeds_by_fuzzy_match(query, top_k)

        # Try embedding-based search
        try:
            return self._find_seeds_by_embedding(query, top_k)
        except Exception as e:
            if not self.enable_fallback:
                logger.error(f"Embedding-based seed search failed: {e}")
                raise ValueError(
                    f"Embedding-based search failed: {e}. Set enable_fallback=True to use fuzzy matching fallback."
                ) from e

            if self.log_fallback_warnings:
                logger.warning(
                    f"Embedding-based seed search failed: {e}. Falling back to fuzzy matching."
                )
            return self._find_seeds_by_fuzzy_match(query, top_k)

    def _find_seeds_by_embedding(
        self, query: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Find seed nodes using embedding similarity search.

        Args:
            query: Query string
            top_k: Number of seeds to return

        Returns:
            List of (node_id, score) tuples sorted by similarity
        """
        # Ensure embedding index is built
        if self._embedding_index is None:
            self._build_embedding_index()

        if self._embedding_index is None or len(self._node_id_to_index) == 0:
            logger.warning(
                "No embeddings available for nodes, falling back to fuzzy matching"
            )
            return self._find_seeds_by_fuzzy_match(query, top_k)

        # Compute query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Validate dimensions
        if query_vector.shape[1] != self._embedding_dimension:
            logger.error(
                f"Query embedding dimension mismatch: expected {self._embedding_dimension}, got {query_vector.shape[1]}"
            )
            return self._find_seeds_by_fuzzy_match(query, top_k)

        # Search FAISS index
        try:
            import faiss

            # Request more results from FAISS since we'll expand to multiple nodes per embedding
            # This ensures we get top_k unique embeddings, which may map to more nodes
            search_k = min(top_k * 2, len(self._index_to_node_ids))
            distances, indices = self._embedding_index.search(query_vector, search_k)

            # Convert FAISS results to (node_id, score) tuples
            # FAISS returns L2 distances, convert to similarity scores
            seed_nodes = []
            seen_indices = set()

            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._index_to_node_ids):
                    continue

                # Skip if we've already processed this embedding (deduplication)
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)

                # Convert L2 distance to similarity score (higher is better)
                # Use exponential decay: score = exp(-distance)
                similarity_score = np.exp(-dist)

                if similarity_score >= self.min_score:
                    # Get all nodes that share this embedding
                    node_ids = self._index_to_node_ids[idx]
                    # Add all nodes with the same score
                    for node_id in node_ids:
                        seed_nodes.append((node_id, float(similarity_score)))

                # Stop once we have enough seed nodes
                if len(seed_nodes) >= top_k:
                    break

            # Trim to top_k if we got too many
            seed_nodes = seed_nodes[:top_k]

            logger.debug(
                f"Found {len(seed_nodes)} seed nodes from {len(seen_indices)} unique embeddings via similarity search"
            )
            return seed_nodes

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return self._find_seeds_by_fuzzy_match(query, top_k)

    def _find_seeds_by_fuzzy_match(
        self, query: str, top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Find seed nodes using fuzzy string matching and term overlap.

        Args:
            query: Query string
            top_k: Number of seeds to return

        Returns:
            List of (node_id, score) tuples sorted by relevance
        """
        try:
            from rapidfuzz import fuzz

            use_rapidfuzz = True
        except ImportError:
            logger.warning("rapidfuzz not available, using basic string matching")
            use_rapidfuzz = False

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Get all nodes from graph store
        nodes = self._get_all_nodes()

        scored_nodes = []
        orphaned_count = 0

        for node_id, node_data in nodes.items():
            # Track nodes without vector_id
            if "vector_id" not in node_data and "properties" not in node_data:
                orphaned_count += 1
            elif (
                "properties" in node_data and "vector_id" not in node_data["properties"]
            ):
                orphaned_count += 1

            # Compute fuzzy match score
            if use_rapidfuzz:
                score = self._score_node_fuzzy(node_data, query_lower, query_terms)
            else:
                score = self._score_node(node_data, query_terms, query_lower)

            if score >= self.min_score:
                scored_nodes.append((node_id, score))

        self._orphaned_nodes_count = orphaned_count
        if orphaned_count > 0:
            logger.debug(f"Found {orphaned_count} nodes without vector_id references")

        # Sort by score and take top_k
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return scored_nodes[:top_k]

    def _score_node_fuzzy(
        self, node_data: Dict[str, Any], query_lower: str, query_terms: Set[str]
    ) -> float:
        """
        Score node using rapidfuzz fuzzy string matching.

        Args:
            node_data: Node attributes/metadata
            query_lower: Lowercased query string
            query_terms: Set of query terms

        Returns:
            Relevance score (0-100)
        """
        from rapidfuzz import fuzz

        score = 0.0

        # Check node name/label with fuzzy matching
        name = node_data.get("name", "") or node_data.get("properties", {}).get(
            "name", ""
        )
        if name:
            # Partial ratio handles substring matches well
            fuzzy_score = fuzz.partial_ratio(query_lower, name.lower())
            score += fuzzy_score * 0.03  # Scale to 0-3 range

        # Check node type
        node_type = node_data.get("type", "").lower()
        if node_type:
            type_score = fuzz.partial_ratio(query_lower, node_type)
            score += type_score * 0.01  # Scale to 0-1 range

        # Check properties
        properties = node_data.get("properties", {})
        for key, value in properties.items():
            if key in [
                "name",
                "vector_id",
                "chunk_id",
                "embedding_timestamp",
                "embedding_source",
            ]:
                continue
            if isinstance(value, str) and value:
                prop_score = fuzz.partial_ratio(query_lower, value.lower())
                score += prop_score * 0.005  # Scale to 0-0.5 range

        # Normalize to 0-1 range (max possible score ~ 4.5, normalize to 1)
        return min(score / 4.5, 1.0)

    def _build_embedding_index(self):
        """
        Build FAISS index from node embeddings in vector store.

        This queries the vector store for embeddings using vector_id references
        stored in node properties.
        """
        if not self.vector_store:
            logger.warning("No vector store available for building embedding index")
            return

        try:
            from ragdoll.vector_stores.adapter import VectorStoreAdapter
            import faiss
        except ImportError as e:
            logger.error(f"Required dependency missing for embedding index: {e}")
            return

        # Get all nodes and extract vector_ids
        nodes = self._get_all_nodes()
        node_embeddings = {}

        adapter = VectorStoreAdapter(self.vector_store)

        # Collect vector_ids from node properties
        # Track which nodes map to which vector_ids (many nodes can share one vector_id)
        vector_id_to_nodes = {}  # Maps vector_id -> list of node_ids

        for node_id, node_data in nodes.items():
            vector_id = None

            # Try to get vector_id from properties
            if "properties" in node_data:
                vector_id = node_data["properties"].get("vector_id")
            elif "vector_id" in node_data:
                vector_id = node_data["vector_id"]

            if vector_id:
                if vector_id not in vector_id_to_nodes:
                    vector_id_to_nodes[vector_id] = []
                vector_id_to_nodes[vector_id].append(node_id)

        if not vector_id_to_nodes:
            self._orphaned_nodes_count = len(nodes)
            if self.log_fallback_warnings:
                logger.warning(
                    f"No nodes with vector_id references found. All {self._orphaned_nodes_count} nodes are orphaned. "
                    "Graph retrieval will fall back to fuzzy matching."
                )
            return

        # Get unique vector_ids for fetching embeddings (deduplicated)
        unique_vector_ids = list(vector_id_to_nodes.keys())
        total_nodes = sum(len(node_list) for node_list in vector_id_to_nodes.values())

        # Fetch embeddings from vector store (deduplicated IDs)
        logger.info(
            f"Attempting to fetch embeddings for {total_nodes} nodes "
            f"({len(unique_vector_ids)} unique vector_ids) from vector store"
        )
        logger.debug(f"Sample vector_ids: {unique_vector_ids[:5]}")
        embeddings_dict = adapter.get_embeddings_by_ids(unique_vector_ids)

        if not embeddings_dict:
            logger.warning(
                f"Failed to retrieve any embeddings from vector store (requested {len(unique_vector_ids)} unique IDs)"
            )
            logger.debug(
                f"Vector store type: {type(self.vector_store)}, Adapter backend: {adapter._backend}"
            )
            return

        # Build index mapping and embedding matrix
        # Multiple nodes can share the same embedding (from same document chunk)
        embeddings_list = []
        self._node_id_to_index = {}
        self._index_to_node_ids = (
            {}
        )  # Maps index -> list of all node_ids with that embedding
        orphaned_count = 0

        for vector_id, node_list in vector_id_to_nodes.items():
            if vector_id in embeddings_dict:
                embedding = embeddings_dict[vector_id]
                # Add this embedding once to the list
                embedding_index = len(embeddings_list)
                embeddings_list.append(embedding)

                # All nodes with this vector_id map to the same embedding index
                for node_id in node_list:
                    self._node_id_to_index[node_id] = embedding_index
                # Store all node_ids that share this embedding
                self._index_to_node_ids[embedding_index] = node_list
            else:
                orphaned_count += len(
                    node_list
                )  # Track orphaned nodes (nodes with vector_id but no embedding retrieved)
        # All nodes are accounted for in vector_id_to_nodes
        self._orphaned_nodes_count = orphaned_count

        if self._orphaned_nodes_count > 0 and self.log_fallback_warnings:
            logger.debug(
                f"{self._orphaned_nodes_count} nodes lack embeddings (orphaned). "
                "These nodes can still be reached via graph traversal."
            )

        if not embeddings_list:
            if self.log_fallback_warnings:
                logger.warning("No valid embeddings retrieved from vector store")
            return

        # Create FAISS index
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        self._embedding_dimension = embeddings_matrix.shape[1]

        # Use L2 index for similarity search
        self._embedding_index = faiss.IndexFlatL2(self._embedding_dimension)
        self._embedding_index.add(embeddings_matrix)

        indexed_nodes = len(self._node_id_to_index)
        logger.info(
            f"Built FAISS index with {len(embeddings_list)} unique embeddings "
            f"covering {indexed_nodes} nodes (dimension={self._embedding_dimension})"
        )

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

            # Extract relationship triples for metadata
            relationship_triples = []
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
                            # Add structured triple: (subject, predicate, object)
                            relationship_triples.append((name, rel_type, other))
                        else:
                            other = nodes.get(edge["source"], {}).get(
                                "name", edge["source"]
                            )
                            content_parts.append(f"    <- {rel_type} <- {other}")
                            # Add structured triple: (subject, predicate, object)
                            relationship_triples.append((other, rel_type, name))

            # Try to get source passage from vector store if vector_id is available
            source_doc = None
            vector_id = None
            if "properties" in node_data:
                vector_id = node_data["properties"].get("vector_id")
            elif "vector_id" in node_data:
                vector_id = node_data.get("vector_id")

            # Retrieve source document if we have vector_id and vector_store
            if vector_id and self.vector_store:
                try:
                    from ragdoll.vector_stores.adapter import VectorStoreAdapter

                    adapter = VectorStoreAdapter(self.vector_store)
                    docs_dict = adapter.get_documents_by_ids([vector_id])
                    if vector_id in docs_dict:
                        source_doc = docs_dict[vector_id]
                        logger.debug(
                            f"Retrieved source passage for entity {name} (vector_id: {vector_id[:8]}...)"
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not retrieve source passage for {vector_id}: {e}"
                    )

            # Build metadata - prioritize source passage metadata if available
            if source_doc:
                # Use source passage metadata but add graph context
                doc_metadata = dict(source_doc.get("metadata", {}))
                doc_metadata.update(
                    {
                        "node_id": node_id,
                        "node_type": node_type,
                        "entity_name": name,
                        "relevance_score": node_data.get("relevance_score", 0),
                        "hop_distance": node_data.get("hop_distance", 0),
                        "retrieval_method": "graph_expanded",
                    }
                )
                # Add relationship triples if available
                if relationship_triples:
                    doc_metadata["relationship_triples"] = relationship_triples
                # Use source passage content instead of entity description
                page_content = source_doc.get("page_content", "\n".join(content_parts))
            else:
                # Fallback to entity description if source not found
                doc_metadata = {
                    "source": "graph_retrieval",
                    "node_id": node_id,
                    "node_type": node_type,
                    "entity_name": name,
                    "relevance_score": node_data.get("relevance_score", 0),
                    "hop_distance": node_data.get("hop_distance", 0),
                }
                # Add relationship triples if available
                if relationship_triples:
                    doc_metadata["relationship_triples"] = relationship_triples
                page_content = "\n".join(content_parts)

            # Create document
            doc = Document(
                page_content=page_content,
                metadata=doc_metadata,
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
            # For GraphStoreWrapper, access the underlying store
            if hasattr(self.graph_store, "store"):
                return dict(self.graph_store.store.nodes[node_id])
            else:
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
            "embedding_search_enabled": self.vector_store is not None
            and self.embedding_model is not None,
            "orphaned_nodes": self._orphaned_nodes_count,
        }

        # Try to get node/edge counts
        try:
            nodes = self._get_all_nodes()
            stats["node_count"] = len(nodes)

            if hasattr(self.graph_store, "number_of_edges"):
                stats["edge_count"] = self.graph_store.number_of_edges()

            # Add embedding index stats
            if self._embedding_index is not None:
                stats["indexed_nodes"] = len(self._node_id_to_index)
                stats["embedding_dimension"] = self._embedding_dimension
        except:
            pass

        return stats
