"""
PageRank-based Graph Retriever

Graph-first retrieval using personalized PageRank on local subgraphs.
Inspired by fast-graphrag and HippoRAG approaches.
"""

import logging
from typing import List, Optional, Dict, Any, Set, Tuple
from collections import deque
import numpy as np
from langchain_core.documents import Document

from ragdoll.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class PageRankGraphRetriever(BaseRetriever):
    """
    PageRank-based graph retriever with local subgraph extraction.

    Given a query, this retriever:
    1. Selects seed nodes via embedding similarity or keyword matching
    2. Builds a local subgraph around seeds via bounded BFS
    3. Runs personalized PageRank on the local subgraph
    4. Maps high-scoring nodes back to chunks in the vector store
    5. Returns ranked chunks as context

    Args:
        graph_store: Graph persistence service instance
        vector_store: Optional vector store for embedding-based seed selection
        embedding_model: Optional embedding model for query encoding
        top_k: Maximum number of final chunks to return
        max_nodes: Maximum nodes in the local subgraph
        max_hops: Maximum hop radius from seeds for subgraph extraction
        seed_strategy: "embedding" (vector similarity) or "keyword" (fuzzy match)
        num_seed_chunks: Number of chunks to use as seeds (for embedding strategy)
        damping_factor: Teleport probability in PPR (alpha), typically 0.15
        max_iter: Maximum iterations for PPR power iteration
        tol: Convergence tolerance for PPR
        allowed_node_types: List of node types to include in ranking
        min_score: Minimum PageRank score threshold
        dedup_on_vector_id: If True, collapse multiple nodes mapping to same chunk
        include_edges: Whether to include relationship information in results
        enable_fallback: If True, fall back to vector search if subgraph empty
        log_fallback_warnings: If True, log warnings when fallback occurs
    """

    def __init__(
        self,
        graph_store,
        vector_store=None,
        embedding_model=None,
        top_k: int = 5,
        max_nodes: int = 200,
        max_hops: int = 3,
        seed_strategy: str = "embedding",
        num_seed_chunks: int = 5,
        damping_factor: float = 0.15,
        max_iter: int = 50,
        tol: float = 1e-6,
        allowed_node_types: Optional[List[str]] = None,
        min_score: float = 0.0,
        dedup_on_vector_id: bool = True,
        include_edges: bool = True,
        enable_fallback: bool = True,
        log_fallback_warnings: bool = True,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.max_nodes = max_nodes
        self.max_hops = max_hops
        self.seed_strategy = seed_strategy.lower()
        self.num_seed_chunks = num_seed_chunks
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tol = tol
        self.allowed_node_types = allowed_node_types or [
            "entity",
            "event",
            "document",
        ]
        self.min_score = min_score
        self.dedup_on_vector_id = dedup_on_vector_id
        self.include_edges = include_edges
        self.enable_fallback = enable_fallback
        self.log_fallback_warnings = log_fallback_warnings

        if self.seed_strategy not in ["embedding", "keyword"]:
            raise ValueError(
                f"Unknown seed_strategy: {seed_strategy}. Must be 'embedding' or 'keyword'."
            )

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve documents using personalized PageRank on local subgraph.

        Args:
            query: Query string
            **kwargs: Override retrieval parameters (top_k, max_hops, max_nodes)

        Returns:
            List of Document objects ranked by PageRank score
        """
        if not self.graph_store:
            logger.warning("No graph store available for PageRank retrieval")
            if self.enable_fallback and self.vector_store:
                return self.vector_store.similarity_search(query, k=self.top_k)
            return []

        # Allow runtime parameter overrides
        top_k = kwargs.get("top_k", self.top_k)
        max_hops = kwargs.get("max_hops", self.max_hops)
        max_nodes = kwargs.get("max_nodes", self.max_nodes)

        try:
            # Step 1: Select seed nodes from query
            seed_nodes = self._select_seeds(query)

            if not seed_nodes:
                logger.debug("No seed nodes found, attempting fallback")
                if self.enable_fallback and self.vector_store:
                    return self.vector_store.similarity_search(query, k=top_k)
                return []

            logger.debug(f"Selected {len(seed_nodes)} seed nodes")

            # Step 2: Build local subgraph around seeds
            subgraph, node_to_idx, idx_to_node = self._build_local_subgraph(
                seed_nodes, max_hops=max_hops, max_nodes=max_nodes
            )

            if not subgraph:
                logger.debug("Empty subgraph, attempting fallback")
                if self.enable_fallback and self.vector_store:
                    return self.vector_store.similarity_search(query, k=top_k)
                return []

            logger.debug(
                f"Built subgraph with {len(node_to_idx)} nodes and {len(subgraph)} edges"
            )

            # Step 3: Run personalized PageRank
            seed_node_ids = {node_id for node_id, _ in seed_nodes}
            seed_indices = {
                node_to_idx[nid] for nid in seed_node_ids if nid in node_to_idx
            }
            scores = self._run_personalized_pagerank(
                subgraph, node_to_idx, seed_indices
            )

            # Step 4: Rank nodes and select vector IDs
            vector_ids = self._rank_nodes_and_select_vector_ids(
                node_to_idx, idx_to_node, scores, top_k
            )

            if not vector_ids:
                logger.debug("No vector IDs found after ranking, attempting fallback")
                if self.enable_fallback and self.vector_store:
                    return self.vector_store.similarity_search(query, k=top_k)
                return []

            # Step 5: Fetch final documents from vector store
            documents = self._fetch_documents_by_vector_ids(vector_ids)

            # Enrich metadata with PageRank info
            for doc in documents:
                if "vector_id" in doc.metadata:
                    doc.metadata["retrieval_source"] = "pagerank"

            logger.debug(f"Retrieved {len(documents)} final documents")
            return documents

        except Exception as e:
            logger.error(f"PageRank retrieval error: {e}", exc_info=True)
            if self.enable_fallback and self.vector_store:
                return self.vector_store.similarity_search(query, k=self.top_k)
            return []

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Async version of get_relevant_documents.

        Note: Currently delegates to sync version as most graph stores
        don't have native async support.
        """
        return self.get_relevant_documents(query, **kwargs)

    def _select_seeds(self, query: str) -> List[Tuple[str, float]]:
        """
        Select seed nodes from the query using configured strategy.

        Args:
            query: Query string

        Returns:
            List of (node_id, seed_score) tuples
        """
        if self.seed_strategy == "embedding":
            return self._select_seeds_embedding(query)
        else:
            return self._select_seeds_keyword(query)

    def _select_seeds_embedding(self, query: str) -> List[Tuple[str, float]]:
        """
        Select seeds by embedding similarity on chunks.

        1. Use vector store to find top chunks similar to query
        2. Extract their vector_ids
        3. Resolve all graph nodes that reference these vector_ids
        """
        if not self.vector_store:
            logger.warning("No vector store available for embedding seed selection")
            return self._select_seeds_keyword(query)

        try:
            # Find top chunks by similarity
            chunks = self.vector_store.similarity_search_with_scores(
                query, k=self.num_seed_chunks
            )

            if not chunks:
                logger.debug("No similar chunks found in vector store")
                return []

            # Extract vector_ids from chunks
            vector_ids = set()
            for doc, score in chunks:
                if "vector_id" in doc.metadata:
                    vector_ids.add(doc.metadata["vector_id"])

            if not vector_ids:
                logger.debug("No vector_ids found in chunk metadata")
                return []

            # Resolve nodes that reference these vector_ids
            seed_nodes = []
            nodes = self._get_all_nodes()

            for node_id, node_data in nodes.items():
                # Check for vector_id in node properties or metadata
                node_vector_id = None
                if "properties" in node_data:
                    node_vector_id = node_data["properties"].get("vector_id")
                elif "vector_id" in node_data:
                    node_vector_id = node_data["vector_id"]

                if node_vector_id and node_vector_id in vector_ids:
                    # Find the corresponding chunk score
                    chunk_score = next(
                        (
                            s
                            for d, s in chunks
                            if d.metadata.get("vector_id") == node_vector_id
                        ),
                        0.5,
                    )
                    seed_nodes.append((node_id, float(chunk_score)))

            logger.debug(
                f"Found {len(seed_nodes)} seed nodes from {len(vector_ids)} vector_ids"
            )
            return seed_nodes

        except Exception as e:
            logger.warning(
                f"Embedding seed selection failed: {e}, falling back to keyword"
            )
            return self._select_seeds_keyword(query)

    def _select_seeds_keyword(self, query: str) -> List[Tuple[str, float]]:
        """
        Select seeds by fuzzy keyword matching on node names/labels.

        Args:
            query: Query string

        Returns:
            List of (node_id, score) tuples
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("rapidfuzz not available, using basic string matching")
            fuzz = None

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        nodes = self._get_all_nodes()
        scored_nodes = []

        for node_id, node_data in nodes.items():
            score = 0.0

            # Check node name
            name = node_data.get("name", "")
            if not name and "properties" in node_data:
                name = node_data["properties"].get("name", "")

            if name:
                if fuzz:
                    fuzzy_score = fuzz.partial_ratio(query_lower, name.lower())
                    score += fuzzy_score * 0.03
                else:
                    # Basic term overlap
                    name_terms = set(name.lower().split())
                    overlap = len(query_terms & name_terms)
                    score += overlap * 0.1

            # Check node type
            node_type = node_data.get("type", "").lower()
            if node_type and node_type in query_terms:
                score += 0.1

            if score >= self.min_score:
                scored_nodes.append((node_id, min(score / 4.5, 1.0)))

        # Sort by score and take top seeds
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return scored_nodes[: self.num_seed_chunks]

    def _build_local_subgraph(
        self,
        seed_nodes: List[Tuple[str, float]],
        max_hops: int,
        max_nodes: int,
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, int], Dict[int, str]]:
        """
        Build local subgraph around seeds via bounded BFS.

        Returns:
            Tuple of (adjacency_dict, node_to_index, index_to_node)
            where adjacency_dict maps node_id -> [(neighbor_id, edge_weight), ...]
        """
        visited = set()
        queue = deque()
        adjacency = {}

        # Initialize with seed nodes
        for node_id, seed_score in seed_nodes:
            if node_id not in visited:
                queue.append((node_id, 0))  # (node_id, depth)
                visited.add(node_id)
                adjacency[node_id] = []

        # BFS traversal
        while queue and len(visited) < max_nodes:
            node_id, depth = queue.popleft()

            # Don't explore beyond max_hops
            if depth >= max_hops:
                continue

            # Get neighbors
            try:
                neighbors = list(self.graph_store.neighbors(node_id))
            except Exception:
                neighbors = []

            for neighbor_id in neighbors:
                if neighbor_id not in visited and len(visited) < max_nodes:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
                    adjacency[neighbor_id] = []

                if neighbor_id in visited or neighbor_id in adjacency.get(node_id, []):
                    continue

                # Get edge weight
                edge_weight = self._get_edge_weight(node_id, neighbor_id)

                # Add edge in both directions (undirected for PageRank)
                adjacency.setdefault(node_id, []).append((neighbor_id, edge_weight))
                adjacency.setdefault(neighbor_id, []).append((node_id, edge_weight))

        # Create index mappings
        node_to_idx = {node_id: i for i, node_id in enumerate(sorted(visited))}
        idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}

        logger.debug(f"Subgraph: {len(visited)} nodes, {len(adjacency)} edges")
        return adjacency, node_to_idx, idx_to_node

    def _get_edge_weight(self, src_id: str, dst_id: str) -> float:
        """
        Get edge weight from graph store, default to 1.0.

        Args:
            src_id: Source node ID
            dst_id: Destination node ID

        Returns:
            Edge weight (float)
        """
        try:
            edge_data = self.graph_store.get_edge_data(src_id, dst_id)
            if isinstance(edge_data, dict):
                return float(edge_data.get("weight", 1.0))
        except Exception:
            pass
        return 1.0

    def _run_personalized_pagerank(
        self,
        adjacency: Dict[str, List[Tuple[str, float]]],
        node_to_idx: Dict[str, int],
        seed_indices: Set[int],
    ) -> np.ndarray:
        """
        Run personalized PageRank using power iteration.

        Args:
            adjacency: Adjacency dict mapping node_id -> [(neighbor_id, weight), ...]
            node_to_idx: Mapping from node_id to vector index
            seed_indices: Set of seed node indices in the vector

        Returns:
            Array of PageRank scores indexed by node index
        """
        n = len(node_to_idx)

        # Handle empty subgraph
        if n == 0:
            return np.array([])

        # Initialize personalization vector: uniform over seeds
        personalization = np.zeros(n)
        if seed_indices:
            for idx in seed_indices:
                personalization[idx] = 1.0 / len(seed_indices)
        else:
            personalization[:] = 1.0 / n

        # Power iteration
        scores = personalization.copy()

        for iteration in range(self.max_iter):
            scores_new = self.damping_factor * personalization.copy()

            # For each node, apply damped transition: (1-alpha) * A_hat @ scores
            for node_id, neighbors in adjacency.items():
                node_idx = node_to_idx[node_id]

                if not neighbors:
                    # Dangling node: distribute score uniformly to all via teleport
                    scores_new += (1 - self.damping_factor) * scores[node_idx] / n
                    continue

                # Compute outgoing transition probabilities
                total_weight = sum(weight for _, weight in neighbors)
                for neighbor_id, weight in neighbors:
                    neighbor_idx = node_to_idx.get(neighbor_id)
                    if neighbor_idx is not None:
                        transition_prob = (
                            (weight / total_weight) if total_weight > 0 else 0
                        )
                        scores_new[neighbor_idx] += (
                            (1 - self.damping_factor)
                            * scores[node_idx]
                            * transition_prob
                        )

            # Check convergence
            l1_diff = np.abs(scores_new - scores).sum()
            scores = scores_new

            if l1_diff < self.tol:
                logger.debug(f"PageRank converged after {iteration + 1} iterations")
                break

        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    def _rank_nodes_and_select_vector_ids(
        self,
        node_to_idx: Dict[str, int],
        idx_to_node: Dict[int, str],
        scores: np.ndarray,
        top_k: int,
    ) -> List[str]:
        """
        Rank nodes by PageRank score and select top_k vector IDs.

        Args:
            node_to_idx: Mapping from node_id to index
            idx_to_node: Mapping from index to node_id
            scores: PageRank scores array
            top_k: Number of results to return

        Returns:
            List of vector_ids (sorted by score descending)
        """
        nodes = self._get_all_nodes()

        # Build list of (node_id, score) tuples
        ranked_nodes = []
        for idx, score in enumerate(scores):
            if score >= self.min_score:
                node_id = idx_to_node[idx]
                node_data = nodes.get(node_id, {})

                # Filter by node type if configured
                node_type = node_data.get("type", "")
                if node_type not in self.allowed_node_types:
                    continue

                ranked_nodes.append((node_id, float(score), node_data))

        # Sort by score descending
        ranked_nodes.sort(key=lambda x: x[1], reverse=True)

        # Select vector IDs, deduplicating on vector_id if configured
        selected_vector_ids = []
        seen_vector_ids = set()

        for node_id, score, node_data in ranked_nodes:
            if len(selected_vector_ids) >= top_k:
                break

            # Extract vector_id from node
            vector_id = None
            if "properties" in node_data:
                vector_id = node_data["properties"].get("vector_id")
            elif "vector_id" in node_data:
                vector_id = node_data["vector_id"]

            if not vector_id:
                continue

            # Skip if already seen and dedup enabled
            if self.dedup_on_vector_id and vector_id in seen_vector_ids:
                continue

            selected_vector_ids.append(vector_id)
            if self.dedup_on_vector_id:
                seen_vector_ids.add(vector_id)

        return selected_vector_ids

    def _fetch_documents_by_vector_ids(self, vector_ids: List[str]) -> List[Document]:
        """
        Fetch documents from vector store by vector IDs.

        Args:
            vector_ids: List of vector IDs to fetch

        Returns:
            List of Document objects
        """
        if not self.vector_store or not vector_ids:
            return []

        try:
            # Try get_by_ids if available
            if hasattr(self.vector_store, "get_by_ids"):
                return self.vector_store.get_by_ids(vector_ids)

            # Fallback: use similarity_search for each (less efficient)
            logger.warning("Vector store does not support get_by_ids, using fallback")
            # Return empty list as we don't have the query anymore
            return []

        except Exception as e:
            logger.error(f"Failed to fetch documents by vector IDs: {e}")
            return []

    def _get_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all nodes from graph store.

        Returns:
            Dict mapping node_id -> node_data
        """
        try:
            nodes_dict = {}
            for node_id, node_data in self.graph_store.nodes(data=True):
                nodes_dict[node_id] = node_data
            return nodes_dict
        except Exception as e:
            logger.error(f"Failed to retrieve nodes from graph store: {e}")
            return {}
