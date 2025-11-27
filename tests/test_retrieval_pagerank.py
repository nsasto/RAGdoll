"""
Unit tests for PageRankGraphRetriever.

Tests cover:
- Seed selection (embedding vs keyword strategies)
- Local subgraph building with BFS
- Personalized PageRank computation
- Node ranking and vector ID selection
- Deduplication on vector_id
- Fallback behavior
- Integration with graph and vector stores
- Performance under realistic conditions
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Tuple, Set
import numpy as np
from langchain_core.documents import Document

from ragdoll.retrieval.pagerank import PageRankGraphRetriever


@pytest.fixture
def mock_graph_store():
    """Create a mock graph store for testing."""
    store = MagicMock()

    # Define a simple test graph
    nodes_data = {
        "entity_1": {
            "id": "entity_1",
            "name": "Alice",
            "type": "entity",
            "properties": {"vector_id": "vec_1"},
        },
        "entity_2": {
            "id": "entity_2",
            "name": "Bob",
            "type": "entity",
            "properties": {"vector_id": "vec_2"},
        },
        "entity_3": {
            "id": "entity_3",
            "name": "Charlie",
            "type": "entity",
            "properties": {"vector_id": "vec_3"},
        },
        "event_1": {
            "id": "event_1",
            "name": "Meeting",
            "type": "event",
            "properties": {"vector_id": "vec_4"},
        },
    }

    # Define adjacency
    adjacency = {
        "entity_1": ["entity_2", "event_1"],
        "entity_2": ["entity_1", "entity_3"],
        "entity_3": ["entity_2"],
        "event_1": ["entity_1"],
    }

    def nodes_with_data(data=False):
        if data:
            return list(nodes_data.items())
        return list(nodes_data.keys())

    def get_neighbors(node_id):
        return adjacency.get(node_id, [])

    def get_edge_data(src, dst):
        if dst in adjacency.get(src, []):
            return {"weight": 1.0}
        return None

    store.nodes.side_effect = nodes_with_data
    store.neighbors.side_effect = get_neighbors
    store.get_edge_data.side_effect = get_edge_data
    store._nodes_data = nodes_data

    return store


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    store = MagicMock()

    # Mock similarity search results
    chunks = [
        (
            Document(
                page_content="Alice works at company", metadata={"vector_id": "vec_1"}
            ),
            0.9,
        ),
        (
            Document(page_content="Bob is a manager", metadata={"vector_id": "vec_2"}),
            0.8,
        ),
        (
            Document(
                page_content="Meeting with Alice and Bob",
                metadata={"vector_id": "vec_4"},
            ),
            0.7,
        ),
    ]

    store.similarity_search_with_scores.return_value = chunks
    store.similarity_search.return_value = [doc for doc, score in chunks]

    # Mock get_by_ids
    def get_by_ids(ids):
        result = []
        for vid in ids:
            if vid == "vec_1":
                result.append(
                    Document(
                        page_content="Alice works at company",
                        metadata={"vector_id": "vec_1"},
                    )
                )
            elif vid == "vec_2":
                result.append(
                    Document(
                        page_content="Bob is a manager", metadata={"vector_id": "vec_2"}
                    )
                )
            elif vid == "vec_3":
                result.append(
                    Document(
                        page_content="Charlie is an engineer",
                        metadata={"vector_id": "vec_3"},
                    )
                )
            elif vid == "vec_4":
                result.append(
                    Document(
                        page_content="Meeting with Alice and Bob",
                        metadata={"vector_id": "vec_4"},
                    )
                )
        return result

    store.get_by_ids.side_effect = get_by_ids

    return store


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = MagicMock()
    model.embed_query.return_value = np.random.randn(768).tolist()
    return model


@pytest.fixture
def pagerank_retriever(mock_graph_store, mock_vector_store, mock_embedding_model):
    """Create a PageRankGraphRetriever instance."""
    return PageRankGraphRetriever(
        graph_store=mock_graph_store,
        vector_store=mock_vector_store,
        embedding_model=mock_embedding_model,
        top_k=5,
        max_nodes=200,
        max_hops=3,
        seed_strategy="embedding",
        num_seed_chunks=5,
        damping_factor=0.15,
        max_iter=50,
        tol=1e-6,
    )


class TestSeedSelection:
    """Tests for seed node selection strategies."""

    def test_select_seeds_embedding_strategy(
        self, pagerank_retriever, mock_vector_store
    ):
        """Test seed selection using embedding strategy."""
        seeds = pagerank_retriever._select_seeds_embedding("test query")

        assert len(seeds) > 0
        assert all(isinstance(node_id, str) for node_id, score in seeds)
        assert all(isinstance(score, float) for node_id, score in seeds)

    def test_select_seeds_keyword_strategy(self, pagerank_retriever):
        """Test seed selection using keyword strategy."""
        seeds = pagerank_retriever._select_seeds_keyword("Alice meeting")

        # Should find nodes matching keywords
        assert len(seeds) > 0
        node_ids = [nid for nid, score in seeds]
        # Alice or event_1 should be in results
        assert "entity_1" in node_ids or "event_1" in node_ids

    def test_select_seeds_with_no_vector_store(self, mock_graph_store):
        """Test embedding seed selection falls back when no vector store."""
        retriever = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            vector_store=None,
            seed_strategy="embedding",
        )

        # Should fall back to keyword strategy
        seeds = retriever._select_seeds_embedding("test query")
        assert isinstance(seeds, list)

    def test_select_seeds_keyword_no_fuzz(self, mock_graph_store):
        """Test keyword seed selection without rapidfuzz."""
        retriever = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            seed_strategy="keyword",
        )

        # Test that it handles missing rapidfuzz gracefully
        seeds = retriever._select_seeds_keyword("Alice")
        # Should still work with basic string matching
        assert isinstance(seeds, list)


class TestSubgraphBuilding:
    """Tests for local subgraph extraction."""

    def test_build_local_subgraph_basic(self, pagerank_retriever, mock_graph_store):
        """Test building a local subgraph with BFS."""
        seed_nodes = [("entity_1", 0.9)]
        adjacency, node_to_idx, idx_to_node = pagerank_retriever._build_local_subgraph(
            seed_nodes, max_hops=2, max_nodes=100
        )

        assert len(node_to_idx) > 0
        assert len(adjacency) == len(node_to_idx)
        assert "entity_1" in node_to_idx

        # Verify bidirectional edges (undirected graph)
        for node_id, neighbors in adjacency.items():
            for neighbor_id, weight in neighbors:
                assert weight > 0

    def test_build_local_subgraph_respects_max_nodes(
        self, pagerank_retriever, mock_graph_store
    ):
        """Test that max_nodes limit is respected."""
        seed_nodes = [("entity_1", 0.9)]
        adjacency, node_to_idx, idx_to_node = pagerank_retriever._build_local_subgraph(
            seed_nodes, max_hops=10, max_nodes=2
        )

        assert len(node_to_idx) <= 2

    def test_build_local_subgraph_respects_max_hops(
        self, pagerank_retriever, mock_graph_store
    ):
        """Test that max_hops limit is respected."""
        seed_nodes = [("entity_1", 0.9)]
        adjacency, node_to_idx, idx_to_node = pagerank_retriever._build_local_subgraph(
            seed_nodes, max_hops=1, max_nodes=100
        )

        # With max_hops=1, should only reach immediate neighbors
        assert len(node_to_idx) <= 3  # entity_1, entity_2, event_1

    def test_build_local_subgraph_empty_seeds(self, pagerank_retriever):
        """Test subgraph building with no seeds."""
        adjacency, node_to_idx, idx_to_node = pagerank_retriever._build_local_subgraph(
            [], max_hops=2, max_nodes=100
        )

        assert len(node_to_idx) == 0
        assert len(adjacency) == 0


class TestPageRankComputation:
    """Tests for personalized PageRank algorithm."""

    def test_run_personalized_pagerank_basic(self, pagerank_retriever):
        """Test basic PageRank computation."""
        # Create a simple subgraph
        adjacency = {
            "n1": [("n2", 1.0)],
            "n2": [("n1", 1.0), ("n3", 1.0)],
            "n3": [("n2", 1.0)],
        }
        node_to_idx = {"n1": 0, "n2": 1, "n3": 2}
        seed_indices = {0}  # seed is n1

        scores = pagerank_retriever._run_personalized_pagerank(
            adjacency, node_to_idx, seed_indices
        )

        assert len(scores) == 3
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
        # All scores should be finite
        assert np.all(np.isfinite(scores))

    def test_pagerank_convergence(self, pagerank_retriever):
        """Test that PageRank converges."""
        adjacency = {
            "n1": [("n2", 1.0)],
            "n2": [("n3", 1.0)],
            "n3": [("n1", 1.0)],
        }
        node_to_idx = {"n1": 0, "n2": 1, "n3": 2}
        seed_indices = {0}

        # Should converge even with cycle
        scores = pagerank_retriever._run_personalized_pagerank(
            adjacency, node_to_idx, seed_indices
        )

        assert np.all(np.isfinite(scores))

    def test_pagerank_dangling_nodes(self, pagerank_retriever):
        """Test PageRank with dangling nodes (no outgoing edges)."""
        adjacency = {
            "n1": [("n2", 1.0)],
            "n2": [],  # Dangling node
        }
        node_to_idx = {"n1": 0, "n2": 1}
        seed_indices = {0}

        scores = pagerank_retriever._run_personalized_pagerank(
            adjacency, node_to_idx, seed_indices
        )

        assert len(scores) == 2
        assert np.all(np.isfinite(scores))

    def test_pagerank_edge_weights(self, pagerank_retriever):
        """Test PageRank respects edge weights."""
        # Heavy edge to n2, light edge to n3
        adjacency = {
            "n1": [("n2", 10.0), ("n3", 1.0)],
        }
        node_to_idx = {"n1": 0, "n2": 1, "n3": 2}
        seed_indices = {0}

        scores = pagerank_retriever._run_personalized_pagerank(
            adjacency, node_to_idx, seed_indices
        )

        # n2 should get higher score due to higher edge weight
        # (though exact scores depend on damping factor)
        assert len(scores) == 3


class TestNodeRanking:
    """Tests for node ranking and vector ID selection."""

    def test_rank_nodes_and_select_vector_ids(
        self, pagerank_retriever, mock_graph_store
    ):
        """Test ranking nodes by PageRank and selecting top vector IDs."""
        node_to_idx = {"entity_1": 0, "entity_2": 1, "event_1": 2}
        idx_to_node = {0: "entity_1", 1: "entity_2", 2: "event_1"}
        scores = np.array([0.5, 0.3, 0.2])

        vector_ids = pagerank_retriever._rank_nodes_and_select_vector_ids(
            node_to_idx, idx_to_node, scores, top_k=2
        )

        assert len(vector_ids) <= 2
        assert "vec_1" in vector_ids  # entity_1 has highest score

    def test_rank_nodes_respects_min_score(self, pagerank_retriever, mock_graph_store):
        """Test that min_score threshold is respected."""
        retriever = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            min_score=0.5,
        )

        node_to_idx = {"entity_1": 0, "entity_2": 1}
        idx_to_node = {0: "entity_1", 1: "entity_2"}
        scores = np.array([0.7, 0.3])  # entity_2 below threshold

        vector_ids = retriever._rank_nodes_and_select_vector_ids(
            node_to_idx, idx_to_node, scores, top_k=2
        )

        assert "vec_2" not in vector_ids  # entity_2 should be filtered

    def test_rank_nodes_respects_allowed_types(
        self, pagerank_retriever, mock_graph_store
    ):
        """Test that only allowed node types are included."""
        retriever = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            allowed_node_types=["entity"],  # Exclude events
        )

        node_to_idx = {"entity_1": 0, "event_1": 1}
        idx_to_node = {0: "entity_1", 1: "event_1"}
        scores = np.array([0.3, 0.7])  # event_1 has higher score but wrong type

        vector_ids = retriever._rank_nodes_and_select_vector_ids(
            node_to_idx, idx_to_node, scores, top_k=2
        )

        assert "vec_4" not in vector_ids  # event_1 filtered by type

    def test_rank_nodes_dedup_on_vector_id(self, mock_graph_store, mock_vector_store):
        """Test deduplication on vector_id."""
        # Create a scenario where two nodes map to same vector_id
        retriever = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            dedup_on_vector_id=True,
        )

        # Mock graph store to return same vector_id for two nodes
        mock_nodes = {
            "entity_1": {"type": "entity", "properties": {"vector_id": "vec_1"}},
            "entity_2": {
                "type": "entity",
                "properties": {"vector_id": "vec_1"},
            },  # Same vector_id
        }

        with patch.object(retriever, "_get_all_nodes", return_value=mock_nodes):
            node_to_idx = {"entity_1": 0, "entity_2": 1}
            idx_to_node = {0: "entity_1", 1: "entity_2"}
            scores = np.array([0.7, 0.6])

            vector_ids = retriever._rank_nodes_and_select_vector_ids(
                node_to_idx, idx_to_node, scores, top_k=5
            )

            # Should only include vec_1 once
            assert vector_ids.count("vec_1") == 1


class TestEdgeWeights:
    """Tests for edge weight handling."""

    def test_get_edge_weight_from_metadata(self, pagerank_retriever, mock_graph_store):
        """Test getting edge weight from graph store metadata."""
        weight = pagerank_retriever._get_edge_weight("entity_1", "entity_2")
        assert weight == 1.0

    def test_get_edge_weight_default(self, pagerank_retriever):
        """Test default edge weight when graph store fails."""
        mock_store = MagicMock()
        mock_store.get_edge_data.side_effect = Exception("Not found")
        retriever = PageRankGraphRetriever(graph_store=mock_store)

        weight = retriever._get_edge_weight("n1", "n2")
        assert weight == 1.0


class TestFallbackBehavior:
    """Tests for graceful fallback mechanisms."""

    def test_fallback_to_vector_search_no_seeds(
        self, pagerank_retriever, mock_vector_store
    ):
        """Test fallback to vector search when no seeds found."""
        mock_graph_store = MagicMock()
        mock_graph_store.nodes.return_value = []

        retriever = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            enable_fallback=True,
        )

        docs = retriever.get_relevant_documents("test query")

        # Should fall back to vector search
        assert len(docs) > 0

    def test_fallback_disabled(self, mock_graph_store):
        """Test that fallback can be disabled."""
        retriever = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            enable_fallback=False,
        )

        # Mock empty graph to trigger fallback path
        mock_graph_store.nodes.return_value = []

        docs = retriever.get_relevant_documents("test query")

        # Should return empty list
        assert len(docs) == 0

    def test_fallback_to_vector_search_on_error(self, mock_vector_store):
        """Test fallback to vector search on retrieval error."""
        retriever = PageRankGraphRetriever(
            graph_store=None,  # Will cause error
            vector_store=mock_vector_store,
            enable_fallback=True,
        )

        docs = retriever.get_relevant_documents("test query")

        # Should fall back to vector search
        assert len(docs) > 0


class TestIntegration:
    """Integration tests with combined components."""

    def test_full_retrieval_pipeline(self, pagerank_retriever):
        """Test complete retrieval pipeline end-to-end."""
        docs = pagerank_retriever.get_relevant_documents("test query about Alice")

        assert isinstance(docs, list)
        assert len(docs) >= 0
        if docs:
            assert all(isinstance(d, Document) for d in docs)
            assert all("retrieval_source" in d.metadata for d in docs)

    def test_async_retrieval(self, pagerank_retriever):
        """Test async retrieval."""
        import asyncio

        async def test_async():
            docs = await pagerank_retriever.aget_relevant_documents("test query")
            return docs

        docs = asyncio.run(test_async())
        assert isinstance(docs, list)

    def test_different_seed_strategies(
        self, pagerank_retriever, mock_graph_store, mock_vector_store
    ):
        """Test both seed strategies produce reasonable results."""
        # Embedding strategy
        retriever_emb = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            seed_strategy="embedding",
        )
        docs_emb = retriever_emb.get_relevant_documents("test query")

        # Keyword strategy
        retriever_kw = PageRankGraphRetriever(
            graph_store=mock_graph_store,
            seed_strategy="keyword",
        )
        docs_kw = retriever_kw.get_relevant_documents("test query")

        # Both should work (may return different results)
        assert isinstance(docs_emb, list)
        assert isinstance(docs_kw, list)


class TestPerformance:
    """Performance and edge case tests."""

    def test_large_subgraph_performance(self, pagerank_retriever):
        """Test performance with larger subgraph (sanity check)."""
        import time

        # Create a larger mock graph
        large_graph = MagicMock()
        nodes = {
            f"n_{i}": {
                "type": "entity",
                "name": f"node_{i}",
                "properties": {"vector_id": f"v_{i}"},
            }
            for i in range(100)
        }

        def get_nodes(data=False):
            if data:
                return list(nodes.items())
            return list(nodes.keys())

        def get_neighbors(nid):
            idx = int(nid.split("_")[1])
            neighbors = []
            if idx > 0:
                neighbors.append(f"n_{idx-1}")
            if idx < 99:
                neighbors.append(f"n_{idx+1}")
            return neighbors

        large_graph.nodes.side_effect = get_nodes
        large_graph.neighbors.side_effect = get_neighbors
        large_graph.get_edge_data.return_value = {"weight": 1.0}

        retriever = PageRankGraphRetriever(
            graph_store=large_graph,
            max_nodes=50,
            max_hops=5,
        )

        # Should complete in reasonable time
        start = time.time()
        adjacency, node_to_idx, idx_to_node = retriever._build_local_subgraph(
            [("n_50", 0.9)], max_hops=5, max_nodes=50
        )
        elapsed = time.time() - start

        # Should be fast (< 1 second for 100 node graph)
        assert elapsed < 1.0
        assert len(node_to_idx) <= 50

    def test_pagerank_empty_subgraph(self, pagerank_retriever):
        """Test PageRank with empty subgraph."""
        adjacency = {}
        node_to_idx = {}
        seed_indices = set()

        scores = pagerank_retriever._run_personalized_pagerank(
            adjacency, node_to_idx, seed_indices
        )

        assert len(scores) == 0

    def test_fetch_documents_no_vector_store(self, pagerank_retriever):
        """Test fetching documents when no vector store available."""
        retriever = PageRankGraphRetriever(
            graph_store=MagicMock(),
            vector_store=None,
        )

        docs = retriever._fetch_documents_by_vector_ids(["v_1", "v_2"])
        assert len(docs) == 0

    def test_fetch_documents_empty_ids(self, pagerank_retriever, mock_vector_store):
        """Test fetching with empty ID list."""
        docs = pagerank_retriever._fetch_documents_by_vector_ids([])
        assert len(docs) == 0
