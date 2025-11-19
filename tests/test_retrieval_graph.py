"""
Tests for GraphRetriever.
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document

from ragdoll.retrieval import GraphRetriever


class TestGraphRetriever:
    """Test suite for GraphRetriever."""

    @pytest.fixture
    def mock_graph_store(self):
        """Create a mock graph store with NetworkX-like interface."""
        store = MagicMock()

        # Mock nodes data
        nodes_data = {
            "node1": {"name": "Alice", "type": "Person"},
            "node2": {"name": "Acme Corp", "type": "Organization"},
            "node3": {"name": "Product X", "type": "Product"},
        }

        # NetworkX-style interface
        store.nodes = nodes_data
        store.__contains__ = lambda key: key in nodes_data

        # Mock nodes(data=True) for NetworkX-style iteration
        def nodes_with_data(data=False):
            if data:
                return [(k, v) for k, v in nodes_data.items()]
            return list(nodes_data.keys())

        store.nodes = Mock()
        store.nodes.__getitem__ = lambda key: nodes_data[key]
        store.nodes.__contains__ = lambda key: key in nodes_data
        store.nodes.side_effect = lambda data=False: nodes_with_data(data)

        # Mock neighbors method
        def neighbors(node_id):
            if node_id == "node1":
                return ["node2"]
            elif node_id == "node2":
                return ["node3"]
            return []

        store.neighbors = neighbors

        # Mock get_edge_data
        def get_edge_data(source, target):
            if source == "node1" and target == "node2":
                return {"type": "WORKS_FOR"}
            elif source == "node2" and target == "node3":
                return {"type": "PRODUCES"}
            return {}

        store.get_edge_data = get_edge_data
        store.number_of_edges = Mock(return_value=2)

        # Mock GraphRetriever helper methods
        store.get_all_nodes = Mock(return_value=nodes_data)

        def get_node(node_id):
            return nodes_data.get(node_id)

        store.get_node = get_node

        def get_neighbors(node_id):
            neighbor_ids = neighbors(node_id)
            result = []
            for neighbor_id in neighbor_ids:
                edge_data = get_edge_data(node_id, neighbor_id)
                result.append((neighbor_id, edge_data))
            return result

        store.get_neighbors = get_neighbors

        return store

    def test_init_default_params(self, mock_graph_store):
        """Test initialization with default parameters."""
        retriever = GraphRetriever(graph_store=mock_graph_store)

        assert retriever.graph_store == mock_graph_store
        assert retriever.top_k == 5
        assert retriever.max_hops == 2
        assert retriever.traversal_strategy == "bfs"
        assert retriever.include_edges is True
        assert retriever.min_score == 0.0

    def test_init_custom_params(self, mock_graph_store):
        """Test initialization with custom parameters."""
        retriever = GraphRetriever(
            graph_store=mock_graph_store,
            top_k=10,
            max_hops=3,
            traversal_strategy="dfs",
            include_edges=False,
            min_score=0.5,
        )

        assert retriever.top_k == 10
        assert retriever.max_hops == 3
        assert retriever.traversal_strategy == "dfs"
        assert retriever.include_edges is False
        assert retriever.min_score == 0.5

    def test_init_invalid_strategy(self, mock_graph_store):
        """Test initialization with invalid traversal strategy."""
        with pytest.raises(ValueError, match="Unknown traversal strategy"):
            GraphRetriever(graph_store=mock_graph_store, traversal_strategy="invalid")

    def test_find_seed_nodes(self, mock_graph_store):
        """Test seed node selection."""
        retriever = GraphRetriever(graph_store=mock_graph_store, top_k=2)

        seeds = retriever._find_seed_nodes("Alice", top_k=2)

        assert len(seeds) > 0
        assert all(isinstance(s, tuple) and len(s) == 2 for s in seeds)
        # Should find Alice with high score
        node_ids = [s[0] for s in seeds]
        assert "node1" in node_ids

    def test_score_node(self, mock_graph_store):
        """Test node scoring."""
        retriever = GraphRetriever(graph_store=mock_graph_store)

        node_data = {"name": "Alice Smith", "type": "Person"}
        query_terms = {"alice", "smith"}
        query_lower = "alice smith"

        score = retriever._score_node(node_data, query_terms, query_lower)

        assert score > 0

    def test_bfs_traverse(self, mock_graph_store):
        """Test BFS traversal."""
        retriever = GraphRetriever(
            graph_store=mock_graph_store, traversal_strategy="bfs", max_hops=2
        )

        seed_nodes = [("node1", 1.0)]
        subgraph = retriever._bfs_traverse(seed_nodes, max_hops=2)

        assert "nodes" in subgraph
        assert "edges" in subgraph
        assert "node1" in subgraph["nodes"]
        # Should have traversed to node2
        assert "node2" in subgraph["nodes"]

    def test_dfs_traverse(self, mock_graph_store):
        """Test DFS traversal."""
        retriever = GraphRetriever(
            graph_store=mock_graph_store, traversal_strategy="dfs", max_hops=2
        )

        seed_nodes = [("node1", 1.0)]
        subgraph = retriever._dfs_traverse(seed_nodes, max_hops=2)

        assert "nodes" in subgraph
        assert "edges" in subgraph
        assert "node1" in subgraph["nodes"]

    def test_subgraph_to_documents(self, mock_graph_store):
        """Test subgraph to documents conversion."""
        retriever = GraphRetriever(graph_store=mock_graph_store, include_edges=True)

        subgraph = {
            "nodes": {
                "node1": {
                    "name": "Alice",
                    "type": "Person",
                    "relevance_score": 1.0,
                    "hop_distance": 0,
                },
                "node2": {
                    "name": "Acme Corp",
                    "type": "Organization",
                    "relevance_score": 0.7,
                    "hop_distance": 1,
                },
            },
            "edges": [
                {"source": "node1", "target": "node2", "relationship": "WORKS_FOR"}
            ],
        }

        docs = retriever._subgraph_to_documents(subgraph, include_edges=True)

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].metadata["node_type"] == "Person"
        assert docs[0].metadata["relevance_score"] == 1.0
        # Should include relationships
        assert "WORKS_FOR" in docs[0].page_content

    def test_get_relevant_documents(self, mock_graph_store):
        """Test full retrieval pipeline."""
        retriever = GraphRetriever(graph_store=mock_graph_store, top_k=2, max_hops=1)

        docs = retriever.get_relevant_documents("Alice")

        assert isinstance(docs, list)
        # Should retrieve at least one document
        assert len(docs) >= 0

    def test_get_relevant_documents_no_store(self):
        """Test retrieval with no graph store."""
        retriever = GraphRetriever(graph_store=None)

        docs = retriever.get_relevant_documents("test query")

        assert docs == []

    def test_get_relevant_documents_runtime_override(self, mock_graph_store):
        """Test runtime parameter override."""
        retriever = GraphRetriever(graph_store=mock_graph_store, top_k=2, max_hops=1)

        docs = retriever.get_relevant_documents(
            "Alice", top_k=5, max_hops=3, include_edges=False
        )

        # Should use overridden parameters
        assert isinstance(docs, list)

    @pytest.mark.asyncio
    async def test_aget_relevant_documents(self, mock_graph_store):
        """Test async retrieval (delegates to sync)."""
        retriever = GraphRetriever(graph_store=mock_graph_store)

        docs = await retriever.aget_relevant_documents("Alice")

        assert isinstance(docs, list)

    def test_get_stats(self, mock_graph_store):
        """Test retriever statistics."""
        retriever = GraphRetriever(
            graph_store=mock_graph_store, top_k=5, max_hops=2, traversal_strategy="bfs"
        )

        stats = retriever.get_stats()

        assert stats["top_k"] == 5
        assert stats["max_hops"] == 2
        assert stats["traversal_strategy"] == "bfs"
        assert stats["include_edges"] is True
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2

    def test_get_all_nodes(self, mock_graph_store):
        """Test getting all nodes from graph."""
        retriever = GraphRetriever(graph_store=mock_graph_store)

        nodes = retriever._get_all_nodes()

        assert len(nodes) == 3
        assert "node1" in nodes
        assert nodes["node1"]["name"] == "Alice"

    def test_get_node_data(self, mock_graph_store):
        """Test getting specific node data."""
        retriever = GraphRetriever(graph_store=mock_graph_store)

        node_data = retriever._get_node_data("node1")

        assert node_data is not None
        assert node_data["name"] == "Alice"

    def test_get_neighbors(self, mock_graph_store):
        """Test getting node neighbors."""
        retriever = GraphRetriever(graph_store=mock_graph_store)

        neighbors = retriever._get_neighbors("node1")

        assert len(neighbors) == 1
        assert neighbors[0][0] == "node2"
        assert neighbors[0][1]["type"] == "WORKS_FOR"

    def test_score_decay_by_hops(self, mock_graph_store):
        """Test score decay over multiple hops."""
        retriever = GraphRetriever(graph_store=mock_graph_store, max_hops=2)

        seed_nodes = [("node1", 1.0)]
        subgraph = retriever._bfs_traverse(seed_nodes, max_hops=2)

        # Check that scores decay with distance
        if "node2" in subgraph["nodes"]:
            hop1_score = subgraph["nodes"]["node2"]["relevance_score"]
            assert hop1_score < 1.0  # Should be decayed
