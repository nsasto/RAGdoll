from langchain_core.documents import Document

from ragdoll.entity_extraction.models import Graph, GraphEdge, GraphNode
from ragdoll.retrievers import RagdollRetriever


class _FakeVectorStore:
    def similarity_search(self, query: str, k: int = 4):
        return [
            Document(
                page_content=f"Vector hit for {query}",
                metadata={"source": "vec1", "score": 0.9},
            )
        ]


class _FakeGraphRetriever:
    def __init__(self) -> None:
        # Minimal graph to allow hop expansion without errors.
        self.graph = Graph(
            nodes=[
                GraphNode(id="n1", name="Alice", type="ENTITY"),
                GraphNode(id="n2", name="Bob", type="ENTITY"),
            ],
            edges=[GraphEdge(source="n1", target="n2", type="KNOWS")],
        )

    def get_relevant_documents(self, query: str):
        return [
            Document(
                page_content=f"Graph hit for {query}",
                metadata={"node_id": "n1", "score": 1.0},
            )
        ]


def test_ragdoll_retriever_merges_vector_and_graph():
    retriever = RagdollRetriever(
        vector_store=_FakeVectorStore(),
        graph_retriever=_FakeGraphRetriever(),
        graph_hops=1,
        top_k_vector=3,
        top_k_graph=3,
        max_results=10,
    )

    docs = retriever.get_relevant_documents("test query")

    # Expect both a vector hit and at least one graph node.
    kinds = {doc.metadata.get("source_kind") for doc in docs}
    assert "vector" in kinds
    assert "graph" in kinds

    # Ensure scores are attached for downstream sorting.
    for doc in docs:
        assert "score_total" in (doc.metadata or {})
