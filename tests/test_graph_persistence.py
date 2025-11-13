from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ragdoll.entity_extraction.graph_persistence import (
    GraphPersistenceService,
    SimpleGraphRetriever,
)
from ragdoll.entity_extraction.models import Graph, GraphEdge, GraphNode


def _build_graph() -> Graph:
    node_a = GraphNode(id="a", type="Person", name="Barack Obama", metadata={"role": "president"})
    node_b = GraphNode(id="b", type="Location", name="Honolulu", metadata={"country": "USA"})
    edge = GraphEdge(source="a", target="b", type="BORN_IN")
    return Graph(nodes=[node_a, node_b], edges=[edge])


def test_simple_graph_retriever_returns_documents():
    graph = _build_graph()
    retriever = SimpleGraphRetriever(graph, top_k=1)

    docs = retriever.invoke("Obama")

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.metadata["node_id"] == "a"
    assert "b" in doc.metadata["connected_to"]


def test_graph_persistence_service_create_retriever_uses_last_graph():
    graph = _build_graph()
    service = GraphPersistenceService()
    service.save(graph)

    retriever = service.create_retriever()
    docs = retriever.invoke("Honolulu")

    assert docs  # ensures retriever returns hits
    assert docs[0].metadata["node_type"] == "Location"


def test_graph_persistence_service_create_retriever_requires_graph():
    service = GraphPersistenceService()

    with pytest.raises(ValueError):
        service.create_retriever()


def test_graph_persistence_custom_factory_receives_config():
    graph = _build_graph()
    captured = {}

    def factory(graph_obj: Graph, cfg: dict[str, object]):
        captured["graph"] = graph_obj
        captured["config"] = cfg
        return "custom"

    service = GraphPersistenceService(retriever_factory=factory)
    service.save(graph)

    retriever = service.create_retriever(foo="bar")

    assert retriever == "custom"
    assert captured["graph"] == graph
    assert captured["config"]["foo"] == "bar"
