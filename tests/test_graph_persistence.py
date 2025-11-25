from __future__ import annotations

import pytest
from pathlib import Path
import json

from ragdoll.entity_extraction.graph_persistence import GraphPersistenceService
from ragdoll.entity_extraction.models import Graph, GraphEdge, GraphNode


def _build_graph() -> Graph:
    node_a = GraphNode(id="a", type="Person", name="Barack Obama", metadata={"role": "president"})
    node_b = GraphNode(id="b", type="Location", name="Honolulu", metadata={"country": "USA"})
    edge = GraphEdge(source="a", target="b", type="BORN_IN")
    return Graph(nodes=[node_a, node_b], edges=[edge])


def test_graph_persistence_save_json(tmp_path):
    graph = _build_graph()
    output_file = tmp_path / "graph.json"
    
    service = GraphPersistenceService(output_format="json", output_path=str(output_file))
    saved_graph = service.save(graph)
    
    assert saved_graph == graph
    assert output_file.exists()
    
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1


def test_graph_persistence_save_custom_object():
    graph = _build_graph()
    service = GraphPersistenceService(output_format="custom_graph_object")
    saved_graph = service.save(graph)
    assert saved_graph == graph
