# Graph Stores

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending Graph Stores](#extending-graph-stores)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/graph_stores/`

## Purpose

Graph stores manage relationships between document chunks, entities, or concepts. They enable advanced retrieval, reasoning, and knowledge graph applications.

## Key Components

- `base_graph_store.py`: Abstract base for graph store implementations.
- `__init__.py`: Module exports.

## Features

- Flexible graph schema.
- Relationship and entity management.
- Useful for advanced retrieval and reasoning.

---

## How It Works

1. **Nodes**: Represent document chunks, entities, or concepts.
2. **Edges**: Represent relationships between nodes.
3. **Querying**: Supports graph queries for advanced retrieval.

---

## Public API and Function Documentation

### `BaseGraphStore`

#### `add_node(node_id: str, properties: Dict[str, Any]) -> None`

Add a node to the graph with the given properties.

#### `add_edge(source_id: str, target_id: str, relationship: str, properties: Dict[str, Any]) -> None`

Add an edge (relationship) between two nodes.

#### `query_graph(query: str) -> List[Dict[str, Any]]`

Query the graph and return results as a list of dictionaries.

---

## Usage Example

```python
from ragdoll.graph_stores.base_graph_store import BaseGraphStore
# Implement a concrete graph store subclass, then:
store = MyGraphStore()
store.add_node("doc1", {"title": "Document 1"})
store.add_edge("doc1", "doc2", "references", {})
results = store.query_graph("MATCH (n)-[r]->(m) RETURN n, r, m")
```

---

## Extending Graph Stores

- Subclass `BaseGraphStore` and implement required methods for new backends.
- Add support for new query languages or graph features as needed.

---

## Best Practices

- Use meaningful node and edge properties for richer queries.
- Validate graph integrity when adding nodes/edges.
- Optimize queries for performance on large graphs.

---

## Related Modules

- [Vector Stores](vector_stores.md)
- [Ingestion](ingestion.md)
