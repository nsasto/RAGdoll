"""
Quick diagnostic to check graph quality for hybrid mode debugging.
"""

import pickle
from pathlib import Path


def analyze_graph(graph_path: Path):
    """Analyze a pickled NetworkX graph."""
    if not graph_path.exists():
        print(f"❌ Graph not found: {graph_path}")
        return

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    print(f"\n📊 Graph Analysis: {graph_path.name}")
    print("=" * 70)

    # Basic stats
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Density: {num_edges / max(num_nodes * (num_nodes - 1), 1):.4f}")

    # Node types
    node_types = {}
    entity_nodes = 0
    chunk_nodes = 0

    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1

        if node_type == "entity":
            entity_nodes += 1
        elif node_type == "chunk":
            chunk_nodes += 1

    print(f"\nNode Types:")
    for ntype, count in sorted(node_types.items()):
        print(f"  {ntype}: {count}")

    # Edge types
    edge_types = {}
    for u, v, data in graph.edges(data=True):
        edge_type = data.get("relationship", "unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print(f"\nEdge Types:")
    for etype, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        print(f"  {etype}: {count}")

    # Sample entities
    print(f"\nSample Entities (first 10):")
    entity_count = 0
    for node, data in graph.nodes(data=True):
        if data.get("type") == "entity" and entity_count < 10:
            entity_type = data.get("entity_type", "unknown")
            print(f"  - {node} ({entity_type})")
            entity_count += 1

    # Connectivity check
    if entity_nodes > 0 and chunk_nodes > 0:
        print(
            f"\n✅ Graph has both entities ({entity_nodes}) and chunks ({chunk_nodes})"
        )
    elif entity_nodes == 0:
        print(f"\n❌ WARNING: No entity nodes found! Graph retrieval will fail.")
    elif chunk_nodes == 0:
        print(
            f"\n⚠️  No chunk nodes found. Entities may not link back to source passages."
        )

    # Check vector_id linkage
    nodes_with_vector_id = 0
    for node, data in graph.nodes(data=True):
        if "properties" in data and "vector_id" in data["properties"]:
            nodes_with_vector_id += 1
        elif "vector_id" in data:
            nodes_with_vector_id += 1

    print(f"\n🔗 Vector ID Linkage:")
    print(f"   Nodes with vector_id: {nodes_with_vector_id}/{num_nodes}")
    if num_nodes > 0:
        coverage = nodes_with_vector_id / num_nodes * 100
        print(f"   Coverage: {coverage:.1f}%")
        if coverage < 50:
            print(f"   ⚠️  Low coverage - entities may not link to source passages")

    print("=" * 70)


if __name__ == "__main__":
    import sys

    # Check all benchmark databases
    benchmarks_dir = Path(__file__).parent
    db_dir = benchmarks_dir / "db"

    if not db_dir.exists():
        print(f"❌ No db directory found: {db_dir}")
        sys.exit(1)

    # Find all graph.pkl files
    graph_files = list(db_dir.rglob("graph.pkl"))

    if not graph_files:
        print(f"❌ No graph.pkl files found in {db_dir}")
        print("\nRun the benchmark with --create first:")
        print(
            "  python ragdoll_benchmark.py -d 2wikimultihopqa -n 51 --mode hybrid --create"
        )
        sys.exit(1)

    for graph_file in graph_files:
        analyze_graph(graph_file)
