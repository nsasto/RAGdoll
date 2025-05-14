# examples/entity_extraction_example.py

import asyncio
import logging
from typing import List, Dict

from langchain.docstore.document import Document
from ragdoll.entity_extraction.entity_extraction_service import GraphCreationService
from ragdoll.llms.base_llm import BaseLLM
from ragdoll.config.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple mock LLM implementation for example purposes
class MockLLM(BaseLLM):
    def call(self, prompt: str) -> str:
        """
        Simple implementation that returns mock JSON responses
        for graph creation tasks.
        """
        if "entity_extraction_llm" in prompt or "Extract entities" in prompt:
            return '''[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "President", "type": "ROLE"},
                {"text": "United States", "type": "GPE"},
                {"text": "Hawaii", "type": "GPE"},
                {"text": "Michelle Obama", "type": "PERSON"},
                {"text": "Malia Ann Obama", "type": "PERSON"},
                {"text": "Sasha Obama", "type": "PERSON"},
                {"text": "Senator", "type": "ROLE"},
                {"text": "Illinois", "type": "GPE"}
            ]'''
        elif "extract_relationships" in prompt or "relationships" in prompt:
            # Make sure this is correctly detected as a list
            return '''[
                {"subject": "Barack Obama", "relationship": "HAS_ROLE", "object": "President"},
                {"subject": "Barack Obama", "relationship": "BORN_IN", "object": "Hawaii"},
                {"subject": "Barack Obama", "relationship": "SPOUSE_OF", "object": "Michelle Obama"},
                {"subject": "Barack Obama", "relationship": "PARENT_OF", "object": "Malia Ann Obama"},
                {"subject": "Barack Obama", "relationship": "PARENT_OF", "object": "Sasha Obama"},
                {"subject": "Barack Obama", "relationship": "HAS_ROLE", "object": "Senator"},
                {"subject": "Senator", "relationship": "REPRESENTS", "object": "Illinois"}
            ]'''
        elif "coreference_resolution" in prompt:
            # Return text with resolved coreferences
            return "Barack Obama was the 44th President of the United States. Barack Obama was born in Honolulu, Hawaii. Barack Obama's wife is Michelle Obama, and they have two daughters, Malia Ann Obama and Sasha Obama. Barack Obama served as a U.S. Senator from Illinois before becoming President."
        elif "entity_relationship_continue_extraction" in prompt:
            return '''{"nodes": [], "edges": []}'''
        elif "entity_relationship_gleaning_done_extraction" in prompt:
            return "done"
        else:
            return '[]'  # Return empty list by default, not empty dict


async def main():
    """
    Demonstrates how to use the GraphCreationService.
    """
    # Create a mock LLM implementation
    llm = MockLLM()

    # Get configuration from config manager
    config_manager = ConfigManager()
    config = config_manager.get_entity_extraction_service_config()
    
    # Create an instance of the GraphCreationService
    graph_service = GraphCreationService(config)

    # Configure debug output
    print("== Configuration ==")
    print(f"Entity extraction methods: {config['entity_extraction_methods']}")
    print(f"Relationship extraction method: {config['relationship_extraction_method']}")
    print(f"Entity types: {config['entity_types']}")
    print(f"Relationship types: {config['relationship_types']}")

    # Define sample text
    sample_text = (
        "Barack Obama was the 44th President of the United States. "
        "He was born in Honolulu, Hawaii. "
        "His wife is Michelle Obama, and they have two daughters, Malia Ann Obama and Sasha Obama. "
        "Obama served as a U.S. Senator from Illinois before becoming President."
    )

    print(f"Processing text:\n{sample_text}\n")

    # Create a Langchain Document object
    sample_doc = Document(
        page_content=sample_text,
        metadata={
            "source": "example_doc_1",
            "id": "doc1"
        }
    )

    # Extract entities and relationships to form a graph
    graph = await graph_service.extract(
        documents=[sample_doc],
        llm=llm,
        entity_types=config['entity_types'],
        relationship_types=config['relationship_types']
    )

    print(f"\nExtracted {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Display the extracted graph
    print("\n=== Extracted Knowledge Graph ===")
    
    # Display nodes
    print("\nNodes:")
    for node in graph.nodes:
        print(f"• {node.id} - {node.text} ({node.type})")

    # Display edges
    print("\nRelationships:")
    nodes_dict = {node.id: node for node in graph.nodes}
    for edge in graph.edges:
        source_node = nodes_dict.get(edge.source, None)
        target_node = nodes_dict.get(edge.target, None)
        if source_node and target_node:
            print(f"• {source_node.text} --[{edge.type}]--> {target_node.text}")
        else:
            print(f"• {edge.source} --[{edge.type}]--> {edge.target}")

    # Create a NetworkX graph visualization (if available)
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in graph.nodes:
            G.add_node(node.id, label=node.text, type=node.type)
            
        # Add edges with attributes
        for edge in graph.edges:
            G.add_edge(edge.source, edge.target, label=edge.type)
        
        # Set up plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, alpha=0.8, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20)
        
        # Node labels
        node_labels = {node_id: G.nodes[node_id]["label"] for node_id in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # Edge labels
        edge_labels = {(source, target): G.edges[source, target]["label"] 
                     for source, target in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.axis("off")
        plt.title("Knowledge Graph Visualization", size=15)
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig("knowledge_graph.png", dpi=300, bbox_inches="tight")
        print("\nGraph visualization saved as 'knowledge_graph.png'")
        
        # Save graph data as JSON
        if hasattr(graph, "model_dump_json"):  # Pydantic v2
            with open("graph_output.json", "w") as f:
                f.write(graph.model_dump_json(indent=2))
        else:  # Pydantic v1
            with open("graph_output.json", "w") as f:
                f.write(graph.json(indent=2))
                
        print("Graph data saved as 'graph_output.json'")
        
    except ImportError:
        print("\nNote: Install networkx and matplotlib for graph visualization:")
        print("pip install networkx matplotlib")


if __name__ == "__main__":
    asyncio.run(main())