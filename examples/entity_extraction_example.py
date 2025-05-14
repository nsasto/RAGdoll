# examples/entity_extraction_example.py

import asyncio
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

from langchain.docstore.document import Document
from ragdoll.entity_extraction.entity_extraction_service import GraphCreationService
from ragdoll.config.config_manager import ConfigManager
from ragdoll.llms import get_litellm_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a mock LLM function for examples/testing
def mock_llm(prompt: str) -> str:
    """Mock LLM function that returns predefined responses for examples and testing."""
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
        return "Barack Obama was the 44th President of the United States. Barack Obama was born in Honolulu, Hawaii. Barack Obama's wife is Michelle Obama, and they have two daughters, Malia Ann Obama and Sasha Obama. Barack Obama served as a U.S. Senator from Illinois before becoming President."
    elif "entity_relationship_continue_extraction" in prompt:
        return '''{"nodes": [], "edges": []}'''
    elif "entity_relationship_gleaning_done_extraction" in prompt:
        return "done"
    else:
        return '[]'


async def main(use_mock_llm: bool = True, model_name: Optional[str] = None):
    """
    Demonstrates how to use the GraphCreationService.
    
    Args:
        use_mock_llm: If True, use the mock LLM function. If False, use LiteLLM.
        model_name: Optional name of the model to use (if not using mock LLM)
    """
    # Load environment variables for API keys
    load_dotenv()
    
    # Get configuration
    config_manager = ConfigManager()
    entity_extraction_config = config_manager.get_entity_extraction_service_config()
    
    # Determine which LLM to use
    if use_mock_llm:
        print("Using mock LLM for entity extraction (predefined responses)")
        llm = mock_llm
    else:
        # If no specific model provided, use the default from config
        model_name = model_name or config_manager._config.get("llms", {}).get("default_model", "gpt-3.5-turbo")
        print(f"Using LiteLLM with model: {model_name}")
        llm = get_litellm_model(model_name, config_manager)
    
    # Create the service and process the text
    graph_service = GraphCreationService(entity_extraction_config)
    
    # Define sample text
    sample_text = (
        "Barack Obama was the 44th President of the United States. "
        "He was born in Honolulu, Hawaii. "
        "His wife is Michelle Obama, and they have two daughters, Malia Ann Obama and Sasha Obama. "
        "Obama served as a U.S. Senator from Illinois before becoming President."
    )

    print(f"\nProcessing text:\n{sample_text}\n")

    # Create a Langchain Document
    sample_doc = Document(
        page_content=sample_text,
        metadata={
            "source": "example_doc_1",
            "id": "doc1"
        }
    )

    # Extract the graph
    graph = await graph_service.extract(
        documents=[sample_doc],
        llm=llm,
        entity_types=entity_extraction_config.get('entity_types'),
        relationship_types=entity_extraction_config.get('relationship_types')
    )

    # Display results
    print(f"\nExtracted {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Visualization code remains the same...
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
        plt.title("Knowledge Graph Visualization", size=15)
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig("knowledge_graph.png", dpi=300, bbox_inches="tight")
        print("\nGraph visualization saved as 'knowledge_graph.png'")
        
        # Save graph data as JSON
        try:
            # Try Pydantic v2 method first
            with open("graph_output.json", "w") as f:
                f.write(graph.model_dump_json(indent=2))
        except AttributeError:
            # Fall back to Pydantic v1 method
            with open("graph_output.json", "w") as f:
                f.write(graph.json(indent=2))
                
        print("Graph data saved as 'graph_output.json'")
        
    except ImportError:
        print("\nNote: Install networkx and matplotlib for graph visualization:")
        print("pip install networkx matplotlib")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entity extraction example")
    parser.add_argument('--use-real-llm', action='store_true', help='Use real LLM instead of mock')
    parser.add_argument('--model', type=str, default=None, help='Specify a model name (from config)')
    args = parser.parse_args()
    
    asyncio.run(main(use_mock_llm=not args.use_real_llm, model_name=args.model))