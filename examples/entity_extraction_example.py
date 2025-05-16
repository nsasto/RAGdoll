# examples/entity_extraction_example.py

import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv

from langchain.docstore.document import Document
from ragdoll.entity_extraction.entity_extraction_service import GraphCreationService
from ragdoll.config.config_manager import ConfigManager
from ragdoll.llms import get_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(model_name: Optional[str] = None):
    """
    Demonstrates how to use the GraphCreationService with a real LLM.
    
    Args:
        model_name: Optional name of the model to use. Can be a model name or a model type ('default', 'basic', 'reasoning', 'vision')
    """
    # Load environment variables for API keys
    load_dotenv()
    
    # Get configuration
    config_manager = ConfigManager()
    entity_extraction_config = config_manager.entity_extraction_config.model_dump()

    # Use the real LLM
    model_name = model_name or "gpt-4o"#"gpt-3.5-turbo"
    print(f"Using get_llm with model: {model_name}")
    llm = get_llm(model_name, config_manager)

    # Create the service
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
        metadata={"source": "example_doc_1", "id": "doc1"}
    )

    # Extract the graph
    graph = await graph_service.extract(
        documents=[sample_doc],
        llm=llm,
        entity_types=entity_extraction_config.get('entity_types'),
        relationship_types=entity_extraction_config.get('relationship_types')
    )

    print(f"\nExtracted {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Optional visualization
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        for node in graph.nodes:
            G.add_node(node.id, label=node.text, type=node.type)
        for edge in graph.edges:
            G.add_edge(edge.source, edge.target, label=edge.type)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=2000, alpha=0.8, node_color="lightblue")
        nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20)
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes}, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(s, t): G.edges[s, t]["label"] for s, t in G.edges}, font_size=8)
        plt.axis("off")
        plt.title("Knowledge Graph Visualization", size=15)
        plt.tight_layout()
        plt.savefig("knowledge_graph.png", dpi=300, bbox_inches="tight")
        print("\nGraph visualization saved as 'knowledge_graph.png'")

        with open("graph_output.json", "w") as f:
            try:
                f.write(graph.model_dump_json(indent=2))
            except AttributeError:
                f.write(graph.json(indent=2))
        print("Graph data saved as 'graph_output.json'")

    except ImportError:
        print("\nInstall networkx and matplotlib to visualize the graph: pip install networkx matplotlib")


if __name__ == "__main__":

    logging.getLogger("ragdoll.entity_extraction").setLevel(logging.DEBUG)
    import argparse

    parser = argparse.ArgumentParser(description="Entity extraction example")
    parser.add_argument('--model', type=str, default=None, help='Specify a model name (from config) or a model type (default, reasoning, vision)')
    args = parser.parse_args()

    asyncio.run(main(model_name=args.model))
