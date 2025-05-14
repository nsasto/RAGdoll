import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from langchain.docstore.document import Document
from ragdoll.llms.base_llm import BaseLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple models for the example
class GraphNode:
    def __init__(self, id=None, text="", type="", metadata=None):
        self.id = id or str(uuid.uuid4())
        self.text = text
        self.type = type
        self.metadata = metadata or {}


class GraphEdge:
    def __init__(self, id=None, source="", target="", type="", metadata=None, source_document_id=None):
        self.id = id or str(uuid.uuid4())
        self.source = source
        self.target = target
        self.type = type
        self.metadata = metadata or {}
        self.source_document_id = source_document_id


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
    
    def json(self):
        """Return a JSON representation of the graph"""
        return json.dumps({
            "nodes": [self._node_to_dict(node) for node in self.nodes],
            "edges": [self._edge_to_dict(edge) for edge in self.edges]
        }, indent=2)
        
    def _node_to_dict(self, node):
        return {
            "id": node.id,
            "text": node.text,
            "type": node.type,
            "metadata": node.metadata
        }
        
    def _edge_to_dict(self, edge):
        return {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "type": edge.type,
            "metadata": edge.metadata,
            "source_document_id": edge.source_document_id
        }


# Simple mock LLM implementation for example purposes
class MockLLM(BaseLLM):
    def call(self, prompt: str) -> str:
        """
        Simple implementation that returns mock JSON responses
        for graph creation tasks.
        """
        if "entity_extraction_llm" in prompt or "Extract" in prompt:
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
        elif "extract_relationships" in prompt:
            return '''[
                {"subject": "Barack Obama", "relationship": "HAS_ROLE", "object": "President"},
                {"subject": "Barack Obama", "relationship": "BORN_IN", "object": "Hawaii"},
                {"subject": "Barack Obama", "relationship": "SPOUSE_OF", "object": "Michelle Obama"},
                {"subject": "Barack Obama", "relationship": "PARENT_OF", "object": "Malia Ann Obama"},
                {"subject": "Barack Obama", "relationship": "PARENT_OF", "object": "Sasha Obama"},
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
            return '{}'


@dataclass
class MockGraphCreationService:
    config: Dict[str, Any]
    
    def __init__(self, config=None):
        self.config = config or {}
        self.llm = None
    
    async def extract(self, documents, llm, entity_types=None, relationship_types=None):
        """
        Simplified mock implementation of the graph extraction process
        """
        self.llm = llm
        graph = Graph()
        entity_id_map = {}
        
        for doc in documents:
            logger.info(f"Processing document: {doc.metadata.get('id', 'unknown')}")
            
            # Extract entities
            entity_data = json.loads(self.llm.call(f"Extract entities from: {doc.page_content}"))
            
            # Create nodes
            for entity in entity_data:
                node_id = str(uuid.uuid4())
                node = GraphNode(
                    id=node_id,
                    text=entity["text"],
                    type=entity["type"],
                    metadata={"source_doc": doc.metadata.get("id")}
                )
                graph.nodes.append(node)
                entity_id_map[entity["text"].lower()] = node_id
            
            # Extract relationships
            relationship_data = json.loads(self.llm.call(
                f"Given these entities: {[e['text'] for e in entity_data]}, extract relationships from: {doc.page_content}"
            ))
            
            # Create edges
            for rel in relationship_data:
                subject_text = rel["subject"].lower()
                object_text = rel["object"].lower()
                
                if subject_text in entity_id_map and object_text in entity_id_map:
                    edge = GraphEdge(
                        source=entity_id_map[subject_text],
                        target=entity_id_map[object_text],
                        type=rel["relationship"],
                        source_document_id=doc.metadata.get("id")
                    )
                    graph.edges.append(edge)
            
        logger.info(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph


async def main():
    """
    Demonstrates how to use a simplified version of the GraphCreationService.
    """
    # Create a mock LLM implementation
    llm = MockLLM()

    # Configure the service
    config = {
        "output_format": "json",
        "graph_database_config": {
            "output_file": "graph_output.json"
        }
    }
    
    # Create an instance of the mock GraphCreationService
    graph_service = MockGraphCreationService(config)

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
        entity_types=["PERSON", "GPE", "ORG", "ROLE"],
        relationship_types=["HAS_ROLE", "BORN_IN", "SPOUSE_OF", "PARENT_OF", "REPRESENTS"]
    )

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
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig("knowledge_graph.png", dpi=300, bbox_inches="tight")
        print("\nGraph visualization saved as 'knowledge_graph.png'")
        
        # Save graph as JSON
        with open("graph_output.json", "w") as f:
            f.write(graph.json())
        print("Graph data saved as 'graph_output.json'")
        
    except ImportError:
        print("\nNote: Install networkx and matplotlib for graph visualization:")
        print("pip install networkx matplotlib")
        
        # Still save graph as JSON even without visualization
        with open("graph_output.json", "w") as f:
            f.write(graph.json())
        print("Graph data saved as 'graph_output.json'")

if __name__ == "__main__":
    asyncio.run(main())