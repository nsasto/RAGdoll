from typing import Type, Optional
from pydantic import BaseModel, ValidationError
import json
import re

def fix_json(s: str) -> str:
    """
    Attempts to fix common JSON errors like missing quotes,
    trailing commas, and unescaped characters. This is a simplified
    fixer and may not handle all cases.
    """
    s = re.sub(r'([{,]\s*)([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', s)  # Fix unquoted keys
    s = re.sub(r',\s*}', '}', s)  # Remove trailing commas
    s = re.sub(r',\s*]', ']', s)  # Remove trailing commas
    s = s.replace('\\', '\\\\')  # Escape backslashes
    s = re.sub(r'([^\\])"', r'\1\\"', s)  # Basic string escaping
    return s

def json_parse(response: str, pydantic_object: Optional[Type[BaseModel]] = dict, max_retries: int = 3) -> Optional[BaseModel]:
    """
    Robustly parses a string response into a Pydantic object or a dictionary if no Pydantic model is provided.
    Handles common LLM output issues.

    Args:
        response: The string response to parse (typically from an LLM).
        pydantic_object: The Pydantic model to parse the response into, or dict if not specified.
        max_retries: Maximum number of attempts to parse.

    Returns:
        The parsed Pydantic object or dictionary, or None if parsing fails after multiple retries.
    """
    for attempt in range(max_retries):
        try:
            # 1. Attempt direct JSON parsing first. This is the ideal case.
            try:
                json_response = json.loads(response)
                if pydantic_object == dict:
                    return json_response
                return pydantic_object.model_validate(json_response)
            except json.JSONDecodeError:
                pass  # If it's not valid JSON, move to the next attempt

            # 2. Clean up common LLM formatting issues:
            cleaned_response = response.strip()
            cleaned_response = re.sub(r'```(json)?\n?', '', cleaned_response)  # Remove ```json and ```
            cleaned_response = re.sub(r'```', '', cleaned_response)
            cleaned_response = re.sub(r'\n+', '\n', cleaned_response)  # Reduce multiple newlines
            cleaned_response = cleaned_response.strip()  # Remove leading/trailing whitespace

            # 3. Attempt to extract a JSON-like substring.
            if "{" in cleaned_response and "}" in cleaned_response:
                start_index = cleaned_response.find("{")
                end_index = cleaned_response.rfind("}") + 1
                json_like_substring = cleaned_response[start_index:end_index]
                try:
                    json_response = json.loads(json_like_substring)
                    if pydantic_object == dict:
                        return json_response
                    return pydantic_object.model_validate(json_response)
                except json.JSONDecodeError:
                    pass

            # 4. Handle potential issues with extra newlines or incomplete JSON
            try:
                json_response = json.loads(cleaned_response)
                if pydantic_object == dict:
                    return json_response
                return pydantic_object.model_validate(json_response)
            except json.JSONDecodeError:
                pass

            # 5. Use the fix_json function to fix common JSON errors.
            fixed_json_response = fix_json(cleaned_response)
            try:
                json_response = json.loads(fixed_json_response)
                if pydantic_object == dict:
                    return json_response
                return pydantic_object.model_validate(json_response)
            except json.JSONDecodeError:
                pass

            # 6. Try parsing the original response after fixes.
            try:
                json_response = json.loads(cleaned_response)
                if pydantic_object == dict:
                    return json_response
                return pydantic_object.model_validate(json_response)
            except ValidationError as e:
                if attempt < max_retries - 1:
                    print(f"Parsing failed on attempt {attempt + 1}: {e}. Retrying...")
                else:
                    print(f"Parsing failed after {max_retries} attempts: {e}")
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Parsing failed on attempt {attempt + 1}: {e}. Retrying...")
                else:
                    print(f"Parsing failed after {max_retries} attempts: {e}")
                    return None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"An unexpected error occurred on attempt {attempt + 1}: {e}. Retrying...")
            else:
                print(f"An unexpected error occurred after {max_retries} attempts: {e}")
                return None

    return None  # Return None if all attempts fail

def visualize_graph(graph, output_image_path="knowledge_graph.png", output_json_path="graph_output.json"):
    """
    Visualizes the knowledge graph and saves it as an image and JSON file.

    Args:
        graph: The graph object to visualize.
        output_image_path: Path to save the graph visualization image.
        output_json_path: Path to save the graph data as JSON.
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nInstall networkx and matplotlib to visualize the graph: pip install networkx matplotlib")
        return

    try:
        # Create a directed graph
        G = nx.DiGraph()
        for node in graph.nodes:
            G.add_node(node.id, label=node.name, type=node.type)
        for edge in graph.edges:
            G.add_edge(edge.source, edge.target, label=edge.type)

        # Plot the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=2000, alpha=0.8, node_color="lightblue")
        nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20)
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes}, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(s, t): G.edges[s, t]["label"] for s, t in G.edges}, font_size=8)
        plt.axis("off")
        plt.title("Knowledge Graph Visualization", size=15)
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"\nGraph visualization saved as '{output_image_path}'")

        # Save graph data as JSON
        with open(output_json_path, "w") as f:
            try:
                f.write(graph.model_dump_json(indent=2))
            except AttributeError:
                f.write(graph.json(indent=2))
        print(f"Graph data saved as '{output_json_path}'")

    except ImportError:
        print("\nInstall networkx and matplotlib to visualize the graph: pip install networkx matplotlib")
