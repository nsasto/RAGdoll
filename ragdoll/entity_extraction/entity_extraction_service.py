import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Literal, Any, Union

import networkx as nx
import spacy
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class GraphNode(BaseModel):
    """Represents a node in the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    text: str
    metadata: Dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Represents an edge/relationship in the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    target: str
    type: str
    metadata: Dict = Field(default_factory=dict)
    source_document_id: Optional[str] = None  # Link back to the originating document


class Graph(BaseModel):
    """Represents a complete knowledge graph with nodes and edges."""
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


@dataclass
class GraphCreationService:
    """
    Service for creating knowledge graphs from documents using a combination of
    spaCy NER and LLM-based entity and relationship extraction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the GraphCreationService with configuration options.
        
        Args:
            config: Dictionary containing configuration parameters.
        """
        self.config = config or {}
        self.llm = None  # Will be set in the extract method
        
        # Load spaCy model
        spacy_model = self.config.get("spacy_model", "en_core_web_sm")
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model {spacy_model}: {e}")
            logger.info("Attempting to download the model...")
            try:
                spacy.cli.download(spacy_model)
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Successfully downloaded and loaded model: {spacy_model}")
            except Exception as download_error:
                logger.error(f"Failed to download model: {download_error}")
                raise

    async def _chunk_document(self, document: Document) -> List[Document]:
        """
        Splits a Langchain Document into smaller documents based on the configured strategy.
        
        Args:
            document: The document to split.
            
        Returns:
            A list of document chunks.
        """
        try:
            strategy = self.config.get("chunking_strategy", "none")
            
            if strategy == "none":
                logger.debug(f"No chunking applied for document {document.metadata.get('id', 'unknown')}")
                return [document]
            
            elif strategy == "fixed":
                chunk_size = self.config.get("chunk_size", 500)
                chunk_overlap = self.config.get("chunk_overlap", 50)
                logger.debug(f"Applying fixed chunking with size={chunk_size}, overlap={chunk_overlap}")
                
                text_splitter = CharacterTextSplitter(
                    separator="\n\n",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )
                return text_splitter.split_documents([document])
            
            elif strategy == "semantic":
                splitter_type = self.config.get("splitter_type", "markdown")
                chunk_size = self.config.get("chunk_size", 1000)
                chunk_overlap = self.config.get("chunk_overlap", 0)
                logger.debug(f"Applying semantic chunking with type={splitter_type}, size={chunk_size}, overlap={chunk_overlap}")
                
                if splitter_type == "markdown":
                    text_splitter = MarkdownTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                elif splitter_type == "python":
                    text_splitter = PythonCodeTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                return text_splitter.split_documents([document])
            else:
                raise ValueError(f"Invalid chunking strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Error during document chunking: {e}")
            return [document]  # Fall back to the entire document

    async def _resolve_coreferences(self, text: str) -> str:
        """
        Resolves coreferences in the input text based on the configured method.
        
        Args:
            text: The text to process.
            
        Returns:
            Text with resolved coreferences.
        """
        try:
            method = self.config.get("coreference_resolution_method", "none")
            
            if method == "none":
                return text
                
            elif method == "rule_based":
                logger.debug("Applying rule-based coreference resolution")
                # Simple rule-based coreference resolution
                resolved_text = re.sub(r'\b(he|she|it|they|him|her|them)\b', '[ENTITY]', text, flags=re.IGNORECASE)
                return resolved_text
                
            elif method == "llm":
                logger.debug("Applying LLM-based coreference resolution")
                if not self.llm:
                    logger.warning("LLM not available for coreference resolution")
                    return text
                    
                prompt_template = self.config.get("llm_prompt_templates", {}).get(
                    "coreference_resolution",
                    "Resolve coreferences in the following text by replacing pronouns with their referents. Text: {text}"
                )
                prompt = prompt_template.format(text=text)
                response = await self._call_llm(prompt)
                return response
                
            else:
                logger.warning(f"Unknown coreference resolution method: {method}")
                return text
                
        except Exception as e:
            logger.error(f"Error during coreference resolution: {e}")
            return text  # Return original text on error

    async def _extract_entities_ner(self, text: str) -> List[Dict]:
        """
        Extracts entities from text using spaCy's NER.
        
        Args:
            text: The text to extract entities from.
            
        Returns:
            A list of extracted entities with their types and positions.
        """
        try:
            logger.debug("Extracting entities using spaCy NER")
            doc = self.nlp(text)
            entities = [
                {
                    "text": ent.text,
                    "type": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char
                }
                for ent in doc.ents
            ]
            
            # Filter by entity types if specified
            entity_types = self.config.get("entity_types", [])
            if entity_types:
                entities = [
                    ent for ent in entities 
                    if ent["type"] in entity_types or ent["type"].upper() in entity_types
                ]
                
            logger.debug(f"NER extracted {len(entities)} entities")
            return entities
        except Exception as e:
            logger.error(f"Error during NER entity extraction: {e}")
            return []

    async def _extract_entities_llm(self, text: str, prompt_template: str = None) -> List[Dict]:
        """
        Extracts entities from text using the LLM.
        
        Args:
            text: The text to extract entities from.
            prompt_template: Optional custom prompt template. If None, uses configured template.
            
        Returns:
            A list of entities extracted by the LLM.
        """
        if not self.llm:
            logger.warning("LLM not available for entity extraction")
            return []
            
        try:
            logger.debug("Extracting entities using LLM")
            
            entity_types = self.config.get("entity_types", [])
            entity_types_str = ", ".join(entity_types) if entity_types else "any important entities"
            
            # Use the provided prompt if given, otherwise use configured template
            template = prompt_template or self.config.get("llm_prompt_templates", {}).get(
                "entity_extraction_llm",
                "Extract {entity_types} from the following text. Return as a JSON array of objects with 'text' and 'type' properties. Text: {text}"
            )
            
            prompt = template.format(text=text, entity_types=entity_types_str)
            response = await self._call_llm(prompt)
            
            # Parse LLM response
            try:
                import json
                entities = json.loads(response)
                if not isinstance(entities, list):
                    logger.warning(f"LLM entity extraction returned non-list: {type(entities)}")
                    entities = []
                    
                # Ensure each entity has required fields
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    if "text" not in entity:
                        entity["text"] = ""
                    if "type" not in entity:
                        entity["type"] = "UNKNOWN"
                        
                logger.debug(f"LLM extracted {len(entities)} entities")
                return entities
                
            except json.JSONDecodeError:
                # If JSON parsing fails, try simple extraction
                logger.warning("Failed to parse LLM entity extraction response as JSON")
                entity_matches = re.findall(r'([A-Z][a-zA-Z\s]+):\s*([A-Za-z\s]+)', response)
                entities = [{"text": text, "type": ent_type} for ent_type, text in entity_matches]
                
                # If that also fails, split by commas or lines
                if not entities:
                    lines = [line.strip() for line in response.split('\n') if line.strip()]
                    if not lines or len(lines) == 1:
                        items = [item.strip() for item in response.split(',') if item.strip()]
                        entities = [{"text": item, "type": "UNKNOWN"} for item in items]
                    else:
                        entities = [{"text": line, "type": "UNKNOWN"} for line in lines]
                
                logger.debug(f"LLM extracted {len(entities)} entities (fallback parsing)")
                return entities
                
        except Exception as e:
            logger.error(f"Error during LLM entity extraction: {e}")
            return []

    async def _extract_relationships_llm(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        Extracts relationships between entities using the LLM.
        
        Args:
            text: The source text.
            entities: The entities extracted from the text.
            
        Returns:
            A list of extracted relationships.
        """
        if not self.llm:
            logger.warning("LLM not available for relationship extraction")
            return []
            
        if not entities:
            logger.debug("No entities provided for relationship extraction")
            return []
            
        try:
            logger.debug(f"Extracting relationships for {len(entities)} entities using LLM")
            
            # Format entities for prompt
            entity_str = "\n".join([f"- {entity['text']} (Type: {entity['type']})" for entity in entities])
            
            relationship_types = self.config.get("relationship_types", [])
            relationship_types_str = ", ".join(relationship_types) if relationship_types else "any relevant relationships"
            
            prompt_template = self.config.get("llm_prompt_templates", {}).get(
                "extract_relationships",
                """
                Extract {relationship_types} from the text, given these entities:
                {entities}
                
                Text: {text}
                
                Return the relationships as a JSON array with 'subject', 'relationship', and 'object' properties.
                """
            )
            
            prompt = prompt_template.format(
                text=text,
                entities=entity_str,
                relationship_types=relationship_types_str
            )
            
            response = await self._call_llm(prompt)
            
            # Parse LLM response
            try:
                import json
                relationships = json.loads(response)
                if not isinstance(relationships, list):
                    logger.warning(f"LLM relationship extraction returned non-list: {type(relationships)}")
                    relationships = []
                    
                # Ensure each relationship has required fields
                valid_relationships = []
                for rel in relationships:
                    if not isinstance(rel, dict):
                        continue
                    if "subject" in rel and "object" in rel and "relationship" in rel:
                        valid_relationships.append(rel)
                    else:
                        logger.debug(f"Skipping invalid relationship: {rel}")
                
                logger.debug(f"LLM extracted {len(valid_relationships)} relationships")
                return valid_relationships
                
            except json.JSONDecodeError:
                # Fallback parsing for non-JSON responses
                logger.warning("Failed to parse LLM relationship extraction response as JSON")
                relationships = []
                
                # Try to extract "Subject: X, Relationship: Y, Object: Z" patterns
                pattern = r"Subject:?\s*(.*?),\s*Relationship:?\s*(.*?),\s*Object:?\s*(.*?)(?:;|$|\n)"
                matches = re.findall(pattern, response, re.IGNORECASE)
                
                for match in matches:
                    subject = match[0].strip()
                    relationship = match[1].strip()
                    obj = match[2].strip()
                    if subject and relationship and obj:
                        relationships.append({
                            "subject": subject,
                            "relationship": relationship,
                            "object": obj
                        })
                
                logger.debug(f"LLM extracted {len(relationships)} relationships (fallback parsing)")
                return relationships
                
        except Exception as e:
            logger.error(f"Error during relationship extraction: {e}")
            return []

    async def _glean_graph(self, initial_graph: Graph, text: str, history: List[Dict]) -> Graph:
        """
        Iteratively refines the graph by querying the LLM.
        
        Args:
            initial_graph: The initial graph to refine.
            text: The source text.
            history: Conversation history for the LLM.
            
        Returns:
            The refined graph.
        """
        if not self.config.get("gleaning_enabled", False):
            logger.debug("Gleaning is disabled, skipping")
            return initial_graph
            
        if not self.llm:
            logger.warning("LLM not available for gleaning")
            return initial_graph
            
        try:
            logger.debug("Starting graph gleaning process")
            current_graph = initial_graph
            max_gleaning_steps = self.config.get("max_gleaning_steps", 3)
            
            for step in range(max_gleaning_steps):
                logger.debug(f"Gleaning step {step+1}/{max_gleaning_steps}")
                
                # Format the current graph
                graph_summary = f"Nodes: {len(current_graph.nodes)}, Edges: {len(current_graph.edges)}"
                
                # List of node texts and types
                nodes_str = "\n".join([
                    f"- {node.text} (Type: {node.type}, ID: {node.id})" 
                    for node in current_graph.nodes
                ])
                
                # List of relationships
                edges_str = "\n".join([
                    f"- {self._get_node_text(current_graph, edge.source)} --[{edge.type}]--> {self._get_node_text(current_graph, edge.target)}"
                    for edge in current_graph.edges
                ])
                
                prompt_template = self.config.get("llm_prompt_templates", {}).get(
                    "entity_relationship_continue_extraction",
                    """
                    Continue extracting entities and relationships from the text. 
                    
                    Current graph summary: {graph_summary}
                    
                    Nodes:
                    {nodes}
                    
                    Relationships:
                    {edges}
                    
                    Original text:
                    {text}
                    
                    Return any additional entities and relationships you can identify as a JSON object with 'nodes' and 'edges' arrays.
                    Each node should have 'id', 'type', and 'text' properties.
                    Each edge should have 'source', 'target', and 'type' properties.
                    """
                )
                
                prompt = prompt_template.format(
                    graph_summary=graph_summary,
                    nodes=nodes_str,
                    edges=edges_str,
                    text=text
                )
                
                response = await self._call_llm(prompt)
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": response})
                
                # Parse response into a Graph object
                try:
                    import json
                    result = json.loads(response)
                    
                    # Extract nodes
                    new_nodes = []
                    if "nodes" in result and isinstance(result["nodes"], list):
                        for node_data in result["nodes"]:
                            if isinstance(node_data, dict) and "text" in node_data and "type" in node_data:
                                node_id = node_data.get("id", str(uuid.uuid4()))
                                new_node = GraphNode(
                                    id=node_id,
                                    text=node_data["text"],
                                    type=node_data["type"]
                                )
                                new_nodes.append(new_node)
                    
                    # Extract edges
                    new_edges = []
                    if "edges" in result and isinstance(result["edges"], list):
                        for edge_data in result["edges"]:
                            if isinstance(edge_data, dict) and "source" in edge_data and "target" in edge_data and "type" in edge_data:
                                edge_id = edge_data.get("id", str(uuid.uuid4()))
                                new_edge = GraphEdge(
                                    id=edge_id,
                                    source=edge_data["source"],
                                    target=edge_data["target"],
                                    type=edge_data["type"]
                                )
                                new_edges.append(new_edge)
                    
                    logger.debug(f"Gleaning extracted {len(new_nodes)} new nodes and {len(new_edges)} new edges")
                    
                    # Add new nodes and edges to current graph
                    current_graph.nodes.extend(new_nodes)
                    current_graph.edges.extend(new_edges)
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error parsing gleaning result: {e}")
                
                # Check if we should continue gleaning
                if step < max_gleaning_steps - 1:
                    prompt_template = self.config.get("llm_prompt_templates", {}).get(
                        "entity_relationship_gleaning_done_extraction",
                        "Are you done extracting entities and relationships? Respond with 'done' or 'continue'."
                    )
                    
                    prompt = prompt_template
                    response = await self._call_llm(prompt)
                    history.append({"role": "user", "content": prompt})
                    history.append({"role": "assistant", "content": response})
                    
                    if "done" in response.lower():
                        logger.debug(f"Stopping gleaning early at step {step+1}/{max_gleaning_steps}")
                        break
            
            return current_graph
            
        except Exception as e:
            logger.error(f"Error during graph gleaning: {e}")
            return initial_graph

    def _get_node_text(self, graph: Graph, node_id: str) -> str:
        """Helper method to get node text from a node ID."""
        for node in graph.nodes:
            if node.id == node_id:
                return node.text
        return f"[Unknown Node: {node_id}]"

    async def _link_entities(self, graph: Graph) -> Graph:
        """
        Links entities in the graph based on the configured method.
        
        Args:
            graph: The graph to process.
            
        Returns:
            The graph with linked entities.
        """
        if not self.config.get("entity_linking_enabled", False):
            logger.debug("Entity linking is disabled, skipping")
            return graph
            
        try:
            logger.debug("Starting entity linking process")
            method = self.config.get("entity_linking_method", "string_similarity")
            
            if method == "string_similarity":
                threshold = self.config.get("entity_linking_threshold", 0.8)
                logger.debug(f"Applying string similarity linking with threshold {threshold}")
                
                # Create node ID mapping to track which nodes get merged
                node_id_mapping = {node.id: node.id for node in graph.nodes}
                nodes_to_remove = set()
                
                # Find similar nodes
                for i, node1 in enumerate(graph.nodes):
                    if node1.id in nodes_to_remove:
                        continue
                        
                    for j in range(i + 1, len(graph.nodes)):
                        node2 = graph.nodes[j]
                        if node2.id in nodes_to_remove:
                            continue
                            
                        # Compare texts for similarity
                        similarity = SequenceMatcher(None, node1.text.lower(), node2.text.lower()).ratio()
                        
                        # Also check if one is contained in the other
                        text1_lower = node1.text.lower()
                        text2_lower = node2.text.lower()
                        contained = text1_lower in text2_lower or text2_lower in text1_lower
                        
                        # Link if similar or contained
                        if similarity > threshold or contained:
                            logger.debug(f"Linking entities: '{node1.text}' and '{node2.text}' (similarity: {similarity:.2f})")
                            
                            # Update the node ID mapping
                            node_id_mapping[node2.id] = node1.id
                            nodes_to_remove.add(node2.id)
                
                # Update edges to reflect merged nodes
                for edge in graph.edges:
                    edge.source = node_id_mapping.get(edge.source, edge.source)
                    edge.target = node_id_mapping.get(edge.target, edge.target)
                
                # Remove merged nodes
                graph.nodes = [node for node in graph.nodes if node.id not in nodes_to_remove]
                
            elif method == "knowledge_graph":
                logger.warning("Knowledge graph linking is not implemented yet")
                
            else:
                logger.warning(f"Unknown entity linking method: {method}")
                
            return graph
            
        except Exception as e:
            logger.error(f"Error during entity linking: {e}")
            return graph

    async def _postprocess_graph(self, graph: Graph) -> Graph:
        """
        Applies post-processing steps to the graph.
        
        Args:
            graph: The graph to process.
            
        Returns:
            The processed graph.
        """
        steps = self.config.get("postprocessing_steps", [])
        if not steps:
            logger.debug("No postprocessing steps configured, skipping")
            return graph
            
        try:
            logger.debug(f"Applying postprocessing steps: {steps}")
            
            if "merge_similar_entities" in steps:
                graph = await self._merge_similar_entities(graph)
                
            if "normalize_relations" in steps:
                graph = await self._normalize_relations(graph)
                
            if "remove_low_confidence" in steps:
                graph = await self._remove_low_confidence(graph)
                
            return graph
            
        except Exception as e:
            logger.error(f"Error during graph postprocessing: {e}")
            return graph

    async def _merge_similar_entities(self, graph: Graph) -> Graph:
        """
        Merges entities with similar text.
        
        Args:
            graph: The graph to process.
            
        Returns:
            The graph with merged entities.
        """
        try:
            logger.debug("Merging similar entities")
            
            # Track node IDs to merge
            node_id_mapping = {}
            nodes_to_remove = set()
            
            # Find nodes with identical text (case insensitive)
            text_to_id = {}
            for node in graph.nodes:
                text_lower = node.text.lower()
                if text_lower in text_to_id:
                    # We've seen this text before, mark for merging
                    primary_id = text_to_id[text_lower]
                    node_id_mapping[node.id] = primary_id
                    nodes_to_remove.add(node.id)
                else:
                    # First time seeing this text
                    text_to_id[text_lower] = node.id
            
            # Update edges to use primary node IDs
            for edge in graph.edges:
                if edge.source in node_id_mapping:
                    edge.source = node_id_mapping[edge.source]
                if edge.target in node_id_mapping:
                    edge.target = node_id_mapping[edge.target]
            
            # Remove merged nodes
            graph.nodes = [node for node in graph.nodes if node.id not in nodes_to_remove]
            
            logger.debug(f"Merged {len(nodes_to_remove)} similar entities")
            return graph
            
        except Exception as e:
            logger.error(f"Error during entity merging: {e}")
            return graph

    async def _normalize_relations(self, graph: Graph) -> Graph:
        """
        Normalizes relationship types.
        
        Args:
            graph: The graph to process.
            
        Returns:
            The graph with normalized relationships.
        """
        try:
            logger.debug("Normalizing relationship types")
            
            # Define relationship type mapping
            # This could be moved to config
            mapping = {
                "works for": "works_for",
                "is a": "is_a",
                "is an": "is_a",
                "located in": "located_in",
                "located at": "located_in",
                "born in": "born_in",
                "lives in": "lives_in",
                "married to": "married_to",
                "spouse of": "spouse_of",
                "parent of": "parent_of",
                "child of": "child_of",
                "works with": "works_with"
            }
            
            # Normalize edge types
            normalized_count = 0
            for edge in graph.edges:
                edge_type_lower = edge.type.lower()
                if edge_type_lower in mapping:
                    edge.type = mapping[edge_type_lower]
                    normalized_count += 1
            
            logger.debug(f"Normalized {normalized_count} relationship types")
            return graph
            
        except Exception as e:
            logger.error(f"Error during relationship normalization: {e}")
            return graph

    async def _remove_low_confidence(self, graph: Graph) -> Graph:
        """
        Removes entities or relations with low confidence.
        
        Args:
            graph: The graph to process.
            
        Returns:
            The graph with low-confidence elements removed.
        """
        # This is a placeholder - in a real implementation, you would need to track
        # confidence scores during extraction
        logger.debug("Low confidence removal not implemented yet")
        return graph

    async def _call_llm(self, prompt: str) -> str:
        """
        Calls the LLM and handles errors.
        
        Args:
            prompt: The prompt to send to the LLM.
            
        Returns:
            The LLM's response.
        """
        try:
            if not self.llm:
                raise ValueError("LLM not initialized")
                
            # For async LLMs
            if hasattr(self.llm, "agenerate") and callable(getattr(self.llm, "agenerate")):
                generation = await self.llm.agenerate([prompt])
                response = generation.generations[0][0].text
                return response
                
            # For sync LLMs
            elif hasattr(self.llm, "generate") and callable(getattr(self.llm, "generate")):
                generation = self.llm.generate([prompt])
                response = generation.generations[0][0].text
                return response
                
            # For BaseLLM with __call__
            elif hasattr(self.llm, "__call__"):
                response = self.llm(prompt)
                return response
                
            # Fallback to basic .call() interface
            else:
                response = self.llm.call(prompt)
                return response
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return ""  # Return empty string on error

    async def _store_graph(self, graph: Graph):
        """
        Stores the graph in a configured graph database or format.
        
        Args:
            graph: The graph to store.
        """
        db_config = self.config.get("graph_database_config")
        if not db_config:
            logger.debug("No graph database configuration, skipping storage")
            return
            
        try:
            output_format = self.config.get("output_format", "custom_graph_object")
            logger.info(f"Storing graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges in format: {output_format}")
            
            if output_format == "custom_graph_object":
                # This is a placeholder for custom storage logic
                logger.debug(f"Graph object ready for custom storage")
                
            elif output_format == "json":
                import json
                # Fix: use model_dump_json instead of json() method with indent parameter
                graph_json = graph.model_dump_json(indent=2)  # For newer pydantic versions
                # For older pydantic versions, use: graph.json(indent=2)
                logger.debug(f"Graph JSON: {graph_json}")
                
                # Optionally save to a file
                if "output_file" in db_config:
                    with open(db_config["output_file"], "w") as f:
                        f.write(graph_json)
                    logger.info(f"Graph saved to {db_config['output_file']}")
                    
            elif output_format == "networkx":
                # Convert to networkx graph
                g = nx.DiGraph()
                
                # Add nodes
                for node in graph.nodes:
                    g.add_node(node.id, type=node.type, text=node.text, **node.metadata)
                    
                # Add edges
                for edge in graph.edges:
                    g.add_edge(
                        edge.source, 
                        edge.target, 
                        type=edge.type, 
                        source_document_id=edge.source_document_id,
                        **edge.metadata
                    )
                
                logger.debug(f"NetworkX graph created: {g}")
                
                # Optionally save to a file
                if "output_file" in db_config:
                    import pickle
                    with open(db_config["output_file"], "wb") as f:
                        pickle.dump(g, f)
                    logger.info(f"NetworkX graph saved to {db_config['output_file']}")
                    
            elif output_format == "neo4j":
                # Example Neo4j storage using py2neo
                if "uri" not in db_config or "user" not in db_config or "password" not in db_config:
                    logger.error("Missing Neo4j connection parameters")
                    return
                    
                try:
                    from py2neo import Graph as Neo4jGraph, Node, Relationship
                    
                    # Connect to Neo4j
                    neo4j_graph = Neo4jGraph(
                        db_config["uri"], 
                        auth=(db_config["user"], db_config["password"])
                    )
                    
                    # Create transaction
                    tx = neo4j_graph.begin()
                    
                    # Create nodes
                    neo4j_nodes = {}
                    for node in graph.nodes:
                        neo4j_node = Node(
                            node.type,
                            id=node.id,
                            text=node.text,
                            **node.metadata
                        )
                        neo4j_nodes[node.id] = neo4j_node
                        tx.create(neo4j_node)
                    
                    # Create relationships
                    for edge in graph.edges:
                        if edge.source in neo4j_nodes and edge.target in neo4j_nodes:
                            source_node = neo4j_nodes[edge.source]
                            target_node = neo4j_nodes[edge.target]
                            relationship = Relationship(
                                source_node, 
                                edge.type, 
                                target_node,
                                id=edge.id,
                                source_document_id=edge.source_document_id,
                                **edge.metadata
                            )
                            tx.create(relationship)
                    
                    # Commit transaction
                    tx.commit()
                    logger.info(f"Graph stored in Neo4j at {db_config['uri']}")
                    
                except ImportError:
                    logger.error("py2neo not installed. Install with: pip install py2neo")
                except Exception as neo4j_error:
                    logger.error(f"Neo4j storage error: {neo4j_error}")
            else:
                logger.warning(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error storing graph: {e}")

    async def extract(
        self, 
        documents: List[Document], 
        llm: BaseLLM, 
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ) -> Graph:
        """
        Main method to extract a graph from a list of Langchain Documents.
        
        Args:
            documents: List of Langchain Document objects
            llm: Language model to use for extraction
            entity_types: Optional list of entity types to extract
            relationship_types: Optional list of relationship types to extract
            
        Returns:
            The extracted knowledge graph
        """
        # Store the LLM instance
        self.llm = llm
        
        # Update config with passed arguments
        if entity_types:
            self.config["entity_types"] = entity_types
        if relationship_types:
            self.config["relationship_types"] = relationship_types
            
        # Initialize final graph
        final_graph = Graph()
        
        try:
            logger.info(f"Starting graph extraction for {len(documents)} documents")
            
            # Process each document
            for doc_idx, doc in enumerate(documents):
                logger.info(f"Processing document {doc_idx+1}/{len(documents)}: {doc.metadata.get('source', 'unknown')}")
                
                # Create a graph for this document
                document_graph = Graph()
                
                try:
                    # Split document into chunks
                    chunks = await self._chunk_document(doc)
                    logger.debug(f"Document split into {len(chunks)} chunks")
                    
                    # Process each chunk
                    for chunk_idx, chunk in enumerate(chunks):
                        logger.debug(f"Processing chunk {chunk_idx+1}/{len(chunks)}")
                        text = chunk.page_content
                        
                        # Resolve coreferences if configured
                        if self.config.get("coreference_resolution_method") != "none":
                            text = await self._resolve_coreferences(text)
                            
                        # Extract entities based on configured methods
                        chunk_entities = []
                        if "ner" in self.config.get("entity_extraction_methods", ["ner"]):
                            ner_entities = await self._extract_entities_ner(text)
                            chunk_entities.extend(ner_entities)
                            
                        if "llm" in self.config.get("entity_extraction_methods", ["ner"]):
                            llm_entities = await self._extract_entities_llm(text)
                            chunk_entities.extend(llm_entities)
                            
                        # Extract relationships
                        chunk_relationships = []
                        if chunk_entities and self.config.get("relationship_extraction_method", "llm") == "llm":
                            chunk_relationships = await self._extract_relationships_llm(text, chunk_entities)
                            
                        # Convert entities to GraphNodes
                        for ent in chunk_entities:
                            node_id = str(uuid.uuid4())
                            node = GraphNode(
                                id=node_id,
                                text=ent["text"],
                                type=ent["type"],
                                metadata={
                                    "start_char": ent.get("start_char"),
                                    "end_char": ent.get("end_char"),
                                    "source_document_id": doc.metadata.get("source"),
                                    "chunk_idx": chunk_idx
                                }
                            )
                            document_graph.nodes.append(node)
                            
                        # Create a mapping from entity text to node ID
                        entity_text_to_id = {}
                        for node in document_graph.nodes:
                            entity_text_to_id[node.text.lower()] = node.id
                            
                        # Convert relationships to GraphEdges
                        for rel in chunk_relationships:
                            subject_text = rel.get("subject", "").lower()
                            object_text = rel.get("object", "").lower()
                            rel_type = rel.get("relationship")
                            
                            # Try to find subject and object nodes
                            subject_id = entity_text_to_id.get(subject_text)
                            object_id = entity_text_to_id.get(object_text)
                            
                            # If not found by exact match, try substring matching
                            if not subject_id:
                                for text, node_id in entity_text_to_id.items():
                                    if subject_text in text or text in subject_text:
                                        subject_id = node_id
                                        break
                                        
                            if not object_id:
                                for text, node_id in entity_text_to_id.items():
                                    if object_text in text or text in object_text:
                                        object_id = node_id
                                        break
                            
                            # Create edge if both nodes were found
                            if subject_id and object_id and rel_type:
                                edge = GraphEdge(
                                    source=subject_id,
                                    target=object_id,
                                    type=rel_type,
                                    source_document_id=doc.metadata.get("source"),
                                    metadata={
                                        "chunk_idx": chunk_idx
                                    }
                                )
                                document_graph.edges.append(edge)
                    
                    # Gleaning to refine the document graph
                    if self.config.get("gleaning_enabled", False):
                        # Add the full document text for context
                        history = []
                        document_graph = await self._glean_graph(document_graph, doc.page_content, history)
                    
                    # Entity linking within the document
                    if self.config.get("entity_linking_enabled", False):
                        document_graph = await self._link_entities(document_graph)
                        
                    # Merge document graph into the final graph
                    self._merge_document_into_final_graph(document_graph, final_graph, doc.metadata.get("source"))
                    
                except Exception as doc_error:
                    logger.error(f"Error processing document {doc_idx+1}: {doc_error}")
                    continue
            
            # Apply post-processing to the final graph
            if self.config.get("postprocessing_steps"):
                final_graph = await self._postprocess_graph(final_graph)
                
            # Store the graph if configured
            await self._store_graph(final_graph)
            
            logger.info(f"Graph extraction complete: {len(final_graph.nodes)} nodes, {len(final_graph.edges)} edges")
            return final_graph
            
        except Exception as e:
            logger.error(f"Error during graph extraction: {e}")
            return final_graph

    def _merge_document_into_final_graph(self, document_graph: Graph, final_graph: Graph, doc_id: Optional[str]):
        """
        Merges a document graph into the final graph, handling duplicates.
        
        Args:
            document_graph: The graph extracted from a document
            final_graph: The final graph to merge into
            doc_id: ID of the source document
        """
        try:
            logger.debug(f"Merging document graph ({len(document_graph.nodes)} nodes, {len(document_graph.edges)} edges) into final graph ({len(final_graph.nodes)} nodes, {len(final_graph.edges)} edges)")
            
            # Create mapping from text to existing node IDs
            existing_text_to_id = {node.text.lower(): node.id for node in final_graph.nodes}
            
            # Mapping from document node IDs to final graph node IDs
            node_id_mapping = {}
            
            # Add nodes to final graph, handling duplicates
            for doc_node in document_graph.nodes:
                node_text_lower = doc_node.text.lower()
                
                # Check if this node text already exists in final graph
                if node_text_lower in existing_text_to_id:
                    # Map this node to the existing one
                    final_node_id = existing_text_to_id[node_text_lower]
                    node_id_mapping[doc_node.id] = final_node_id
                    
                    # Optionally merge metadata or update existing node
                    # For simplicity, we just link the node here
                else:
                    # Add as new node to final graph
                    final_graph.nodes.append(doc_node)
                    existing_text_to_id[node_text_lower] = doc_node.id
                    node_id_mapping[doc_node.id] = doc_node.id  # Identity mapping
            
            # Add edges to final graph
            for doc_edge in document_graph.edges:
                # Get mapped node IDs
                source_id = node_id_mapping.get(doc_edge.source)
                target_id = node_id_mapping.get(doc_edge.target)
                
                if source_id and target_id:
                    # Check for duplicate edges
                    is_duplicate = any(
                        edge.source == source_id and 
                        edge.target == target_id and 
                        edge.type == doc_edge.type
                        for edge in final_graph.edges
                    )
                    
                    if not is_duplicate:
                        # Create a new edge using the mapped node IDs
                        edge = GraphEdge(
                            source=source_id,
                            target=target_id,
                            type=doc_edge.type,
                            source_document_id=doc_edge.source_document_id or doc_id,
                            metadata=doc_edge.metadata
                        )
                        final_graph.edges.append(edge)
                        
            logger.debug(f"Final graph now has {len(final_graph.nodes)} nodes and {len(final_graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error merging document graph: {e}")

