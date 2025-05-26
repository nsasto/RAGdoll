import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Literal, Any, Union, Callable

import networkx as nx
import spacy

from langchain.output_parsers import PydanticOutputParser

from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel, Field
from ragdoll.chunkers.chunker import Chunker  # Import the Chunker class
from ragdoll.config.config_manager import ConfigManager
from ragdoll.llms import get_llm, call_llm
from ragdoll.utils import json_parse
from .models import Entity, Relationship, EntityList, RelationshipList, GraphNode, GraphEdge, Graph
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(module)s][%(levelname)s] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", 
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

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
        config_manager = ConfigManager()
        default_config = config_manager.entity_extraction_config.model_dump()  # Get default config as a dict
        default_config["prompts"] = config_manager.get_default_prompt_templates()  # Get default prompts
        
        # Override defaults with any provided config
        if config:
            merged_config = {**default_config, **config}
        else:
            merged_config = default_config

        self.config = merged_config
        self.llm = None  # Will be set in the extract method
        self.chunker = Chunker(config=self.config)  # Create a Chunker instance
        logger.debug(config_manager.print_graph_creation_pipeline(merged_config))

        
        # Load spaCy model
        spacy_model = self.config.get("spacy_model", "en_core_web_sm")
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.info(f"spaCy model '{spacy_model}' not found. Downloading...")
            try:
                spacy.cli.download(spacy_model)
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Downloaded and loaded spaCy model: {spacy_model}")
            except Exception as download_error:
                logger.error(f"Error downloading spaCy model: {download_error}")
                raise

    async def _chunk_document(self, document: Document) -> List[Document]:
        """
        Splits a Langchain Document into smaller documents based on the configured strategy.
        
        Args:
            document: The document to split.
            
        Returns:
            A list of document chunks.
        """
        strategy = self.config.get("chunking_strategy", "none")
        chunk_size = self.config.get("chunk_size", None)
        chunk_overlap = self.config.get("chunk_overlap", None)
        splitter_type = self.config.get("splitter_type", None)
        
        return self.chunker.chunk_document(
            document=document,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=splitter_type
        )

    async def _resolve_coreferences(self, text: str, prompt_template: str = None) -> str:
        """
        Resolves coreferences in the input text based on the configured method.
        
        Args:
            text: The text to process.
            prompt_template: Optional custom prompt template. If None, uses configured template.
            
        Returns:
            Text with resolved coreferences.
        """
        try:
            # Always log the input text length
            logger.debug(f"Resolving coreferences in text of length: {len(text)}")
            
            method = self.config.get("coreference_resolution_method", "none")
            
            if method == "none":
                return text
                    
            elif method == "rule_based":
                logger.debug("Applying rule-based coreference resolution")

                # Define a mapping of pronouns to placeholder entities
                pronoun_map = {
                    r'\bhe\b': '[MALE_ENTITY]',
                    r'\bshe\b': '[FEMALE_ENTITY]',
                    r'\bit\b': '[NEUTRAL_ENTITY]',
                    r'\bthey\b': '[PLURAL_ENTITY]',
                    r'\bhim\b': '[MALE_ENTITY]',
                    r'\bher\b': '[FEMALE_ENTITY]',
                    r'\bthem\b': '[PLURAL_ENTITY]',
                }

                resolved_text = text
                for pattern, replacement in pronoun_map.items():
                    resolved_text = re.sub(pattern, replacement, resolved_text, flags=re.IGNORECASE)

                return resolved_text
                    
            elif method == "llm":
                logger.debug("Applying LLM-based coreference resolution")
                if not self.llm:
                    logger.warning("LLM not available for coreference resolution")
                    return text
                    
                # Use provided template or get from config
                prompt_template = prompt_template or self.config.get("prompts", {}).get("coreference_resolution")
                
                if not prompt_template:
                    logger.error("No 'coreference_resolution' prompt template specified in the config")
                    return text  # Return original text instead of failing
                    
                prompt = prompt_template.format(text=text)
                #logger.debug(f"Co-reference resolution prompt [first 100 chars]: {prompt[:100]}")
                
                # Call LLM with better error handling
                try:
                    response = await self._call_llm(prompt)
                    
                    logger.debug(f"LLM coreference response length: {len(response or '')}")
                    logger.debug(f"LLM Response: {response[:100]}...")  # Log the first 100 characters of the response
                    
                    # Check if the response looks like JSON (starts with [ or {)
                    if response and (response.strip().startswith('[') or response.strip().startswith('{')):
                        logger.warning("LLM returned JSON instead of plain text for coreference resolution")
                        try:
                            # Try to extract text content from JSON
                            import json
                            json_response = json.loads(response)
                            if isinstance(json_response, list) and len(json_response) > 0:
                                # If it's a list, see if we can extract text from an item
                                if isinstance(json_response[0], dict) and "text" in json_response[0]:
                                    return json_response[0]["text"]
                                elif isinstance(json_response[0], str):
                                    return json_response[0]
                            # If JSON parsing worked but we couldn't find text, fall back to original
                            logger.warning("Couldn't extract plain text from JSON response")
                            return text
                        except:
                            # If JSON parsing failed, continue with the response as is
                            pass
                    
                    # Critical fix: Never return empty text
                    if not response or len(response.strip()) == 0:
                        logger.warning("LLM returned empty response for coreference resolution")
                        return text  # Return original text if response is empty
                        
                    return response
                except Exception as llm_error:
                    logger.error(f"LLM call failed during coreference resolution: {llm_error}")
                    return text  # Return original text on LLM failure
                    
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
                    "name": ent.text,  # Use 'name' instead of 'text'
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
            
            # Use provided template or get from config
            prompt_template = prompt_template or self.config.get("prompts", {}).get("entity_extraction")
            
            if not prompt_template:
                raise ValueError("No 'entity_extraction' prompt template specified in the config")
            
            #logger.debug(f'Using prompt template for entity extraction with entity types {entity_types_str}')    
            prompt = prompt_template.format(text=text, entity_types=entity_types_str)
            response = await self._call_llm(prompt)
            logger.debug(f"LLM response: {response}")
            # Parse LLM response
            # Use PydanticOutputParser
            # parser = PydanticOutputParser(pydantic_object=EntityList)
            # ent = parser.parse(response)
            ent = json_parse(response, EntityList)
            result = [e.model_dump() for e in ent.entities]
#             
            logger.debug(f"LLM extracted {len(result)} entities")
            return result  # Convert to list of dicts                        

        except Exception as e:
            logger.error(f"Error during LLM entity extraction: {e}", exc_info=True)
            logger.error(f"LLM response that caused error: {response if 'response' in locals() else 'No response variable'}")
            return []

    async def _extract_relationships_llm(self, text: str, entities: List[Dict], prompt_template: str = None) -> List[Dict]:
        """
        Extracts relationships between entities using the LLM.
        
        Args:
            text: The source text.
            entities: The entities extracted from the text.
            prompt_template: Optional custom prompt template. If None, uses configured template.
            
        Returns:
            A list of extracted relationships.
        """
        if not text:
            logger.warning("No reference text provided for llm relationship extraction")

        if not self.llm:
            logger.warning("LLM not available for relationship extraction")
            return []
            
        if not entities:
            logger.debug("No entities provided for relationship extraction")
            return []
            
        try:
            logger.debug(f"Extracting relationships for {len(entities)} entities using LLM")
            
            # Format entities for prompt
            entity_str = "\n".join([f"- {entity['name']} (Type: {entity['type']})" for entity in entities])
            
            # Get relationship types from config and format them for the prompt
            relationship_types = self.config.get("relationship_types", [])
            
            # Format relationship types - either as a comma-separated list or with more structure
            if relationship_types:
                relationship_types_str = ", ".join(relationship_types)
            else:
                relationship_types_str = "any relevant relationships"
            
            #logger.debug(f"Using relationship types: {relationship_types_str}")

            # Use provided template or get from config
            prompt_template = prompt_template or self.config.get("prompts", {}).get("extract_relationships")
            
            if not prompt_template:
                raise ValueError("No 'extract_relationships' prompt template specified in the config")

            prompt = prompt_template.format(
                source_text=text,
                entities=entity_str,
                relationship_types=relationship_types_str
            )
            #logger.debug(f"First 100 characters of extract_relationships prompt:\n {prompt[:100]}\n")
            response = await self._call_llm(prompt)

            # Parse LLM response
            # Use PydanticOutputParser
            # parser = PydanticOutputParser(pydantic_object=RelationshipList)
            # rel = parser.parse(response)
            rel = json_parse(response, RelationshipList)
            result = [e.model_dump() for e in rel.relationships]

            return result
                
        except Exception as e:
            logger.error(f"Error during LLM relationship extraction: {e}", exc_info=True)
            logger.error(f"LLM response that caused error: {response if 'response' in locals() else 'No response variable'}")
            return []

    async def _glean_graph(self, initial_graph: Graph, text: str, history: List[Dict], prompt_template: str = None) -> Graph:
        """
        Iteratively refines the graph by querying the LLM.
        
        Args:
            initial_graph: The initial graph to refine.
            text: The source text.
            history: Conversation history for the LLM.
            prompt_template: Optional custom prompt template. If None, uses configured template.
            
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
                
                nodes_str = "\n".join([
                    f"- {node.name} (Type: {node.type}, ID: {node.id})"
                    for node in current_graph.nodes
                ])

                # List of relationships
                edges_str = "\n".join([
                    f"- {self._get_node_name(current_graph, edge.source)} --[{edge.type}]--> {self._get_node_name(current_graph, edge.target)}"
                    for edge in current_graph.edges
                ])
                

                # name provided template or get from config
                
                gleaning_prompt_template = prompt_template or self.config.get("prompts", {}).get("entity_relationship_gleaning")
                
                if not gleaning_prompt_template:
                    raise ValueError("No 'entity_relationship_gleaning' prompt template specified in the config")

                prompt = gleaning_prompt_template.format(
                    graph_summary=f"Nodes: {len(current_graph.nodes)}, Edges: {len(current_graph.edges)}",
                    nodes=nodes_str,
                    edges=edges_str,
                    text=text
                )

                response = await self._call_llm(prompt)
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": response})
                # Parse response into a Graph object
                try:
                    result = json_parse(response) #parse to a dict

                    # Extract nodes
                    new_nodes = []
                    if "nodes" in result and isinstance(result["nodes"], list):
                        for node_data in result["nodes"]:
                            if isinstance(node_data, dict) and "name" in node_data and "type" in node_data:
                                node_id = node_data.get("id", str(uuid.uuid4()))
                                new_node = GraphNode(
                                    id=node_id,
                                    text=node_data["name"],
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
                    # Check if the previous gleaning step returned any new nodes or edges
                    if not new_nodes and not new_edges:
                        logger.debug(f"Stopping gleaning early at step {step+1}/{max_gleaning_steps} due to no new nodes or edges")
                        break
            
            return current_graph
            
        except Exception as e:
            logger.error(f"Error during graph gleaning: {e}")
            return initial_graph

    def _get_node_name(self, graph: Graph, node_id: str) -> str:
        """Helper method to get node name from a node ID."""
        for node in graph.nodes:
            if node.id == node_id:
                return node.name
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
                        similarity = SequenceMatcher(None, node1.name.lower(), node2.name.lower()).ratio()
                        
                        # Also check if one is contained in the other
                        text1_lower = node1.name.lower()
                        text2_lower = node2.name.lower()
                        contained = text1_lower in text2_lower or text2_lower in text1_lower
                        
                        # Link if similar or contained
                        if similarity > threshold or contained:
                            logger.debug(f"Linking entities: '{node1.name}' and '{node2.name}' (similarity: {similarity:.2f})")
                            
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
            name_to_id = {}
            for node in graph.nodes:
                name_lower = node.name.lower()
                if name_lower in name_to_id:
                    # We've seen this text before, mark for merging
                    primary_id = name_to_id[name_lower]
                    node_id_mapping[node.id] = primary_id
                    nodes_to_remove.add(node.id)
                else:
                    # First time seeing this text
                    name_to_id[name_lower] = node.id
            
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

            # Get the mapping and relationship types from the config
            mapping = self.config.get("relationship_type_mapping", {})
            relationship_types = self.config.get("relationship_types", [])

            # Normalize edge types
            normalized_count = 0
            for edge in graph.edges:
                edge_type_lower = edge.type.lower()
                if edge_type_lower in mapping:
                    normalized_type = mapping[edge_type_lower]
                    edge.type = normalized_type
                    normalized_count += 1

                    # Add the normalized type to the relationship_types list if missing
                    if normalized_type not in relationship_types:
                        relationship_types.append(normalized_type)

            # Update the config with the updated relationship types
            self.config["relationship_types"] = relationship_types

            logger.debug(f"Normalized {normalized_count} relationship types")
            logger.debug(f"Updated relationship types: {relationship_types}")
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
        Call the LLM with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        # Use the centralized helper from the llms module
        return await call_llm(self.llm, prompt)

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

                nodes_str = "\n".join([
                    f"- {node.name} (Type: {node.type}, ID: {node.id})"
                    for node in graph.nodes
                ])

                # List of relationships
                edges_str = "\n".join([
                    f"- {self._get_node_name(graph, edge.source)} --[{edge.type}]--> {self._get_node_name(graph, edge.target)}"
                    for edge in graph.edges
                ])

                logger.debug(f"\nGraph parsed from JSON:\n {nodes_str}\n{edges_str}\n")

                # name save to a file
                if "output_file" in db_config:
                    with open(db_config["output_file"], "w") as f:
                        f.write(graph_json)
                    logger.info(f"Graph saved to {db_config['output_file']}")
                    
            elif output_format == "networkx":
                # Convert to networkx graph
                g = nx.DiGraph()
                
                # Add nodes
                for node in graph.nodes:
                    g.add_node(node.id, type=node.type, name=node.name, **node.metadata)
                    
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
                            name=node.name,
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
        llm: Union[str, Callable[[str], str]], 
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ) -> Graph:
        """
        Extract entities and relationships from a set of documents.
        
        Args:
            documents: List of Document objects
            llm: Either a model type ('default', 'reasoning', 'vision') or a callable function that takes a prompt and returns a response
            entity_types: List of entity types to extract. If None, uses config.
            relationship_types: List of relationship types to extract. If None, uses config.
            
        Returns:
            Graph object with nodes and edges
        """
        # If a string (model type) is provided, get the LangChain model
        if isinstance(llm, str):
            llm_model = get_llm(llm)
            if llm_model is None:
                raise ValueError(f"Failed to initialize LangChain model: {llm}")

            # Wrap the LangChain model in a callable function
            async def llm_fn(prompt: str) -> str:
                return llm_model.invoke(prompt)  # Use .invoke() for ChatModels

            self.llm = llm_fn
        else:
            # Otherwise, use the provided callable directly
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
                                name=ent["name"],
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
                        entity_name_to_id = {}
                        for node in document_graph.nodes:
                            entity_name_to_id[node.name.lower()] = node.id
                            
                        # Convert relationships to GraphEdges
                        for rel in chunk_relationships:
                            subject_text = rel.get("subject", "").lower()
                            object_text = rel.get("object", "").lower()
                            rel_type = rel.get("relationship")
                            
                            # Try to find subject and object nodes
                            subject_id = entity_name_to_id.get(subject_text)
                            object_id = entity_name_to_id.get(object_text)
                            
                            # If not found by exact match, try substring matching
                            if not subject_id:
                                for name, node_id in entity_name_to_id.items():
                                    if subject_text in name or name in subject_text:
                                        subject_id = node_id
                                        break
                                        
                            if not object_id:
                                for name, node_id in entity_name_to_id.items():
                                    if object_text in name or name in object_text:
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
            existing_name_to_id = {node.name.lower(): node.id for node in final_graph.nodes}
            
            # Mapping from document node IDs to final graph node IDs
            node_id_mapping = {}
            
            # Add nodes to final graph, handling duplicates
            for doc_node in document_graph.nodes:
                node_name_lower = doc_node.name.lower()
                
                # Check if this node text already exists in final graph
                if node_name_lower in existing_name_to_id:
                    # Map this node to the existing one
                    final_node_id = existing_name_to_id[node_name_lower]
                    node_id_mapping[doc_node.id] = final_node_id
                    
                    # Optionally merge metadata or update existing node
                    # For simplicity, we just link the node here
                else:
                    # Add as new node to final graph
                    final_graph.nodes.append(doc_node)
                    existing_name_to_id[node_name_lower] = doc_node.id
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