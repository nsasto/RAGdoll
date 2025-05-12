import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Union, Type, Generic, TypeVar, Protocol
from collections import defaultdict

import uuid
import spacy
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, ValidationError, create_model

class BaseLLMService:
    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "Generated Text" # Mock implementation
    async def generate_text(self, prompt: str, **kwargs) -> str:
        return "Generated Text"   # Mock implementation

    async def get_response_model(self, prompt_key: str):
        return TGraph  # Mock implementation

async def format_and_send_prompt(prompt_key: str, llm: BaseLLMService, format_kwargs: Dict[str, str], response_model: type[BaseModel], history_messages: List[Dict[str, str]] = []) -> tuple[BaseModel, List[Dict[str, str]]]:
    # print(f"Prompt Key: {prompt_key}, Format Kargs: {format_kwargs}")   # For debugging
    # Mock the LLM response
    # We need to create mock response models that match the structure expected by the calling code
    # For entity_extraction_query, the expected response_model is TQueryEntities
    mock_response_models = {
        "entity_extraction_query": create_model("MockTQueryEntities", named=(List[str], Field(default=[])), generic=(List[str], Field(default=[]))),
        "entity_relationship_extraction": create_model("MockTGraph", entities=(List[dict], Field(default=[])), relationships=(List[dict], Field(default=[]))),
        "entity_relationship_continue_extraction": create_model("MockTGraph", entities=(List[dict], Field(default=[])), relationships=(List[dict], Field(default=[]))),
        "entity_relationship_gleaning_done_extraction": create_model("MockTGleaningStatus", status=(Literal["done", "continue"], "done")),
        "extract_relationships": List[Dict[str, str]], # Assuming relationships are returned as a list of dicts
    }

    mock_response_data = {}
    if prompt_key == "entity_extraction_query":
         mock_response_data = {"named": ["MockEntity1"], "generic": ["MockEntity2"]}
    elif prompt_key == "entity_relationship_extraction" or prompt_key == "entity_relationship_continue_extraction":
        # Mock some entities and relationships based on format_kwargs, if possible
        text = format_kwargs.get("input_text", "")
        mock_entities_data = []
        mock_relationships_data = []
        if "Text:" in format_kwargs.get("input_text", ""): # Simple heuristic to check if it's chunk text
             # Add some mock entities for testing
             mock_entities_data = [
                 {"id": str(uuid.uuid4()), "type": "PERSON", "text": "John Doe", "start_char": 0, "end_char": 8},
                 {"id": str(uuid.uuid4()), "type": "ORG", "text": "Example Corp", "start_char": 20, "end_char": 32},
             ]
             # Add some mock relationships
             if len(mock_entities_data) > 1:
                 mock_relationships_data = [
                     {"id": str(uuid.uuid4()), "type": "WORKS_FOR", "subject_id": mock_entities_data[0]["id"], "object_id": mock_entities_data[1]["id"], "chunks": []}
                 ]

        mock_response_data = {"entities": mock_entities_data, "relationships": mock_relationships_data}
    elif prompt_key == "entity_relationship_gleaning_done_extraction":
         mock_response_data = {"status": "done"}
    elif prompt_key == "extract_relationships":
         # Mock relationships based on provided entities and text
         entities = format_kwargs.get("entities", [])
         text = format_kwargs.get("text", "")
         mock_relationships_data = []
         if len(entities) >= 2:
             # Create a mock relationship between the first two entities
             mock_relationships_data.append({
                 "subject_id": str(uuid.uuid4()), # We'll need to map back to actual entity IDs later
                 "object_id": str(uuid.uuid4()), # We'll need to map back to actual entity IDs later
                 "relationship_type": "related_to"
             })
         mock_response_data = mock_relationships_data


    try:
        if response_model in mock_response_models.values() or response_model is List[Dict[str, str]]:
             if response_model is List[Dict[str, str]]:
                 parsed_response = mock_response_data # Assuming mock_response_data is already in the correct list of dicts format
             else:
                 # Instantiate the mock response model with mock data
                 parsed_response = response_model(**mock_response_data)
        else:
            # Attempt to parse the mock data using the provided response_model
            # This might fail if the response_model expects a different structure
            parsed_response = response_model(**mock_response_data)
    except ValidationError as e:
        print(f"Mock LLM response validation error for prompt_key '{prompt_key}': {e}")
        # Depending on the expected behavior, you might want to return an empty response
        if response_model is List[Dict[str, str]]:
            return [], history_messages
        else:
            return response_model(), history_messages
    except Exception as e:
        print(f"Unexpected error parsing mock LLM response for prompt_key '{prompt_key}': {e}")
        if response_model is List[Dict[str, str]]:
            return [], history_messages
        else:
             return response_model(), history_messages

    # Mock history update
    history_messages.append({"role": "user", "content": "Mock user query"})
    history_messages.append({"role": "assistant", "content": "Mock assistant response"})

    return parsed_response, history_messages

"""
    if response_model.__name__ == "TQueryEntities":
        return TQueryEntities(named=["entity1"], generic=["entity2"]), []
    elif response_model.__name__ == "TGleaningStatus":
        return TGleaningStatus(status="done"), []
    elif response_model == TGraph:
        return TGraph(entities=[], relationships=[]), [{"role": "user", "content": "mock user message"}] #Added mock history
    else:
        return response_model(), []

    return response_model(), [] #Added to prevent error

# Define a type variable for the generic type
"""
T = TypeVar('T')
BaseModelAlias = BaseModel

from dataclasses import dataclass
from typing import Any, Hashable, List, TypeVar, Generic, Protocol, Dict
import numpy as np
from pydantic import Field, BaseModel
import re
from collections import defaultdict

BTNode = TypeVar("BTNode")
"""The type of the Graph DB Node base model."""

BTEdge = TypeVar("BTEdge")
"""The type of the Graph DB Edge base model."""

BTChunk = TypeVar("BTChunk")
"""The type of the Graph DB Chunk base model."""

TSerializable = Union[Dict, List, str, int, float, bool, None]

GTNode = TypeVar("GTNode", bound=BaseModel)
"""The type of the graph Nodes."""

GTEdge = TypeVar("GTEdge", bound=BaseModel)
"""The type of the graph Edges."""

GTChunk = TypeVar("GTChunk", bound=BaseModel)
"""The type of the graph Chunks."""

TEmbeddingType = Literal["openai", "azure_openai"]

TEmbedding = np.ndarray
"""The type of the Embeddings."""

THash = int
"""The type of the Hashes."""

TScore = float
"""The type of the Scores."""

TIndex = int
"""The type of the Indexes."""

TId = TypeVar("TId", bound=Hashable)
"""The type of the identifiers."""

TDocument = TypeVar("TDocument", bound=BaseModel)
"""The type of the Document model."""

TGraph = TypeVar("TGraph", bound=BaseModel)
"""The type of the Graph model."""

TContext = TypeVar("TContext", bound=BaseModel)
"""The type of the Context model."""

TQueryResponse = TypeVar("TQueryResponse", bound=BaseModel)
"""The type of the Query Response model."""


# Mock the missing modules and classes from fast_graphrag._storage._base
# Replace these with actual imports when available
class BaseGraphStorage(Generic[TEntity, TRelation, GTId], Protocol): # Inherit from Generic and Protocol

    def __init__(self, config):
        self.config = config
        self.relations = []
        self.entities = {}
        self.relations = {}

    async def insert_start(self):
        pass

    async def insert_done(self):
        pass

    async def insert_entity(self, entity: "TEntity"):
        if entity.id not in self.entities:
            self.entities[entity.id] = entity
        # Handle duplicate entities if necessary (e.g., merge properties)

    async def insert_relation(self, relation: "TRelation"):
        # Simplified the logic for inserting relations
        # In a real graph storage, you would add edges between nodes
        # For this mock, we just store the relation object
        self.relations[relation.id] = relation

    def get_all_entities(self) -> List["TEntity"]:
        return list(self.entities.values())
    def get_entity_by_id(self, entity_id: str) -> Optional["TEntity"]:
        return self.entities.get(entity_id)

    def get_all_relations(self) -> List["TRelation"]:
        return self.relations

    async def close(self):
        pass
    
    async def get_or_create_node(self, entity: "TEntity") -> "TEntity":
        if entity.id not in self.entities:
            self.entities[entity.id] = entity
        return self.entities[entity.id]

    async def get_or_create_relation(self, relation: "TRelation") -> "TRelation":
        """Get or create relation, assumes that nodes already exist"""
        # In a real implementation, you would add an edge to the graph
        return relation # simplified the logic
    
    async def graph_upsert(self,entities: List["TEntity"], relationships: List["TRelation"]):
          for entity in entities:
             await self.insert_entity(entity)
          for relation in relationships:
             await self.insert_relation(relation)
    
    async def graph_upsert(self, entities: List["TEntity"], relationships: List["TRelation"]):
        """Placeholder for graph upsert logic."""
        pass

# Mock the missing models and classes from fast_graphrag._models
# Replace these with actual imports when available
class TQueryEntities(BaseModel):
    named: List[str] = Field(default_factory=list)
    generic: List[str] = Field(default_factory=list)

@dataclass
class _Chunk:
    text: str
    id: Optional[THash] = None
    chunk_id: Optional[THash] = None  # Renamed to chunk_id for clarity if needed
    document_id: Optional[THash] = None
    start_index: Optional[TIndex] = None
    end_index: Optional[TIndex] = None

@dataclass
class _Document:
    text: str
    id: Optional[THash] = None
    document_id: Optional[THash] = None # Renamed for clarity
    url: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    chunk_ids: List[THash] = Field(default_factory=list)

@dataclass
class _ReferenceList:
    reference_ids: List[THash] = Field(default_factory=list)

# Update the mocked TChunk, TEntity, TRelation, TGraph to match the example
# These might need to be adjusted based on the actual definition in fast_graphrag._types
class TChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    document_id: Optional[str] = None # Added for consistency

class TEntity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    text: str
    start_char: int
    end_char: int

class TRelation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    subject_id: str
    object_id: str
    chunks: List[str] = Field(default_factory=list) # Made default_factory

class TGraph(BaseModel):
    entities: List[TEntity] = Field(default_factory=list)
    relationships: List[TRelation] = Field(default_factory=list)

class IGraphStorageConfig(BaseModel):
    entity_type: type
    relation_type: type


class IGraphStorage(BaseGraphStorage):
    def __init__(self, config: IGraphStorageConfig):
        super().__init__(config)

# Mock GTId if not defined elsewhere
# It seems GTId is used as a type variable, so defining it as a TypeVar is more appropriate
GTId = TypeVar("GTId") # Defined GTId as TypeVar

# Mock logger if not defined elsewhere
def logger(*args, **kwargs):
    """Mock logger function."""
    print("MOCK_LOGGER:", *args, **kwargs)


# def logger(): #defined
#     pass


class TGleaningStatus(BaseModel):
    status: Literal["done", "continue"] = Field(
        description="done if all entities and relationship have been extracted, continue otherwise"
    )

# Mock BaseInformationExtractionService if not defined elsewhere
class BaseInformationExtractionService:
    pass

class GTId:
    pass

@dataclass
class EntityExtractionService(BaseInformationExtractionService):
    """Entity and relationship extractor using a hybrid approach."""
    _nlp: spacy.Language = field(default_factory=lambda: spacy.load('en_core_web_sm')) # Load spaCy model here
    def extract(
        self,
        llm: BaseLLMService,
        documents: Iterable[Iterable[TChunk]],
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, GTId]]]]:
        """Extract both entities and relationships from the given data using a hybrid approach."""
        return [
            asyncio.create_task(self._extract_document_hybrid(llm, document, prompt_kwargs, entity_types)) for document in documents
        ]

    async def _extract_document_hybrid(
        self, llm: BaseLLMService, chunks: Iterable[TChunk], prompt_kwargs: Dict[str, str], entity_types: List[str]
    ) -> Optional[BaseGraphStorage[TEntity, TRelation, GTId]]:
        """Extract entities and relationships from a document using the hybrid approach."""
        all_entities: List[TEntity] = []
        all_relationships: List[TRelation] = []

        for chunk in chunks:
            resolved_text = self._resolve_coreferences(chunk.content)
            entities = self._extract_entities_with_ner(resolved_text)
            relationships = await self._extract_relationships_with_llm(llm, resolved_text, entities)
            filtered_relationships = self._filter_relationships_with_rules(relationships)

            # Convert extracted entities and relationships to TEntity and TRelation objects
            all_entities.extend([
                TEntity(
                    text=entity["text"],
                    type=entity["type"],
                    start_char=entity["start_char"],
                    end_char=entity["end_char"],
                )
                for entity in entities
            ])
            all_relationships.extend([
                TRelation(
                    type=rel["relationship_type"],  # Assuming this is the correct key
                    subject_id=rel["subject_id"],  # Assuming these are the correct keys
                    object_id=rel["object_id"],
                    chunks=[chunk.id],  # Add the chunk ID here
                )
                for rel in filtered_relationships
            ])

        # Combine entities and relationships into a TGraph and merge into storage
        graph = TGraph(entities=all_entities, relationships=all_relationships)
        graph_storage = await self._merge(llm, [graph])
        return graph_storage

    async def extract_entities_from_query(
        self, llm: BaseLLMService, query: str, prompt_kwargs: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Extract entities from the given query."""
        prompt_kwargs["query"] = query
        entities, _ = await format_and_send_prompt(
            prompt_key="entity_extraction_query",
            llm=llm,
            format_kwargs=prompt_kwargs,
            response_model=TQueryEntities,
        )
        return {
            "named": entities.named,
            "generic": entities.generic
        }

    async def _extract(
        # This is the original fast_graphrag method, keeping it for comparison or potential reuse
        self, llm: BaseLLMService, chunks: Iterable[TChunk], prompt_kwargs: Dict[str, str], entity_types: List[str]
    ) -> Optional[BaseGraphStorage[TEntity, TRelation, GTId]]:
        """Extract both entities and relationships from the given chunks."""
        # Extract entities and relatioships from each chunk
        try:
            chunk_graphs = await asyncio.gather(
                *[self._extract_from_chunk(llm, chunk, prompt_kwargs, entity_types) for chunk in chunks]
            )
            if not chunk_graphs:  # Changed from len(chunk_graphs) == 0:
                return None

            # Combine chunk graphs in document graph
            return await self._merge(llm, chunk_graphs)
        except Exception as e:
            #  logger.error(f"Error during information extraction from document: {e}") # Removed logger
            print(f"Error during information extraction from document: {e}")
            return None

    async def _gleaning(
        self, llm: BaseLLMService, initial_graph: TGraph, history: list[dict[str, str]]
    ) -> Optional[TGraph]:
        """Do gleaning steps until the llm says we are done or we reach the max gleaning steps."""
        # Prompts
        current_graph = initial_graph
        self.max_gleaning_steps = 3 # added max gleaning steps.

        try:
            for gleaning_count in range(self.max_gleaning_steps):
                # Do gleaning step
                gleaning_result, history = await format_and_send_prompt(
                    prompt_key="entity_relationship_continue_extraction",
                    llm=llm,
                    format_kwargs={},
                    response_model=TGraph,
                    history_messages=history,
                )

                # Combine new entities, relationships with previously obtained ones
                current_graph.entities.extend(gleaning_result.entities)
                current_graph.relationships.extend(gleaning_result.relationships)

                # Stop gleaning if we don't need to keep going
                if gleaning_count == self.max_gleaning_steps - 1:
                    break

                # Ask llm if we are done extracting entities and relationships
                gleaning_status, _ = await format_and_send_prompt(
                    prompt_key="entity_relationship_gleaning_done_extraction",
                    llm=llm,
                    format_kwargs={},
                    response_model=TGleaningStatus,
                    history_messages=history,
                )

                # If we are done parsing, stop gleaning
                if gleaning_status.status == "done": # Changed from Literal["done"]
                    break
        except Exception as e:
            # logger.error(f"Error during gleaning: {e}") # Removed logger
            print(f"Error during gleaning: {e}")
            return None

        return current_graph

    async def _extract_from_chunk(
        self, llm: BaseLLMService, chunk: TChunk, prompt_kwargs: Dict[str, str], entity_types: List[str]
    ) -> TGraph:
        """Extract entities and relationships from the given chunk."""
        prompt_kwargs["input_text"] = chunk.content

        chunk_graph, history = await format_and_send_prompt(
            prompt_key="entity_relationship_extraction",
            llm=llm,
            format_kwargs=prompt_kwargs,
            response_model=TGraph,
        )

        # Do gleaning
        chunk_graph_with_gleaning = await self._gleaning(llm, chunk_graph, history)
        if chunk_graph_with_gleaning:
            chunk_graph = chunk_graph_with_gleaning

        _clean_entity_types = [re.sub("[ _]", "", entity_type).upper() for entity_type in entity_types]
        for entity in chunk_graph.entities:
            if re.sub("[ _]", "", entity.type).upper() not in _clean_entity_types:
                entity.type = "UNKNOWN"

        # Assign chunk ids to relationships
        for relationship in chunk_graph.relationships:
            relationship.chunks = [chunk.id]

        return chunk_graph

    async def _merge(self, llm: BaseLLMService, graphs: List[TGraph]) -> BaseGraphStorage[TEntity, TRelation, GTId]:
        """Merge the given graphs into a single graph storage."""
        graph_storage = IGraphStorage(config=IGraphStorageConfig(entity_type=TEntity, relation_type=TRelation)) # Added entity_type and relation_type

        await graph_storage.insert_start()

        try:
            # This is synchronous since each sub graph is inserted into the graph storage and conflicts are resolved
            for graph in graphs:
                await graph_storage.graph_upsert(graph.entities, graph.relationships) #changed from self to graph_storage
        finally:
            await graph_storage.insert_done()
        return graph_storage

    def _extract_entities_with_ner(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """
        Extracts entities from the input text using spaCy's Named Entity Recognition.

        Args:
            text: The input text to extract entities from.

        Returns:
            A list of dictionaries, where each dictionary represents an entity
            and contains its text, type, and start and end character offsets.
        """
        doc = self._nlp(text)
        entities: List[Dict[str, Union[str, int]]] = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
        return entities

    async def _extract_relationships_with_llm(
        self, llm: BaseLLMService, text: str, entities: List[Dict[str, Union[str, int]]]
    ) -> List[Dict[str, str]]:
        """
        Extracts relationships between entities using an LLM.

        Args:
            llm: The Language Model service to use.
            text: The input text.
            entities: A list of extracted entities.

        Returns:
            A list of dictionaries, where each dictionary represents a relationship.
        """
        # Craft a prompt for the LLM to extract relationships
        prompt = f"""
        Extract relationships between the following entities found in the text:
        {entities}

        Text:
        {text}

        Provide the relationships as a list of dictionaries, where each dictionary represents a relationship.
        Each dictionary should have the following keys: "subject_id", "object_id", and "relationship_type".
        """
        # Use format_and_send_prompt to interact with the LLM
        # You might need a specific prompt key and response model for relationships
        try:
            llm_response, _ = await format_and_send_prompt(
                prompt_key="extract_relationships",  # You'll need to define this prompt key
                llm=llm,
                format_kwargs={"text": text, "entities": entities},
                response_model=List[Dict[str, str]],  # Adjust the response model as needed
            )
            return llm_response
        except ValidationError as e:
            print(f"Error in LLM response: {e}")
            return []  # Return an empty list in case of an error

    def _filter_relationships_with_rules(
        self, relationships: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Filters relationships based on predefined rules.

        Args:
            relationships: A list of extracted relationships.

        Returns:
            A list of valid relationships.
        """
        filtered = []
        for rel in relationships:
            #  Example rule: Filter relationships where subject or object is empty
            if rel.get("subject_id") and rel.get("object_id") and rel.get("relationship_type"):
                filtered.append(rel)
        return filtered

    def _resolve_coreferences(self, text: str) -> str:
        """
        Resolves coreferences in the input text.

        Args:
            text: The input text.

        Returns:
            The text with resolved coreferences.
        """
        # Simple coreference resolution: replace common pronouns with placeholder "[ENTITY]"
        # A proper implementation would require a dedicated library like neuralcoref (though older)
        # or a more advanced rule-based system or LLM-based approach.
        resolved_text = re.sub(r'\b(he|she|it|they|him|her|them)\b', '[ENTITY]', text, flags=re.IGNORECASE)
        return resolved_text

    def _postprocess_graph(
        self, graph_storage: BaseGraphStorage[TEntity, TRelation, GTId]
    ) -> BaseGraphStorage[TEntity, TRelation, GTId]:
        """
        Applies post-processing steps to the graph storage.

        Args:
            graph_storage: The graph storage to post-process.

        Returns:
            The post-processed graph storage.
        """
        # This is a simplified example using networkx for graph operations.
        # In a real scenario, you would work directly with your BaseGraphStorage implementation.
        graph = nx.DiGraph()

        # Add entities as nodes
        for entity in graph_storage.get_all_entities():  # Assuming get_all_entities exists
            graph.add_node(entity.id, type=entity.type, text=entity.text)

        # Add relationships as edges
        for relation in graph_storage.get_all_relations():  # Assuming get_all_relations exists
            graph.add_edge(relation.subject_id, relation.object_id, type=relation.type)

        # Example post-processing: Merge nodes with similar text (basic string matching)
        nodes_to_merge = []
        for node1, data1 in graph.nodes(data=True):
            for node2, data2 in graph.nodes(data=True):
                if node1 != node2 and data1['text'].lower() == data2['text'].lower():
                    nodes_to_merge.append((node1, node2))
        # Implement merging logic (e.g., using networkx.contract)
        for node1_id, node2_id in nodes_to_merge:
            try:
                graph = nx.contract_nodes(graph, node1_id, node2_id, self_loops=False)
            except nx.NetworkXError as e:
                print(f"Error merging nodes {node1_id} and {node2_id}: {e}")

        # Example post-processing: Normalize relationship types
        relationship_type_mapping = {"works for": "works_for", "is a": "is_a"}  # Define your mapping
        for u, v, data in graph.edges(data=True):
            if data['type'].lower() in relationship_type_mapping:
                data['type'] = relationship_type_mapping[data['type'].lower()]

        # After post-processing the networkx graph, you would update your graph_storage
        # This would involve deleting old nodes/edges and adding new ones, or using update methods
        # provided by your storage implementation.
        # This is a conceptual outline and the actual implementation depends heavily
        # on how your BaseGraphStorage handles updates and merging.

        # Create a new GraphStorage to return the modified graph data
        postprocessed_graph_storage = IGraphStorage(config=IGraphStorageConfig(entity_type=TEntity, relation_type=TRelation))
        for node_id, node_data in graph.nodes(data=True):
            entity = TEntity(
                id=node_id,
                type=node_data["type"],
                text=node_data["text"],
                start_char=0,  # These values might not be directly transferable from the networkx graph
                end_char=0,    # You might need to re-derive them if needed.
            )
            postprocessed_graph_storage.insert_entity(entity)

        for source, target, edge_data in graph.edges(data=True):
             relation = TRelation(
                subject_id=source,
                object_id=target,
                type=edge_data["type"],
                chunks=[],  #  You might need to re-derive these.
            )
             postprocessed_graph_storage.insert_relation(relation)
        return postprocessed_graph_storage

