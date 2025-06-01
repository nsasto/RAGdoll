from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import uuid

class Entity(BaseModel):
    name: str
    type: str
    desc: Optional[str] = None

class Relationship(BaseModel):
    subject: str
    relationship: str
    object: str

class EntityList(BaseModel):
    entities: List[Entity]

class RelationshipList(BaseModel):
    relationships: List[Relationship]

class GraphNode(BaseModel):
    """Represents a node in the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    name: str
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