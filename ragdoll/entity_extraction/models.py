from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator
import uuid


class Entity(BaseModel):
    name: str
    type: str
    desc: Optional[str] = None


class Relationship(BaseModel):
    subject: str
    relationship: str
    object: str

    @field_validator("subject", "relationship", "object", mode="before")
    @classmethod
    def coerce_to_string(cls, v: Any) -> str:
        """Coerce any type to string - LLMs sometimes return integers."""
        if v is None:
            return ""
        return str(v)


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
    label: Optional[str] = None  # Optional display label for the node
    properties: Optional[Dict] = (
        None  # Additional properties including vector references
    )


class GraphEdge(BaseModel):
    """Represents an edge/relationship in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    target: str
    type: str
    metadata: Dict = Field(default_factory=dict)
    source_document_id: Optional[str] = None  # Link back to the originating document

    @field_validator("source_document_id", mode="before")
    @classmethod
    def coerce_document_id_to_string(cls, v: Any) -> Optional[str]:
        """Coerce document ID to string - often comes as hash integer."""
        if v is None:
            return None
        return str(v)


class Graph(BaseModel):
    """Represents a complete knowledge graph with nodes and edges."""

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
