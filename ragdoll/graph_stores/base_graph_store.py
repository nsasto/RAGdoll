from abc import ABC, abstractmethod
from typing import Any, List, Dict

class BaseGraphStore(ABC):
    @abstractmethod
    def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Add a node to the graph."""
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Dict[str, Any]) -> None:
        """Add an edge to the graph."""
        pass

    @abstractmethod
    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Query the graph and return results."""
        pass