from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split the text and return a list of chunks."""
        pass