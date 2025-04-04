from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document

class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> List[Document]:
        """Load data and return it as a list of Documents."""
        pass