from typing import List, Dict, Any, Optional, Union
from langchain.docstore.document import Document
from langchain_core.document_loaders.base import BaseLoader as LangchainBaseLoader

class BaseLoader(LangchainBaseLoader):
    """
    Base loader class that extends LangChain's BaseLoader.
    
    This allows any LangChain loader to be used directly with RAGdoll's ingestion system.
    """
    
    def load(self) -> List[Document]:
        """
        Load data and return it as a list of Documents.
        
        Returns:
            List of Document objects.
        """
        return super().load()
    
    def load_and_split(self, text_splitter=None):
        """
        Load documents and split them using the provided text splitter.
        
        Args:
            text_splitter: The text splitter to use.
            
        Returns:
            List of split Document objects.
        """
        return super().load_and_split(text_splitter)

# Type alias for compatibility with both RAGdoll loaders and LangChain loaders
LoaderType = Union[BaseLoader, LangchainBaseLoader]

def ensure_loader_compatibility(loader: LoaderType) -> LoaderType:
    """
    Ensures the loader is compatible with the RAGdoll system.
    
    Args:
        loader: A BaseLoader or LangchainBaseLoader instance.
        
    Returns:
        Compatible loader instance.
    """
    if isinstance(loader, LangchainBaseLoader):
        return loader
    elif isinstance(loader, BaseLoader):
        return loader
    else:
        raise TypeError(f"Unsupported loader type: {type(loader)}")