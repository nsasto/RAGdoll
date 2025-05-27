from abc import ABC
from typing import Optional, List, Tuple  # Added Tuple here
from langchain.docstore.document import Document
from langchain.text_splitter import (
    TextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
)
import logging
from ragdoll.config import ConfigManager

logger = logging.getLogger(__name__)

class Chunker:
    """Class for chunking text documents"""
    
    def __init__(self, config_manager=None, config=None):
        """
        Initialize the chunker.
        
        Args:
            config_manager: Optional ConfigManager instance.
            config: Optional configuration dictionary (alternative to config_manager).
        """
        if config_manager is not None:
            # Use the provided ConfigManager
            self.config_manager = config_manager
        elif config is not None:
            # Create a ConfigManager with the provided config
            self.config_manager = ConfigManager()
            # Update the config with the provided dictionary
            # This merges at the top level, so we can update specific sections like "chunker"
            if isinstance(config, dict):
                if "chunker" in config:
                    # If config has a chunker key, use it directly
                    self.config_manager._config["chunker"] = config["chunker"]
                else:
                    # Otherwise treat the entire config as chunker settings
                    self.config_manager._config["chunker"] = config
        else:
            # Create a default ConfigManager
            self.config_manager = ConfigManager()
            
        self._text_splitter = None
        self._current_config = {}
    
    @classmethod
    def from_config(cls, config_manager=None):
        """
        Create a Chunker instance from config.
        
        Args:
            config_manager: Optional ConfigManager instance.
            
        Returns:
            A Chunker instance.
        """
        return cls(config_manager)
    
    def get_text_splitter(
        self,
        splitter_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        markdown_headers: Optional[List[Tuple[str, int]]] = None,
    ) -> TextSplitter:
        """
        Get a text splitter based on configuration.
        
        Args:
            splitter_type: Type of text splitter ('recursive' or 'markdown').
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between chunks.
            separators: List of separators for recursive splitting.
            markdown_headers: List of markdown headers for markdown splitting.
            
        Returns:
            A TextSplitter instance.
            
        Raises:
            ValueError: If the splitter type is invalid.
            KeyError: If required configuration is missing.
        """
        # Get chunker config
        chunker_config = self.config_manager._config.get("chunker", {})
        
        # Determine splitter type (priority to direct parameter, then config)
        actual_splitter_type = splitter_type or chunker_config.get("default_splitter")
        if not actual_splitter_type:
            raise KeyError("No splitter type specified and no default found in config")
        
        # Create a config dict for the current request
        current_request_config = {
            "splitter_type": actual_splitter_type,
            "chunk_size": chunk_size or chunker_config.get("chunk_size", 1000),
            "chunk_overlap": chunk_overlap or chunker_config.get("chunk_overlap", 200),
        }
        
        # Add type-specific parameters
        if actual_splitter_type == "recursive":
            current_request_config["separators"] = separators or chunker_config.get("separators", ["\n\n", "\n", " ", ""])
        elif actual_splitter_type == "markdown":
            # Ensure consistent order for the headers list - this matters for comparison in tests
            headers = markdown_headers or chunker_config.get("markdown_headers", [("#", 1), ("##", 2), ("###", 3)])
            current_request_config["markdown_headers"] = sorted(headers, key=lambda x: x[1])
        else:
            raise ValueError(f"Invalid default_splitter type: {actual_splitter_type}")
        
        # If config hasn't changed, return the cached splitter
        if self._text_splitter and self._current_config == current_request_config:
            return self._text_splitter
        
        # Create a new splitter based on the type
        if actual_splitter_type == "recursive":
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=current_request_config["chunk_size"],
                chunk_overlap=current_request_config["chunk_overlap"],
                separators=current_request_config["separators"],
            )
            # Store these values as properties for easier test access
            self._text_splitter.chunk_size = current_request_config["chunk_size"]
            self._text_splitter.chunk_overlap = current_request_config["chunk_overlap"]
            self._text_splitter.separators = current_request_config["separators"]
        elif actual_splitter_type == "markdown":
            self._text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=current_request_config["markdown_headers"],
            )
        
        # Store the current config for caching
        self._current_config = current_request_config
        
        return self._text_splitter

    def chunk_document(
        self, 
        document: Document, 
        strategy: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        splitter_type: str = None
    ) -> List[Document]:
        """
        Splits a Langchain Document into smaller documents based on the specified strategy.
        
        Args:
            document: The document to split.
            strategy: Chunking strategy ('none', 'fixed', 'semantic', 'recursive', 'header')
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            splitter_type: Type of text splitter for semantic strategy ('markdown', 'python', etc.)
            
        Returns:
            A list of document chunks.
        """
        try:
            # Get chunker config
            chunker_config = self.config_manager._config.get("chunker", {})
            
            # Use default strategy from config if not provided
            if strategy is None:
                strategy = chunker_config.get("chunking_strategy", "fixed")
                
            logger.debug(f"Chunking document using strategy: {strategy}")
            
            if strategy == "none":
                logger.debug(f"No chunking applied for document {document.metadata.get('id', 'unknown')}")
                return [document]
            
            # Get appropriate text splitter
            text_splitter = self.get_text_splitter(
                splitter_type=splitter_type,
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            # Split the document
            result = text_splitter.split_documents([document])
            logger.debug(f"Document split into {len(result)} chunks")
            return result
                
        except Exception as e:
            logger.error(f"Error during document chunking: {e}")
            return [document]  # Fall back to the entire document

    def split_documents(
        self, 
        documents: List[Document], 
        strategy: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        splitter_type: str = None
    ) -> List[Document]:
        """
        Splits multiple Langchain Documents into smaller documents.
        
        Args:
            documents: List of documents to split.
            strategy: Chunking strategy ('none', 'fixed', 'semantic', 'recursive', 'header')
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            splitter_type: Type of text splitter for semantic strategy ('markdown', 'python', etc.)
            
        Returns:
            A list of document chunks from all input documents.
        """
        if not documents:
            return []
            
        result = []
        for doc in documents:
            result.extend(self.chunk_document(
                document=doc,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter_type=splitter_type
            ))
        
        return result