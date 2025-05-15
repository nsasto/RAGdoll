from abc import ABC
from typing import Optional, List
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
from ragdoll.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class Chunker:
    def __init__(self, config: Optional[dict] = None, text_splitter: Optional[TextSplitter] = None):
        """
        Initializes the Chunker with an optional text splitter.

        Args:
            config: An optional configuration object. If None, the default configuration is loaded.
            text_splitter: An optional LangChain text splitter to use.
        """
        if text_splitter is not None and not isinstance(text_splitter, TextSplitter):
            raise TypeError("text_splitter must be an instance of TextSplitter")
        
        # Load configuration
        if config is None:
            config_manager = ConfigManager()
            self.config = config_manager._config  # Load default config
        else:
            self.config = config
            
        self.text_splitter = text_splitter
        self.chunker_config = self.config.get("chunker", {})

    @classmethod
    def from_config(cls):
        """
        Create a Chunker instance from the default configuration.
        
        Returns:
            Chunker: A configured chunker instance
        """
        config_manager = ConfigManager()
        config = config_manager._config
        return cls(config=config)

    def get_text_splitter(
        self, 
        strategy: str = None, 
        chunk_size: int = None, 
        chunk_overlap: int = None,
        splitter_type: str = None
    ) -> TextSplitter:
        """
        Returns a text splitter object based on the specified strategy.
        
        Args:
            strategy: Chunking strategy ('none', 'fixed', 'semantic', 'recursive', 'header')
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            splitter_type: Type of text splitter for semantic strategy ('markdown', 'python', etc.)
            
        Returns:
            TextSplitter: A configured text splitter
        """
        # Return the existing text splitter if it's already been initialized
        if hasattr(self, "text_splitter") and self.text_splitter is not None:
            return self.text_splitter

        # Use provided values or fall back to config
        if strategy is None:
            strategy = self.chunker_config.get("chunking_strategy", "fixed")
            
        # Use provided values or fall back to config
        _chunk_size = chunk_size or self.chunker_config.get("chunk_size", 1000)
        _chunk_overlap = chunk_overlap or self.chunker_config.get("chunk_overlap", 200)
        _splitter_type = splitter_type or self.chunker_config.get("default_splitter", "markdown")

        if strategy == "none":
            # Return a dummy splitter that doesn't actually split
            class NoSplitTextSplitter(TextSplitter):
                def split_text(self, text):
                    return [text]
            return NoSplitTextSplitter()
            
        elif strategy == "fixed":
            logger.debug(f"Creating fixed-size chunker: size={_chunk_size}, overlap={_chunk_overlap}")
            return CharacterTextSplitter(
                separator="\n\n",
                chunk_size=_chunk_size,
                chunk_overlap=_chunk_overlap,
                length_function=len,
            )
            
        elif strategy == "semantic":
            logger.debug(f"Creating semantic chunker: type={_splitter_type}, size={_chunk_size}, overlap={_chunk_overlap}")
            
            if _splitter_type == "markdown":
                return MarkdownTextSplitter(
                    chunk_size=_chunk_size,
                    chunk_overlap=_chunk_overlap
                )
            elif _splitter_type == "python":
                return PythonCodeTextSplitter(
                    chunk_size=_chunk_size,
                    chunk_overlap=_chunk_overlap
                )
            else:
                logger.warning(f"Unknown splitter type: {_splitter_type}, using RecursiveCharacterTextSplitter")
                return RecursiveCharacterTextSplitter(
                    chunk_size=_chunk_size,
                    chunk_overlap=_chunk_overlap,
                    length_function=len,
                )
                
        elif strategy == "recursive":
            logger.debug(f"Creating recursive chunker: size={_chunk_size}, overlap={_chunk_overlap}")
            return RecursiveCharacterTextSplitter(
                chunk_size=_chunk_size,
                chunk_overlap=_chunk_overlap,
                length_function=len,
            )
            
        elif strategy == "header":
            logger.debug(f"Creating header-based chunker")
            headers_to_split = [                
                ("###", 3),  # h3
                ("##", 2),   # h2
                ("#", 1),    # h1
            ]
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
            
            # For header splitter, we also need a recursive splitter for further splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=_chunk_size,
                chunk_overlap=_chunk_overlap,
                length_function=len,
            )
            
            # Return a composite splitter
            class HeaderThenChunkerSplitter(TextSplitter):
                def split_text(self, text):
                    header_chunks = header_splitter.split_text(text)
                    result = []
                    for chunk in header_chunks:
                        result.extend(text_splitter.split_text(chunk))
                    return result
                    
                def split_documents(self, documents):
                    return self._split_documents(documents)
                    
            return HeaderThenChunkerSplitter()
            
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}, using fixed chunking")
            return CharacterTextSplitter(
                separator="\n\n",
                chunk_size=_chunk_size,
                chunk_overlap=_chunk_overlap,
                length_function=len,
            )

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
            # Use default strategy from config if not provided
            if strategy is None:
                strategy = self.chunker_config.get("chunking_strategy", "fixed")
                
            logger.debug(f"Chunking document using strategy: {strategy}")
            
            if strategy == "none":
                logger.debug(f"No chunking applied for document {document.metadata.get('id', 'unknown')}")
                return [document]
            
            # Get appropriate text splitter
            text_splitter = self.get_text_splitter(
                strategy=strategy, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                splitter_type=splitter_type
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