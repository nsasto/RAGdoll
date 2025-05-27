"""
Chunker utilities for RAGdoll.

This module provides utilities for creating and configuring text splitters
for document chunking.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

from langchain.docstore.document import Document
from langchain.text_splitter import (
    TextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    TokenTextSplitter
)

from ragdoll.config import ConfigManager

logger = logging.getLogger("ragdoll.chunkers")

# Cache for text splitters to avoid recreation with same parameters
_splitter_cache = {}

def get_text_splitter(
    splitter_type: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    config_manager = None,
    config: Dict[str, Any] = None,
    language: str = None,
    separators: List[str] = None,
    markdown_headers: List[Tuple[str, int]] = None,
    **kwargs
) -> TextSplitter:
    """
    Factory function to create appropriate TextSplitter based on configuration.
    
    Args:
        splitter_type: Type of text splitter ('recursive', 'character', 'markdown', 'code', 'token')
        chunk_size: Size of each chunk in characters or tokens
        chunk_overlap: Overlap between chunks
        config_manager: Optional ConfigManager instance
        config: Optional configuration dictionary
        language: Programming language for code splitters
        separators: List of separators for recursive splitting
        markdown_headers: List of markdown headers for markdown splitting
        **kwargs: Additional arguments passed to the text splitter
        
    Returns:
        A configured TextSplitter instance
    """
    # Initialize config
    chunker_config = {}
    if config_manager is not None:
        chunker_config = config_manager._config.get("chunker", {})
    elif config is not None:
        if isinstance(config, dict):
            if "chunker" in config:
                chunker_config = config["chunker"]
            else:
                chunker_config = config
    
    # Determine splitter type (priority: parameter > config > default)
    actual_splitter_type = splitter_type or chunker_config.get("splitter_type")
    if not actual_splitter_type:
        actual_splitter_type = chunker_config.get("default_splitter", "recursive")
    
    # Get chunk parameters with appropriate defaults
    actual_chunk_size = chunk_size or chunker_config.get("chunk_size", 1000)
    actual_chunk_overlap = chunk_overlap or chunker_config.get("chunk_overlap", 200)
    
    # Make sure default chunk_overlap respects chunk_size
    if actual_chunk_size and actual_chunk_overlap:
        if actual_chunk_overlap >= actual_chunk_size:
            logger.warning(f"Reducing chunk_overlap ({actual_chunk_overlap}) to be smaller than chunk_size ({actual_chunk_size})")
            actual_chunk_overlap = actual_chunk_size // 4  # Set to 25% of chunk size as a reasonable default
    
    # Prepare a cache key based on splitter type and parameters
    cache_key_parts = [
        f"type={actual_splitter_type}",
        f"size={actual_chunk_size}",
        f"overlap={actual_chunk_overlap}"
    ]
    
    # Create or retrieve the appropriate text splitter based on type
    if actual_splitter_type == "recursive":
        actual_separators = separators or chunker_config.get("separators", ["\n\n", "\n", " ", ""])
        cache_key_parts.append(f"seps={''.join(actual_separators)}")
        
        cache_key = ":".join(cache_key_parts)
        if cache_key in _splitter_cache:
            return _splitter_cache[cache_key]
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            separators=actual_separators,
            **kwargs
        )
        _splitter_cache[cache_key] = splitter
        return splitter
        
    elif actual_splitter_type == "character":
        separator = kwargs.get("separator") or chunker_config.get("separator", "\n\n")
        cache_key_parts.append(f"sep={separator}")
        
        cache_key = ":".join(cache_key_parts)
        if cache_key in _splitter_cache:
            return _splitter_cache[cache_key]
            
        splitter = CharacterTextSplitter(
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            separator=separator,
            **kwargs
        )
        _splitter_cache[cache_key] = splitter
        return splitter
        
    elif actual_splitter_type == "markdown":
        # Create or get markdown headers
        headers = markdown_headers or chunker_config.get("markdown_headers", [
            ("#", 1), ("##", 2), ("###", 3), ("####", 4)
        ])
        # Sort headers by level for consistency
        sorted_headers = sorted(headers, key=lambda x: x[1])
        cache_key_parts.append(f"headers={len(sorted_headers)}")
        
        cache_key = ":".join(cache_key_parts)
        if cache_key in _splitter_cache:
            return _splitter_cache[cache_key]
            
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=sorted_headers,
            **kwargs
        )
        _splitter_cache[cache_key] = splitter
        return splitter
        
    elif actual_splitter_type == "code":
        actual_language = language or chunker_config.get("language", "python").lower()
        cache_key_parts.append(f"lang={actual_language}")
        
        cache_key = ":".join(cache_key_parts)
        if cache_key in _splitter_cache:
            return _splitter_cache[cache_key]
        
        if actual_language == "python":
            splitter = PythonCodeTextSplitter(
                chunk_size=actual_chunk_size,
                chunk_overlap=actual_chunk_overlap,
                **kwargs
            )
        else:
            # Fallback to recursive with code-friendly separators
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=actual_chunk_size,
                chunk_overlap=actual_chunk_overlap,
                separators=["\n\n", "\n", ";", "}", "{", "(", ")", " ", ""],
                **kwargs
            )
        _splitter_cache[cache_key] = splitter
        return splitter
        
    elif actual_splitter_type == "token":
        encoding_name = kwargs.get("encoding_name") or chunker_config.get("encoding_name", "cl100k_base")
        cache_key_parts.append(f"encoding={encoding_name}")
        
        cache_key = ":".join(cache_key_parts)
        if cache_key in _splitter_cache:
            return _splitter_cache[cache_key]
            
        splitter = TokenTextSplitter(
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            encoding_name=encoding_name,
            **kwargs
        )
        _splitter_cache[cache_key] = splitter
        return splitter
    
    else:
        logger.warning(f"Unknown splitter type: {actual_splitter_type}, defaulting to recursive")
        return get_text_splitter(
            splitter_type="recursive",
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            **kwargs
        )

# Fix the split_markdown_documents function

def split_markdown_documents(
    documents: List[Document],
    markdown_splitter: MarkdownHeaderTextSplitter,
    **kwargs
) -> List[Document]:
    """
    Helper function to split markdown documents using a header text splitter.
    
    Args:
        documents: List of documents to split
        markdown_splitter: MarkdownHeaderTextSplitter instance
        
    Returns:
        List of split documents with metadata
    """
    result = []
    for doc in documents:
        # The MarkdownHeaderTextSplitter needs to be used differently:
        # First split the text into chunks with headers
        headers_to_split_on = markdown_splitter.headers_to_split_on
        splits = markdown_splitter.split_text(doc.page_content)
        
        # Create new documents with original metadata
        for i, split_text in enumerate(splits):
            metadata = doc.metadata.copy()
            metadata["chunk"] = i
            result.append(Document(page_content=split_text, metadata=metadata))
    
    return result

# Then update the split_documents function to handle markdown splitters
def split_documents(
    documents: List[Document],
    splitter_type: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    config_manager = None,
    config: Dict[str, Any] = None,
    strategy: str = None,
    **kwargs
) -> List[Document]:
    """
    Split documents using a configured text splitter.
    
    Args:
        documents: List of documents to split
        splitter_type: Type of text splitter ('recursive', 'character', 'markdown', 'code', 'token')
        chunk_size: Size of each chunk in characters or tokens
        chunk_overlap: Overlap between chunks
        config_manager: Optional ConfigManager instance
        config: Optional configuration dictionary
        strategy: Optional chunking strategy. If 'none', returns documents unchanged
        **kwargs: Additional arguments for the text splitter
        
    Returns:
        A list of document chunks
    """
    if not documents:
        return []
    
    # Check if chunking should be skipped
    chunker_config = {}
    if config_manager is not None:
        chunker_config = config_manager._config.get("chunker", {})
    elif config is not None and isinstance(config, dict):
        chunker_config = config.get("chunker", config)
        
    actual_strategy = strategy or chunker_config.get("chunking_strategy", "fixed")
    
    if actual_strategy == "none":
        logger.debug("No chunking strategy applied")
        return documents
    
    # Get the splitter and apply it
    try:
        text_splitter = get_text_splitter(
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            config_manager=config_manager,
            config=config,
            **kwargs
        )
        
        # Special handling for markdown splitters which don't have split_documents
        if isinstance(text_splitter, MarkdownHeaderTextSplitter):
            return split_markdown_documents(documents, text_splitter)
        # Otherwise use the standard split_documents method
        else:
            result = text_splitter.split_documents(documents)
            logger.debug(f"Split {len(documents)} documents into {len(result)} chunks")
            return result
    
    except Exception as e:
        logger.error(f"Error during document chunking: {e}")
        return documents  # Return original documents on error

