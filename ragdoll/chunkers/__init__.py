"""
Chunker utilities for RAGdoll.

This module provides utilities for creating and configuring text splitters
for document chunking.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import (
    TextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    TokenTextSplitter,
)

logger = logging.getLogger("ragdoll.chunkers")

# Cache for text splitters to avoid recreation with same parameters
_splitter_cache = {}


from ragdoll import settings
from ragdoll.app_config import AppConfig


def get_text_splitter(
    splitter_type: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    config_manager=None,
    app_config: Optional[AppConfig] = None,
    config: Dict[str, Any] = None,
    language: str = None,
    separators: List[str] = None,
    markdown_headers: List[Tuple[str, int]] = None,
    **kwargs,
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
    chunker_config: Dict[str, Any] = {}
    if app_config is not None:
        chunker_config = app_config.config._config.get("chunker", {})
    elif config_manager is not None:
        chunker_config = config_manager._config.get("chunker", {})
    elif config is not None:
        if isinstance(config, dict):
            if "chunker" in config:
                chunker_config = config["chunker"]
            else:
                chunker_config = config
    else:
        chunker_config = settings.get_app().config._config.get("chunker", {})

    # Determine splitter type (priority: parameter > config > default)
    # First check explicit splitter_type, then chunking_strategy (if it's a valid splitter type),
    # then default_splitter, and finally fall back to "recursive"
    actual_splitter_type = splitter_type
    if not actual_splitter_type:
        chunking_strategy = chunker_config.get("chunking_strategy")
        # If chunking is disabled, return None to signal no splitting
        if chunking_strategy == "none":
            logger.info("Chunking disabled (strategy='none') - returning None splitter")
            return None
        elif chunking_strategy in [
            "recursive",
            "character",
            "markdown",
            "code",
            "token",
        ]:
            actual_splitter_type = chunking_strategy
        else:
            actual_splitter_type = chunker_config.get("default_splitter", "recursive")

    # Filter out chunking_strategy from kwargs before passing to LangChain splitters
    # (LangChain TextSplitter doesn't accept this parameter)
    kwargs = {k: v for k, v in kwargs.items() if k != "chunking_strategy"}

    # Get chunk parameters with appropriate defaults
    actual_chunk_size = chunk_size or chunker_config.get("chunk_size", 1000)
    actual_chunk_overlap = chunk_overlap or chunker_config.get("chunk_overlap", 200)

    # Make sure default chunk_overlap respects chunk_size
    if actual_chunk_size and actual_chunk_overlap:
        if actual_chunk_overlap >= actual_chunk_size:
            logger.warning(
                f"Reducing chunk_overlap ({actual_chunk_overlap}) to be smaller than chunk_size ({actual_chunk_size})"
            )
            actual_chunk_overlap = (
                actual_chunk_size // 4
            )  # Set to 25% of chunk size as a reasonable default

    # Prepare a cache key based on splitter type and parameters
    cache_key_parts = [
        f"type={actual_splitter_type}",
        f"size={actual_chunk_size}",
        f"overlap={actual_chunk_overlap}",
    ]

    # Create or retrieve the appropriate text splitter based on type
    if actual_splitter_type == "recursive":
        actual_separators = separators or chunker_config.get(
            "separators", ["\n\n", "\n", " ", ""]
        )
        cache_key_parts.append(f"seps={''.join(actual_separators)}")

        cache_key = ":".join(cache_key_parts)
        if cache_key in _splitter_cache:
            return _splitter_cache[cache_key]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            separators=actual_separators,
            **kwargs,
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
            **kwargs,
        )
        _splitter_cache[cache_key] = splitter
        return splitter

    elif actual_splitter_type == "markdown":
        cache_key = ":".join(cache_key_parts)
        if cache_key in _splitter_cache:
            return _splitter_cache[cache_key]

        # Use MarkdownTextSplitter instead of MarkdownHeaderTextSplitter
        splitter = MarkdownTextSplitter(
            chunk_size=actual_chunk_size, chunk_overlap=actual_chunk_overlap, **kwargs
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
                **kwargs,
            )
        else:
            # Fallback to recursive with code-friendly separators
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=actual_chunk_size,
                chunk_overlap=actual_chunk_overlap,
                separators=["\n\n", "\n", ";", "}", "{", "(", ")", " ", ""],
                **kwargs,
            )
        _splitter_cache[cache_key] = splitter
        return splitter

    elif actual_splitter_type == "token":
        encoding_name = kwargs.get("encoding_name") or chunker_config.get(
            "encoding_name", "cl100k_base"
        )
        cache_key_parts.append(f"encoding={encoding_name}")

        cache_key = ":".join(cache_key_parts)
        if cache_key in _splitter_cache:
            return _splitter_cache[cache_key]

        splitter = TokenTextSplitter(
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            encoding_name=encoding_name,
            **kwargs,
        )
        _splitter_cache[cache_key] = splitter
        return splitter

    else:
        logger.warning(
            f"Unknown splitter type: {actual_splitter_type}, defaulting to recursive"
        )
        return get_text_splitter(
            splitter_type="recursive",
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            **kwargs,
        )


# Fix the split_markdown_documents function


def split_markdown_documents(
    documents: List[Document], markdown_splitter: MarkdownHeaderTextSplitter, **kwargs
) -> List[Document]:
    """
    Helper function to split markdown documents using a header text splitter.

    Args:
        documents: List of documents to split
        markdown_splitter: MarkdownHeaderTextSplitter instance

    Returns:
        List of split documents with metadata
    """
    import hashlib

    result = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)

        # Generate stable source hash for chunk IDs
        source = doc.metadata.get("source", "unknown")
        source_hash = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:8]

        for i, chunk in enumerate(splits):
            metadata = dict(doc.metadata or {})
            page_text = chunk
            if isinstance(chunk, Document):
                page_text = chunk.page_content
                metadata.update(chunk.metadata or {})
            metadata["chunk"] = i
            # Add versioned chunk ID for stable references to vector store
            metadata["chunk_id"] = f"{source_hash}_{i}_v1"
            result.append(Document(page_content=str(page_text), metadata=metadata))

    return result


# Then update the split_documents function to handle markdown splitters
def split_documents(
    documents: List[Document],
    splitter_type: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    config_manager=None,
    config: Dict[str, Any] = None,
    strategy: str = None,
    text_splitter=None,
    **kwargs,
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
        text_splitter: Optional pre-initialized text splitter to use directly
        **kwargs: Additional arguments for the text splitter

    Returns:
        A list of document chunks
    """
    if not documents:
        return []

    splitter_override = kwargs.pop("splitter", None)
    if text_splitter is None and splitter_override is not None:
        text_splitter = splitter_override

    # First, fix any nested Document objects
    valid_documents = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, Document):
            logger.error(f"Non-Document object found at position {i}: {type(doc)}")
            continue

        if not isinstance(doc.page_content, str):
            logger.error(
                f"Document at position {i} has non-string page_content: {type(doc.page_content)}"
            )
            # Fix nested Document
            if isinstance(doc.page_content, Document):
                nested_doc = doc.page_content
                logger.warning(f"Fixing nested Document detected at position {i}")
                # Create new document with nested content and combined metadata
                combined_metadata = {**doc.metadata, **nested_doc.metadata}
                valid_documents.append(
                    Document(
                        page_content=nested_doc.page_content, metadata=combined_metadata
                    )
                )
            continue

        # If we got here, document seems OK
        valid_documents.append(doc)

    if len(valid_documents) != len(documents):
        logger.info(
            f"Fixed {len(documents) - len(valid_documents)} problematic documents, proceeding with {len(valid_documents)} valid documents"
        )

    # Use valid_documents instead of original documents
    documents = valid_documents

    # Check if chunking should be skipped
    chunker_config = {}
    if config_manager is not None:
        chunker_config = config_manager._config.get("chunker", {})
    elif config is not None and isinstance(config, dict):
        chunker_config = config.get("chunker", config)

    actual_strategy = strategy or chunker_config.get("chunking_strategy", "markdown")

    logger.info(
        f"Chunking strategy: '{actual_strategy}' (from strategy={strategy}, config={chunker_config.get('chunking_strategy')})"
    )

    if actual_strategy == "none":
        logger.info("No chunking strategy applied - returning documents unchanged")
        return documents

    # Get or use the text splitter
    try:
        # If a text splitter is provided directly, use it
        # BUT if text_splitter is None, that means chunking was disabled at get_text_splitter level
        if text_splitter is None:
            logger.info(
                "No text splitter provided (chunking disabled) - returning documents unchanged"
            )
            return documents
        elif text_splitter is not None:
            logger.debug(
                f"Using provided text splitter: {text_splitter.__class__.__name__}"
            )
        # Otherwise create a new one using the factory
        else:
            # If the strategy is a valid splitter type, use it as the splitter_type
            splitter_to_use = splitter_type
            if not splitter_to_use and actual_strategy in [
                "recursive",
                "character",
                "markdown",
                "code",
                "token",
            ]:
                splitter_to_use = actual_strategy

            # Filter out chunking_strategy from kwargs before passing to get_text_splitter
            # (LangChain splitters don't accept this parameter)
            splitter_kwargs = {
                k: v for k, v in kwargs.items() if k != "chunking_strategy"
            }

            text_splitter = get_text_splitter(
                splitter_type=splitter_to_use,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                config_manager=config_manager,
                config=config,
                **splitter_kwargs,
            )

        if isinstance(text_splitter, MarkdownHeaderTextSplitter):
            result = split_markdown_documents(documents, text_splitter, **kwargs)
        else:
            result = text_splitter.split_documents(documents)
            # Add versioned chunk IDs for non-markdown splitters (only for Document objects)
            import hashlib

            for idx, doc in enumerate(result):
                if isinstance(doc, Document) and "chunk_id" not in doc.metadata:
                    source = doc.metadata.get("source", "unknown")
                    source_hash = hashlib.md5(str(source).encode("utf-8")).hexdigest()[
                        :8
                    ]
                    doc.metadata["chunk_id"] = f"{source_hash}_{idx}_v1"

        logger.debug(f"Split {len(documents)} documents into {len(result)} chunks")
        return result

    except Exception as e:
        logger.error(f"Error during document chunking: {e}")
        return documents  # Return original documents on error
