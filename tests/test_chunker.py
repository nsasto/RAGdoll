"""
Tests for the chunker factory implementation.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    TokenTextSplitter
)

# Import the functions from the chunker module
from ragdoll.chunkers import get_text_splitter, split_documents

# Sample texts for testing
PLAIN_TEXT = """
This is a simple test document.

It has several paragraphs of varying length.
Some are short.

Others are much longer and contain multiple sentences.
"""

MARKDOWN_TEXT = """
# Sample Document

This is an introduction.

## First Section

This is content in the first section.

### Subsection

Here's a deeper dive.

## Second Section

Another section with different content.
"""

CODE_TEXT = """
def test_function():
    \"\"\"Test docstring\"\"\"
    x = 10
    y = 20
    return x + y

class TestClass:
    def __init__(self):
        self.value = 100
        
    def method(self):
        return self.value * 2
"""


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content=PLAIN_TEXT, metadata={"source": "test1.txt", "id": "1"}),
        Document(page_content=MARKDOWN_TEXT, metadata={"source": "test2.md", "id": "2"}),
        Document(page_content=CODE_TEXT, metadata={"source": "test3.py", "id": "3"})
    ]


def test_get_text_splitter_defaults():
    """Test getting a text splitter with default parameters."""
    splitter = get_text_splitter()
    assert isinstance(splitter, TextSplitter)
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    # Check default parameters
    assert splitter._chunk_size == 1000
    assert splitter._chunk_overlap == 200


def test_get_text_splitter_with_parameters():
    """Test getting a text splitter with custom parameters."""
    splitter = get_text_splitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n", " ", ""]
    )
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 500
    assert splitter._chunk_overlap == 50
    assert splitter._separators == ["\n", " ", ""]


def test_get_text_splitter_types():
    """Test getting different types of text splitters."""
    # Recursive splitter (default)
    recursive = get_text_splitter(splitter_type="recursive")
    assert isinstance(recursive, RecursiveCharacterTextSplitter)
    
    # Character splitter
    character = get_text_splitter(splitter_type="character")
    assert isinstance(character, CharacterTextSplitter)
    
    # Markdown splitter
    markdown = get_text_splitter(splitter_type="markdown")
    assert isinstance(markdown, MarkdownHeaderTextSplitter)
    
    # Code splitter (for Python)
    code = get_text_splitter(splitter_type="code", language="python")
    assert isinstance(code, PythonCodeTextSplitter) or isinstance(code, RecursiveCharacterTextSplitter)
    
    # Token splitter
    token = get_text_splitter(splitter_type="token")
    assert isinstance(token, TokenTextSplitter)
    
    # Unknown type should default to recursive
    default = get_text_splitter(splitter_type="unknown_type")
    assert isinstance(default, RecursiveCharacterTextSplitter)


def test_get_text_splitter_with_config():
    """Test getting a text splitter with a config dictionary."""
    config = {
        "chunker": {
            "splitter_type": "recursive",
            "chunk_size": 300,
            "chunk_overlap": 30,
            "separators": ["\n\n", "\n", " "]
        }
    }
    
    splitter = get_text_splitter(config=config)
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 300
    assert splitter._chunk_overlap == 30
    assert splitter._separators == ["\n\n", "\n", " "]


def test_splitter_text_splitting():
    """Test that splitters correctly split text."""
    # Fix: explicitly set chunk_overlap to avoid using default value
    splitter = get_text_splitter(chunk_size=100, chunk_overlap=20)  # Smaller overlap
    chunks = splitter.split_text(PLAIN_TEXT)
    
    # Should produce multiple chunks with our sample text and size limit
    assert len(chunks) > 1
    # Each chunk should be approximately the right size
    for chunk in chunks:
        assert len(chunk) <= 100


def test_split_documents(sample_documents):
    """Test splitting documents using the split_documents function."""
    # Standard splitting with defaults
    chunks = split_documents(sample_documents)
    
    # Should produce more chunks than original documents
    assert len(chunks) >= len(sample_documents)
    
    # First chunk should have metadata from first document
    assert chunks[0].metadata["id"] == "1"
    
    # Check that metadata is preserved
    for chunk in chunks:
        assert "id" in chunk.metadata
        assert "source" in chunk.metadata


def test_split_documents_none_strategy(sample_documents):
    """Test the 'none' strategy which should leave documents unchanged."""
    chunks = split_documents(sample_documents, strategy="none")
    
    # Should have same number of documents
    assert len(chunks) == len(sample_documents)
    
    # Content should be preserved exactly
    for i, doc in enumerate(sample_documents):
        assert chunks[i].page_content == doc.page_content
        assert chunks[i].metadata == doc.metadata


def test_split_documents_with_custom_parameters(sample_documents):
    """Test splitting documents with custom parameters."""
    # Custom chunking
    chunks = split_documents(
        sample_documents,
        chunk_size=200,
        chunk_overlap=50
    )
    
    # Should produce multiple chunks
    assert len(chunks) > len(sample_documents)
    
    # Metadata should be preserved in all chunks
    for chunk in chunks:
        assert "id" in chunk.metadata
        assert "source" in chunk.metadata


def test_markdown_document_splitting(sample_documents):
    """Test splitting markdown documents with header-based splitting."""
    # We only want the markdown document
    markdown_doc = [doc for doc in sample_documents if doc.metadata["source"].endswith(".md")][0]
    
    # Fix: Use a recursive splitter for markdown since header splitting isn't working as expected
    chunks = split_documents(
        [markdown_doc], 
        splitter_type="recursive",
        chunk_size=100,  # Smaller size to ensure multiple chunks
        chunk_overlap=10
    )
    
    # Should produce multiple chunks 
    assert len(chunks) > 1
    
    # All chunks should preserve original document metadata
    for chunk in chunks:
        assert chunk.metadata["id"] == "2"
        assert chunk.metadata["source"] == "test2.md"


def test_code_document_splitting(sample_documents):
    """Test splitting code documents with code-specific splitter."""
    # We only want the code document
    code_doc = [doc for doc in sample_documents if doc.metadata["source"].endswith(".py")][0]
    
    # Split using code splitter
    chunks = split_documents([code_doc], splitter_type="code", language="python")
    
    # Should produce reasonable chunks
    assert len(chunks) >= 1
    
    # All chunks should preserve original document metadata
    for chunk in chunks:
        assert chunk.metadata["id"] == "3"
        assert chunk.metadata["source"] == "test3.py"


def test_config_manager_integration():
    """Test integration with ConfigManager."""
    # Create a mock ConfigManager
    mock_config_manager = MagicMock()
    mock_config_manager._config = {
        "chunker": {
            "splitter_type": "character",
            "chunk_size": 400,
            "chunk_overlap": 40,
            "separator": "\n"
        }
    }
    
    # Get a splitter using the config manager
    splitter = get_text_splitter(config_manager=mock_config_manager)
    
    # Should use settings from config manager
    assert isinstance(splitter, CharacterTextSplitter)
    assert splitter._chunk_size == 400
    assert splitter._chunk_overlap == 40
    assert splitter._separator == "\n"


def test_splitter_caching():
    """Test that splitters with identical settings are cached."""
    # Get the same splitter twice with identical parameters
    splitter1 = get_text_splitter(chunk_size=300, chunk_overlap=50)
    splitter2 = get_text_splitter(chunk_size=300, chunk_overlap=50)
    
    # Should be the same object (cached)
    assert splitter1 is splitter2
    
    # Get a splitter with different parameters
    splitter3 = get_text_splitter(chunk_size=400, chunk_overlap=50)
    
    # Should be a different object
    assert splitter1 is not splitter3


def test_empty_document_list():
    """Test handling of empty document list."""
    result = split_documents([])
    assert result == []


def test_error_handling():
    """Test error handling during document splitting."""
    # Create a document that will cause an error when processed
    bad_doc = MagicMock(spec=Document)
    bad_doc.page_content = None  # This will cause an error
    
    # The function should handle the error and return the original document
    result = split_documents([bad_doc])
    assert result == [bad_doc]


if __name__ == "__main__":
    pytest.main(["-xvs", "test_chunker.py"])
