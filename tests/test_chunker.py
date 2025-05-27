import pytest
from unittest.mock import patch
from ragdoll.chunkers.chunker import Chunker
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TextSplitter,
)


@pytest.fixture
def mock_config_manager():
    with patch("ragdoll.chunkers.chunker.ConfigManager") as mock:
        mock_instance = mock.return_value
        # Set up a default chunker configuration
        mock_instance._config = {
            "chunker": {
                "default_splitter": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]
            }
        }
        yield mock_instance


def test_invalid_splitter_type(mock_config_manager):
    """Test that an invalid splitter type raises a ValueError."""
    mock_config_manager._config = {"chunker": {"default_splitter": "invalid_type"}}
    chunker = Chunker()
    with pytest.raises(ValueError, match="Invalid default_splitter type: invalid_type"):
        chunker.get_text_splitter()


def test_markdown_splitter_no_config_params(mock_config_manager):
    """Test markdown splitter creation with minimal config."""
    mock_config_manager._config["chunker"]["default_splitter"] = "markdown"
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, MarkdownHeaderTextSplitter)
    
    # Order-insensitive comparison - convert to sets of tuples
    expected_headers = {("#", 1), ("##", 2), ("###", 3)}
    actual_headers = set(tuple(header) for header in splitter.headers_to_split_on)
    assert actual_headers == expected_headers


def test_recursive_splitter_default_params(mock_config_manager):
    """Test recursive splitter uses default parameters when not specified."""
    mock_config_manager._config["chunker"]["default_splitter"] = "recursive"
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter.chunk_size == 1000
    assert splitter.chunk_overlap == 200
    assert splitter.separators == ["\n\n", "\n", " ", ""]


def test_recursive_splitter_custom_params(mock_config_manager):
    """Test recursive splitter with custom parameters."""
    mock_config_manager._config["chunker"] = {
        "default_splitter": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "separators": ["\n", " ", ""]
    }
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 50
    assert splitter.separators == ["\n", " ", ""]


def test_chunker_config_precedence(mock_config_manager):
    """Test that direct parameters take precedence over config."""
    mock_config_manager._config["chunker"] = {
        "default_splitter": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    chunker = Chunker()
    # Override config with direct parameters
    splitter = chunker.get_text_splitter(
        splitter_type="recursive",
        chunk_size=300,
        chunk_overlap=50
    )
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter.chunk_size == 300
    assert splitter.chunk_overlap == 50


def test_empty_config_raises_error():
    """Test that an empty config raises KeyError."""
    with patch("ragdoll.chunkers.chunker.ConfigManager") as mock:
        mock_instance = mock.return_value
        mock_instance._config = {}
        chunker = Chunker()
        with pytest.raises(KeyError):
            chunker.get_text_splitter()


def test_missing_splitter_type_raises_error(mock_config_manager):
    """Test that missing splitter type raises KeyError."""
    mock_config_manager._config["chunker"] = {}  # No default_splitter
    chunker = Chunker()
    with pytest.raises(KeyError):
        chunker.get_text_splitter()


def test_markdown_splitter_headers(mock_config_manager):
    """Test markdown splitter with custom headers."""
    mock_config_manager._config["chunker"] = {
        "default_splitter": "markdown",
        "markdown_headers": [("####", 1), ("#####", 2)]
    }
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, MarkdownHeaderTextSplitter)
    
    # Order-insensitive comparison - convert to sets of tuples
    expected_headers = {("####", 1), ("#####", 2)}
    actual_headers = set(tuple(header) for header in splitter.headers_to_split_on)
    assert actual_headers == expected_headers


def test_text_splitter_not_recreated(mock_config_manager):
    """Test that the text splitter is not recreated if parameters don't change."""
    chunker = Chunker()
    splitter1 = chunker.get_text_splitter()
    splitter2 = chunker.get_text_splitter()
    # Should return the same cached instance
    assert splitter1 is splitter2


def test_from_config_class_method(mock_config_manager):
    """Test the from_config class method."""
    mock_config_manager._config["chunker"]["default_splitter"] = "recursive"
    chunker = Chunker.from_config()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter.chunk_size == 1000
    assert splitter.chunk_overlap == 200
