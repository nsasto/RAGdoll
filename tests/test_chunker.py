import pytest
from unittest.mock import patch
from ragdoll.chunkers.chunker import Chunker
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TextSplitter,
)


@pytest.fixture
def mock_config_manager():
    with patch("ragdoll.chunkers.chunker.ConfigManager") as mock:
        yield mock


def test_invalid_splitter_type(mock_config_manager):
    """Test that an invalid splitter type raises a ValueError."""
    mock_config = {"chunker": {"default_splitter": "invalid_type"}}
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    with pytest.raises(ValueError, match="Invalid default_splitter type: invalid_type"):
        chunker.get_text_splitter()


def test_markdown_splitter_no_config_params(mock_config_manager):
    """Test markdown splitter creation with minimal config."""
    mock_config = {"chunker": {"default_splitter": "markdown"}}
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, MarkdownHeaderTextSplitter)
    assert splitter.headers_to_split_on == [("###", 1), ("##", 2), ("#", 3)]


def test_recursive_splitter_default_params(mock_config_manager):
    """Test recursive splitter uses default parameters when not specified."""
    mock_config = {"chunker": {"default_splitter": "recursive"}}
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 1000  # Default value
    assert splitter._chunk_overlap == 200  # Default value


def test_recursive_splitter_custom_params(mock_config_manager):
    """Test recursive splitter with custom parameters."""
    mock_config = {
        "chunker": {
            "default_splitter": "recursive",
            "chunk_size": 400,
            "chunk_overlap": 50,
        }
    }
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 400
    assert splitter._chunk_overlap == 50


def test_chunker_config_precedence():
    """Test that passed config takes precedence over default config."""
    custom_config = {
        "chunker": {
            "default_splitter": "recursive",
            "chunk_size": 300,
            "chunk_overlap": 25,
        }
    }
    chunker = Chunker(config=custom_config)
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 300
    assert splitter._chunk_overlap == 25


def test_empty_config_raises_error(mock_config_manager):
    """Test that empty config raises KeyError."""
    mock_config = {}
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    with pytest.raises(KeyError):
        chunker.get_text_splitter()


def test_missing_splitter_type_raises_error(mock_config_manager):
    """Test that missing splitter type raises KeyError."""
    mock_config = {"chunker": {}}
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    with pytest.raises(KeyError):
        chunker.get_text_splitter()


def test_custom_splitter_type_validation():
    """Test that custom text splitter must be instance of TextSplitter."""
    with pytest.raises(TypeError):
        Chunker(text_splitter="not a splitter")


def test_markdown_splitter_headers(mock_config_manager):
    """Test that the markdown splitter is created with the correct headers."""
    mock_config = {"chunker": {"default_splitter": "markdown"}}
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, MarkdownHeaderTextSplitter)
    assert splitter.headers_to_split_on == [("###", 1), ("##", 2), ("#", 3)]


def test_text_splitter_not_recreated(mock_config_manager):
    """Test that the text splitter is not recreated if already set."""
    custom_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)
    chunker = Chunker(text_splitter=custom_splitter)
    splitter = chunker.get_text_splitter()
    assert splitter is custom_splitter
    assert splitter._chunk_size == 300
    assert splitter._chunk_overlap == 75

    # Test with a configured Chunker
    mock_config = {
        "chunker": {
            "default_splitter": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 100,
        }
    }
    mock_config_manager.return_value._config = mock_config
    chunker_from_config = Chunker()
    splitter_from_config = chunker_from_config.get_text_splitter()
    assert isinstance(splitter_from_config, RecursiveCharacterTextSplitter)
    assert splitter_from_config._chunk_size == 500
    assert splitter_from_config._chunk_overlap == 100

    # Test markdown splitter as well
    mock_config_markdown = {"chunker": {"default_splitter": "markdown"}}
    mock_config_manager.return_value._config = mock_config_markdown
    chunker_markdown = Chunker()
    splitter_markdown = chunker_markdown.get_text_splitter()
    assert isinstance(splitter_markdown, MarkdownHeaderTextSplitter)
    assert splitter_markdown.headers_to_split_on == [("###", 1), ("##", 2), ("#", 3)]


def test_custom_splitter_override():
    """Test that a custom splitter overrides the default."""
    custom_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunker = Chunker(text_splitter=custom_splitter)
    splitter = chunker.get_text_splitter()
    assert splitter is custom_splitter
    assert splitter._chunk_size == 200
    assert splitter._chunk_overlap == 50


def test_from_config_class_method(mock_config_manager):
    """Test the from_config class method."""
    mock_config = {
        "chunker": {
            "default_splitter": "recursive",
            "chunk_size": 700,
            "chunk_overlap": 150,
        }
    }
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker.from_config()
    assert isinstance(chunker, Chunker)
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 700
    assert splitter._chunk_overlap == 150
