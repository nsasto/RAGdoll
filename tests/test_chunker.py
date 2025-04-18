import pytest
from unittest import mock
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TextSplitter,
)
from ragdoll.config.config_manager import ConfigManager

from ragdoll.chunkers.chunker import Chunker  # Import the Chunker class


@pytest.fixture
def mock_config_manager():
    with mock.patch(
        "ragdoll.chunkers.chunker.ConfigManager"
    ) as MockConfigManager:
        yield MockConfigManager

def test_default_splitter_from_config(mock_config_manager):
    """Test that the default splitter is loaded from the config."""
    mock_config = {
        "chunker": {"default_splitter": "recursive", "chunk_size": 500, "chunk_overlap": 100}
    }
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 500
    assert splitter._chunk_overlap == 100

    mock_config = {"chunker": {"default_splitter": "markdown"}}
    mock_config_manager.return_value._config = mock_config
    chunker = Chunker()
    splitter = chunker.get_text_splitter()
    assert isinstance(splitter, MarkdownHeaderTextSplitter)

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