import pytest
from pytest import fixture
from unittest.mock import patch, MagicMock
from ragdoll.embeddings.embeddings import RagdollEmbeddings


@fixture
def mock_config_manager():
    with patch("ragdoll.embeddings.embeddings.ConfigManager") as MockConfigManager:
        yield MockConfigManager

def test_initialization_with_config():
    config = {"embeddings": {"default_model": "openai", "openai": {"model": "test-model"}}}
    embeddings = RagdollEmbeddings(config=config)
    assert embeddings.config == config

def test_initialization_without_config(mock_config_manager):
    mock_config = {"embeddings": {"default_model": "openai"}}
    mock_config_manager.return_value._config = mock_config
    embeddings = RagdollEmbeddings()
    assert embeddings.config == mock_config

def test_from_config(mock_config_manager):
    mock_config = {"embeddings": {"default_model": "huggingface"}}
    mock_config_manager.return_value._config = mock_config
    embeddings = RagdollEmbeddings.from_config()
    assert embeddings.config == mock_config

def test_get_embeddings_model_pre_initialized(mock_config_manager):
    mock_config = {"embeddings": {"default_model": "openai"}}
    mock_config_manager.return_value._config = mock_config
    mock_model = MagicMock()
    embeddings = RagdollEmbeddings(embeddings_model=mock_model)
    assert embeddings.get_embeddings_model() == mock_model
            

def test_get_embeddings_model_openai_from_config(mock_config_manager):
    with patch("ragdoll.embeddings.embeddings.OpenAIEmbeddings") as MockOpenAIEmbeddings:
        mock_config = {"embeddings": {"default_model": "openai", "openai": {"model": "test-openai-model", "dimensions": 128}}}
        mock_config_manager.return_value._config = mock_config
        embeddings = RagdollEmbeddings()
        embeddings.get_embeddings_model()
        MockOpenAIEmbeddings.assert_called_once_with(model="test-openai-model", dimensions=128)

def test_get_embeddings_model_huggingface_from_config(mock_config_manager):
    with patch("ragdoll.embeddings.embeddings.HuggingFaceEmbeddings") as MockHuggingFaceEmbeddings:
        mock_config = {"embeddings": {"default_model": "huggingface", "huggingface": {"model_name": "test-hf-model"}}}
        mock_config_manager.return_value._config = mock_config
        embeddings = RagdollEmbeddings()
        embeddings.get_embeddings_model()
        MockHuggingFaceEmbeddings.assert_called_once_with(model_name="test-hf-model")

def test_get_embeddings_model_default(mock_config_manager):
    with patch("ragdoll.embeddings.embeddings.OpenAIEmbeddings") as MockOpenAIEmbeddings:
        mock_config = {"embeddings": {}}
        mock_config_manager.return_value._config = mock_config
        embeddings = RagdollEmbeddings()
        embeddings.get_embeddings_model()
        MockOpenAIEmbeddings.assert_called_once()

def test_get_embeddings_model_invalid_model_type(mock_config_manager):
    with patch("ragdoll.embeddings.embeddings.OpenAIEmbeddings") as MockOpenAIEmbeddings:
        mock_config = {"embeddings": {"default_model": "invalid"}}
        mock_config_manager.return_value._config = mock_config
        embeddings = RagdollEmbeddings()
        embeddings.get_embeddings_model()
        MockOpenAIEmbeddings.assert_called_once()

def test_get_embeddings_model_openai_no_params(mock_config_manager):
    with patch("ragdoll.embeddings.embeddings.OpenAIEmbeddings") as MockOpenAIEmbeddings:
        mock_config = {"embeddings": {"default_model": "openai"}}
        mock_config_manager.return_value._config = mock_config
        embeddings = RagdollEmbeddings()
        embeddings.get_embeddings_model()
        MockOpenAIEmbeddings.assert_called_once()

def test_get_embeddings_model_huggingface_no_params(mock_config_manager):
    with patch("ragdoll.embeddings.embeddings.HuggingFaceEmbeddings") as MockHuggingFaceEmbeddings:
        mock_config = {"embeddings": {"default_model": "huggingface"}}
        mock_config_manager.return_value._config = mock_config
        embeddings = RagdollEmbeddings()
        embeddings.get_embeddings_model()
        MockHuggingFaceEmbeddings.assert_called_once()

def test_get_embeddings_model_default_openai_params(mock_config_manager):
    with patch("ragdoll.embeddings.embeddings.OpenAIEmbeddings") as MockOpenAIEmbeddings:
        mock_config = {"embeddings": {"default_model": "openai", "openai": {}}}
        mock_config_manager.return_value._config = mock_config
        embeddings = RagdollEmbeddings()
        embeddings.get_embeddings_model()
        MockOpenAIEmbeddings.assert_called_once_with(model="text-embedding-3-large")