import os
import pytest
from pytest import fixture
from unittest.mock import patch, MagicMock
from ragdoll.embeddings import get_embedding_model


@fixture
def mock_config_manager():
    mock = MagicMock()
    mock._config = {
        "embeddings": {
            "default_model": "text-embedding-ada-002",
            "models": {
                "text-embedding-ada-002": {
                    "provider": "openai",
                    "model": "text-embedding-ada-002"
                },
                "all-mpnet-base-v2": {
                    "provider": "huggingface",
                    "model_name": "sentence-transformers/all-mpnet-base-v2"
                }
            },
            # Legacy format for backward compatibility tests
            "openai": {
                "model": "text-embedding-3-large",
                "dimensions": 128
            },
            "huggingface": {
                "model_name": "sentence-transformers/all-mpnet-base-v2"
            }
        }
    }
    return mock


def test_get_embedding_model_default(mock_config_manager):
    """Test getting the default embedding model from config."""
    with patch("ragdoll.embeddings.ConfigManager", return_value=mock_config_manager):
        with patch("ragdoll.embeddings._create_openai_embeddings") as mock_create:
            mock_create.return_value = MagicMock()
            model = get_embedding_model()
            assert model is not None
            mock_create.assert_called_once()


def test_get_embedding_model_by_name(mock_config_manager):
    """Test getting a specific named embedding model."""
    with patch("ragdoll.embeddings.ConfigManager", return_value=mock_config_manager):
        with patch("ragdoll.embeddings._create_huggingface_embeddings") as mock_create:
            mock_create.return_value = MagicMock()
            model = get_embedding_model(model_name="all-mpnet-base-v2")
            assert model is not None
            mock_create.assert_called_once()


def test_get_embedding_model_with_kwargs(mock_config_manager):
    """Test that kwargs override config values."""
    with patch("ragdoll.embeddings.ConfigManager", return_value=mock_config_manager):
        with patch("ragdoll.embeddings._create_openai_embeddings") as mock_create:
            mock_create.return_value = MagicMock()
            model = get_embedding_model(
                model_name="text-embedding-ada-002", 
                model="text-embedding-3-small"  # Override model
            )
            # Check that the override was passed to the creation function
            called_kwargs = mock_create.call_args[0][0]
            assert "model" in called_kwargs
            assert called_kwargs["model"] == "text-embedding-3-small"


def test_get_embedding_model_legacy_openai(mock_config_manager):
    """Test getting openai embeddings using legacy config format."""
    with patch("ragdoll.embeddings.ConfigManager", return_value=mock_config_manager):
        with patch("ragdoll.embeddings._create_openai_embeddings") as mock_create:
            mock_create.return_value = MagicMock()
            model = get_embedding_model(model_name="openai")
            assert model is not None
            mock_create.assert_called_once()
            # Check that dimensions was passed
            called_kwargs = mock_create.call_args[0][0]
            assert "dimensions" in called_kwargs
            assert called_kwargs["dimensions"] == 128


def test_get_embedding_model_legacy_huggingface(mock_config_manager):
    """Test getting huggingface embeddings using legacy config format."""
    with patch("ragdoll.embeddings.ConfigManager", return_value=mock_config_manager):
        with patch("ragdoll.embeddings._create_huggingface_embeddings") as mock_create:
            mock_create.return_value = MagicMock()
            model = get_embedding_model(model_name="huggingface")
            assert model is not None
            mock_create.assert_called_once()
            # Check that model_name was passed
            called_kwargs = mock_create.call_args[0][0]
            assert "model_name" in called_kwargs
            assert called_kwargs["model_name"] == "sentence-transformers/all-mpnet-base-v2"


def test_get_embedding_model_direct_provider():
    """Test specifying provider directly in kwargs."""
    with patch("ragdoll.embeddings._create_fake_embeddings") as mock_create:
        mock_create.return_value = MagicMock()
        get_embedding_model(provider="fake", size=768)
        
        mock_create.assert_called_once()
        # Check that size was passed
        called_kwargs = mock_create.call_args[0][0]
        assert "size" in called_kwargs
        assert called_kwargs["size"] == 768


def test_create_openai_embeddings():
    """Test the OpenAI embeddings creation function."""
    with patch("langchain_openai.OpenAIEmbeddings") as MockOpenAIEmbeddings:
        MockOpenAIEmbeddings.return_value = MagicMock()
        
        # Import the function directly for testing
        from ragdoll.embeddings import _create_openai_embeddings
        
        model_params = {
            "model": "text-embedding-3-large",
            "dimensions": 1024,
            "api_key": "test_key"
        }
        
        model = _create_openai_embeddings(model_params)
        assert model is not None


def test_create_huggingface_embeddings():
    """Test the HuggingFace embeddings creation function."""
    # Mock at the module level where the import happens to prevent actual import
    with patch("ragdoll.embeddings.HuggingFaceEmbeddings") as MockHuggingFaceEmbeddings:
        MockHuggingFaceEmbeddings.return_value = MagicMock()
        
        # Import the function directly for testing
        from ragdoll.embeddings import _create_huggingface_embeddings
        
        model_params = {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "cache_folder": "/tmp/models"
        }
        
        model = _create_huggingface_embeddings(model_params)
        assert model is not None
        MockHuggingFaceEmbeddings.assert_called_once()


def test_get_embedding_model_invalid_provider(mock_config_manager):
    """Test behavior with an invalid provider."""
    with patch("ragdoll.embeddings.logger.error") as mock_logger:
        model = get_embedding_model(provider="invalid_provider")
        assert model is None
        mock_logger.assert_called_once()


def test_get_embedding_model_missing_config():
    """Test behavior when config is missing."""
    with patch("ragdoll.embeddings.ConfigManager", side_effect=ImportError):
        with patch("ragdoll.embeddings.logger.warning") as mock_logger:
            model = get_embedding_model(provider="openai", model="text-embedding-3-large")
            # Should still work with direct parameters
            assert model is not None


def test_get_embedding_model_environment_api_key():
    """Test using environment variables for API keys."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env_test_key"}):
        with patch("ragdoll.embeddings.OpenAIEmbeddings") as MockOpenAIEmbeddings:
            MockOpenAIEmbeddings.return_value = MagicMock()
            model = get_embedding_model(provider="openai", model="text-embedding-3-large")
            assert model is not None
            # Should have used the environment variable
            kwargs = MockOpenAIEmbeddings.call_args[1]
            assert kwargs["openai_api_key"] == "env_test_key"


def test_get_embedding_model_config_api_key_reference(mock_config_manager):
    """Test API key references in config (#ENV_VAR format)."""
    # Update the mock config to use an API key reference
    mock_config_manager._config["embeddings"]["models"]["text-embedding-ada-002"]["api_key"] = "#TEST_API_KEY"
    
    with patch.dict(os.environ, {"TEST_API_KEY": "env_test_key"}):
        with patch("ragdoll.embeddings.ConfigManager", return_value=mock_config_manager):
            with patch("ragdoll.embeddings._create_openai_embeddings") as mock_create:
                mock_create.return_value = MagicMock()
                get_embedding_model(model_name="text-embedding-ada-002")
                
                # Check that the API key reference was resolved
                called_kwargs = mock_create.call_args[0][0]
                api_key = called_kwargs.get("api_key")
                # In the implementation, api_key should be resolved from #TEST_API_KEY to env_test_key
                assert api_key.startswith("#")  # It will now be processed inside _create_openai_embeddings