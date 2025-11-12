from unittest.mock import MagicMock, patch

import pytest

from ragdoll.embeddings import get_embedding_model


@pytest.fixture
def config_manager():
    manager = MagicMock()
    manager._config = {}
    return manager


def test_get_embedding_model_uses_models_section(config_manager):
    config_manager._config = {
        "embeddings": {
            "default_model": "custom-openai",
            "models": {
                "custom-openai": {
                    "provider": "openai",
                    "model": "text-embedding-3-large",
                    "dimensions": 1024,
                }
            },
        }
    }
    with patch("ragdoll.embeddings._initialize_model_by_provider") as mock_provider:
        sentinel = object()
        mock_provider.return_value = sentinel
        result = get_embedding_model(config_manager=config_manager)
        assert result is sentinel
        mock_provider.assert_called_once_with(
            "openai",
            {"model": "text-embedding-3-large", "dimensions": 1024},
        )


def test_get_embedding_model_falls_back_to_openai_section(config_manager):
    config_manager._config = {
        "embeddings": {
            "default_model": "openai",
            "openai": {"model": "text-embedding-3-small", "dimensions": 256},
        }
    }
    with patch("ragdoll.embeddings._create_openai_embeddings") as mock_openai:
        sentinel = object()
        mock_openai.return_value = sentinel
        result = get_embedding_model(config_manager=config_manager)
        assert result is sentinel
        mock_openai.assert_called_once_with(
            {"model": "text-embedding-3-small", "dimensions": 256}
        )


def test_get_embedding_model_falls_back_to_huggingface_section(config_manager):
    config_manager._config = {
        "embeddings": {
            "default_model": "huggingface",
            "huggingface": {"model_name": "sentence-transformers/all-mpnet-base-v2"},
        }
    }
    with patch("ragdoll.embeddings._create_huggingface_embeddings") as mock_hf:
        sentinel = object()
        mock_hf.return_value = sentinel
        result = get_embedding_model(config_manager=config_manager)
        assert result is sentinel
        mock_hf.assert_called_once_with(
            {"model_name": "sentence-transformers/all-mpnet-base-v2"}
        )


def test_get_embedding_model_accepts_direct_provider():
    with patch("ragdoll.embeddings._initialize_model_by_provider") as mock_provider:
        sentinel = object()
        mock_provider.return_value = sentinel
        result = get_embedding_model(provider="fake", size=42)
        assert result is sentinel
        mock_provider.assert_called_once_with("fake", {"size": 42})


def test_get_embedding_model_inferrs_provider_from_name(config_manager):
    config_manager._config = {"embeddings": {}}
    with patch("ragdoll.embeddings._create_openai_embeddings") as mock_openai:
        sentinel = object()
        mock_openai.return_value = sentinel
        result = get_embedding_model(
            model_name="text-embedding-3-large",
            config_manager=config_manager,
            temperature=0.3,
        )
        assert result is sentinel
        mock_openai.assert_called_once_with({"temperature": 0.3})


def test_get_embedding_model_inferrs_huggingface_from_path(config_manager):
    config_manager._config = {"embeddings": {}}
    with patch("ragdoll.embeddings._create_huggingface_embeddings") as mock_hf:
        sentinel = object()
        mock_hf.return_value = sentinel
        result = get_embedding_model(
            model_name="sentence-transformers/all-mpnet-base-v2",
            config_manager=config_manager,
            normalize=True,
        )
        assert result is sentinel
        mock_hf.assert_called_once_with(
            {"model_name": "sentence-transformers/all-mpnet-base-v2", "normalize": True}
        )
