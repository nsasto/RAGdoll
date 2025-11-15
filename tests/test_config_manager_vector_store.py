from pathlib import Path

from ragdoll.config.config_manager import ConfigManager


def _write_config(tmp_path, text: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_vector_store_config_supports_legacy_schema(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
vector_store:
  enabled: false
  store_type: faiss
  params:
    index_path: "./legacy.faiss"
""",
    )

    manager = ConfigManager(str(config_path))
    vector_config = manager.vector_store_config

    assert vector_config.enabled is False
    assert vector_config.store_type == "faiss"
    assert vector_config.params["index_path"] == "./legacy.faiss"


def test_vector_store_config_normalizes_multi_store_schema(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
vector_stores:
  enabled: true
  default_store: chroma
  stores:
    chroma:
      collection_name: "demo"
      params:
        persist_directory: "./data/chroma"
    faiss:
      distance_strategy: "cosine"
""",
    )

    manager = ConfigManager(str(config_path))
    vector_config = manager.vector_store_config

    assert vector_config.enabled is True
    assert vector_config.store_type == "chroma"
    assert vector_config.params["collection_name"] == "demo"
    assert vector_config.params["persist_directory"] == "./data/chroma"
