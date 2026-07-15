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


def test_scaled_runtime_configuration_is_typed(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
execution:
  adapter: celery
  broker_url: os.environ/CELERY_BROKER_URL
job_store:
  adapter: postgres
  dsn: os.environ/RAGDOLL_POSTGRES_DSN
corpus_index:
  adapter: vector
  state_adapter: postgres
  dsn: os.environ/RAGDOLL_POSTGRES_DSN
vector_store:
  enabled: true
  store_type: qdrant
  params:
    collection_name: docs
""",
    )

    manager = ConfigManager(str(config_path))

    assert manager.execution_config.adapter == "celery"
    assert manager.job_store_runtime_config.adapter == "postgres"
    assert manager.corpus_index_runtime_config.state_adapter == "postgres"
    assert manager.vector_store_config.store_type == "qdrant"
