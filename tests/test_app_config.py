import tempfile
from pathlib import Path

from ragdoll.app_config import AppConfig, bootstrap_app
from ragdoll.cache.cache_manager import CacheManager
from ragdoll.config import Config


def test_bootstrap_app_overrides_apply_to_config():
    app = bootstrap_app(
        overrides={
            "ingestion": {
                "batch_size": 2,
            }
        }
    )

    assert app.config._config["ingestion"]["batch_size"] == 2


def test_app_config_cache_manager_override():
    base_config = Config(None)
    app = AppConfig(config=base_config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        custom_cache = CacheManager(cache_dir=str(cache_dir), ttl_seconds=1)
        app.set_cache_manager(custom_cache)
        assert app.get_cache_manager() is custom_cache
