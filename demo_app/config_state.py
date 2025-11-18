from __future__ import annotations

import os
from pathlib import Path

import ragdoll.config.config_manager as config_manager
from ragdoll import settings
from ragdoll.app_config import AppConfig, bootstrap_app, CONFIG_ENV_VAR
from ragdoll.config import Config

from . import state

# Global AppConfig instance shared across all endpoints
_global_app_config: AppConfig | None = None

DEFAULT_CONFIG_PATH = Path(config_manager.__file__).resolve().parent / "default_config.yaml"
CUSTOM_CONFIG_PATH = state.STATE_DIR / "custom_config.yaml"


def _maybe_set_env_override() -> None:
    """Ensure the override path is used if it already exists."""

    if CUSTOM_CONFIG_PATH.exists() and not os.environ.get(CONFIG_ENV_VAR):
        os.environ[CONFIG_ENV_VAR] = str(CUSTOM_CONFIG_PATH.resolve())


_maybe_set_env_override()


def _active_config_path() -> Path:
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        resolved = Path(env_path)
        if resolved.exists():
            return resolved.resolve()
    if CUSTOM_CONFIG_PATH.exists():
        return CUSTOM_CONFIG_PATH.resolve()
    return DEFAULT_CONFIG_PATH


def current_config_yaml() -> str:
    """Return the YAML text for the currently active configuration."""

    return _active_config_path().read_text(encoding="utf-8")


def default_config_yaml() -> str:
    """Return the bundled default YAML configuration."""

    return DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def current_config_source() -> str:
    """Human-readable label describing where the current config came from."""

    active_path = _active_config_path()
    if active_path == DEFAULT_CONFIG_PATH:
        return f"Default ({DEFAULT_CONFIG_PATH.name})"
    return f"Custom override ({active_path.name})"


def apply_config_yaml(yaml_text: str) -> None:
    """
    Validate and persist a new configuration, then refresh the shared AppConfig.
    """
    global _global_app_config

    state.ensure_state_dirs()
    temp_path = CUSTOM_CONFIG_PATH.with_suffix(".tmp")
    temp_path.write_text(yaml_text, encoding="utf-8")
    try:
        Config(str(temp_path))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    temp_path.replace(CUSTOM_CONFIG_PATH)
    os.environ[CONFIG_ENV_VAR] = str(CUSTOM_CONFIG_PATH.resolve())
    settings.get_app.cache_clear()
    
    # Reload the global app config with the new configuration
    _global_app_config = bootstrap_app(config_path=str(CUSTOM_CONFIG_PATH))


def get_app_config() -> AppConfig:
    """
    Get the global AppConfig instance. Initializes it if not already created.
    """
    global _global_app_config
    if _global_app_config is None:
        config_path = _active_config_path()
        _global_app_config = bootstrap_app(config_path=str(config_path))
    return _global_app_config


def initialize_app_config() -> AppConfig:
    """
    Initialize the global AppConfig on application startup.
    """
    global _global_app_config
    config_path = _active_config_path()
    _global_app_config = bootstrap_app(config_path=str(config_path))
    return _global_app_config
