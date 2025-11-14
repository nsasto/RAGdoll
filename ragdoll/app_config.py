from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from ragdoll.config import Config

CONFIG_ENV_VAR = "RAGDOLL_CONFIG_PATH"

if TYPE_CHECKING:
    from ragdoll.cache.cache_manager import CacheManager
    from ragdoll.metrics.metrics_manager import MetricsManager


@dataclass
class AppConfig:
    """
    Container for shared configuration and runtime dependencies.

    The dataclass stores the validated :class:`Config` instance and lazily
    instantiates optional helpers (cache manager, metrics, prompt templates)
    when they are first requested. This keeps bootstrap lightweight while
    ensuring downstream components share the same singletons.
    """

    config: Config
    _cache_manager: Optional["CacheManager"] = field(default=None, init=False, repr=False)
    _metrics_manager: Optional["MetricsManager"] = field(default=None, init=False, repr=False)
    _prompt_templates: Optional[Dict[str, str]] = field(default=None, init=False, repr=False)

    def get_cache_manager(self) -> "CacheManager":
        """Return the shared CacheManager, creating it on first access."""

        if self._cache_manager is None:
            from ragdoll.cache.cache_manager import CacheManager

            self._cache_manager = CacheManager(app_config=self)
        return self._cache_manager

    def set_cache_manager(self, manager: "CacheManager") -> None:
        """Override the cache manager reference (useful for tests)."""

        self._cache_manager = manager

    def get_metrics_manager(self) -> "MetricsManager":
        """Return the shared MetricsManager, creating it on demand."""

        if self._metrics_manager is None:
            from ragdoll.metrics.metrics_manager import MetricsManager

            self._metrics_manager = MetricsManager()
        return self._metrics_manager

    def set_metrics_manager(self, manager: "MetricsManager") -> None:
        """Override the metrics manager reference."""

        self._metrics_manager = manager

    def get_prompt_templates(self) -> Dict[str, str]:
        """Fetch prompt templates from config once and cache the result."""

        if self._prompt_templates is None:
            self._prompt_templates = self.config.get_default_prompt_templates()
        return self._prompt_templates

    def set_prompt_templates(self, templates: Dict[str, str]) -> None:
        """Override the cached prompt templates."""

        self._prompt_templates = templates


def bootstrap_app(
    config_path: Optional[str] = None,
    *,
    config: Optional[Config] = None,
    overrides: Optional[Dict[str, Any]] = None,
    cache_manager: Optional["CacheManager"] = None,
    metrics_manager: Optional["MetricsManager"] = None,
    prompt_templates: Optional[Dict[str, str]] = None,
) -> AppConfig:
    """
    Create a fully-initialized :class:`AppConfig`.

    Args:
        config_path: Optional path to a YAML configuration file. When omitted,
            falls back to ``RAGDOLL_CONFIG_PATH`` or the package default.
        config: Pre-built :class:`Config` instance. Useful in tests where the
            caller already performed custom hydration.
        overrides: Optional dictionary merged into the raw configuration dict
            before typed models are materialized.
        cache_manager: Optional cache manager override.
        metrics_manager: Optional metrics manager override.
        prompt_templates: Preloaded prompt mapping.

    Returns:
        :class:`AppConfig` with shared dependencies.
    """

    resolved_path = config_path or os.environ.get(CONFIG_ENV_VAR)
    config_instance = config or Config(resolved_path)
    if overrides:
        config_instance._config.update(overrides)

    app_config = AppConfig(config=config_instance)
    if cache_manager:
        app_config.set_cache_manager(cache_manager)
    if metrics_manager:
        app_config.set_metrics_manager(metrics_manager)
    if prompt_templates:
        app_config.set_prompt_templates(prompt_templates)
    return app_config


__all__ = ["AppConfig", "bootstrap_app", "CONFIG_ENV_VAR"]
