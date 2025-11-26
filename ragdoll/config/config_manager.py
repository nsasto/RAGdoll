import json
import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Type

import yaml

from ragdoll.config.base_config import (
    CacheConfig,
    EmbeddingsConfig,
    EntityExtractionConfig,
    GraphDatabaseConfig,
    IngestionConfig,
    LLMPromptsConfig,
    LoadersConfig,
    MonitorConfig,
    VectorStoreConfig,
)
from ragdoll.prompts import get_prompt, list_prompts


class ConfigManager:
    """Manages configuration loading and validation."""

    logger = logging.getLogger(__name__)

    def __init__(self, config_path: str | None = None):
        """
        Initialize the config manager.

        Args:
            config_path: Optional path to the configuration file. When omitted,
                the default configuration bundled with the package is used.
        """

        if not config_path:
            config_path = Path(__file__).parent / "default_config.yaml"

        self.config_path = config_path
        self._config = self._load_config()
        self._log_loaded_config()
        self._initialize_prompts()
        self._ensure_entity_extraction_defaults()
        self._ensure_retriever_defaults()
        self._normalize_vector_store_config()

    def _log_loaded_config(self) -> None:
        """Emit a debug log with the loaded configuration."""

        try:
            pretty_config = json.dumps(self._config, indent=2)
        except TypeError:
            pretty_config = str(self._config)
        self.logger.debug("Loaded config:\n%s", pretty_config)

    def _initialize_prompts(self) -> None:
        """Load prompt names available in the prompts package."""

        try:
            self.available_prompts = set(list_prompts())
            self.logger.debug("Available prompts: %s", self.available_prompts)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Could not load available prompts: %s", exc)
            self.available_prompts = set()

    def _ensure_entity_extraction_defaults(self) -> None:
        """
        Ensure useful defaults exist for entity extraction relationship metadata.

        The function operates directly on the underlying config dictionary so the
        defaults are visible to both the Pydantic models and consumer code.
        """

        entity_config = self._config.setdefault("entity_extraction", {})
        entity_config.setdefault("relationship_types", [])
        entity_config.setdefault(
            "relationship_type_mapping",
            {
                "works for": "WORKS_FOR",
                "is a": "IS_A",
                "is an": "IS_A",
                "located in": "LOCATED_IN",
                "located at": "LOCATED_IN",
                "born in": "BORN_IN",
                "lives in": "LOCATED_IN",
                "married to": "SPOUSE_OF",
                "spouse of": "SPOUSE_OF",
                "parent of": "PARENT_OF",
                "child of": "PARENT_OF",
                "works with": "AFFILIATED_WITH",
            },
        )
        entity_config.setdefault(
            "graph_database_config", {"output_file": "graph_output.json"}
        )
        parsing_defaults = entity_config.setdefault("relationship_parsing", {})
        parsing_defaults.setdefault("preferred_format", "auto")
        parsing_defaults.setdefault("parser_class", None)
        parsing_defaults.setdefault("schema", None)
        parsing_defaults.setdefault("parser_kwargs", {})

        prompt_defaults = entity_config.setdefault("relationship_prompts", {})
        prompt_defaults.setdefault("default", "relationship_extraction")
        prompt_defaults.setdefault("providers", {})

    def _ensure_retriever_defaults(self) -> None:
        """
        Ensure defaults exist for the unified retriever configuration.

        This creates a comprehensive retriever configuration with:
        - Vector retrieval settings
        - Graph retrieval settings
        - PageRank retrieval settings
        - Hybrid combination strategy
        """
        retriever_config = self._config.setdefault("retriever", {})

        # Vector retrieval defaults
        vector_config = retriever_config.setdefault("vector", {})
        vector_config.setdefault("enabled", True)
        vector_config.setdefault("top_k", 3)
        vector_config.setdefault("search_type", "similarity")
        vector_config.setdefault("search_kwargs", {})

        # Graph retrieval defaults
        graph_config = retriever_config.setdefault("graph", {})
        graph_config.setdefault("enabled", True)
        graph_config.setdefault("backend", "networkx")
        graph_config.setdefault("top_k", 5)
        graph_config.setdefault("max_hops", 2)
        graph_config.setdefault("include_edges", True)
        graph_config.setdefault("traversal_strategy", "bfs")
        graph_config.setdefault("min_score", 0.0)

        # PageRank retrieval defaults
        pagerank_config = retriever_config.setdefault("pagerank", {})
        pagerank_config.setdefault("enabled", False)
        pagerank_config.setdefault("top_k", 5)
        pagerank_config.setdefault("max_nodes", 200)
        pagerank_config.setdefault("max_hops", 3)
        pagerank_config.setdefault("seed_strategy", "embedding")
        pagerank_config.setdefault("num_seed_chunks", 5)
        pagerank_config.setdefault("damping_factor", 0.15)
        pagerank_config.setdefault("max_iter", 50)
        pagerank_config.setdefault("tol", 1e-6)
        pagerank_config.setdefault(
            "allowed_node_types", ["entity", "event", "document"]
        )
        pagerank_config.setdefault("min_score", 0.0)
        pagerank_config.setdefault("dedup_on_vector_id", True)
        pagerank_config.setdefault("include_edges", True)
        pagerank_config.setdefault("enable_fallback", True)
        pagerank_config.setdefault("log_fallback_warnings", True)

        # Hybrid combination defaults
        hybrid_config = retriever_config.setdefault("hybrid", {})
        hybrid_config.setdefault("mode", "concat")
        hybrid_config.setdefault("vector_weight", 0.6)
        hybrid_config.setdefault("graph_weight", 0.4)
        hybrid_config.setdefault("deduplicate", True)

    def _normalize_vector_store_config(self) -> None:
        """
        Support both legacy ``vector_store`` and the newer ``vector_stores`` schema.

        The new format lets users define multiple backends and select a default. We
        normalize it here so downstream code can continue to reference the legacy
        ``vector_store`` block without caring which syntax the user chose.
        """

        if "vector_store" in self._config:
            return

        block = self._config.get("vector_stores")
        normalized = self._coerce_vector_store_block(block)
        if normalized:
            self._config["vector_store"] = normalized
        elif block is not None:
            self.logger.warning(
                "Unable to normalize 'vector_stores' configuration. Falling back to defaults."
            )

    def _coerce_vector_store_block(
        self, block: Dict[str, Any] | None
    ) -> Dict[str, Any] | None:
        if not isinstance(block, dict):
            return None

        stores = block.get("stores") or {}
        default_store = block.get("default_store")
        if not default_store and stores:
            default_store = next(iter(stores))

        store_entry = stores.get(default_store) if default_store else None
        if store_entry is None:
            return None

        enabled = block.get("enabled")
        params: Dict[str, Any] = {}
        store_type = default_store

        if isinstance(store_entry, dict):
            store_type = store_entry.get("store_type", store_type)
            if "enabled" in store_entry:
                enabled = store_entry["enabled"]

            nested_params = store_entry.get("params")
            values = {
                key: value
                for key, value in store_entry.items()
                if key not in {"store_type", "enabled", "params"}
            }
            params.update(values)
            if isinstance(nested_params, dict):
                params.update(nested_params)
        else:
            store_type = store_entry or store_type

        top_level_params = block.get("params")
        if isinstance(top_level_params, dict):
            params.update(top_level_params)

        return {
            "enabled": True if enabled is None else bool(enabled),
            "store_type": store_type,
            "params": params,
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML and environment overrides."""

        if not os.path.exists(self.config_path):
            self.logger.debug(
                "Config file not found at %s. Using default values.", self.config_path
            )
            return {}

        self.logger.debug("Loading config file from %s.", self.config_path)
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        self._hydrate_embedding_defaults(config)
        return config

    def _hydrate_embedding_defaults(self, config: Dict[str, Any]) -> None:
        """Apply environment-driven defaults to the embeddings block."""

        embeddings = config.get("embeddings")
        if not isinstance(embeddings, dict):
            return

        # Backfill OpenAI API keys when omitted.
        openai_section = embeddings.get("openai")
        if isinstance(openai_section, dict):
            api_key = openai_section.get("api_key") or openai_section.get(
                "openai_api_key"
            )
            if not api_key:
                openai_section["api_key"] = os.environ.get("OPENAI_API_KEY")

        # Provide a default model alias for the modern schema.
        if "default_model" not in embeddings:
            if isinstance(embeddings.get("models"), dict) and embeddings["models"]:
                embeddings["default_model"] = next(iter(embeddings["models"]))
            else:
                embeddings["default_model"] = "text-embedding-3-large"

        # Provide a default client for older client-based schema.
        if "default_client" not in embeddings:
            embeddings["default_client"] = "openai"

        # Ensure a clients block exists for backwards compatibility.
        if "clients" not in embeddings:
            clients: Dict[str, Any] = {}
            for provider in ("openai", "huggingface"):
                provider_cfg = embeddings.get(provider)
                if isinstance(provider_cfg, dict):
                    clients[provider] = provider_cfg
            if clients:
                embeddings["clients"] = clients

    @property
    def embeddings_config(self) -> EmbeddingsConfig:
        """Typed embeddings configuration."""

        return EmbeddingsConfig.model_validate(self._config.get("embeddings", {}))

    @property
    def cache_config(self) -> CacheConfig:
        """Typed cache configuration."""

        return CacheConfig.model_validate(self._config.get("cache", {}))

    @property
    def monitor_config(self) -> MonitorConfig:
        """Typed monitor configuration."""

        return MonitorConfig.model_validate(self._config.get("monitor", {}))

    @property
    def vector_store_config(self) -> VectorStoreConfig:
        """Typed vector store configuration."""

        return VectorStoreConfig.model_validate(self._config.get("vector_store", {}))

    @property
    def ingestion_config(self) -> IngestionConfig:
        """Typed ingestion configuration."""

        return IngestionConfig.model_validate(self._config.get("ingestion", {}))

    @property
    def entity_extraction_config(self) -> EntityExtractionConfig:
        """Typed entity extraction configuration."""

        return EntityExtractionConfig.model_validate(
            self._config.get("entity_extraction", {})
        )

    @property
    def graph_database_config(self) -> GraphDatabaseConfig:
        """Typed graph database configuration."""

        entity_section = self._config.get("entity_extraction", {})
        graph_config = entity_section.get("graph_database_config", {})
        return GraphDatabaseConfig.model_validate(graph_config)

    @property
    def llm_prompts_config(self) -> LLMPromptsConfig:
        """Typed LLM prompts configuration."""

        return LLMPromptsConfig.model_validate(self._config.get("llm_prompts", {}))

    def get_loader_mapping(self) -> Dict[str, Type | str]:
        """
        Get the loader mapping with deferred imports.

        Returns:
            Dictionary mapping file extensions to loader classes or import paths.
        """

        loaders_config: LoadersConfig | None = self.ingestion_config.loaders
        result: Dict[str, Type | str] = {}
        file_mappings = getattr(loaders_config, "file_mappings", None)
        self.logger.info(
            "Raw file mappings from config:\n%s",
            json.dumps(file_mappings or {}, indent=2),
        )

        if file_mappings:
            for ext, class_path in file_mappings.items():
                result[ext] = class_path
                self.logger.debug(
                    "Registered loader path for extension '%s': %s", ext, class_path
                )
        else:
            self.logger.warning("No file extension loaders were registered.")

        return result

    def get_source_loader_mapping(self) -> Dict[str, Type]:
        """
        Get the source loader mapping with imported classes.

        Returns:
            Dictionary mapping source types to loader classes.
        """

        source_loaders: Dict[str, Type] = {}
        loaders_config = self.ingestion_config.loaders
        file_mappings = getattr(loaders_config, "file_mappings", None)

        if file_mappings:
            for ext, class_path in file_mappings.items():
                if ext in source_loaders:
                    continue
                try:
                    module_path, class_name = class_path.rsplit(".", 1)
                    module = import_module(module_path)
                    loader_class = getattr(module, class_name)
                    source_loaders[ext] = loader_class
                    self.logger.debug(
                        "Loaded source loader for '%s' via %s",
                        ext,
                        class_path,
                    )
                    try:
                        from ragdoll.ingestion import register_loader_class

                        norm_ext = (
                            ext.lstrip(".").lower() if isinstance(ext, str) else ext
                        )
                        register_loader_class(norm_ext, loader_class)
                    except Exception:
                        pass
                except (ImportError, AttributeError, ValueError) as exc:
                    self.logger.warning(
                        "Error loading loader for source %s: %s. This source type will "
                        "not be supported.",
                        ext,
                        exc,
                    )

        # Backward compatibility with custom arxiv retriever key.
        arxiv_retriever = getattr(loaders_config, "arxiv_retriever", None)
        if arxiv_retriever and "arxiv" not in source_loaders:
            try:
                module_path, class_name = arxiv_retriever.rsplit(".", 1)
                module = import_module(module_path)
                loader_class = getattr(module, class_name)
                source_loaders["arxiv"] = loader_class
                self.logger.debug("Loaded loader for arxiv source: %s", arxiv_retriever)
            except (ImportError, AttributeError, ValueError) as exc:
                self.logger.warning(
                    "Error loading arxiv retriever %s: %s. This source type will not be supported.",
                    arxiv_retriever,
                    exc,
                )

        self.logger.info("Loaded %s source loaders.", len(source_loaders))
        return source_loaders

    def get_default_prompt_templates(self) -> Dict[str, str]:
        """
        Build a mapping of prompt keys to template content.

        Returns:
            Dictionary keyed by config identifiers with file contents as values.
        """

        prompt_templates: Dict[str, str] = {}
        prompt_mapping = self.llm_prompts_config.model_dump()

        for config_key, prompt_name in prompt_mapping.items():
            try:
                prompt_template = get_prompt(prompt_name)
                if prompt_template:
                    prompt_templates[config_key] = prompt_template
                    self.logger.debug(
                        "Loaded prompt template '%s' as '%s'.",
                        prompt_name,
                        config_key,
                    )
                else:
                    self.logger.warning(
                        "Prompt template %s not found for %s.",
                        prompt_name,
                        config_key,
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(
                    "Error loading prompt template %s for %s: %s",
                    prompt_name,
                    config_key,
                    exc,
                )

        return prompt_templates

    def print_graph_creation_pipeline(self, config: Dict[str, Any]) -> str:
        """
        Generates a formatted string describing the graph creation pipeline configuration.

        Args:
            config: Dictionary containing configuration parameters.

        Returns:
            A formatted string describing the graph creation pipeline.
        """

        log_lines = ["Graph creation pipeline:"]
        step_number = 1

        ee_methods = config.get("entity_extraction_methods", [])
        ee_chunking = config.get("chunking_strategy", "none")
        coref_method = config.get("coreference_resolution_method", "none")
        log_lines.append(
            f"\t{step_number}. Coreference Resolution: method='{coref_method}'"
        )
        step_number += 1

        entity_line = (
            f"\t{step_number}. Entity Extraction: chunking_strategy='{ee_chunking}', "
            f"methods={ee_methods}"
        )
        if "ner" in ee_methods:
            entity_line += f", ner model={config.get('spacy_model', 'en_core_web_sm')}"
        log_lines.append(entity_line)
        step_number += 1

        linking_enabled = config.get("entity_linking_enabled", False)
        linking_method = config.get("entity_linking_method", "none")
        log_lines.append(
            f"\t{step_number}. Entity Linking: enabled={linking_enabled}, "
            f"method='{linking_method}'"
        )
        step_number += 1

        relation_method = config.get("relationship_extraction_method", "none")
        log_lines.append(
            f"\t{step_number}. Relationship Extraction: method='{relation_method}'"
        )
        step_number += 1

        gleaning_enabled = config.get("gleaning_enabled", False)
        max_gleaning = (
            config.get("max_gleaning_steps", "none") if gleaning_enabled else "none"
        )
        log_lines.append(
            f"\t{step_number}. Gleaning: enabled={gleaning_enabled}, max_steps={max_gleaning}"
        )
        step_number += 1

        postprocessing = config.get("postprocessing_steps", [])
        log_lines.append(
            f"\t{step_number}. Postprocessing: steps="
            f"{postprocessing if postprocessing else 'none'}"
        )
        log_lines.append("")

        entity_types = config.get("entity_types", [])
        if entity_types:
            lines = ["Configured entity_types:"]
            for i in range(0, len(entity_types), 5):
                lines.append("  " + ", ".join(entity_types[i : i + 5]))
            log_lines.append("\t".join(lines))
        else:
            log_lines.append("No entity_types configured.")

        relationship_types = config.get("relationship_types", [])
        if relationship_types:
            lines = ["Configured relationship_types:"]
            for i in range(0, len(relationship_types), 5):
                lines.append("  " + ", ".join(relationship_types[i : i + 5]))
            log_lines.append("\t".join(lines))
        else:
            log_lines.append("No relationship_types configured.")

        log_lines.append("")  # Final newline
        return "\n".join(log_lines)
