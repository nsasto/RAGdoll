"""Tools related to web search and query generation."""

from __future__ import annotations

import ast
import logging
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from langchain_core.llms import LLM
from langchain_core.tools import BaseTool
from langchain_google_community import GoogleSearchAPIWrapper
from pydantic import BaseModel, Field

from ragdoll import settings
from ragdoll.app_config import AppConfig
from ragdoll.config import Config
from ragdoll.llms.callers import BaseLLMCaller, LangChainLLMCaller, call_llm_sync
from ragdoll.prompts import get_prompt


class SearchToolsInput(BaseModel):
    query: str = Field(description="The search query string.")
    num_results: Optional[int] = Field(
        default=3,
        description="The number of search results to return. Defaults to 3.",
    )


class SearchInternetTool(BaseTool):
    name = "search_internet"
    description = (
        "Use Google Custom Search via langchain-google-community to retrieve documents. "
        "Skips obvious noise such as YouTube results."
    )
    args_schema = SearchToolsInput

    def __init__(
        self,
        *,
        google_search: Optional[GoogleSearchAPIWrapper] = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.google")
        self.logger.setLevel(log_level)
        self._search = google_search or GoogleSearchAPIWrapper()

    def _run(self, query: str, num_results: int = 3) -> List[Dict]:
        """Synchronously run a Google search."""
        self.logger.info("Searching the web for: %s", query)
        results = self._search.results(query, num_results)
        if not results:
            return []

        filtered: List[Dict] = []
        for result in results:
            if "youtube.com" in result.get("link", ""):
                continue
            filtered.append(
                {
                    "title": result.get("title"),
                    "href": result.get("link"),
                    "snippet": result.get("snippet"),
                }
            )
        return filtered

    async def _arun(self, *args, **kwargs):  # pragma: no cover - sync only
        raise NotImplementedError("SearchInternetTool does not implement async execution.")


class SuggestedSearchTermsInput(BaseModel):
    query: str = Field(description="The query to generate suggested search terms for.")
    num_suggestions: Optional[int] = Field(
        default=3,
        description="Number of suggested search terms to return. Defaults to 3.",
    )


class SuggestedSearchTermsTool(BaseTool):
    name = "generate_suggested_search_terms"
    description = (
        "Generate related Google queries for a topic using the configured LLM and prompt templates."
    )
    args_schema = SuggestedSearchTermsInput

    DEFAULT_PROMPT_KEY = "search_queries"

    def __init__(
        self,
        llm: Optional[LLM] = None,
        *,
        llm_caller: Optional[BaseLLMCaller] = None,
        config_manager: Optional[Config] = None,
        app_config: Optional[AppConfig] = None,
        prompt_key: str = DEFAULT_PROMPT_KEY,
        log_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.suggestions")
        self.logger.setLevel(log_level)
        self.app_config = app_config
        self.llm_caller = self._resolve_llm_caller(llm=llm, llm_caller=llm_caller)
        self.prompt_template = self._resolve_prompt(
            prompt_key, config_manager=config_manager
        )

    def _run(self, query: str, num_suggestions: int = 3) -> List[str]:
        prompt = self.prompt_template.format(
            query=query.strip(),
            query_count=max(1, num_suggestions),
            current_date=datetime.utcnow().strftime("%Y-%m-%d"),
        )
        self.logger.debug("Generating search queries with prompt:\n%s", prompt)
        if not self.llm_caller:
            self.logger.warning("No LLM configured for SuggestedSearchTermsTool.")
            return []

        try:
            raw_output = call_llm_sync(self.llm_caller, prompt)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to generate search suggestions: %s", exc)
            return []
        suggestions = self._parse_suggestions(raw_output)

        # Deduplicate while preserving order and enforce requested length.
        ordered: List[str] = []
        for suggestion in suggestions:
            if suggestion and suggestion not in ordered:
                ordered.append(suggestion)
            if len(ordered) >= num_suggestions:
                break

        self.logger.info("Generated %s suggested queries.", len(ordered))
        return ordered

    async def _arun(self, *args, **kwargs):  # pragma: no cover - sync only
        raise NotImplementedError("SuggestedSearchTermsTool does not implement async execution.")

    def _resolve_prompt(
        self,
        prompt_key: str,
        config_manager: Optional[Config],
    ) -> str:
        manager = config_manager
        if manager is None:
            if self.app_config is not None:
                manager = self.app_config.config
            else:
                manager = settings.get_app().config

        if self.app_config is not None:
            templates = self.app_config.get_prompt_templates()
        else:
            templates = manager.get_default_prompt_templates()

        template = templates.get(prompt_key)
        if template:
            return template

        try:
            return get_prompt(prompt_key)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Prompt '{prompt_key}' could not be loaded from ragdoll.prompts."
            ) from exc

    def _parse_suggestions(self, raw_output: str | Sequence[str]) -> List[str]:
        if isinstance(raw_output, list):
            return [str(item).strip() for item in raw_output if str(item).strip()]

        text = str(raw_output).strip()
        if not text:
            return []

        # Try to interpret the response as a Python/JSON list first.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

        # Fallback: treat each line as a suggestion.
        lines = [line.strip().lstrip("-*") for line in text.splitlines()]
        return [line for line in lines if line]

    def _resolve_llm_caller(
        self,
        *,
        llm: Optional[LLM],
        llm_caller: Optional[BaseLLMCaller],
    ) -> Optional[BaseLLMCaller]:
        if llm_caller is not None:
            return llm_caller
        if llm is None:
            return None
        return LangChainLLMCaller(llm)
