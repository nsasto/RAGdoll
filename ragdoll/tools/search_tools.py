"""Tools related to search functionality."""

from typing import List, Dict, Optional
import logging
from langchain_google_community import GoogleSearchAPIWrapper
from ragdoll.config import Config
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from ragdoll_v1.prompts import generate_search_queries_prompt
from langchain_core.llms import LLM


class SearchToolsInput(BaseModel):
    query: str = Field(description="The search query string.")
    num_results: Optional[int] = Field(default=3, description="The number of search results to return. Defaults to 3.")


class SearchInternetTool(BaseTool):
    name = "search_internet"
    description = (
        "A tool for performing internet searches using Google. "
        "Input should be a query string, and optionally the number of results to return. "
        "Returns a list of search results, where each result is a dictionary "
        "containing the 'title', 'href' (URL), and 'snippet' (text summary) of the page."
    )
    args_schema = SearchToolsInput

    def __init__(self, config: Config, ):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)

    def _run(self, query: str, num_results: int = 3) -> List[Dict]:
        """Performs a Google search with the given query."""
        self.logger.info(f"  🌐 Searching with query {query}...")

        google_search = GoogleSearchAPIWrapper()
        results = google_search.results(query, num_results)

        if results is None:
            return []
        search_results = []

        for result in results:
            # skip youtube results
            if "youtube.com" in result["link"]:
                continue
            search_result = {
                "title": result["title"],
                "href": result["link"],
                "snippet": result["snippet"],
            }
            search_results.append(search_result)

        return search_results


class SuggestedSearchTermsInput(BaseModel):
    query: str = Field(description="The query to generate suggested search terms for.")
    num_suggestions: Optional[int] = Field(default=3, description="The number of suggested search terms to return. Defaults to 3.")


class SuggestedSearchTermsTool(BaseTool):
    name = "generate_suggested_search_terms"
    description = (
        "A tool for generating suggested search terms related to a given query. "
        "Input should be a query string, and optionally the number of suggestions to return. "
        "Returns a list of suggested search terms (strings)."
    )
    args_schema = SuggestedSearchTermsInput

    def __init__(self, config: Config, llm: LLM):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)
        self.llm = llm
        
    def _run(self, query: str, num_suggestions: int = 3) -> List[str]:
        """Generates suggested search terms for a given query."""
        prompt = generate_search_queries_prompt(query, num_suggestions)

        self.logger.info(
            f"🧠 Generating potential search queries with prompt:\n {query}"
        )
        # use the shared LLM instance
        result = self.llm.invoke(prompt)
        values = result.content if hasattr(result, "content") else result
        self.logger.info(f"🧠 Generated potential search queries: {values}")
        import ast
        return ast.literal_eval(values)
