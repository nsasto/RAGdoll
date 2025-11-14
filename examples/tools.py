"""Examples for the search tools in ragdoll/tools/search_tools.py."""

import logging
from dotenv import load_dotenv

from ragdoll import settings
from ragdoll.tools.search_tools import SearchInternetTool, SuggestedSearchTermsTool
from langchain_openai import OpenAI  # Requires langchain-openai installed


def main() -> None:
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    app = settings.get_app()

    print("--- SearchInternetTool Example ---")
    search_tool = SearchInternetTool()
    query = "What is the capital of France?"
    results = search_tool._run(query=query, num_results=2)
    for result in results:
        logger.info("%s (%s)", result["title"], result["href"])

    print("\n--- SuggestedSearchTermsTool Example ---")
    openai_llm = OpenAI()  # Relies on OPENAI_API_KEY
    suggest_tool = SuggestedSearchTermsTool(app_config=app, llm=openai_llm)
    suggestions = suggest_tool._run(query="Paris", num_suggestions=3)
    for suggestion in suggestions:
        logger.info("Suggested: %s", suggestion)


if __name__ == "__main__":
    main()
