"""Examples for the search tools in ragdoll/tools/search_tools.py."""

import logging
from ragdoll.config import Config  # Assuming Config is in ragdoll.config
from ragdoll.config import Config  # Assuming ConfigManager is in ragdoll.config.config_manager
from ragdoll.tools.search_tools import SearchInternetTool, SuggestedSearchTermsTool
from langchain_openai import OpenAI  # Import OpenAI LLM

# Load OpenAI API key from .env file, overriding existing environment variables
load_dotenv(override=True)
logger = logging.getLogger(__name__)
# Create a configuration manager
config_manager = Config()
config = config_manager._config #access the loaded config as a dictionary
logger.setLevel(config.get("log_level",logging.INFO)) # Set logging level if needed
logging.basicConfig(level=config.get("log_level",logging.INFO)) # Set logging level if needed
# --- Example for SearchInternetTool ---
print("--- SearchInternetTool Example ---")
search_tool = SearchInternetTool(config=config)

query = "What is the capital of France?"
num_results = 2
search_results = search_tool._run(query=query, num_results=num_results)

print(f"Search results for '{query}' (limited to {num_results} results):")
for result in search_results:
    print(f"  Title: {result['title']}")
    print(f"  URL: {result['href']}")
    print(f"  Snippet: {result['snippet']}")
    print("-" * 20)

# --- Example for SuggestedSearchTermsTool ---
print("\n--- SuggestedSearchTermsTool Example ---")

# Use OpenAI LLM instead of MockLLM
openai_llm = OpenAI() # You can specify model parameters here if needed
suggest_tool = SuggestedSearchTermsTool(config=config, llm=openai_llm)
query = "Paris"
num_suggestions = 3
suggestions = suggest_tool._run(query=query, num_suggestions=num_suggestions)

print(f"Suggested search terms for '{query}' (limited to {num_suggestions} suggestions):")
for suggestion in suggestions:
    print(f"  - {suggestion}")
