# examples/entity_extraction_example.py

import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from ragdoll.entity_extraction.entity_extraction_service import (
    EntityExtractionService,
)
from ragdoll import settings
from ragdoll.llms import get_llm_caller
from ragdoll.utils import visualize_graph

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def main(model_name: Optional[str] = None):
    """
    Demonstrates how to use the EntityExtractionService with a real LLM.

    Args:
        model_name: Optional name of the model to use. Can be a model name or a model type ('default', 'basic', 'reasoning', 'vision')
    """
    # Load environment variables for API keys
    load_dotenv(override=True)

    # Get configuration
    app_config = settings.get_app()
    config_manager = app_config.config
    entity_extraction_config = config_manager.entity_extraction_config.model_dump()

    # Use the real LLM
    model_name = model_name or "gpt-4o"  # "gpt-4o" "gpt-3.5-turbo"
    print(f"Using get_llm_caller with model: {model_name}")
    llm_caller = get_llm_caller(model_name, app_config=app_config)
    if llm_caller is None:
        raise RuntimeError("Unable to load LLM caller. Check your configuration/API keys.")

    # Create the service
    graph_service = EntityExtractionService(
        config=entity_extraction_config,
        llm_caller=llm_caller,
        app_config=app_config,
    )

    # Define sample text
    sample_text = (
        "Barack Obama was the 44th President of the United States. "
        "He was born in Honolulu, Hawaii. "
        "His wife is Michelle Obama, and they have two daughters, Malia Ann Obama and Sasha Obama. "
        "Obama served as a U.S. Senator from Illinois before becoming President."
    )

    # Another sample text (tangentially related, not directly about Obama)
    sample_text_2 = (
        "Angela Merkel served as the Chancellor of Germany from 2005 to 2021. "
        "She was born in Hamburg, West Germany. "
        "Merkel is known for her scientific background and pragmatic leadership style. "
        "During her tenure, she played a key role in managing the European financial crisis."
    )

    print(f"\nProcessing text:\n{sample_text}\n")

    # Create a Langchain Document
    sample_doc = Document(
        page_content=sample_text, metadata={"source": "example_doc_1", "id": "doc1"}
    )
    sample_doc2 = Document(
        page_content=sample_text_2, metadata={"source": "example_doc_2", "id": "doc2"}
    )

    # Extract the graph
    graph = await graph_service.extract(documents=[sample_doc, sample_doc2])

    print(f"\nExtracted {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Optional visualization
    visualize_graph(graph)


if __name__ == "__main__":

    logging.getLogger("ragdoll.entity_extraction").setLevel(logging.DEBUG)
    import argparse

    parser = argparse.ArgumentParser(description="Entity extraction example")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specify a model name (from config) or a model type (default, reasoning, vision)",
    )
    args = parser.parse_args()

    asyncio.run(main(model_name=args.model))

