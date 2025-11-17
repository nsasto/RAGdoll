import asyncio
import sys
from pathlib import Path
from typing import Optional
import logging

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)


# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from ragdoll.pipeline import ingest_documents, IngestionOptions
from ragdoll.llms import get_llm_caller
from ragdoll.utils import visualize_graph


class ExampleFallbackLLM:
    """Tiny fallback so the example can run without remote LLM credentials."""

    async def call(self, prompt: str) -> str:  # type: ignore[override]
        return (
            "This is a fallback response generated locally because no API key "
            "was available. Prompt preview: "
            f"{prompt[:200]}..."
        )


async def main(model_name: Optional[str] = None, visualize: bool = False):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize LLM
    model_name = model_name or "gpt-4o-mini"
    print(f"Using get_llm_caller with model: {model_name}")
    llm_caller = get_llm_caller(model_name)
    if llm_caller is None:
        print(
            "Unable to initialize the requested LLM. Falling back to a local stub. "
            "Set OPENAI_API_KEY or choose a configured provider to use a real model."
        )
        llm_caller = ExampleFallbackLLM()
    else:
        print("LLM caller initialized.")

    # Get the absolute path to the test file
    test_data_dir = Path(__file__).parent.parent / "tests" / "test_data"
    txt_file = (
        test_data_dir / "test_docx.docx"
    )  # Use a simpler file type that we know works

    # Verify the file exists
    if not txt_file.exists():
        print(f"Error: File not found at {txt_file}")
        return

    # Sources can be file paths, URLs, or other identifiers
    sources = [str(txt_file)]

    vector_store_dir = Path("./data/vector_stores/my_graph_rag")
    graph_store_dir = Path("./data/graph_stores/my_graph_rag")
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    graph_store_dir.mkdir(parents=True, exist_ok=True)
    graph_store_file = graph_store_dir / "graph.pkl"

    print(f"Processing file: {sources[0]}")

    # Configure options for the ingestion process
    options = IngestionOptions(
        batch_size=5,
        parallel_extraction=False,  # Set to False for easier debugging
        extract_entities=True,
        chunking_options={"chunk_size": 1000, "chunk_overlap": 200},
        # Use Chroma so we do not need to bootstrap a FAISS index/docstore.
        vector_store_options={
            "store_type": "chroma",
            "params": {
                "collection_name": "graph_rag_demo",
                "persist_directory": str(vector_store_dir),
            },
        },
        graph_store_options={
            "store_type": "networkx",
            "output_file": str(graph_store_file),
        },
        llm_caller=llm_caller,
        # Pass entity extraction specific options
        entity_extraction_options={
            "entity_types": ["Person", "Organization", "Location", "Date"],
            "relationship_types": ["works_for", "born_in", "located_in"],
        },
    )

    # Run the ingestion
    result = await ingest_documents(sources, options=options)
    stats = result.get("stats", {})
    graph = result.get("graph")

    # Print results
    print(f"\n✅ Ingestion complete!")
    print(f"Documents processed: {stats.get('documents_processed')}")
    print(f"Chunks created: {stats.get('chunks_created')}")
    print(f"Entities extracted: {stats.get('entities_extracted')}")
    print(f"Relationships extracted: {stats.get('relationships_extracted')}")
    print(f"Vector entries added: {stats.get('vector_entries_added')}")
    print(f"Graph entries added: {stats.get('graph_entries_added')}")

    if visualize and graph:
        visualize_graph(graph)

    if stats.get("errors"):
        print(f"\n⚠️ Warnings/Errors:")
        for error in stats["errors"]:
            print(f"  - {error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph RAG ingestion example")
    parser.add_argument("--model", type=str, default=None, help="Specify a model name")
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Render the resulting graph to graph_output.json/png",
    )
    args = parser.parse_args()

    asyncio.run(main(model_name=args.model, visualize=args.visualize))
