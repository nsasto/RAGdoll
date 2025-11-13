import asyncio
import sys
from pathlib import Path
from typing import Optional
import logging

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from ragdoll.pipeline import ingest_documents, IngestionOptions
from ragdoll.llms import get_llm_caller

async def main(model_name: Optional[str] = None):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize LLM
    model_name = model_name or "gpt-3.5-turbo"
    print(f"Using get_llm_caller with model: {model_name}")
    llm_caller = get_llm_caller(model_name)
    if llm_caller is None:
        print("Unable to initialize the requested LLM. Check your configuration or API keys.")
        return
    print("LLM caller initialized.")
    
    # Get the absolute path to the test file
    test_data_dir = Path(__file__).parent.parent / "tests" / "test_data"
    txt_file = test_data_dir / "test_docx.docx"  # Use a simpler file type that we know works
    
    # Verify the file exists
    if not txt_file.exists():
        print(f"Error: File not found at {txt_file}")
        return
        
    # Sources can be file paths, URLs, or other identifiers
    sources = [str(txt_file)]
    
    print(f"Processing file: {sources[0]}")
    
    # Configure options for the ingestion process
    options = IngestionOptions(
        batch_size=5,
        parallel_extraction=False,  # Set to False for easier debugging
        extract_entities=True,
        chunking_options={
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        vector_store_options={
            "store_type": "faiss",
            "persist_directory": "./data/vector_stores/my_graph_rag"
        },
        graph_store_options={
            "store_type": "networkx",
            "persist_directory": "./data/graph_stores/my_graph_rag"
        },
        llm_caller=llm_caller,
        # Pass entity extraction specific options
        entity_extraction_options={
            "entity_types": ["Person", "Organization", "Location", "Date"],
            "relationship_types": ["works_for", "born_in", "located_in"],
        }
    )
    
    # Run the ingestion
    stats = await ingest_documents(sources, options=options)
    
    # Print results
    print(f"\n✅ Ingestion complete!")
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Entities extracted: {stats['entities_extracted']}")
    print(f"Relationships extracted: {stats['relationships_extracted']}")
    print(f"Vector entries added: {stats['vector_entries_added']}")
    print(f"Graph entries added: {stats['graph_entries_added']}")
    
    if stats["errors"]:
        print(f"\n⚠️ Warnings/Errors:")
        for error in stats["errors"]:
            print(f"  - {error}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Graph RAG ingestion example")
    parser.add_argument('--model', type=str, default=None, help='Specify a model name')
    args = parser.parse_args()
    
    asyncio.run(main(model_name=args.model))
