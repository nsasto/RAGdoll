import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from ragdoll.ingestion.pipeline import ingest_documents, IngestionOptions

async def main():
    # Sources can be file paths, URLs, or other identifiers
    sources = [
        "../test/test_data/test_docx.docx",
    ]
    
    # Configure options for the ingestion process
    options = IngestionOptions(
        batch_size=5,
        parallel_extraction=True,
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
    asyncio.run(main())