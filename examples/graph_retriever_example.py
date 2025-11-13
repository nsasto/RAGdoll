"""
Demonstrate how to enable the graph retriever and query it via the Ragdoll facade.

This example keeps everything in-memory by constructing langchain `Document`
objects manually, so it runs without touching the filesystem. In a real workflow
you would point `ingest_with_graph()` at file paths or URLs instead.
"""

from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from ragdoll.ragdoll import Ragdoll
from ragdoll.pipeline import IngestionOptions


async def main() -> None:
    # Sample content to ingest; replace with file paths or URLs in real usage.
    documents = [
        Document(
            page_content=(
                "Ada Lovelace worked with Charles Babbage on the Analytical Engine. "
                "She is often credited as one of the first computer programmers."
            ),
            metadata={"source": "example-doc"},
        )
    ]

    # Enable the graph retriever and keep vector-store writes disabled for brevity.
    options = IngestionOptions(
        skip_vector_store=True,
        entity_extraction_options={
            "config": {
                "graph_retriever": {
                    "enabled": True,
                    "backend": "simple",
                    "top_k": 5,
                }
            }
        },
    )

    ragdoll = Ragdoll()

    result = await ragdoll.ingest_with_graph(documents, options=options)
    print("Ingestion stats:", result["stats"])

    retriever = result["graph_retriever"]
    if not retriever:
        print("Graph retriever was not created (check your config).")
        return

    hits = retriever.invoke("Who collaborated with Charles Babbage?")
    print("\nGraph retriever hits:")
    for doc in hits:
        print("-", doc.page_content, doc.metadata)


if __name__ == "__main__":
    asyncio.run(main())
