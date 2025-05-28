try:
    from ragdoll.ingestion import ContentExtractionService, Source
    print("✅ Import from ragdoll.ingestion successful")
except ImportError as e:
    print(f"❌ Import from ragdoll.ingestion failed: {e}")

try:
    from ragdoll.content_extraction import ContentExtractionService, Source
    print("✅ Import from ragdoll.content_extraction successful")
except ImportError as e:
    print(f"❌ Import from ragdoll.content_extraction failed: {e}")