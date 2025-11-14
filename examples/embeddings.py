"""
Example demonstrating RAGdoll's embeddings functionality.

This script shows how to:
1. Initialize embedding models
2. Generate embeddings using various providers
3. Configure custom embedding parameters
"""

import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add parent directory to path to allow imports from RAGdoll
sys.path.append(str(Path(__file__).parent.parent))

from ragdoll.embeddings import get_embedding_model
from ragdoll import settings
from ragdoll.app_config import bootstrap_app

# Get API keys from .env file
load_dotenv(override=True)


def print_divider():
    """Print a divider for better readability."""
    print("\n" + "=" * 80 + "\n")


def print_embedding_info(embedding, model_name):
    """Print information about an embedding."""
    print(f"Successfully embedded text with {model_name}.")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First few values: {embedding[:5]}...")


print_divider()
print("RAGdoll Embeddings Example")
print_divider()

# Load configuration
print("Loading configuration...")
app_config = settings.get_app()
config_manager = app_config.config

# Sample text to embed
text_to_embed = "This is an example sentence that we will convert to a vector embedding."

# Example 1: Default Embeddings
print("EXAMPLE 1: DEFAULT EMBEDDINGS")
try:
    print("Getting default embedding model...")
    default_model = get_embedding_model(config_manager=config_manager)

    print("Generating embedding...")
    default_embedding = default_model.embed_query(text_to_embed)

    print_embedding_info(default_embedding, "default model")

except Exception as exc:
    print(f"Error with default embeddings: {exc}")

print_divider()

# Example 2: OpenAI Embeddings
print("EXAMPLE 2: OPENAI EMBEDDINGS")
try:
    print("Getting OpenAI embedding model...")
    openai_model = get_embedding_model(
        provider="openai",
        model_name="text-embedding-3-small",
        dimensions=1536,
    )

    print("Generating embedding...")
    openai_embedding = openai_model.embed_query(text_to_embed)

    print_embedding_info(openai_embedding, "OpenAI")

except Exception as exc:
    print(f"Error with OpenAI embeddings: {exc}")

print_divider()

# Example 3: HuggingFace Embeddings
print("EXAMPLE 3: HUGGINGFACE EMBEDDINGS")
try:
    print("Getting HuggingFace embedding model...")
    hf_model = get_embedding_model(
        provider="huggingface",
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    print("Generating embedding...")
    hf_embedding = hf_model.embed_query(text_to_embed)

    print_embedding_info(hf_embedding, "HuggingFace")

except Exception as exc:
    print(f"Error with HuggingFace embeddings: {exc}")

print_divider()

# Example 4: Configuration-Based Embeddings
print("EXAMPLE 4: CONFIGURATION-BASED EMBEDDINGS")
try:
    # Create a custom configuration override
    custom_config = {
        "embeddings": {
            "default_model": "cohere",
            "cohere": {
                "model": "embed-english-v3.0",
                "api_key": os.getenv("COHERE_API_KEY"),
            },
        }
    }

    print("Getting Cohere embedding model from configuration overrides...")
    cohere_app = bootstrap_app(overrides=custom_config)
    config_model = get_embedding_model(app_config=cohere_app)

    print("Generating embedding...")
    config_embedding = config_model.embed_query(text_to_embed)

    print_embedding_info(config_embedding, "Cohere (from overrides)")

except Exception as exc:
    print(f"Error with configuration-based embeddings: {exc}")

print_divider()

# Example 5: Batch Embeddings
print("EXAMPLE 5: BATCH EMBEDDINGS")
try:
    print("Getting embedding model for batch operations...")
    batch_model = get_embedding_model(provider="openai")

    # Multiple texts to embed
    texts = [
        "This is the first example sentence.",
        "Here is another completely different sentence.",
        "And a third one to demonstrate batch processing.",
    ]

    print("Generating batch embeddings...")
    batch_embeddings = batch_model.embed_documents(texts)

    print(f"Successfully embedded {len(batch_embeddings)} texts in batch.")
    print(f"Each embedding has {len(batch_embeddings[0])} dimensions")

except Exception as exc:
    print(f"Error with batch embeddings: {exc}")

print_divider()
print("Examples completed!")
