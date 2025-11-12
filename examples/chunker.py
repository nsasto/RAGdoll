"""
Example script demonstrating various ways to use the RAGdoll chunking functionality.

This script shows how to:
1. Create different types of text splitters
2. Configure chunk size and overlap
3. Split documents with various options
4. Handle different content types (text, markdown, code)
"""

import os
import sys
from pathlib import Path
from pprint import pprint

# Add parent directory to path to allow imports from RAGdoll
sys.path.append(str(Path(__file__).parent.parent))

from ragdoll.chunkers import get_text_splitter, split_documents
from langchain_core.documents import Document

# Sample texts for demonstration
PLAIN_TEXT = """
This is a simple plain text document.

It has several paragraphs of varying length.
Some are short.

Others are much longer and contain multiple sentences. These sentences might talk about different topics, but they're all part of the same paragraph. The chunker will need to decide where to split this content.

This paragraph adds additional content to demonstrate how chunking works with longer inputs.
"""

MARKDOWN_TEXT = """
# Sample Markdown Document

This is an introduction paragraph.

## First Section

This is content in the first section. It contains some details about the topic.

### Subsection A

Here's a deeper dive into a specific aspect.

- List item one
- List item two
- List item three

## Second Section

Another section with different content.

### Subsection B

More specific information here.

#### Even Deeper

This is a fourth-level heading with some content.
"""

CODE_TEXT = """
def calculate_fibonacci(n):
    \"\"\"Calculate the Fibonacci sequence up to n\"\"\"
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def process(self):
        \"\"\"Process the data\"\"\"
        result = []
        for item in self.data:
            result.append(item * 2)
        self.processed = True
        return result

def main():
    # Generate Fibonacci sequence
    fib = calculate_fibonacci(10)
    print(f"Fibonacci sequence: {fib}")
    
    # Process some data
    processor = DataProcessor([1, 2, 3, 4, 5])
    processed_data = processor.process()
    print(f"Processed data: {processed_data}")

if __name__ == "__main__":
    main()
"""


def print_divider():
    """Print a divider line for better readability."""
    print("\n" + "=" * 80 + "\n")


def print_chunks(chunks, title):
    """Print chunks with a title and chunk numbers."""
    print(f"\n{title}:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} " + "-" * 40)
        print(chunk)
        print("-" * 50)


def create_document(text, metadata=None):
    """Create a LangChain document from text."""
    return Document(page_content=text, metadata=metadata or {})


def example_1_basic_usage():
    """Basic usage of the chunker with default settings."""
    print_divider()
    print("EXAMPLE 1: BASIC USAGE")

    # Get a default text splitter (recursive character splitter)
    splitter = get_text_splitter()

    # Split a simple text
    chunks = splitter.split_text(PLAIN_TEXT)

    print_chunks(chunks, "Default Chunking (Recursive Character Splitter)")


def example_2_custom_parameters():
    """Using custom parameters for chunking."""
    print_divider()
    print("EXAMPLE 2: CUSTOM PARAMETERS")

    # Get a splitter with custom chunk size and overlap
    splitter = get_text_splitter(
        chunk_size=100, chunk_overlap=20  # Smaller chunks  # Smaller overlap
    )

    # Split the same text
    chunks = splitter.split_text(PLAIN_TEXT)

    print_chunks(chunks, "Custom Chunking (100 chars, 20 overlap)")


def example_3_different_splitter_types():
    """Demonstrate different types of text splitters."""
    print_divider()
    print("EXAMPLE 3: DIFFERENT SPLITTER TYPES")

    # Character text splitter
    char_splitter = get_text_splitter(
        splitter_type="character", chunk_size=200, chunk_overlap=0
    )

    # Recursive text splitter
    recursive_splitter = get_text_splitter(
        splitter_type="recursive", chunk_size=200, chunk_overlap=0
    )

    # Compare the results
    char_chunks = char_splitter.split_text(PLAIN_TEXT)
    recursive_chunks = recursive_splitter.split_text(PLAIN_TEXT)

    print_chunks(char_chunks, "Character Text Splitter")
    print_chunks(recursive_chunks, "Recursive Text Splitter")


def example_4_markdown_splitting():
    """Demonstrate markdown-specific splitting."""
    print_divider()
    print("EXAMPLE 4: MARKDOWN SPLITTING")

    # Get a markdown splitter
    markdown_splitter = get_text_splitter(splitter_type="markdown")

    # Create a document with markdown content
    markdown_doc = create_document(MARKDOWN_TEXT, {"source": "sample.md"})

    # Use our helper function to split the document
    from ragdoll.chunkers import split_documents

    markdown_chunks = split_documents([markdown_doc], splitter_type="markdown")

    # Print the chunks with their metadata
    print("Markdown Chunks with Headers:")
    for i, chunk in enumerate(markdown_chunks, 1):
        print(f"\n--- Chunk {i} " + "-" * 40)
        content = chunk.page_content
        # Truncate very long content for display
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"Content: {content}")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)


def example_5_code_splitting():
    """Demonstrate code-specific splitting."""
    print_divider()
    print("EXAMPLE 5: CODE SPLITTING")

    # Get a code splitter for Python
    code_splitter = get_text_splitter(
        splitter_type="code", language="python", chunk_size=300, chunk_overlap=50
    )

    # Create a document with Python code
    code_doc = create_document(CODE_TEXT, {"source": "sample.py"})

    # Split the code document
    code_chunks = code_splitter.split_documents([code_doc])

    print_chunks([chunk.page_content for chunk in code_chunks], "Python Code Chunks")


def example_6_convenience_function():
    """Demonstrate the convenience function for splitting documents."""
    print_divider()
    print("EXAMPLE 6: CONVENIENCE FUNCTION")

    # Create multiple documents of different types
    documents = [
        create_document(PLAIN_TEXT[:200], {"type": "text", "id": "doc1"}),
        create_document(MARKDOWN_TEXT[:200], {"type": "markdown", "id": "doc2"}),
        create_document(CODE_TEXT[:200], {"type": "code", "id": "doc3"}),
    ]

    # Split them all at once
    chunks = split_documents(documents, chunk_size=150, chunk_overlap=20)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    print("\nFirst few chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} " + "-" * 40)
        print(f"Content: {chunk.page_content[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)


def example_7_with_config():
    """Demonstrate using configuration dictionaries."""
    print_divider()
    print("EXAMPLE 7: USING CONFIGURATION")

    # Create a configuration dictionary
    config = {
        "chunker": {
            "splitter_type": "recursive",
            "chunk_size": 250,
            "chunk_overlap": 30,
            "separators": ["\n\n", "\n", ".", " "],
        }
    }

    # Get a splitter using the config
    splitter = get_text_splitter(config=config)

    # Split text
    chunks = splitter.split_text(PLAIN_TEXT)

    print_chunks(chunks, "Chunking with Configuration")


def example_8_no_chunking_strategy():
    """Demonstrate the 'none' chunking strategy."""
    print_divider()
    print("EXAMPLE 8: NO CHUNKING STRATEGY")

    # Create documents
    documents = [
        create_document("Document 1: " + PLAIN_TEXT[:100], {"id": "doc1"}),
        create_document("Document 2: " + PLAIN_TEXT[100:200], {"id": "doc2"}),
    ]

    # Split with 'none' strategy
    result = split_documents(documents, strategy="none")

    print(f"Original documents: {len(documents)}")
    print(f"Result documents: {len(result)}")
    print("\nResult documents are unchanged:")
    for doc in result:
        print(f"- {doc.page_content[:50]}...")


def main():
    """Run all examples."""
    print("\nRAGdoll Chunker Examples\n")

    example_1_basic_usage()
    example_2_custom_parameters()
    example_3_different_splitter_types()
    example_4_markdown_splitting()
    example_5_code_splitting()
    example_6_convenience_function()
    example_7_with_config()
    example_8_no_chunking_strategy()

    print_divider()
    print("All examples completed!")


if __name__ == "__main__":
    main()
