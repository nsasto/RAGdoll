from ragdoll.chunkers.chunker import Chunker

# Example usage with default config
chunker = Chunker.from_config()
splitter = chunker.get_text_splitter()

text = """
# This is a sample Markdown document

This is some introductory text.

## Section 1

This is the content of section 1.

### Subsection A

This is a subsection within section 1.

## Section 2

This is the content of section 2.
"""

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n---")