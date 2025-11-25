import nbformat
import re

def migrate_notebook(notebook_path):
    print(f"Migrating {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Replace imports
            cell.source = re.sub(
                r'from ragdoll\.retriever import RagdollRetriever',
                'from ragdoll import Ragdoll',
                cell.source
            )
            
            # Replace instantiation
            cell.source = re.sub(
                r'ragdoll\s*=\s*RagdollRetriever\(config\)',
                'ragdoll = Ragdoll(config)',
                cell.source
            )
            
            # Replace get_db usage (Ragdoll handles this internally now)
            # We'll comment it out and add a note, as the new API is different
            if 'ragdoll.get_db(' in cell.source:
                cell.source = re.sub(
                    r'(.*ragdoll\.get_db\(.*)',
                    r'# \1\n# Note: Ragdoll now handles vector store creation internally during ingestion.',
                    cell.source
                )

            # Replace get_retriever usage
            # Old: retriever = ragdoll.get_retriever()
            # New: retriever = ragdoll.vector_retriever (or hybrid)
            cell.source = re.sub(
                r'ragdoll\.get_retriever\(\)',
                'ragdoll.vector_retriever',
                cell.source
            )
            
            # Replace get_mq_retriever usage
            # This is more complex as MQ might not be directly exposed the same way
            # For now, we'll map it to vector_retriever as a fallback or hybrid
            cell.source = re.sub(
                r'ragdoll\.get_mq_retriever\(\)',
                'ragdoll.vector_retriever # Multi-query is now a configuration option',
                cell.source
            )
            
            # Replace get_compression_retriever
            cell.source = re.sub(
                r'ragdoll\.get_compression_retriever\((.*)\)',
                r'# Compression retriever is now configured via config.yaml\n# ragdoll.get_compression_retriever(\1)',
                cell.source
            )

            # Replace answer_me_this
            # Old: response = ragdoll.answer_me_this(question, cc_retriever)
            # New: response = ragdoll.query(question, retriever_mode="vector") # simplified
            if 'ragdoll.answer_me_this' in cell.source:
                 cell.source = re.sub(
                    r'ragdoll\.answer_me_this\(([^,]+),.*?\)',
                    r'ragdoll.query(\1)',
                    cell.source
                 )

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Finished migrating {notebook_path}")

if __name__ == "__main__":
    migrate_notebook('c:/dev/RAGdoll/ragdoll_example.ipynb')
    migrate_notebook('c:/dev/RAGdoll/ragdoll_pdf_example.ipynb')
