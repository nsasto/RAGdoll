# RAGdoll Documentation

Welcome to the RAGdoll documentation set. This folder contains topic‑focused guides that mirror the code structure so both humans and automation agents can quickly locate the right information.

## Navigating the Docs

| Topic | File | Highlights |
| --- | --- | --- |
| Architecture | `architecture.md` | High‑level ingestion → graph/vector → query flow diagram. |
| Configuration | `configuration.md` | YAML layout, `ConfigManager`, environment overrides. |
| Ingestion | `ingestion.md` | Loader service, pipeline stages, metrics. |
| Chunking | `chunking.md` | `get_text_splitter`, `split_documents`, supported strategies. |
| Embeddings | `embeddings.md` | Model factory (`get_embedding_model`) and provider hints. |
| LLM Integration | `llm_integration.md` | `get_llm`, `get_llm_caller`, prompt templating. |
| Graph Stores | `graph_stores.md` | Graph persistence options and retriever helpers. |
| Vector Stores | `vector_stores.md` | LangChain wrapper/factory usage. |
| Tools | `tools.md` | Built-in LangChain tools (search, suggested queries, etc.). |
| Metrics & Caching | `metrics.md`, `caching.md` | Runtime monitoring and remote-source cache behavior. |

## How to Use This Folder

- **Users**: Start with `overview.md`, then dive into the component you want to customize.
- **Agents/automation**: Each doc has a consistent structure (Location → Purpose → API) so you can parse sections predictably.
- **Contributors**: When you change code, update the matching doc listed above and reference it in PR descriptions.

## Contributing to Docs

1. Keep titles and section headers consistent (e.g., “Location”, “Purpose”, “Key Components”).
2. Reference concrete modules/functions (`ragdoll.chunkers.get_text_splitter`) instead of abstract names.
3. Cross-link related docs using relative paths (e.g., `[Ingestion](ingestion.md)`).
4. Run spell-checkers/linters if available and keep Markdown ASCII-only unless a doc already uses Unicode.

Keeping these guides accurate makes it much easier for downstream agents (CI bots, support scripts, assistants) to reason about the project without digging through code every time. Thank you!
