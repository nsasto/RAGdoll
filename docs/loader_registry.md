# Loader Registry

Overview

RAGdoll supports a small loader registry so configuration can refer to loader classes by a short, stable name (for example `pdf`, `html`, or `s3`) instead of a full Python import path. The registry improves discoverability, reduces runtime import errors, and makes testing and plugin development easier.

How it works

- Modules can register a loader class with the registry using the `@register_loader("name")` decorator found in `ragdoll.ingestion`.
- The configuration may contain `file_mappings` that map file extensions to either a short registry name (preferred) or a full import string (legacy).
- `ConfigManager.get_loader_mapping()` prefers registry names and falls back to dynamic import when necessary.

Example

Register a loader in your module:

```python
from ragdoll.ingestion import register_loader

@register_loader("pdf")
class PDFLoader:
    def __init__(self, file_path: str):
        ...
    def load(self):
        ...
```

Reference in config (short name):

```yaml
ingestion:
  loaders:
    file_mappings:
      .pdf: pdf
      .md: markdown_loader
```

Legacy import-string example (still supported):

```yaml
ingestion:
  loaders:
    file_mappings:
      .pdf: some.package.pdf_loader:PDFLoader
```

Validation and migration

- To surface misconfigured loader names early, instantiate a `ConfigManager` at startup — it will attempt to resolve configured loaders when `DocumentLoaderService` initializes.
- For a migration path, replace import-strings in your configs with short registry names once you register the desired loader classes.

Debugging

- Use `ragdoll.ingestion.list_loaders()` to see registered loader short names at runtime.

