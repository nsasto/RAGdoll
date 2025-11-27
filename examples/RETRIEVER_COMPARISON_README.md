# Retriever Comparison (examples/retrieval_examples.py)

This example ingests the bundled `examples/retriever_comparison_sample.txt` using `examples/app_config_demo.yaml`, builds vector + NetworkX graph stores, and compares four retrievers: vector, graph, hybrid, and PageRank.

## What it does
- Uses the sample text (already present; not generated) as the corpus.
- Runs ingestion with entity/relationship extraction (spaCy) and builds graph + vector stores.
- Executes a few queries through each retriever.
- Prints latency per retriever, number of docs, a simple keyword-coverage proxy (precision/recall), and top snippets.
- PageRank is enabled via config; if unavailable it is skipped gracefully.

## Prerequisites
- `OPENAI_API_KEY` (or edit `app_config_demo.yaml` to point to a local embedding model).
- `spaCy` with `en_core_web_sm` installed (already required for entity extraction).

## Running
Default (option 5 – comparison):
```
python examples/retrieval_examples.py
```
Explicit option:
```
python examples/retrieval_examples.py 1   # basic hybrid
python examples/retrieval_examples.py 2   # vector only
python examples/retrieval_examples.py 3   # graph only
python examples/retrieval_examples.py 4   # hybrid custom
python examples/retrieval_examples.py 5   # retriever comparison (default)
```

## Notes
- Metrics are based on a small sample corpus; absolute times are not representative of larger deployments.
- Keyword precision/recall is a crude proxy for relevance; adjust queries/keywords in the script to suit your data.
- Tuning knobs live in `examples/app_config_demo.yaml` (e.g., pagerank `max_hops`, `num_seed_chunks`, `min_score`).

## Files involved
- `examples/retrieval_examples.py` – the runnable example.
- `examples/app_config_demo.yaml` – demo config used for ingestion/retrieval.
- `examples/retriever_comparison_sample.txt` – sample corpus ingested by the example.
