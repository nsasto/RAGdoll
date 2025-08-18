# Metrics

## Table of Contents

- [Location](#location)
- [Purpose](#purpose)
- [Key Components](#key-components)
- [Features](#features)
- [How It Works](#how-it-works)
- [Public API and Function Documentation](#public-api-and-function-documentation)
- [Usage Example](#usage-example)
- [Extending Metrics](#extending-metrics)
- [Best Practices](#best-practices)
- [Related Modules](#related-modules)

---

## Location

`ragdoll/metrics/`

## Purpose

Metrics modules track and report on system performance and quality, enabling monitoring, debugging, and optimization of RAG pipelines.

## Key Components

- `metrics_manager.py`: Collects and reports metrics.

## Features

- Extensible metric types.
- Integration with ingestion, retrieval, and LLM modules.

---

## How It Works

1. **Session Tracking**: Metrics are collected for each ingestion session.
2. **Source Tracking**: Metrics are tracked for each source (file, URL, etc.).
3. **Reporting**: Metrics can be aggregated and reported for monitoring.

---

## Public API and Function Documentation

### `MetricsManager`

#### Constructor

```python
MetricsManager(metrics_dir: Optional[str] = None)
```

Initializes the metrics manager with a directory for metrics data.

#### `start_session(input_count: int) -> Dict[str, Any]`

Start a new metrics collection session for a batch of inputs.

#### `start_source(batch_id: int, source_id: str, source_type: str) -> Dict[str, Any]`

Start tracking metrics for a specific source in a batch.

#### `end_source(...)`

End tracking for a source (see code for full signature).

#### `end_session(document_count: int)`

End the current metrics session, recording the total document count.

#### `get_recent_sessions(limit: int = 5)`

Get recent metrics sessions for reporting.

#### `get_aggregate_metrics(days: int = 30)`

Get aggregate metrics for the last N days.

---

## Usage Example

```python
from ragdoll.metrics.metrics_manager import MetricsManager
metrics = MetricsManager()
metrics.start_session(input_count=10)
# ... run ingestion ...
metrics.end_session(document_count=100)
```

---

## Extending Metrics

- Add new metric types or reporting methods by extending `MetricsManager`.
- Integrate metrics with dashboards or alerting systems.

---

## Best Practices

- Track metrics for all major pipeline steps.
- Use metrics to identify bottlenecks and optimize performance.
- Store metrics data in a persistent, queryable format.

---

## Related Modules

- [Ingestion](ingestion.md)
- [Configuration](configuration.md)
