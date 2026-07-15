"""
RAGdoll package exports.

Expose the high-level orchestration class and retrieval components so consumers
can rely on stable import paths without triggering side effects.
"""

from .ragdoll import Ragdoll
from .contracts import (
    CorpusId,
    GenerationId,
    IngestionSpec,
    ItemOutcome,
    ItemStatus,
    JobId,
    TenantId,
)
from .corpus import (
    CorpusIndex,
    InMemoryCorpusIndex,
    IndexCapabilities,
    StagedGeneration,
    VectorCorpusIndex,
)
from .ingestion.jobs import (
    CeleryExecutionAdapter,
    DurableIngestion,
    IngestionJob,
    IngestionResult,
    InlineExecutionAdapter,
    JobStatus,
    PostgresJobStore,
    PreparationResult,
)
from .query import Citation, QueryEngine, QueryOptions, QueryResult, QueryTrace
from .observability import (
    Event,
    EventSink,
    LoggingEventSink,
    NullEventSink,
    OpenTelemetryEventSink,
    RecordingEventSink,
)
from .generation_state import (
    FileGenerationStateStore,
    GenerationRecord,
    GenerationStateStore,
    MemoryGenerationStateStore,
    PostgresGenerationStateStore,
)
from .quarantine import (
    FileQuarantineStore,
    MemoryQuarantineStore,
    QuarantinedSource,
    QuarantineStore,
    PostgresQuarantineStore,
)
from .graph_index import InMemoryGraphBackend, Neo4jGraphBackend, VersionedGraphIndex
from .retrieval import (
    BaseRetriever,
    VectorRetriever,
    GraphRetriever,
    HybridRetriever,
)

__all__ = [
    "Ragdoll",
    "BaseRetriever",
    "VectorRetriever",
    "GraphRetriever",
    "HybridRetriever",
    "TenantId",
    "CorpusId",
    "GenerationId",
    "JobId",
    "IngestionSpec",
    "ItemOutcome",
    "ItemStatus",
    "CorpusIndex",
    "InMemoryCorpusIndex",
    "IndexCapabilities",
    "StagedGeneration",
    "VectorCorpusIndex",
    "DurableIngestion",
    "IngestionJob",
    "IngestionResult",
    "JobStatus",
    "CeleryExecutionAdapter",
    "InlineExecutionAdapter",
    "PostgresJobStore",
    "PreparationResult",
    "Citation",
    "QueryEngine",
    "QueryOptions",
    "QueryResult",
    "QueryTrace",
    "Event",
    "EventSink",
    "LoggingEventSink",
    "NullEventSink",
    "OpenTelemetryEventSink",
    "RecordingEventSink",
    "FileGenerationStateStore",
    "GenerationRecord",
    "GenerationStateStore",
    "MemoryGenerationStateStore",
    "PostgresGenerationStateStore",
    "FileQuarantineStore",
    "MemoryQuarantineStore",
    "QuarantinedSource",
    "QuarantineStore",
    "PostgresQuarantineStore",
    "InMemoryGraphBackend",
    "VersionedGraphIndex",
    "Neo4jGraphBackend",
]
