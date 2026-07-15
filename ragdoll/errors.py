"""Actionable error taxonomy for RAGdoll's public interfaces."""

from __future__ import annotations

from typing import Sequence


class RagdollError(Exception):
    """Base error carrying a stable machine-readable code."""

    code = "ragdoll_error"
    retryable = False

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class TransientError(RagdollError):
    code = "transient_error"
    retryable = True


class RejectedSourceError(RagdollError):
    code = "rejected_source"


class CancelledError(RagdollError):
    code = "cancelled"


class JobLeaseUnavailableError(TransientError):
    code = "job_lease_unavailable"


class BatchWriteError(TransientError):
    """A batch write did not persist every requested item."""

    code = "batch_write_failed"

    def __init__(
        self,
        message: str,
        *,
        failed_count: int,
        succeeded_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.failed_count = failed_count
        self.succeeded_ids = tuple(succeeded_ids)

    @property
    def succeeded_count(self) -> int:
        return len(self.succeeded_ids)


class GraphWriteError(TransientError):
    code = "graph_write_failed"


class GenerationNotFoundError(RagdollError):
    code = "generation_not_found"


class GenerationActiveError(RagdollError):
    code = "generation_is_active"


class TenantIsolationError(RagdollError):
    code = "tenant_isolation_violation"


class QueryTimeoutError(TransientError):
    code = "query_timeout"
