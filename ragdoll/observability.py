"""Structured SDK events with an optional OpenTelemetry adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class Event:
    name: str
    attributes: Mapping[str, Any]


class EventSink(Protocol):
    def emit(self, name: str, attributes: Mapping[str, Any]) -> None: ...


class NullEventSink:
    def emit(self, name: str, attributes: Mapping[str, Any]) -> None:
        return None


class RecordingEventSink:
    """Test/embedding adapter that retains structured events in memory."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, name: str, attributes: Mapping[str, Any]) -> None:
        self.events.append(Event(name, dict(attributes)))


class LoggingEventSink:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ragdoll.events")

    def emit(self, name: str, attributes: Mapping[str, Any]) -> None:
        self.logger.info("%s %s", name, dict(attributes))


class OpenTelemetryEventSink:
    """Optional adapter; importing RAGdoll does not require OpenTelemetry."""

    def __init__(self, instrumentation_name: str = "ragdoll") -> None:
        try:
            from opentelemetry import metrics, trace
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OpenTelemetry events require `pip install opentelemetry-api`"
            ) from exc
        self._tracer = trace.get_tracer(instrumentation_name)
        meter = metrics.get_meter(instrumentation_name)
        self._counter = meter.create_counter("ragdoll.events")

    def emit(self, name: str, attributes: Mapping[str, Any]) -> None:
        safe = {key: _attribute(value) for key, value in attributes.items()}
        with self._tracer.start_as_current_span(name, attributes=safe):
            self._counter.add(1, {"event.name": name})


def _attribute(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if value is None:
        return ""
    return str(value)
