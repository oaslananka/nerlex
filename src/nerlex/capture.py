from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from nerlex.redaction import redact_state
from nerlex.spec import (
    Candidate,
    DecisionRequest,
    DecisionResult,
    DecisionSpec,
    LabelObservation,
    LabelSource,
    validate_request,
)
from nerlex.store import SQLiteTraceStore


@dataclass
class CaptureRun:
    store: SQLiteTraceStore
    spec: DecisionSpec
    request: DecisionRequest

    @property
    def request_id(self) -> UUID:
        return self.request.request_id

    def __enter__(self) -> CaptureRun:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def teacher(self, result: DecisionResult) -> DecisionResult:
        if result.request_id != self.request.request_id:
            raise ValueError("Teacher result request_id does not match the captured request.")
        self.store.record_result(result, kind="teacher")
        return result

    def observe(
        self,
        *,
        value: str | bool,
        source: LabelSource,
        source_id: str | None = None,
        confidence: float | None = None,
        adjudicated: bool = False,
        notes: str | None = None,
    ) -> LabelObservation:
        observation = LabelObservation(
            request_id=self.request.request_id,
            source=source,
            value=value,
            source_id=source_id,
            confidence=confidence,
            adjudicated=adjudicated,
            notes=notes,
        )
        self.store.record_label(observation)
        return observation


def capture(
    store: SQLiteTraceStore,
    spec: DecisionSpec,
    *,
    state: dict[str, Any] | str,
    candidates: tuple[Candidate, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> CaptureRun:
    """Capture one decision request before invoking the existing teacher path."""
    store.initialize()
    store.register_spec(spec)

    request = DecisionRequest(
        decision_id=spec.decision_id,
        spec_version=spec.version,
        state=redact_state(state, spec.sensitive_fields),
        candidates=candidates,
        metadata=dict(metadata or {}),
    )
    validate_request(spec, request)
    store.record_request(request)
    return CaptureRun(store=store, spec=spec, request=request)


def observe_outcome(
    store: SQLiteTraceStore,
    *,
    request_id: UUID,
    value: str | bool,
    source_id: str | None = None,
    notes: str | None = None,
) -> LabelObservation:
    observation = LabelObservation(
        request_id=request_id,
        source=LabelSource.OUTCOME,
        value=value,
        source_id=source_id,
        notes=notes,
    )
    store.record_label(observation)
    return observation
