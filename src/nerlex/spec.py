from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionKind(StrEnum):
    CHOICE = "choice"
    BOOLEAN = "boolean"


class CandidateMode(StrEnum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class DecisionRoute(StrEnum):
    LOCAL = "local"
    TEACHER = "teacher"
    FALLBACK = "fallback"
    HUMAN = "human"


class LabelSource(StrEnum):
    TEACHER = "teacher"
    HUMAN = "human"
    OUTCOME = "outcome"
    RULE = "rule"
    ADJUDICATED = "adjudicated"


class Candidate(StrictModel):
    key: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=4096)


class DecisionSpec(StrictModel):
    decision_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    kind: DecisionKind
    description: str = Field(min_length=1, max_length=4096)
    candidate_mode: CandidateMode | None = None
    candidates: tuple[Candidate, ...] = ()
    sensitive_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_shape(self) -> DecisionSpec:
        keys = [candidate.key for candidate in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("Candidate keys must be unique.")

        if self.kind is DecisionKind.BOOLEAN:
            if self.candidate_mode is not None or self.candidates:
                raise ValueError("Boolean decisions do not define choice candidates.")
            return self

        if self.candidate_mode is None:
            raise ValueError("Choice decisions require candidate_mode.")
        if self.candidate_mode is CandidateMode.STATIC and len(self.candidates) < 2:
            raise ValueError("Static choice decisions require at least two candidates.")
        if self.candidate_mode is CandidateMode.DYNAMIC and self.candidates:
            raise ValueError("Dynamic choice candidates belong on each request, not the spec.")
        return self


class DecisionRequest(StrictModel):
    request_id: UUID = Field(default_factory=uuid4)
    decision_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", min_length=1, max_length=128)
    spec_version: str = Field(min_length=1, max_length=64)
    event_time: datetime = Field(default_factory=_utcnow)
    state: dict[str, Any] | str
    candidates: tuple[Candidate, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_candidates(self) -> DecisionRequest:
        keys = [candidate.key for candidate in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("Request candidate keys must be unique.")
        return self


class DecisionResult(StrictModel):
    request_id: UUID
    selected: str | bool | None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated: bool = False
    backend: str = Field(min_length=1, max_length=256)
    backend_version: str | None = Field(default=None, max_length=256)
    artifact_id: str | None = Field(default=None, max_length=256)
    artifact_ids: dict[str, str] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0.0)
    route: DecisionRoute
    abstained: bool = False
    abstain_reason: str | None = Field(default=None, max_length=256)
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _validate_probabilities(self) -> DecisionResult:
        if any(
            not key or not value or len(key) > 64 or len(value) > 256
            for key, value in self.artifact_ids.items()
        ):
            raise ValueError("Artifact identity keys/values must be non-empty and bounded.")
        if any(value < 0.0 or value > 1.0 for value in self.probabilities.values()):
            raise ValueError("Probabilities must be in [0, 1].")
        if self.probabilities and abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1.")
        if self.abstained and self.selected is not None:
            raise ValueError("An abstained result cannot contain a selected value.")
        if not self.abstained and self.selected is None:
            raise ValueError("A non-abstained result requires a selected value.")
        return self


class LabelObservation(StrictModel):
    observation_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    source: LabelSource
    value: str | bool
    source_id: str | None = Field(default=None, max_length=256)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    adjudicated: bool = False
    observed_at: datetime = Field(default_factory=_utcnow)
    notes: str | None = Field(default=None, max_length=4096)


def validate_request(spec: DecisionSpec, request: DecisionRequest) -> DecisionRequest:
    if request.decision_id != spec.decision_id:
        raise ValueError("Request decision_id does not match the decision spec.")
    if request.spec_version != spec.version:
        raise ValueError("Request spec_version does not match the decision spec.")

    if spec.kind is DecisionKind.BOOLEAN:
        if request.candidates:
            raise ValueError("Boolean requests cannot include candidates.")
        return request

    if spec.candidate_mode is CandidateMode.STATIC:
        if request.candidates:
            raise ValueError("Static choice requests must use candidates from the decision spec.")
        return request

    if len(request.candidates) < 2:
        raise ValueError("Dynamic choice requests require at least two candidates.")
    return request
