from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from nerlex.artifact_io import write_immutable_atomic
from nerlex.calibration import predict_calibrated
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.runtime import RuntimeBundle
from nerlex.spec import (
    DecisionKind,
    LabelSource,
    StrictModel,
    validate_request,
)
from nerlex.trace import TraceRecord

SHADOW_SCHEMA_VERSION: Literal[1] = 1
SHADOW_IMPLEMENTATION_VERSION: Literal["1"] = "1"
_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_FLOAT_TOLERANCE = 1e-9


class ShadowError(ValueError):
    """Raised when shadow evidence cannot be produced safely."""


class ShadowConfig(StrictModel):
    truth_priority: tuple[LabelSource, ...] = (
        LabelSource.ADJUDICATED,
        LabelSource.OUTCOME,
        LabelSource.HUMAN,
        LabelSource.RULE,
    )

    @model_validator(mode="after")
    def _validate_truth_priority(self) -> ShadowConfig:
        if not self.truth_priority:
            raise ValueError("truth_priority must contain at least one source.")
        if LabelSource.TEACHER in self.truth_priority:
            raise ValueError("Teacher output must remain separate from truth resolution.")
        if len(self.truth_priority) != len(set(self.truth_priority)):
            raise ValueError("truth_priority must not contain duplicate sources.")
        return self


class ShadowDecision(StrictModel):
    request_id: UUID
    local_selected: str
    confidence: float = Field(ge=0.0, le=1.0)
    local_eligible: bool
    fallback_required: bool
    teacher_selected: str | None = None
    teacher_ambiguous: bool = False
    local_matches_teacher: bool | None = None
    truth: str | None = None
    truth_source: LabelSource | None = None
    local_truth_correct: bool | None = None


class ShadowRouteOutcomeMatrix(StrictModel):
    local_eligible_correct: int = Field(ge=0)
    local_eligible_incorrect: int = Field(ge=0)
    local_eligible_unlabeled: int = Field(ge=0)
    fallback_required_would_be_correct: int = Field(ge=0)
    fallback_required_would_be_incorrect: int = Field(ge=0)
    fallback_required_unlabeled: int = Field(ge=0)


class ShadowSummary(StrictModel):
    total: int = Field(ge=1)
    local_eligible: int = Field(ge=0)
    gate_abstained: int = Field(ge=0)
    fallback_required: int = Field(ge=0)
    local_coverage: float = Field(ge=0.0, le=1.0)
    teacher_comparable: int = Field(ge=0)
    teacher_agreements: int = Field(ge=0)
    teacher_disagreements: int = Field(ge=0)
    teacher_ambiguous: int = Field(ge=0)
    teacher_agreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    truth_labeled: int = Field(ge=0)
    local_truth_correct: int = Field(ge=0)
    counterfactual_local_truth_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    eligible_truth_labeled: int = Field(ge=0)
    eligible_truth_correct: int = Field(ge=0)
    eligible_truth_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    route_outcome: ShadowRouteOutcomeMatrix

    @model_validator(mode="after")
    def _validate_counts(self) -> ShadowSummary:
        if self.local_eligible + self.gate_abstained != self.total:
            raise ValueError("Shadow route counts must equal total.")
        if self.fallback_required != self.gate_abstained:
            raise ValueError("Shadow fallback_required must equal gate_abstained.")
        if self.teacher_agreements + self.teacher_disagreements != self.teacher_comparable:
            raise ValueError("Teacher comparison counts are inconsistent.")
        if self.local_truth_correct > self.truth_labeled:
            raise ValueError("Truth-correct count exceeds truth-labeled count.")
        if self.eligible_truth_correct > self.eligible_truth_labeled:
            raise ValueError("Eligible truth-correct count exceeds labeled eligible count.")
        return self


class _ShadowReportIdentity(StrictModel):
    schema_version: Literal[1] = SHADOW_SCHEMA_VERSION
    shadow_version: Literal["1"] = SHADOW_IMPLEMENTATION_VERSION
    decision_spec_hash: str = Field(pattern=_SHA256_PATTERN)
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    gate_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    source_trace_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_trace_count: int = Field(ge=1)
    config: ShadowConfig
    decisions: tuple[ShadowDecision, ...]
    summary: ShadowSummary


class ShadowReport(_ShadowReportIdentity):
    report_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_report_id(self) -> ShadowReport:
        identity = _ShadowReportIdentity.model_validate(
            self.model_dump(mode="json", exclude={"report_id"})
        )
        if sha256_hex(identity) != self.report_id:
            raise ValueError("Shadow report identity does not match its content.")
        return self


class _ResolvedTruth(StrictModel):
    value: str
    source: LabelSource


def shadow_replay(
    bundle: RuntimeBundle,
    records: Iterable[TraceRecord],
    *,
    config: ShadowConfig | None = None,
) -> ShadowReport:
    """Replay local inference without changing the authoritative application decision."""
    resolved_config = config or ShadowConfig()
    bundle = RuntimeBundle.model_validate(bundle)
    spec = bundle.compiler.decision_spec
    if spec.kind is not DecisionKind.CHOICE:
        raise ShadowError("v0.1 shadow replay currently supports choice decisions only.")

    ordered = sorted(records, key=lambda record: str(record.request.request_id))
    if not ordered:
        raise ShadowError("Shadow replay requires at least one trace.")

    request_ids = [record.request.request_id for record in ordered]
    if len(request_ids) != len(set(request_ids)):
        raise ShadowError("Shadow replay input contains duplicate request IDs.")

    source_trace_sha256 = sha256_hex(tuple(ordered))
    threshold = bundle.gate.confidence_threshold
    decisions: list[ShadowDecision] = []

    for record in ordered:
        try:
            validate_request(spec, record.request)
        except ValueError as exc:
            raise ShadowError(
                f"Trace request {record.request.request_id} does not match the runtime spec."
            ) from exc

        prediction = predict_calibrated(
            bundle.compiler,
            bundle.calibration,
            record.request.state,
        )
        local_eligible = (
            threshold is not None
            and prediction.confidence + _FLOAT_TOLERANCE >= threshold
        )
        teacher_selected, teacher_ambiguous = _teacher_selection(record)
        truth = _resolve_truth(record, resolved_config.truth_priority)
        _validate_resolved_value(
            prediction.selected,
            spec_candidates={candidate.key for candidate in spec.candidates},
            request_id=record.request.request_id,
            role="local prediction",
        )
        if teacher_selected is not None:
            _validate_resolved_value(
                teacher_selected,
                spec_candidates={candidate.key for candidate in spec.candidates},
                request_id=record.request.request_id,
                role="teacher decision",
            )
        if truth is not None:
            _validate_resolved_value(
                truth.value,
                spec_candidates={candidate.key for candidate in spec.candidates},
                request_id=record.request.request_id,
                role="truth label",
            )

        decisions.append(
            ShadowDecision(
                request_id=record.request.request_id,
                local_selected=prediction.selected,
                confidence=prediction.confidence,
                local_eligible=local_eligible,
                fallback_required=not local_eligible,
                teacher_selected=teacher_selected,
                teacher_ambiguous=teacher_ambiguous,
                local_matches_teacher=(
                    prediction.selected == teacher_selected
                    if teacher_selected is not None and not teacher_ambiguous
                    else None
                ),
                truth=truth.value if truth is not None else None,
                truth_source=truth.source if truth is not None else None,
                local_truth_correct=(
                    prediction.selected == truth.value if truth is not None else None
                ),
            )
        )

    decision_tuple = tuple(decisions)
    summary = _summarize(decision_tuple)
    identity = _ShadowReportIdentity(
        decision_spec_hash=sha256_hex(spec),
        compiler_artifact_id=bundle.compiler.artifact_id,
        calibration_artifact_id=bundle.calibration.artifact_id,
        gate_artifact_id=bundle.gate.artifact_id,
        source_trace_sha256=source_trace_sha256,
        source_trace_count=len(ordered),
        config=resolved_config,
        decisions=decision_tuple,
        summary=summary,
    )
    return ShadowReport(
        report_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def write_shadow_report(report: ShadowReport, root: str | Path) -> Path:
    destination = Path(root) / f"{report.report_id}.shadow.json"
    write_immutable_atomic(
        destination,
        (canonical_json(report) + "\n").encode("utf-8"),
    )
    return destination


def load_shadow_report(path: str | Path) -> ShadowReport:
    return ShadowReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _teacher_selection(record: TraceRecord) -> tuple[str | None, bool]:
    values = {
        observation.result.selected
        for observation in record.observations
        if observation.kind == "teacher"
        and not observation.result.abstained
        and isinstance(observation.result.selected, str)
    }
    if not values:
        return None, False
    if len(values) > 1:
        return None, True
    return next(iter(values)), False


def _resolve_truth(
    record: TraceRecord,
    priority: tuple[LabelSource, ...],
) -> _ResolvedTruth | None:
    for source in priority:
        labels = [label for label in record.labels if label.source is source]
        if not labels:
            continue
        values = {label.value for label in labels}
        if len(values) != 1:
            raise ShadowError(
                f"Conflicting {source.value} truth labels for request "
                f"{record.request.request_id}."
            )
        value = next(iter(values))
        if not isinstance(value, str):
            raise ShadowError("v0.1 shadow replay currently supports choice labels only.")
        return _ResolvedTruth(value=value, source=source)
    return None


def _validate_resolved_value(
    value: str,
    *,
    spec_candidates: set[str],
    request_id: UUID,
    role: str,
) -> None:
    if value not in spec_candidates:
        raise ShadowError(
            f"{role} {value!r} is outside the DecisionSpec for request {request_id}."
        )


def _summarize(decisions: tuple[ShadowDecision, ...]) -> ShadowSummary:
    total = len(decisions)
    local_eligible = sum(decision.local_eligible for decision in decisions)
    gate_abstained = total - local_eligible

    comparable = [
        decision
        for decision in decisions
        if decision.local_matches_teacher is not None
    ]
    teacher_agreements = sum(
        bool(decision.local_matches_teacher) for decision in comparable
    )
    teacher_disagreements = len(comparable) - teacher_agreements
    teacher_ambiguous = sum(decision.teacher_ambiguous for decision in decisions)

    truth_labeled = [
        decision for decision in decisions if decision.local_truth_correct is not None
    ]
    local_truth_correct = sum(
        bool(decision.local_truth_correct) for decision in truth_labeled
    )
    eligible_truth = [
        decision
        for decision in truth_labeled
        if decision.local_eligible
    ]
    eligible_truth_correct = sum(
        bool(decision.local_truth_correct) for decision in eligible_truth
    )

    route_outcome = ShadowRouteOutcomeMatrix(
        local_eligible_correct=sum(
            decision.local_eligible and decision.local_truth_correct is True
            for decision in decisions
        ),
        local_eligible_incorrect=sum(
            decision.local_eligible and decision.local_truth_correct is False
            for decision in decisions
        ),
        local_eligible_unlabeled=sum(
            decision.local_eligible and decision.local_truth_correct is None
            for decision in decisions
        ),
        fallback_required_would_be_correct=sum(
            decision.fallback_required and decision.local_truth_correct is True
            for decision in decisions
        ),
        fallback_required_would_be_incorrect=sum(
            decision.fallback_required and decision.local_truth_correct is False
            for decision in decisions
        ),
        fallback_required_unlabeled=sum(
            decision.fallback_required and decision.local_truth_correct is None
            for decision in decisions
        ),
    )

    return ShadowSummary(
        total=total,
        local_eligible=local_eligible,
        gate_abstained=gate_abstained,
        fallback_required=gate_abstained,
        local_coverage=local_eligible / total,
        teacher_comparable=len(comparable),
        teacher_agreements=teacher_agreements,
        teacher_disagreements=teacher_disagreements,
        teacher_ambiguous=teacher_ambiguous,
        teacher_agreement_rate=(
            teacher_agreements / len(comparable) if comparable else None
        ),
        truth_labeled=len(truth_labeled),
        local_truth_correct=local_truth_correct,
        counterfactual_local_truth_accuracy=(
            local_truth_correct / len(truth_labeled) if truth_labeled else None
        ),
        eligible_truth_labeled=len(eligible_truth),
        eligible_truth_correct=eligible_truth_correct,
        eligible_truth_accuracy=(
            eligible_truth_correct / len(eligible_truth) if eligible_truth else None
        ),
        route_outcome=route_outcome,
    )
