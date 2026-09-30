from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from nerlex.artifact_io import write_immutable_atomic
from nerlex.canary import CanaryDecisionEvidence
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.promotion import CanaryPlan, PromotionAssessment
from nerlex.runtime import RuntimeBundle
from nerlex.spec import (
    Candidate,
    DecisionKind,
    DecisionRoute,
    LabelObservation,
    LabelSource,
    StrictModel,
)

ROLLOUT_SCHEMA_VERSION: Literal[1] = 1
ROLLOUT_IMPLEMENTATION_VERSION: Literal["1"] = "1"

_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_FLOAT_TOLERANCE = 1e-9


class RolloutError(ValueError):
    """Raised when live canary rollout evidence cannot be evaluated safely."""


class RolloutConfig(StrictModel):
    truth_priority: tuple[LabelSource, ...] = (
        LabelSource.ADJUDICATED,
        LabelSource.OUTCOME,
        LabelSource.HUMAN,
        LabelSource.RULE,
    )

    @model_validator(mode="after")
    def _validate_truth_priority(self) -> RolloutConfig:
        if not self.truth_priority:
            raise ValueError("truth_priority must contain at least one source.")
        if LabelSource.TEACHER in self.truth_priority:
            raise ValueError("Teacher output must remain separate from truth resolution.")
        if len(self.truth_priority) != len(set(self.truth_priority)):
            raise ValueError("truth_priority must not contain duplicate sources.")
        return self


class RolloutDecision(StrictModel):
    request_id: UUID
    evidence_id: str = Field(pattern=_SHA256_PATTERN)
    cohort: Literal["control", "canary"]
    route: DecisionRoute
    selected: str | bool
    truth: str | bool | None = None
    truth_source: LabelSource | None = None
    correct: bool | None = None

    @model_validator(mode="after")
    def _validate_truth_shape(self) -> RolloutDecision:
        if self.truth is None:
            if self.truth_source is not None or self.correct is not None:
                raise ValueError("Unlabeled rollout decisions cannot contain truth metadata.")
            return self

        if self.truth_source is None or self.correct is None:
            raise ValueError("Truth-labeled rollout decisions require source and correctness.")
        if self.correct != (self.selected == self.truth):
            raise ValueError("Rollout decision correctness does not match selected/truth values.")
        return self


class RolloutRouteOutcomeMatrix(StrictModel):
    control_correct: int = Field(ge=0)
    control_incorrect: int = Field(ge=0)
    control_unlabeled: int = Field(ge=0)
    canary_local_correct: int = Field(ge=0)
    canary_local_incorrect: int = Field(ge=0)
    canary_local_unlabeled: int = Field(ge=0)
    canary_fallback_correct: int = Field(ge=0)
    canary_fallback_incorrect: int = Field(ge=0)
    canary_fallback_unlabeled: int = Field(ge=0)


class RolloutSummary(StrictModel):
    total: int = Field(ge=1)
    control_assigned: int = Field(ge=0)
    canary_assigned: int = Field(ge=0)
    control_fraction: float = Field(ge=0.0, le=1.0)
    canary_fraction: float = Field(ge=0.0, le=1.0)
    local_served: int = Field(ge=0)
    fallback_served: int = Field(ge=0)
    local_serving_fraction: float = Field(ge=0.0, le=1.0)
    fallback_serving_fraction: float = Field(ge=0.0, le=1.0)
    canary_local: int = Field(ge=0)
    canary_fallback: int = Field(ge=0)
    truth_labeled: int = Field(ge=0)
    truth_correct: int = Field(ge=0)
    observed_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    local_truth_labeled: int = Field(ge=0)
    local_truth_correct: int = Field(ge=0)
    local_truth_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    control_truth_labeled: int = Field(ge=0)
    control_truth_correct: int = Field(ge=0)
    control_truth_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    canary_fallback_truth_labeled: int = Field(ge=0)
    canary_fallback_truth_correct: int = Field(ge=0)
    canary_fallback_truth_risk: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    route_outcome: RolloutRouteOutcomeMatrix

    @model_validator(mode="after")
    def _validate_counts(self) -> RolloutSummary:
        if self.control_assigned + self.canary_assigned != self.total:
            raise ValueError("Rollout cohort counts must equal total.")
        if self.local_served + self.fallback_served != self.total:
            raise ValueError("Rollout route counts must equal total.")
        if self.canary_local + self.canary_fallback != self.canary_assigned:
            raise ValueError("Canary route counts must equal canary_assigned.")
        if self.local_served != self.canary_local:
            raise ValueError("Only canary-assigned decisions may serve locally.")
        if self.fallback_served != self.control_assigned + self.canary_fallback:
            raise ValueError("Fallback count must include control and canary fallback routes.")

        _require_fraction(
            self.control_fraction,
            self.control_assigned / self.total,
            "control_fraction",
        )
        _require_fraction(
            self.canary_fraction,
            self.canary_assigned / self.total,
            "canary_fraction",
        )
        _require_fraction(
            self.local_serving_fraction,
            self.local_served / self.total,
            "local_serving_fraction",
        )
        _require_fraction(
            self.fallback_serving_fraction,
            self.fallback_served / self.total,
            "fallback_serving_fraction",
        )

        matrix = self.route_outcome
        control_total = (
            matrix.control_correct
            + matrix.control_incorrect
            + matrix.control_unlabeled
        )
        canary_local_total = (
            matrix.canary_local_correct
            + matrix.canary_local_incorrect
            + matrix.canary_local_unlabeled
        )
        canary_fallback_total = (
            matrix.canary_fallback_correct
            + matrix.canary_fallback_incorrect
            + matrix.canary_fallback_unlabeled
        )
        if control_total != self.control_assigned:
            raise ValueError("Control outcome matrix does not match control_assigned.")
        if canary_local_total != self.canary_local:
            raise ValueError("Canary-local outcome matrix does not match canary_local.")
        if canary_fallback_total != self.canary_fallback:
            raise ValueError("Canary-fallback outcome matrix does not match canary_fallback.")

        matrix_labeled = (
            matrix.control_correct
            + matrix.control_incorrect
            + matrix.canary_local_correct
            + matrix.canary_local_incorrect
            + matrix.canary_fallback_correct
            + matrix.canary_fallback_incorrect
        )
        matrix_correct = (
            matrix.control_correct
            + matrix.canary_local_correct
            + matrix.canary_fallback_correct
        )
        if self.truth_labeled != matrix_labeled or self.truth_correct != matrix_correct:
            raise ValueError("Rollout truth totals do not match route/outcome counts.")

        _require_truth_metric(
            self.truth_labeled,
            self.truth_correct,
            self.observed_risk,
            "observed_risk",
        )
        _require_group_metric(
            self.local_truth_labeled,
            self.local_truth_correct,
            self.local_truth_risk,
            matrix.canary_local_correct,
            matrix.canary_local_incorrect,
            "local truth",
        )
        _require_group_metric(
            self.control_truth_labeled,
            self.control_truth_correct,
            self.control_truth_risk,
            matrix.control_correct,
            matrix.control_incorrect,
            "control truth",
        )
        _require_group_metric(
            self.canary_fallback_truth_labeled,
            self.canary_fallback_truth_correct,
            self.canary_fallback_truth_risk,
            matrix.canary_fallback_correct,
            matrix.canary_fallback_incorrect,
            "canary fallback truth",
        )
        return self


class _CanaryRolloutReportIdentity(StrictModel):
    schema_version: Literal[1] = ROLLOUT_SCHEMA_VERSION
    rollout_version: Literal["1"] = ROLLOUT_IMPLEMENTATION_VERSION
    decision_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        min_length=1,
        max_length=128,
    )
    spec_version: str = Field(min_length=1, max_length=64)
    decision_spec_hash: str = Field(pattern=_SHA256_PATTERN)
    plan_id: str = Field(pattern=_SHA256_PATTERN)
    assessment_id: str = Field(pattern=_SHA256_PATTERN)
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    gate_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    source_evidence_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_evidence_count: int = Field(ge=1)
    source_label_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_label_count: int = Field(ge=0)
    config: RolloutConfig
    decisions: tuple[RolloutDecision, ...]
    summary: RolloutSummary

    @model_validator(mode="after")
    def _validate_shape(self) -> _CanaryRolloutReportIdentity:
        if self.source_evidence_count != len(self.decisions):
            raise ValueError("source_evidence_count must match rollout decision count.")
        if self.summary.total != len(self.decisions):
            raise ValueError("Rollout summary total must match decision count.")

        request_ids = tuple(decision.request_id for decision in self.decisions)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Rollout report contains duplicate request IDs.")
        if request_ids != tuple(sorted(request_ids, key=str)):
            raise ValueError("Rollout report decisions must be sorted by request ID.")

        evidence_ids = tuple(decision.evidence_id for decision in self.decisions)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Rollout report contains duplicate evidence IDs.")
        return self


class CanaryRolloutReport(_CanaryRolloutReportIdentity):
    report_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_report_id(self) -> CanaryRolloutReport:
        identity = _CanaryRolloutReportIdentity.model_validate(
            self.model_dump(mode="json", exclude={"report_id"})
        )
        if sha256_hex(identity) != self.report_id:
            raise ValueError("Canary rollout report identity does not match its content.")
        return self


def evaluate_canary_rollout(
    bundle: RuntimeBundle,
    assessment: PromotionAssessment,
    plan: CanaryPlan,
    evidence: Iterable[CanaryDecisionEvidence],
    labels: Iterable[LabelObservation],
    *,
    config: RolloutConfig | None = None,
) -> CanaryRolloutReport:
    """Evaluate truth-bearing live canary evidence without reusing teacher output."""
    bundle = RuntimeBundle.model_validate(bundle)
    assessment = PromotionAssessment.model_validate(assessment)
    plan = CanaryPlan.model_validate(plan)
    resolved_config = config or RolloutConfig()
    _validate_lineage(bundle, assessment, plan)

    ordered_evidence = tuple(
        sorted(
            (CanaryDecisionEvidence.model_validate(item) for item in evidence),
            key=lambda item: str(item.request_id),
        )
    )
    if not ordered_evidence:
        raise RolloutError("Canary rollout evaluation requires at least one decision.")

    request_ids = tuple(item.request_id for item in ordered_evidence)
    if len(request_ids) != len(set(request_ids)):
        raise RolloutError("Canary rollout evidence contains duplicate request IDs.")

    evidence_ids = tuple(item.evidence_id for item in ordered_evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise RolloutError("Canary rollout evidence contains duplicate evidence IDs.")

    _validate_evidence_lineage(bundle, assessment, plan, ordered_evidence)

    request_id_set = set(request_ids)
    validated_labels = tuple(LabelObservation.model_validate(label) for label in labels)
    relevant_labels = tuple(
        sorted(
            (
                label
                for label in validated_labels
                if label.request_id in request_id_set
            ),
            key=lambda label: (
                str(label.request_id),
                label.source.value,
                label.observed_at,
                str(label.observation_id),
            ),
        )
    )
    observation_ids = tuple(label.observation_id for label in relevant_labels)
    if len(observation_ids) != len(set(observation_ids)):
        raise RolloutError("Canary rollout labels contain duplicate observation IDs.")

    labels_by_request: dict[UUID, list[LabelObservation]] = {}
    for label in relevant_labels:
        labels_by_request.setdefault(label.request_id, []).append(label)

    spec = bundle.compiler.decision_spec
    decisions: list[RolloutDecision] = []
    for item in ordered_evidence:
        if item.result.abstained or item.result.selected is None:
            raise RolloutError(
                f"Canary evidence {item.evidence_id} does not contain an authoritative result."
            )
        _validate_value(
            spec.kind,
            spec.candidates,
            item.result.selected,
            item.request_id,
            "served decision",
        )

        truth = _resolve_truth(
            labels_by_request.get(item.request_id, []),
            resolved_config.truth_priority,
            item.request_id,
        )
        if truth is not None:
            truth_value, truth_source = truth
            _validate_value(
                spec.kind,
                spec.candidates,
                truth_value,
                item.request_id,
                "truth label",
            )
        else:
            truth_value = None
            truth_source = None

        decisions.append(
            RolloutDecision(
                request_id=item.request_id,
                evidence_id=item.evidence_id,
                cohort=item.cohort,
                route=item.result.route,
                selected=item.result.selected,
                truth=truth_value,
                truth_source=truth_source,
                correct=(
                    item.result.selected == truth_value
                    if truth_value is not None
                    else None
                ),
            )
        )

    decision_tuple = tuple(decisions)
    summary = _summarize(decision_tuple)
    identity = _CanaryRolloutReportIdentity(
        decision_id=assessment.decision_id,
        spec_version=assessment.spec_version,
        decision_spec_hash=assessment.decision_spec_hash,
        plan_id=plan.plan_id,
        assessment_id=assessment.assessment_id,
        compiler_artifact_id=bundle.compiler.artifact_id,
        calibration_artifact_id=bundle.calibration.artifact_id,
        gate_artifact_id=bundle.gate.artifact_id,
        source_evidence_sha256=sha256_hex(ordered_evidence),
        source_evidence_count=len(ordered_evidence),
        source_label_sha256=sha256_hex(relevant_labels),
        source_label_count=len(relevant_labels),
        config=resolved_config,
        decisions=decision_tuple,
        summary=summary,
    )
    return CanaryRolloutReport(
        report_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def write_canary_rollout_report(
    report: CanaryRolloutReport,
    root: str | Path,
) -> Path:
    destination = Path(root) / f"{report.report_id}.canary-rollout.json"
    write_immutable_atomic(
        destination,
        (canonical_json(report) + "\n").encode("utf-8"),
    )
    return destination


def load_canary_rollout_report(path: str | Path) -> CanaryRolloutReport:
    return CanaryRolloutReport.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _validate_lineage(
    bundle: RuntimeBundle,
    assessment: PromotionAssessment,
    plan: CanaryPlan,
) -> None:
    if not assessment.passed:
        raise RolloutError("Canary rollout evaluation requires a passing assessment.")
    if plan.assessment_id != assessment.assessment_id:
        raise RolloutError("Canary plan does not belong to the promotion assessment.")

    spec = bundle.compiler.decision_spec
    expected = (
        spec.decision_id,
        spec.version,
        sha256_hex(spec),
        bundle.compiler.artifact_id,
        bundle.calibration.artifact_id,
        bundle.gate.artifact_id,
    )
    assessment_lineage = (
        assessment.decision_id,
        assessment.spec_version,
        assessment.decision_spec_hash,
        assessment.compiler_artifact_id,
        assessment.calibration_artifact_id,
        assessment.gate_artifact_id,
    )
    plan_lineage = (
        plan.decision_id,
        plan.spec_version,
        plan.decision_spec_hash,
        plan.compiler_artifact_id,
        plan.calibration_artifact_id,
        plan.gate_artifact_id,
    )
    if assessment_lineage != expected:
        raise RolloutError("Promotion assessment does not match the runtime bundle.")
    if plan_lineage != expected:
        raise RolloutError("Canary plan does not match the runtime bundle.")


def _validate_evidence_lineage(
    bundle: RuntimeBundle,
    assessment: PromotionAssessment,
    plan: CanaryPlan,
    evidence: tuple[CanaryDecisionEvidence, ...],
) -> None:
    expected_artifacts = bundle.artifact_ids
    for item in evidence:
        if item.plan_id != plan.plan_id:
            raise RolloutError("Canary evidence contains a different plan ID.")
        if item.assessment_id != assessment.assessment_id:
            raise RolloutError("Canary evidence contains a different assessment ID.")
        if item.compiler_artifact_id != expected_artifacts["compiler"]:
            raise RolloutError("Canary evidence contains a different compiler artifact.")
        if item.calibration_artifact_id != expected_artifacts["calibration"]:
            raise RolloutError("Canary evidence contains a different calibration artifact.")
        if item.gate_artifact_id != expected_artifacts["gate"]:
            raise RolloutError("Canary evidence contains a different gate artifact.")


def _resolve_truth(
    labels: list[LabelObservation],
    priority: tuple[LabelSource, ...],
    request_id: UUID,
) -> tuple[str | bool, LabelSource] | None:
    for source in priority:
        source_labels = [label for label in labels if label.source is source]
        if not source_labels:
            continue

        values = {label.value for label in source_labels}
        if len(values) != 1:
            raise RolloutError(
                f"Conflicting {source.value} truth labels for request {request_id}."
            )
        return next(iter(values)), source
    return None


def _validate_value(
    kind: DecisionKind,
    candidates: tuple[Candidate, ...],
    value: str | bool,
    request_id: UUID,
    role: str,
) -> None:
    if kind is DecisionKind.BOOLEAN:
        if not isinstance(value, bool):
            raise RolloutError(
                f"{role} {value!r} is not boolean for request {request_id}."
            )
        return

    if not isinstance(value, str):
        raise RolloutError(
            f"{role} {value!r} is not a choice key for request {request_id}."
        )
    allowed = {candidate.key for candidate in candidates}
    if value not in allowed:
        raise RolloutError(
            f"{role} {value!r} is outside the DecisionSpec for request {request_id}."
        )


def _summarize(decisions: tuple[RolloutDecision, ...]) -> RolloutSummary:
    total = len(decisions)
    control = [decision for decision in decisions if decision.cohort == "control"]
    canary = [decision for decision in decisions if decision.cohort == "canary"]
    canary_local = [
        decision
        for decision in canary
        if decision.route is DecisionRoute.LOCAL
    ]
    canary_fallback = [
        decision
        for decision in canary
        if decision.route is DecisionRoute.FALLBACK
    ]
    fallback = [
        decision
        for decision in decisions
        if decision.route is DecisionRoute.FALLBACK
    ]

    matrix = RolloutRouteOutcomeMatrix(
        control_correct=_count_outcome(control, True),
        control_incorrect=_count_outcome(control, False),
        control_unlabeled=_count_outcome(control, None),
        canary_local_correct=_count_outcome(canary_local, True),
        canary_local_incorrect=_count_outcome(canary_local, False),
        canary_local_unlabeled=_count_outcome(canary_local, None),
        canary_fallback_correct=_count_outcome(canary_fallback, True),
        canary_fallback_incorrect=_count_outcome(canary_fallback, False),
        canary_fallback_unlabeled=_count_outcome(canary_fallback, None),
    )

    truth_labeled = sum(decision.correct is not None for decision in decisions)
    truth_correct = sum(decision.correct is True for decision in decisions)
    local_truth_labeled = sum(decision.correct is not None for decision in canary_local)
    local_truth_correct = sum(decision.correct is True for decision in canary_local)
    control_truth_labeled = sum(decision.correct is not None for decision in control)
    control_truth_correct = sum(decision.correct is True for decision in control)
    canary_fallback_truth_labeled = sum(
        decision.correct is not None for decision in canary_fallback
    )
    canary_fallback_truth_correct = sum(
        decision.correct is True for decision in canary_fallback
    )

    return RolloutSummary(
        total=total,
        control_assigned=len(control),
        canary_assigned=len(canary),
        control_fraction=len(control) / total,
        canary_fraction=len(canary) / total,
        local_served=len(canary_local),
        fallback_served=len(fallback),
        local_serving_fraction=len(canary_local) / total,
        fallback_serving_fraction=len(fallback) / total,
        canary_local=len(canary_local),
        canary_fallback=len(canary_fallback),
        truth_labeled=truth_labeled,
        truth_correct=truth_correct,
        observed_risk=_risk(truth_labeled, truth_correct),
        local_truth_labeled=local_truth_labeled,
        local_truth_correct=local_truth_correct,
        local_truth_risk=_risk(local_truth_labeled, local_truth_correct),
        control_truth_labeled=control_truth_labeled,
        control_truth_correct=control_truth_correct,
        control_truth_risk=_risk(control_truth_labeled, control_truth_correct),
        canary_fallback_truth_labeled=canary_fallback_truth_labeled,
        canary_fallback_truth_correct=canary_fallback_truth_correct,
        canary_fallback_truth_risk=_risk(
            canary_fallback_truth_labeled,
            canary_fallback_truth_correct,
        ),
        route_outcome=matrix,
    )


def _count_outcome(
    decisions: list[RolloutDecision],
    expected: bool | None,
) -> int:
    return sum(decision.correct is expected for decision in decisions)


def _risk(labeled: int, correct: int) -> float | None:
    return (labeled - correct) / labeled if labeled else None


def _require_fraction(observed: float, expected: float, name: str) -> None:
    if not math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError(f"{name} does not match its count-derived value.")


def _require_truth_metric(
    labeled: int,
    correct: int,
    risk: float | None,
    name: str,
) -> None:
    if correct > labeled:
        raise ValueError(f"{name} correct count exceeds labeled count.")
    expected = _risk(labeled, correct)
    if risk is None or expected is None:
        if risk is not expected:
            raise ValueError(f"{name} nullability does not match labeled count.")
        return
    _require_fraction(risk, expected, name)


def _require_group_metric(
    labeled: int,
    correct: int,
    risk: float | None,
    matrix_correct: int,
    matrix_incorrect: int,
    name: str,
) -> None:
    if labeled != matrix_correct + matrix_incorrect or correct != matrix_correct:
        raise ValueError(f"{name} counts do not match the route/outcome matrix.")
    _require_truth_metric(labeled, correct, risk, f"{name} risk")
