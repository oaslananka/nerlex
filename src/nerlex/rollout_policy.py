from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from nerlex.artifact_io import write_immutable_atomic
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.rollout import CanaryRolloutReport
from nerlex.spec import StrictModel

ROLLOUT_ACTION_SCHEMA_VERSION: Literal[1] = 1
ROLLOUT_ACTION_IMPLEMENTATION_VERSION: Literal["1"] = "1"

_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_FLOAT_TOLERANCE = 1e-9


class RolloutAction(StrEnum):
    ADVANCE = "advance"
    HOLD = "hold"
    ROLLBACK = "rollback"


AdvanceFailure = Literal[
    "insufficient_local_truth",
    "insufficient_control_truth",
    "insufficient_local_serving",
    "local_truth_risk",
    "local_vs_control_risk_delta",
]
RollbackTrigger = Literal[
    "local_truth_risk",
    "local_vs_control_risk_delta",
]


class RolloutActionPolicy(StrictModel):
    advance_min_local_truth_labeled: int = Field(ge=1)
    advance_min_control_truth_labeled: int = Field(default=0, ge=0)
    advance_min_local_serving_fraction: float = Field(ge=0.0, le=1.0)
    advance_max_local_truth_risk: float = Field(ge=0.0, le=1.0)
    advance_max_local_vs_control_risk_delta: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    rollback_min_local_truth_labeled: int = Field(ge=1)
    rollback_min_control_truth_labeled: int = Field(default=0, ge=0)
    rollback_local_truth_risk_at_least: float = Field(ge=0.0, le=1.0)
    rollback_local_vs_control_risk_delta_at_least: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def _validate_threshold_relationships(self) -> RolloutActionPolicy:
        if (
            self.rollback_local_truth_risk_at_least
            <= self.advance_max_local_truth_risk
        ):
            raise ValueError(
                "Rollback local-risk threshold must be above the advance threshold."
            )

        advance_delta = self.advance_max_local_vs_control_risk_delta
        rollback_delta = self.rollback_local_vs_control_risk_delta_at_least
        if advance_delta is not None and self.advance_min_control_truth_labeled < 1:
            raise ValueError(
                "Advance risk-delta criteria require control truth evidence."
            )
        if rollback_delta is not None and self.rollback_min_control_truth_labeled < 1:
            raise ValueError(
                "Rollback risk-delta criteria require control truth evidence."
            )
        if (
            advance_delta is not None
            and rollback_delta is not None
            and rollback_delta <= advance_delta
        ):
            raise ValueError(
                "Rollback risk-delta threshold must be above the advance threshold."
            )
        return self


class _RolloutActionAssessmentIdentity(StrictModel):
    schema_version: Literal[1] = ROLLOUT_ACTION_SCHEMA_VERSION
    action_version: Literal["1"] = ROLLOUT_ACTION_IMPLEMENTATION_VERSION
    report_id: str = Field(pattern=_SHA256_PATTERN)
    plan_id: str = Field(pattern=_SHA256_PATTERN)
    promotion_assessment_id: str = Field(pattern=_SHA256_PATTERN)
    policy: RolloutActionPolicy
    local_truth_labeled: int = Field(ge=0)
    control_truth_labeled: int = Field(ge=0)
    local_serving_fraction: float = Field(ge=0.0, le=1.0)
    local_truth_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    control_truth_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    local_vs_control_risk_delta: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    advance_failures: tuple[AdvanceFailure, ...]
    rollback_triggers: tuple[RollbackTrigger, ...]
    action: RolloutAction

    @model_validator(mode="after")
    def _validate_assessment_shape(self) -> _RolloutActionAssessmentIdentity:
        expected_delta = _risk_delta(
            self.local_truth_risk,
            self.control_truth_risk,
        )
        _require_optional_float(
            self.local_vs_control_risk_delta,
            expected_delta,
            "local_vs_control_risk_delta",
        )

        expected_failures = _advance_failures(
            self.policy,
            local_truth_labeled=self.local_truth_labeled,
            control_truth_labeled=self.control_truth_labeled,
            local_serving_fraction=self.local_serving_fraction,
            local_truth_risk=self.local_truth_risk,
            local_vs_control_risk_delta=expected_delta,
        )
        if self.advance_failures != expected_failures:
            raise ValueError("Advance failure reasons do not match observed evidence.")

        expected_triggers = _rollback_triggers(
            self.policy,
            local_truth_labeled=self.local_truth_labeled,
            control_truth_labeled=self.control_truth_labeled,
            local_truth_risk=self.local_truth_risk,
            local_vs_control_risk_delta=expected_delta,
        )
        if self.rollback_triggers != expected_triggers:
            raise ValueError("Rollback triggers do not match observed evidence.")

        expected_action = _action(expected_failures, expected_triggers)
        if self.action is not expected_action:
            raise ValueError("Rollout action does not match policy evaluation.")
        return self


class RolloutActionAssessment(_RolloutActionAssessmentIdentity):
    assessment_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_assessment_id(self) -> RolloutActionAssessment:
        identity = _RolloutActionAssessmentIdentity.model_validate(
            self.model_dump(mode="json", exclude={"assessment_id"})
        )
        if sha256_hex(identity) != self.assessment_id:
            raise ValueError("Rollout action assessment identity does not match content.")
        return self


def assess_rollout_action(
    report: CanaryRolloutReport,
    policy: RolloutActionPolicy,
) -> RolloutActionAssessment:
    """Recommend advance, hold, or rollback without mutating rollout state."""
    report = CanaryRolloutReport.model_validate(report)
    policy = RolloutActionPolicy.model_validate(policy)
    summary = report.summary

    delta = _risk_delta(
        summary.local_truth_risk,
        summary.control_truth_risk,
    )
    failures = _advance_failures(
        policy,
        local_truth_labeled=summary.local_truth_labeled,
        control_truth_labeled=summary.control_truth_labeled,
        local_serving_fraction=summary.local_serving_fraction,
        local_truth_risk=summary.local_truth_risk,
        local_vs_control_risk_delta=delta,
    )
    triggers = _rollback_triggers(
        policy,
        local_truth_labeled=summary.local_truth_labeled,
        control_truth_labeled=summary.control_truth_labeled,
        local_truth_risk=summary.local_truth_risk,
        local_vs_control_risk_delta=delta,
    )

    identity = _RolloutActionAssessmentIdentity(
        report_id=report.report_id,
        plan_id=report.plan_id,
        promotion_assessment_id=report.assessment_id,
        policy=policy,
        local_truth_labeled=summary.local_truth_labeled,
        control_truth_labeled=summary.control_truth_labeled,
        local_serving_fraction=summary.local_serving_fraction,
        local_truth_risk=summary.local_truth_risk,
        control_truth_risk=summary.control_truth_risk,
        local_vs_control_risk_delta=delta,
        advance_failures=failures,
        rollback_triggers=triggers,
        action=_action(failures, triggers),
    )
    return RolloutActionAssessment(
        assessment_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def write_rollout_action_assessment(
    assessment: RolloutActionAssessment,
    root: str | Path,
) -> Path:
    destination = Path(root) / f"{assessment.assessment_id}.rollout-action.json"
    write_immutable_atomic(
        destination,
        (canonical_json(assessment) + "\n").encode("utf-8"),
    )
    return destination


def load_rollout_action_assessment(path: str | Path) -> RolloutActionAssessment:
    return RolloutActionAssessment.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _advance_failures(
    policy: RolloutActionPolicy,
    *,
    local_truth_labeled: int,
    control_truth_labeled: int,
    local_serving_fraction: float,
    local_truth_risk: float | None,
    local_vs_control_risk_delta: float | None,
) -> tuple[AdvanceFailure, ...]:
    failures: list[AdvanceFailure] = []

    local_sufficient = (
        local_truth_labeled >= policy.advance_min_local_truth_labeled
    )
    control_sufficient = (
        control_truth_labeled >= policy.advance_min_control_truth_labeled
    )
    if not local_sufficient:
        failures.append("insufficient_local_truth")
    if not control_sufficient:
        failures.append("insufficient_control_truth")
    if local_serving_fraction < policy.advance_min_local_serving_fraction:
        failures.append("insufficient_local_serving")

    if (
        local_sufficient
        and (
            local_truth_risk is None
            or local_truth_risk > policy.advance_max_local_truth_risk
        )
    ):
        failures.append("local_truth_risk")

    delta_limit = policy.advance_max_local_vs_control_risk_delta
    if (
        delta_limit is not None
        and local_sufficient
        and control_sufficient
        and (
            local_vs_control_risk_delta is None
            or local_vs_control_risk_delta > delta_limit
        )
    ):
        failures.append("local_vs_control_risk_delta")

    return tuple(failures)


def _rollback_triggers(
    policy: RolloutActionPolicy,
    *,
    local_truth_labeled: int,
    control_truth_labeled: int,
    local_truth_risk: float | None,
    local_vs_control_risk_delta: float | None,
) -> tuple[RollbackTrigger, ...]:
    triggers: list[RollbackTrigger] = []

    local_sufficient = (
        local_truth_labeled >= policy.rollback_min_local_truth_labeled
    )
    if (
        local_sufficient
        and local_truth_risk is not None
        and local_truth_risk >= policy.rollback_local_truth_risk_at_least
    ):
        triggers.append("local_truth_risk")

    delta_threshold = policy.rollback_local_vs_control_risk_delta_at_least
    control_sufficient = (
        control_truth_labeled >= policy.rollback_min_control_truth_labeled
    )
    if (
        delta_threshold is not None
        and local_sufficient
        and control_sufficient
        and local_vs_control_risk_delta is not None
        and local_vs_control_risk_delta >= delta_threshold
    ):
        triggers.append("local_vs_control_risk_delta")

    return tuple(triggers)


def _action(
    advance_failures: tuple[AdvanceFailure, ...],
    rollback_triggers: tuple[RollbackTrigger, ...],
) -> RolloutAction:
    if rollback_triggers:
        return RolloutAction.ROLLBACK
    if not advance_failures:
        return RolloutAction.ADVANCE
    return RolloutAction.HOLD


def _risk_delta(
    local_truth_risk: float | None,
    control_truth_risk: float | None,
) -> float | None:
    if local_truth_risk is None or control_truth_risk is None:
        return None
    return local_truth_risk - control_truth_risk


def _require_optional_float(
    observed: float | None,
    expected: float | None,
    name: str,
) -> None:
    if observed is None or expected is None:
        if observed is not expected:
            raise ValueError(f"{name} nullability does not match observed risks.")
        return
    if not math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError(f"{name} does not match observed risks.")
