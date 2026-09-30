from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from nerlex.artifact_io import write_immutable_atomic
from nerlex.evaluation import EvaluationReport
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.runtime import RuntimeBundle
from nerlex.shadow import ShadowReport
from nerlex.spec import StrictModel

PROMOTION_SCHEMA_VERSION: Literal[1] = 1
PROMOTION_IMPLEMENTATION_VERSION: Literal["1"] = "1"
CANARY_SCHEMA_VERSION: Literal[1] = 1
CANARY_IMPLEMENTATION_VERSION: Literal["1"] = "1"
CANARY_ASSIGNMENT_VERSION: Literal["sha256-bucket-v1"] = "sha256-bucket-v1"

_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_BUCKET_SPACE = 1 << 64


class PromotionError(ValueError):
    """Raised when promotion evidence, plans, or assignments are invalid."""


class PromotionPolicy(StrictModel):
    max_test_selective_risk: float = Field(ge=0.0, le=1.0)
    min_test_selective_coverage: float = Field(ge=0.0, le=1.0)
    min_shadow_eligible_truth_labeled: int = Field(ge=1)
    max_shadow_eligible_truth_risk: float = Field(ge=0.0, le=1.0)
    min_shadow_local_coverage: float = Field(ge=0.0, le=1.0)


PromotionCriterionName = Literal[
    "test_selective_risk",
    "test_selective_coverage",
    "shadow_eligible_truth_labeled",
    "shadow_eligible_truth_risk",
    "shadow_local_coverage",
]
PromotionComparison = Literal["at_most", "at_least"]


class PromotionCriterionResult(StrictModel):
    name: PromotionCriterionName
    comparison: PromotionComparison
    observed: float | int | None
    required: float | int
    passed: bool


_EXPECTED_CRITERIA: tuple[PromotionCriterionName, ...] = (
    "test_selective_risk",
    "test_selective_coverage",
    "shadow_eligible_truth_labeled",
    "shadow_eligible_truth_risk",
    "shadow_local_coverage",
)


class _PromotionAssessmentIdentity(StrictModel):
    schema_version: Literal[1] = PROMOTION_SCHEMA_VERSION
    promotion_version: Literal["1"] = PROMOTION_IMPLEMENTATION_VERSION
    decision_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        min_length=1,
        max_length=128,
    )
    spec_version: str = Field(min_length=1, max_length=64)
    decision_spec_hash: str = Field(pattern=_SHA256_PATTERN)
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    gate_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    evaluation_report_id: str = Field(pattern=_SHA256_PATTERN)
    shadow_report_id: str = Field(pattern=_SHA256_PATTERN)
    policy: PromotionPolicy
    criteria: tuple[PromotionCriterionResult, ...]
    passed: bool

    @model_validator(mode="after")
    def _validate_shape(self) -> _PromotionAssessmentIdentity:
        names = tuple(criterion.name for criterion in self.criteria)
        if names != _EXPECTED_CRITERIA:
            raise ValueError("Promotion assessment criteria are incomplete or out of order.")

        requirements: tuple[
            tuple[PromotionCriterionName, PromotionComparison, float | int], ...
        ] = (
            ("test_selective_risk", "at_most", self.policy.max_test_selective_risk),
            (
                "test_selective_coverage",
                "at_least",
                self.policy.min_test_selective_coverage,
            ),
            (
                "shadow_eligible_truth_labeled",
                "at_least",
                self.policy.min_shadow_eligible_truth_labeled,
            ),
            (
                "shadow_eligible_truth_risk",
                "at_most",
                self.policy.max_shadow_eligible_truth_risk,
            ),
            (
                "shadow_local_coverage",
                "at_least",
                self.policy.min_shadow_local_coverage,
            ),
        )
        for criterion, (name, comparison, required) in zip(
            self.criteria,
            requirements,
            strict=True,
        ):
            if criterion.name != name or criterion.comparison != comparison:
                raise ValueError("Promotion criterion definition does not match the policy.")
            if criterion.required != required:
                raise ValueError("Promotion criterion threshold does not match the policy.")
            expected_pass = criterion.observed is not None and (
                criterion.observed <= required
                if comparison == "at_most"
                else criterion.observed >= required
            )
            if criterion.passed != expected_pass:
                raise ValueError("Promotion criterion result does not match its values.")

        if self.passed != all(criterion.passed for criterion in self.criteria):
            raise ValueError("Promotion assessment result does not match criterion results.")
        return self


class PromotionAssessment(_PromotionAssessmentIdentity):
    assessment_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_assessment_id(self) -> PromotionAssessment:
        identity = _PromotionAssessmentIdentity.model_validate(
            self.model_dump(mode="json", exclude={"assessment_id"})
        )
        if sha256_hex(identity) != self.assessment_id:
            raise ValueError("Promotion assessment identity does not match its content.")
        return self


class _CanaryPlanIdentity(StrictModel):
    schema_version: Literal[1] = CANARY_SCHEMA_VERSION
    canary_version: Literal["1"] = CANARY_IMPLEMENTATION_VERSION
    assignment_version: Literal["sha256-bucket-v1"] = CANARY_ASSIGNMENT_VERSION
    assessment_id: str = Field(pattern=_SHA256_PATTERN)
    decision_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        min_length=1,
        max_length=128,
    )
    spec_version: str = Field(min_length=1, max_length=64)
    decision_spec_hash: str = Field(pattern=_SHA256_PATTERN)
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    gate_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    rollout_fraction: float = Field(ge=0.0, le=1.0)
    assignment_seed: str = Field(min_length=1, max_length=256)
    previous_plan_id: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class CanaryPlan(_CanaryPlanIdentity):
    plan_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_plan_id(self) -> CanaryPlan:
        identity = _CanaryPlanIdentity.model_validate(
            self.model_dump(mode="json", exclude={"plan_id"})
        )
        if sha256_hex(identity) != self.plan_id:
            raise ValueError("Canary plan identity does not match its content.")
        return self


class CanaryAssignment(StrictModel):
    plan_id: str = Field(pattern=_SHA256_PATTERN)
    request_id: UUID
    bucket: float = Field(ge=0.0, lt=1.0)
    cohort: Literal["control", "canary"]


def assess_promotion(
    bundle: RuntimeBundle,
    evaluation: EvaluationReport,
    shadow: ShadowReport,
    policy: PromotionPolicy,
) -> PromotionAssessment:
    """Assess immutable evaluation and shadow evidence against an explicit policy."""
    bundle = RuntimeBundle.model_validate(bundle)
    evaluation = EvaluationReport.model_validate(evaluation)
    shadow = ShadowReport.model_validate(shadow)
    policy = PromotionPolicy.model_validate(policy)
    _validate_evidence_lineage(bundle, evaluation, shadow)

    selective = evaluation.selective
    if selective is None:
        raise PromotionError("Promotion assessment requires a gate-bound evaluation report.")

    shadow_truth_count = shadow.summary.eligible_truth_labeled
    shadow_truth_risk = (
        (shadow_truth_count - shadow.summary.eligible_truth_correct) / shadow_truth_count
        if shadow_truth_count
        else None
    )
    criteria = (
        _at_most(
            "test_selective_risk",
            selective.risk,
            policy.max_test_selective_risk,
        ),
        _at_least(
            "test_selective_coverage",
            selective.coverage,
            policy.min_test_selective_coverage,
        ),
        _at_least(
            "shadow_eligible_truth_labeled",
            shadow_truth_count,
            policy.min_shadow_eligible_truth_labeled,
        ),
        _at_most(
            "shadow_eligible_truth_risk",
            shadow_truth_risk,
            policy.max_shadow_eligible_truth_risk,
        ),
        _at_least(
            "shadow_local_coverage",
            shadow.summary.local_coverage,
            policy.min_shadow_local_coverage,
        ),
    )
    spec = bundle.compiler.decision_spec
    identity = _PromotionAssessmentIdentity(
        decision_id=spec.decision_id,
        spec_version=spec.version,
        decision_spec_hash=sha256_hex(spec),
        compiler_artifact_id=bundle.compiler.artifact_id,
        calibration_artifact_id=bundle.calibration.artifact_id,
        gate_artifact_id=bundle.gate.artifact_id,
        evaluation_report_id=evaluation.report_id,
        shadow_report_id=shadow.report_id,
        policy=policy,
        criteria=criteria,
        passed=all(criterion.passed for criterion in criteria),
    )
    return PromotionAssessment(
        assessment_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def create_canary_plan(
    assessment: PromotionAssessment,
    *,
    rollout_fraction: float,
    assignment_seed: str,
    previous_plan: CanaryPlan | None = None,
) -> CanaryPlan:
    """Create an immutable canary plan from a passing promotion assessment."""
    assessment = PromotionAssessment.model_validate(assessment)
    if not assessment.passed:
        raise PromotionError("Canary plans require a passing promotion assessment.")

    previous = CanaryPlan.model_validate(previous_plan) if previous_plan is not None else None
    if previous is not None:
        _validate_previous_plan(previous, assessment, assignment_seed)

    identity = _CanaryPlanIdentity(
        assessment_id=assessment.assessment_id,
        decision_id=assessment.decision_id,
        spec_version=assessment.spec_version,
        decision_spec_hash=assessment.decision_spec_hash,
        compiler_artifact_id=assessment.compiler_artifact_id,
        calibration_artifact_id=assessment.calibration_artifact_id,
        gate_artifact_id=assessment.gate_artifact_id,
        rollout_fraction=rollout_fraction,
        assignment_seed=assignment_seed,
        previous_plan_id=previous.plan_id if previous is not None else None,
    )
    return CanaryPlan(
        plan_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def assign_canary(
    plan: CanaryPlan,
    *,
    request_id: UUID,
    decision_id: str,
    spec_version: str,
    assignment_key: str,
) -> CanaryAssignment:
    """Assign a stable application-defined exposure key to control or canary."""
    plan = CanaryPlan.model_validate(plan)
    if decision_id != plan.decision_id or spec_version != plan.spec_version:
        raise PromotionError("Canary assignment does not match the plan decision spec.")
    if not 1 <= len(assignment_key) <= 512:
        raise PromotionError("Canary assignment key must contain between 1 and 512 characters.")

    digest = sha256_hex(
        {
            "assignment_version": plan.assignment_version,
            "assignment_seed": plan.assignment_seed,
            "decision_id": decision_id,
            "spec_version": spec_version,
            "assignment_key": assignment_key,
        }
    )
    bucket_value = int(digest[:16], 16)
    bucket = bucket_value / _BUCKET_SPACE
    cohort: Literal["control", "canary"] = (
        "canary" if bucket < plan.rollout_fraction else "control"
    )
    return CanaryAssignment(
        plan_id=plan.plan_id,
        request_id=request_id,
        bucket=bucket,
        cohort=cohort,
    )


def write_promotion_assessment(
    assessment: PromotionAssessment,
    root: str | Path,
) -> Path:
    destination = Path(root) / f"{assessment.assessment_id}.promotion.json"
    write_immutable_atomic(
        destination,
        (canonical_json(assessment) + "\n").encode("utf-8"),
    )
    return destination


def load_promotion_assessment(path: str | Path) -> PromotionAssessment:
    return PromotionAssessment.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_canary_plan(plan: CanaryPlan, root: str | Path) -> Path:
    destination = Path(root) / f"{plan.plan_id}.canary.json"
    write_immutable_atomic(
        destination,
        (canonical_json(plan) + "\n").encode("utf-8"),
    )
    return destination


def load_canary_plan(path: str | Path) -> CanaryPlan:
    return CanaryPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _at_most(
    name: PromotionCriterionName,
    observed: float | int | None,
    required: float | int,
) -> PromotionCriterionResult:
    return PromotionCriterionResult(
        name=name,
        comparison="at_most",
        observed=observed,
        required=required,
        passed=observed is not None and observed <= required,
    )


def _at_least(
    name: PromotionCriterionName,
    observed: float | int | None,
    required: float | int,
) -> PromotionCriterionResult:
    return PromotionCriterionResult(
        name=name,
        comparison="at_least",
        observed=observed,
        required=required,
        passed=observed is not None and observed >= required,
    )


def _validate_evidence_lineage(
    bundle: RuntimeBundle,
    evaluation: EvaluationReport,
    shadow: ShadowReport,
) -> None:
    expected = bundle.artifact_ids
    if evaluation.compiler_artifact_id != expected["compiler"]:
        raise PromotionError("Evaluation report does not match the compiler artifact.")
    if evaluation.calibration_artifact_id != expected["calibration"]:
        raise PromotionError("Evaluation report does not match the calibration artifact.")
    if evaluation.gate_artifact_id != expected["gate"]:
        raise PromotionError("Evaluation report does not match the empirical gate artifact.")
    if evaluation.snapshot_id != bundle.compiler.snapshot_id:
        raise PromotionError("Evaluation report does not match the compiler snapshot.")

    if shadow.compiler_artifact_id != expected["compiler"]:
        raise PromotionError("Shadow report does not match the compiler artifact.")
    if shadow.calibration_artifact_id != expected["calibration"]:
        raise PromotionError("Shadow report does not match the calibration artifact.")
    if shadow.gate_artifact_id != expected["gate"]:
        raise PromotionError("Shadow report does not match the empirical gate artifact.")
    if shadow.decision_spec_hash != sha256_hex(bundle.compiler.decision_spec):
        raise PromotionError("Shadow report does not match the compiled DecisionSpec.")


def _validate_previous_plan(
    previous: CanaryPlan,
    assessment: PromotionAssessment,
    assignment_seed: str,
) -> None:
    if previous.decision_spec_hash != assessment.decision_spec_hash:
        raise PromotionError("Previous canary plan uses a different DecisionSpec.")
    if previous.compiler_artifact_id != assessment.compiler_artifact_id:
        raise PromotionError("Previous canary plan uses a different compiler artifact.")
    if previous.calibration_artifact_id != assessment.calibration_artifact_id:
        raise PromotionError("Previous canary plan uses a different calibration artifact.")
    if previous.gate_artifact_id != assessment.gate_artifact_id:
        raise PromotionError("Previous canary plan uses a different gate artifact.")
    if previous.assignment_seed != assignment_seed:
        raise PromotionError("Canary ramp chains must preserve the assignment seed.")
