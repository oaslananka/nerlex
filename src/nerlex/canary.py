from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from nerlex.artifact_io import write_immutable_atomic
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.promotion import (
    CanaryPlan,
    PromotionAssessment,
    PromotionError,
    assign_canary,
)
from nerlex.runtime import (
    FallbackDecision,
    FallbackProvider,
    LocalCascadeRuntime,
    RuntimeBundle,
    RuntimeConfig,
    RuntimeRequestError,
)
from nerlex.spec import (
    DecisionRequest,
    DecisionResult,
    DecisionRoute,
    StrictModel,
    validate_request,
)

CANARY_DECISION_SCHEMA_VERSION: Literal[1] = 1
CANARY_RUNTIME_VERSION: Literal["1"] = "1"

_SHA256_PATTERN = r"^[a-f0-9]{64}$"


class CanaryRuntimeError(RuntimeError):
    """Base error for live canary serving failures."""


class CanaryArtifactError(CanaryRuntimeError, ValueError):
    """Raised when a canary plan/assessment/runtime lineage is incompatible."""


class CanaryRequestError(CanaryRuntimeError):
    """Raised when a canary assignment request is invalid."""


class _CanaryDecisionEvidenceIdentity(StrictModel):
    schema_version: Literal[1] = CANARY_DECISION_SCHEMA_VERSION
    runtime_version: Literal["1"] = CANARY_RUNTIME_VERSION
    plan_id: str = Field(pattern=_SHA256_PATTERN)
    assessment_id: str = Field(pattern=_SHA256_PATTERN)
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    gate_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    request_id: UUID
    cohort: Literal["control", "canary"]
    bucket: float = Field(ge=0.0, lt=1.0)
    result: DecisionResult

    @model_validator(mode="after")
    def _validate_shape(self) -> _CanaryDecisionEvidenceIdentity:
        if self.result.request_id != self.request_id:
            raise ValueError("Canary evidence/result request IDs do not match.")

        expected_artifacts = {
            "compiler": self.compiler_artifact_id,
            "calibration": self.calibration_artifact_id,
            "gate": self.gate_artifact_id,
        }
        for key, artifact_id in expected_artifacts.items():
            if self.result.artifact_ids.get(key) != artifact_id:
                raise ValueError(
                    f"Canary evidence result does not match the {key} artifact."
                )

        if self.cohort == "control":
            if self.result.route is not DecisionRoute.FALLBACK:
                raise ValueError("Control cohort must use the fallback/control route.")
            if self.result.abstain_reason != "control_cohort":
                raise ValueError("Control cohort result must preserve its control reason.")
        elif self.result.route not in {DecisionRoute.LOCAL, DecisionRoute.FALLBACK}:
            raise ValueError("Canary cohort result must use the local cascade routes.")
        return self


class CanaryDecisionEvidence(_CanaryDecisionEvidenceIdentity):
    evidence_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_evidence_id(self) -> CanaryDecisionEvidence:
        identity = _CanaryDecisionEvidenceIdentity.model_validate(
            self.model_dump(mode="json", exclude={"evidence_id"})
        )
        if sha256_hex(identity) != self.evidence_id:
            raise ValueError("Canary decision evidence identity does not match its content.")
        return self


class CanaryRuntime:
    """Apply an immutable canary plan while preserving route and cohort separately."""

    def __init__(
        self,
        bundle: RuntimeBundle,
        assessment: PromotionAssessment,
        plan: CanaryPlan,
        *,
        control: FallbackProvider | Callable[[DecisionRequest], FallbackDecision],
        config: RuntimeConfig | None = None,
    ) -> None:
        self.bundle = RuntimeBundle.model_validate(bundle)
        self.assessment = PromotionAssessment.model_validate(assessment)
        self.plan = CanaryPlan.model_validate(plan)
        _validate_runtime_lineage(self.bundle, self.assessment, self.plan)
        self._runtime = LocalCascadeRuntime(
            self.bundle,
            fallback=control,
            config=config,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._runtime.close()

    def decide(
        self,
        request: DecisionRequest,
        *,
        assignment_key: str,
    ) -> CanaryDecisionEvidence:
        request = DecisionRequest.model_validate(request)
        try:
            validate_request(self.bundle.compiler.decision_spec, request)
        except ValueError as exc:
            raise RuntimeRequestError(str(exc)) from exc

        try:
            assignment = assign_canary(
                self.plan,
                request_id=request.request_id,
                decision_id=request.decision_id,
                spec_version=request.spec_version,
                assignment_key=assignment_key,
            )
        except PromotionError as exc:
            raise CanaryRequestError(str(exc)) from exc

        result = (
            self._runtime.decide_fallback(request, reason="control_cohort")
            if assignment.cohort == "control"
            else self._runtime.decide(request)
        )
        identity = _CanaryDecisionEvidenceIdentity(
            plan_id=self.plan.plan_id,
            assessment_id=self.assessment.assessment_id,
            compiler_artifact_id=self.bundle.compiler.artifact_id,
            calibration_artifact_id=self.bundle.calibration.artifact_id,
            gate_artifact_id=self.bundle.gate.artifact_id,
            request_id=request.request_id,
            cohort=assignment.cohort,
            bucket=assignment.bucket,
            result=result,
        )
        return CanaryDecisionEvidence(
            evidence_id=sha256_hex(identity),
            **identity.model_dump(mode="python"),
        )


def write_canary_decision_evidence(
    evidence: CanaryDecisionEvidence,
    root: str | Path,
) -> Path:
    destination = Path(root) / f"{evidence.evidence_id}.canary-decision.json"
    write_immutable_atomic(
        destination,
        (canonical_json(evidence) + "\n").encode("utf-8"),
    )
    return destination


def load_canary_decision_evidence(path: str | Path) -> CanaryDecisionEvidence:
    return CanaryDecisionEvidence.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _validate_runtime_lineage(
    bundle: RuntimeBundle,
    assessment: PromotionAssessment,
    plan: CanaryPlan,
) -> None:
    if not assessment.passed:
        raise CanaryArtifactError("Live canary runtime requires a passing assessment.")
    if plan.assessment_id != assessment.assessment_id:
        raise CanaryArtifactError("Canary plan does not belong to the promotion assessment.")

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
        raise CanaryArtifactError("Promotion assessment does not match the runtime bundle.")
    if plan_lineage != expected:
        raise CanaryArtifactError("Canary plan does not match the runtime bundle.")
