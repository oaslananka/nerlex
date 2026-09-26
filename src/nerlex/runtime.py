from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from time import perf_counter
from types import TracebackType
from typing import Protocol, Self

from pydantic import Field, model_validator

from nerlex.calibration import (
    CalibratedPrediction,
    TemperatureCalibrationArtifact,
    predict_calibrated,
)
from nerlex.compiler import CompilerArtifact
from nerlex.evaluation import EmpiricalRiskGateArtifact
from nerlex.spec import (
    DecisionKind,
    DecisionRequest,
    DecisionResult,
    DecisionRoute,
    StrictModel,
    validate_request,
)

_FLOAT_TOLERANCE = 1e-9


class LocalRuntimeError(RuntimeError):
    """Base error for local cascade runtime failures."""


class RuntimeArtifactError(LocalRuntimeError, ValueError):
    """Raised when runtime artifacts do not have compatible lineage."""


class RuntimeRequestError(LocalRuntimeError):
    """Raised when a request is incompatible with the compiled decision spec."""


class FallbackExecutionError(LocalRuntimeError):
    """Raised when the configured fallback fails."""


class FallbackTimeoutError(FallbackExecutionError):
    """Raised when the configured fallback exceeds its runtime timeout."""


class RuntimeConfig(StrictModel):
    fallback_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    fallback_workers: int = Field(default=4, ge=1, le=64)


class RuntimeBundle(StrictModel):
    compiler: CompilerArtifact
    calibration: TemperatureCalibrationArtifact
    gate: EmpiricalRiskGateArtifact

    @model_validator(mode="after")
    def _validate_lineage(self) -> RuntimeBundle:
        compiler = self.compiler
        calibration = self.calibration
        gate = self.gate

        if calibration.compiler_artifact_id != compiler.artifact_id:
            raise RuntimeArtifactError(
                "Calibration artifact does not belong to the compiler artifact."
            )
        if calibration.snapshot_id != compiler.snapshot_id:
            raise RuntimeArtifactError("Calibration/compiler snapshot IDs do not match.")
        if calibration.classes != tuple(sorted(compiler.model.classes)):
            raise RuntimeArtifactError("Calibration/compiler class sets do not match.")

        if gate.compiler_artifact_id != compiler.artifact_id:
            raise RuntimeArtifactError("Gate artifact does not belong to the compiler artifact.")
        if gate.calibration_artifact_id != calibration.artifact_id:
            raise RuntimeArtifactError("Gate artifact does not belong to the calibration artifact.")
        if gate.snapshot_id != compiler.snapshot_id:
            raise RuntimeArtifactError("Gate/compiler snapshot IDs do not match.")
        if gate.calibration_sha256 != calibration.calibration_sha256:
            raise RuntimeArtifactError("Gate/calibration split hashes do not match.")
        if gate.calibration_count != calibration.calibration_count:
            raise RuntimeArtifactError("Gate/calibration split counts do not match.")
        return self

    @property
    def artifact_ids(self) -> dict[str, str]:
        return {
            "compiler": self.compiler.artifact_id,
            "calibration": self.calibration.artifact_id,
            "gate": self.gate.artifact_id,
        }


class FallbackDecision(StrictModel):
    selected: str | bool
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated: bool = False
    backend: str = Field(min_length=1, max_length=256)
    backend_version: str | None = Field(default=None, max_length=256)
    artifact_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _validate_probabilities(self) -> FallbackDecision:
        if any(value < 0.0 or value > 1.0 for value in self.probabilities.values()):
            raise ValueError("Fallback probabilities must be in [0, 1].")
        if self.probabilities:
            if abs(sum(self.probabilities.values()) - 1.0) > 1e-6:
                raise ValueError("Fallback probabilities must sum to 1.")
            selected_key = _selected_key(self.selected)
            if selected_key not in self.probabilities:
                raise ValueError("Fallback selected value must exist in probabilities.")
        return self


class FallbackProvider(Protocol):
    def __call__(self, request: DecisionRequest) -> FallbackDecision: ...


class LocalCascadeRuntime:
    def __init__(
        self,
        bundle: RuntimeBundle,
        *,
        fallback: FallbackProvider | Callable[[DecisionRequest], FallbackDecision] | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.bundle = RuntimeBundle.model_validate(bundle)
        self.fallback = fallback
        self.config = config or RuntimeConfig()
        self._fallback_executor = (
            ThreadPoolExecutor(
                max_workers=self.config.fallback_workers,
                thread_name_prefix="nerlex-fallback",
            )
            if fallback is not None
            else None
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
        executor = self._fallback_executor
        if executor is None:
            return
        self._fallback_executor = None
        executor.shutdown(wait=False, cancel_futures=True)

    def decide(self, request: DecisionRequest) -> DecisionResult:
        started = perf_counter()
        self._validate_request(request)

        try:
            local = predict_calibrated(
                self.bundle.compiler,
                self.bundle.calibration,
                request.state,
            )
        except Exception as exc:
            raise LocalRuntimeError("Local calibrated prediction failed.") from exc

        threshold = self.bundle.gate.confidence_threshold
        if threshold is not None and local.confidence + _FLOAT_TOLERANCE >= threshold:
            return self._local_result(request, local, started)

        abstain_reason = (
            "gate_abstain_all"
            if threshold is None
            else "below_calibrated_confidence_threshold"
        )
        if self.fallback is None:
            return self._local_result(
                request,
                local,
                started,
                abstain_reason=abstain_reason,
            )

        fallback = self._run_fallback(request)
        self._validate_fallback_decision(fallback)

        confidence = fallback.confidence
        if confidence is None and fallback.probabilities:
            key = _selected_key(fallback.selected)
            confidence = fallback.probabilities[key]

        artifact_ids = {
            **self.bundle.artifact_ids,
            **(
                {"fallback": fallback.artifact_id}
                if fallback.artifact_id is not None
                else {}
            ),
        }
        return DecisionResult(
            request_id=request.request_id,
            selected=fallback.selected,
            probabilities=fallback.probabilities,
            confidence=confidence,
            calibrated=fallback.calibrated,
            backend=fallback.backend,
            backend_version=fallback.backend_version,
            artifact_id=fallback.artifact_id,
            artifact_ids=artifact_ids,
            latency_ms=_elapsed_ms(started),
            route=DecisionRoute.FALLBACK,
            abstained=False,
            abstain_reason=abstain_reason,
        )

    def _local_result(
        self,
        request: DecisionRequest,
        prediction: CalibratedPrediction,
        started: float,
        *,
        abstain_reason: str | None = None,
    ) -> DecisionResult:
        abstained = abstain_reason is not None
        return DecisionResult(
            request_id=request.request_id,
            selected=None if abstained else prediction.selected,
            probabilities=prediction.probabilities,
            confidence=prediction.confidence,
            calibrated=True,
            backend=prediction.backend,
            backend_version=self.bundle.compiler.compiler_version,
            artifact_id=self.bundle.compiler.artifact_id,
            artifact_ids=self.bundle.artifact_ids,
            latency_ms=_elapsed_ms(started),
            route=DecisionRoute.LOCAL,
            abstained=abstained,
            abstain_reason=abstain_reason,
        )

    def _validate_request(self, request: DecisionRequest) -> None:
        try:
            validate_request(self.bundle.compiler.decision_spec, request)
        except ValueError as exc:
            raise RuntimeRequestError(str(exc)) from exc

    def _run_fallback(self, request: DecisionRequest) -> FallbackDecision:
        if self.fallback is None:
            raise AssertionError("Fallback execution requires a configured provider.")
        executor = self._fallback_executor
        if executor is None:
            raise FallbackExecutionError("Fallback runtime is closed.")

        future = executor.submit(self.fallback, request)
        try:
            value = future.result(timeout=self.config.fallback_timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise FallbackTimeoutError(
                f"Fallback exceeded {self.config.fallback_timeout_seconds:g} seconds."
            ) from exc
        except Exception as exc:
            raise FallbackExecutionError("Fallback execution failed.") from exc

        try:
            return FallbackDecision.model_validate(value)
        except ValueError as exc:
            raise FallbackExecutionError("Fallback returned an invalid decision.") from exc

    def _validate_fallback_decision(self, decision: FallbackDecision) -> None:
        spec = self.bundle.compiler.decision_spec
        if spec.kind is DecisionKind.CHOICE:
            if not isinstance(decision.selected, str):
                raise FallbackExecutionError(
                    "Choice fallback must return a string candidate key."
                )
            allowed = {candidate.key for candidate in spec.candidates}
            if decision.selected not in allowed:
                raise FallbackExecutionError(
                    "Fallback selected a candidate outside the compiled decision spec."
                )
            if decision.probabilities and set(decision.probabilities) != allowed:
                raise FallbackExecutionError(
                    "Fallback probability classes do not match the compiled decision spec."
                )


def _selected_key(value: str | bool) -> str:
    return str(value).lower() if isinstance(value, bool) else value


def _elapsed_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000.0)
