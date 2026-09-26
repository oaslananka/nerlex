from __future__ import annotations

import time
from uuid import UUID

import pytest

from nerlex.dataset import DatasetSplit
from nerlex.evaluation import EmpiricalRiskGateConfig, fit_empirical_risk_gate
from nerlex.runtime import (
    FallbackDecision,
    FallbackExecutionError,
    FallbackTimeoutError,
    LocalCascadeRuntime,
    RuntimeBundle,
    RuntimeConfig,
    RuntimeRequestError,
)
from nerlex.spec import DecisionRequest, DecisionRoute
from tests.support import support_compiler_calibration


def _bundle(
    *,
    abstain_all: bool = False,
    seed: str = "runtime-test",
) -> RuntimeBundle:
    snapshot, compiler, calibration = support_compiler_calibration(seed=seed)
    calibration_count = len(snapshot.examples[DatasetSplit.CALIBRATION])
    gate = fit_empirical_risk_gate(
        compiler,
        calibration,
        snapshot,
        EmpiricalRiskGateConfig(
            max_empirical_risk=0.0,
            min_accepted=calibration_count + 1 if abstain_all else 1,
        ),
    )
    return RuntimeBundle(compiler=compiler, calibration=calibration, gate=gate)


def _request(state: str, request_id: int = 90_001) -> DecisionRequest:
    return DecisionRequest(
        request_id=UUID(int=request_id),
        decision_id="support-routing",
        spec_version="1",
        state=state,
    )


def test_high_confidence_request_resolves_locally_with_exact_artifact_ids() -> None:
    bundle = _bundle()
    runtime = LocalCascadeRuntime(bundle)

    result = runtime.decide(_request("refund invoice payment card"))

    assert result.route is DecisionRoute.LOCAL
    assert result.abstained is False
    assert result.selected == "billing"
    assert result.calibrated is True
    assert result.artifact_id == bundle.compiler.artifact_id
    assert result.artifact_ids == bundle.artifact_ids
    assert result.backend == bundle.compiler.config.kind


def test_below_threshold_request_defers_to_fallback() -> None:
    bundle = _bundle()

    def fallback(_request: DecisionRequest) -> FallbackDecision:
        return FallbackDecision(
            selected="technical",
            probabilities={"billing": 0.2, "technical": 0.8},
            confidence=0.8,
            calibrated=False,
            backend="fixture-teacher",
            backend_version="1",
            artifact_id="teacher-fixture-v1",
        )

    with LocalCascadeRuntime(bundle, fallback=fallback) as runtime:
        result = runtime.decide(_request("unseen neutral words without training markers"))

    assert result.route is DecisionRoute.FALLBACK
    assert result.selected == "technical"
    assert result.abstained is False
    assert result.abstain_reason == "below_calibrated_confidence_threshold"
    assert result.artifact_ids["compiler"] == bundle.compiler.artifact_id
    assert result.artifact_ids["calibration"] == bundle.calibration.artifact_id
    assert result.artifact_ids["gate"] == bundle.gate.artifact_id
    assert result.artifact_ids["fallback"] == "teacher-fixture-v1"


def test_abstain_all_gate_always_defers() -> None:
    bundle = _bundle(abstain_all=True)
    calls = 0

    def fallback(_request: DecisionRequest) -> FallbackDecision:
        nonlocal calls
        calls += 1
        return FallbackDecision(
            selected="billing",
            probabilities={"billing": 1.0, "technical": 0.0},
            backend="fixture-teacher",
        )

    with LocalCascadeRuntime(bundle, fallback=fallback) as runtime:
        result = runtime.decide(_request("refund invoice payment card"))

    assert calls == 1
    assert result.route is DecisionRoute.FALLBACK
    assert result.abstain_reason == "gate_abstain_all"


def test_missing_fallback_returns_explicit_local_abstention() -> None:
    bundle = _bundle(abstain_all=True)

    result = LocalCascadeRuntime(bundle).decide(_request("refund invoice payment card"))

    assert result.route is DecisionRoute.LOCAL
    assert result.abstained is True
    assert result.selected is None
    assert result.calibrated is True
    assert result.abstain_reason == "gate_abstain_all"


def test_runtime_bundle_fails_closed_on_lineage_mismatch() -> None:
    bundle = _bundle(seed="runtime-primary")
    other = _bundle(seed="runtime-other")

    with pytest.raises(ValueError, match="compiler artifact"):
        RuntimeBundle(
            compiler=bundle.compiler,
            calibration=bundle.calibration,
            gate=other.gate,
        )


def test_request_spec_mismatch_fails_before_prediction() -> None:
    runtime = LocalCascadeRuntime(_bundle())
    request = DecisionRequest(
        request_id=UUID(int=90_002),
        decision_id="another-decision",
        spec_version="1",
        state="refund invoice",
    )

    with pytest.raises(RuntimeRequestError, match="decision_id"):
        runtime.decide(request)


def test_fallback_timeout_is_explicit() -> None:
    bundle = _bundle(abstain_all=True)

    def slow_fallback(_request: DecisionRequest) -> FallbackDecision:
        time.sleep(0.05)
        return FallbackDecision(selected="billing", backend="slow-fixture")

    with (
        LocalCascadeRuntime(
            bundle,
            fallback=slow_fallback,
            config=RuntimeConfig(fallback_timeout_seconds=0.01),
        ) as runtime,
        pytest.raises(FallbackTimeoutError, match="exceeded"),
    ):
        runtime.decide(_request("refund invoice"))


def test_invalid_fallback_candidate_is_rejected() -> None:
    bundle = _bundle(abstain_all=True)

    def invalid_fallback(_request: DecisionRequest) -> FallbackDecision:
        return FallbackDecision(selected="unknown", backend="bad-fixture")

    with (
        LocalCascadeRuntime(bundle, fallback=invalid_fallback) as runtime,
        pytest.raises(FallbackExecutionError, match="outside"),
    ):
        runtime.decide(_request("refund invoice"))
