from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

import nerlex.runtime as runtime_module
from nerlex.canary import (
    CanaryArtifactError,
    CanaryRequestError,
    CanaryRuntime,
    load_canary_decision_evidence,
    write_canary_decision_evidence,
)
from nerlex.evaluation import EmpiricalRiskGateConfig, evaluate, fit_empirical_risk_gate
from nerlex.promotion import PromotionPolicy, assess_promotion, create_canary_plan
from nerlex.runtime import FallbackDecision, FallbackExecutionError, RuntimeBundle
from nerlex.shadow import shadow_replay
from nerlex.spec import DecisionRequest, DecisionRoute
from tests.support import support_compiler_calibration, support_trace


def _fixture(
    *,
    rollout_fraction: float,
    seed: str = "live-canary-test",
):
    snapshot, compiler, calibration = support_compiler_calibration(seed=seed)
    gate = fit_empirical_risk_gate(
        compiler,
        calibration,
        snapshot,
        EmpiricalRiskGateConfig(max_empirical_risk=0.0, min_accepted=1),
    )
    bundle = RuntimeBundle(compiler=compiler, calibration=calibration, gate=gate)
    evaluation = evaluate(compiler, calibration, snapshot, gate=gate)
    shadow = shadow_replay(bundle, [support_trace(index) for index in range(120)])
    policy = PromotionPolicy(
        max_test_selective_risk=1.0,
        min_test_selective_coverage=0.0,
        min_shadow_eligible_truth_labeled=1,
        max_shadow_eligible_truth_risk=1.0,
        min_shadow_local_coverage=0.0,
    )
    assessment = assess_promotion(bundle, evaluation, shadow, policy)
    assert assessment.passed is True
    plan = create_canary_plan(
        assessment,
        rollout_fraction=rollout_fraction,
        assignment_seed="stable-canary-seed",
    )
    return bundle, assessment, plan


def _request(
    state: str = "refund invoice payment card",
    *,
    request_id: int = 130_001,
) -> DecisionRequest:
    return DecisionRequest(
        request_id=UUID(int=request_id),
        decision_id="support-routing",
        spec_version="1",
        state=state,
    )


def _control_decision(_request: DecisionRequest) -> FallbackDecision:
    return FallbackDecision(
        selected="technical",
        probabilities={"billing": 0.2, "technical": 0.8},
        confidence=0.8,
        backend="fixture-control",
        backend_version="1",
        artifact_id="fixture-control-v1",
    )


def test_control_cohort_executes_control_without_local_inference(monkeypatch) -> None:
    bundle, assessment, plan = _fixture(rollout_fraction=0.0)
    request = _request()

    def unexpected_local(*_args, **_kwargs):
        raise AssertionError("control cohort must not execute local inference")

    monkeypatch.setattr(runtime_module, "predict_calibrated", unexpected_local)

    with CanaryRuntime(
        bundle,
        assessment,
        plan,
        control=_control_decision,
    ) as runtime:
        evidence = runtime.decide(request, assignment_key="tenant-1")

    assert evidence.cohort == "control"
    assert evidence.result.route is DecisionRoute.FALLBACK
    assert evidence.result.abstain_reason == "control_cohort"
    assert evidence.result.backend == "fixture-control"
    assert evidence.request_id == request.request_id


def test_canary_cohort_can_serve_local_result_without_control_call() -> None:
    bundle, assessment, plan = _fixture(rollout_fraction=1.0)
    calls = 0

    def control(request: DecisionRequest) -> FallbackDecision:
        nonlocal calls
        calls += 1
        return _control_decision(request)

    with CanaryRuntime(bundle, assessment, plan, control=control) as runtime:
        evidence = runtime.decide(_request(), assignment_key="tenant-1")

    assert calls == 0
    assert evidence.cohort == "canary"
    assert evidence.result.route is DecisionRoute.LOCAL
    assert evidence.result.selected == "billing"
    assert evidence.result.abstained is False


def test_canary_gate_abstention_uses_control_as_fallback() -> None:
    bundle, assessment, plan = _fixture(rollout_fraction=1.0)
    calls = 0

    def control(request: DecisionRequest) -> FallbackDecision:
        nonlocal calls
        calls += 1
        return _control_decision(request)

    request = _request("unseen neutral words without training markers")
    with CanaryRuntime(bundle, assessment, plan, control=control) as runtime:
        evidence = runtime.decide(request, assignment_key="tenant-1")

    assert calls == 1
    assert evidence.cohort == "canary"
    assert evidence.result.route is DecisionRoute.FALLBACK
    assert evidence.result.abstain_reason == "below_calibrated_confidence_threshold"
    assert evidence.result.backend == "fixture-control"


def test_cohort_is_retry_stable_but_request_evidence_stays_distinct() -> None:
    bundle, assessment, plan = _fixture(rollout_fraction=0.5)
    first_request = _request(request_id=130_010)
    retry_request = _request(request_id=130_011)

    with CanaryRuntime(
        bundle,
        assessment,
        plan,
        control=_control_decision,
    ) as runtime:
        first = runtime.decide(first_request, assignment_key="tenant-stable")
        retry = runtime.decide(retry_request, assignment_key="tenant-stable")

    assert first.cohort == retry.cohort
    assert first.bucket == retry.bucket
    assert first.request_id != retry.request_id
    assert first.evidence_id != retry.evidence_id


def test_live_canary_runtime_fails_closed_on_lineage_mismatch() -> None:
    bundle, _, _ = _fixture(rollout_fraction=0.5, seed="canary-primary")
    _, other_assessment, other_plan = _fixture(
        rollout_fraction=0.5,
        seed="canary-other",
    )

    with pytest.raises(CanaryArtifactError, match="runtime bundle"):
        CanaryRuntime(
            bundle,
            other_assessment,
            other_plan,
            control=_control_decision,
        )


def test_invalid_assignment_key_fails_before_control_execution() -> None:
    bundle, assessment, plan = _fixture(rollout_fraction=0.0)
    calls = 0

    def control(request: DecisionRequest) -> FallbackDecision:
        nonlocal calls
        calls += 1
        return _control_decision(request)

    request = _request()
    with (
        CanaryRuntime(bundle, assessment, plan, control=control) as runtime,
        pytest.raises(CanaryRequestError, match="assignment key"),
    ):
        runtime.decide(request, assignment_key="")

    assert calls == 0


def test_control_failure_remains_explicit() -> None:
    bundle, assessment, plan = _fixture(rollout_fraction=0.0)

    def failing_control(_request: DecisionRequest) -> FallbackDecision:
        raise RuntimeError("provider failed")

    request = _request()
    with (
        CanaryRuntime(
            bundle,
            assessment,
            plan,
            control=failing_control,
        ) as runtime,
        pytest.raises(FallbackExecutionError, match="failed"),
    ):
        runtime.decide(request, assignment_key="tenant-1")


def test_canary_decision_evidence_round_trip_and_tamper_detection(
    tmp_path: Path,
) -> None:
    bundle, assessment, plan = _fixture(rollout_fraction=1.0)
    with CanaryRuntime(
        bundle,
        assessment,
        plan,
        control=_control_decision,
    ) as runtime:
        evidence = runtime.decide(_request(), assignment_key="tenant-private")

    path = write_canary_decision_evidence(evidence, tmp_path)
    assert load_canary_decision_evidence(path) == evidence
    assert "tenant-private" not in path.read_text(encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bucket"] = 0.0 if payload["bucket"] != 0.0 else 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_canary_decision_evidence(path)


def test_evidence_preserves_exact_runtime_lineage() -> None:
    bundle, assessment, plan = _fixture(rollout_fraction=0.0)
    with CanaryRuntime(
        bundle,
        assessment,
        plan,
        control=_control_decision,
    ) as runtime:
        evidence = runtime.decide(_request(), assignment_key="tenant-1")

    assert evidence.plan_id == plan.plan_id
    assert evidence.assessment_id == assessment.assessment_id
    assert evidence.compiler_artifact_id == bundle.compiler.artifact_id
    assert evidence.calibration_artifact_id == bundle.calibration.artifact_id
    assert evidence.gate_artifact_id == bundle.gate.artifact_id
    assert evidence.result.artifact_ids["compiler"] == bundle.compiler.artifact_id
