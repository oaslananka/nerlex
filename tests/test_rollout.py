from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from nerlex.canary import CanaryDecisionEvidence, CanaryRuntime
from nerlex.evaluation import EmpiricalRiskGateConfig, evaluate, fit_empirical_risk_gate
from nerlex.promotion import (
    CanaryPlan,
    PromotionAssessment,
    PromotionPolicy,
    assess_promotion,
    assign_canary,
    create_canary_plan,
)
from nerlex.rollout import (
    RolloutError,
    evaluate_canary_rollout,
    load_canary_rollout_report,
    write_canary_rollout_report,
)
from nerlex.runtime import FallbackDecision, RuntimeBundle
from nerlex.shadow import shadow_replay
from nerlex.spec import DecisionRequest, DecisionRoute, LabelObservation, LabelSource
from tests.support import support_compiler_calibration, support_trace


def _fixture(
    *,
    seed: str = "rollout-test",
    rollout_fraction: float = 0.5,
) -> tuple[RuntimeBundle, PromotionAssessment, CanaryPlan]:
    snapshot, compiler, calibration = support_compiler_calibration(seed=seed)
    gate = fit_empirical_risk_gate(
        compiler,
        calibration,
        snapshot,
        EmpiricalRiskGateConfig(max_empirical_risk=0.0, min_accepted=1),
    )
    bundle = RuntimeBundle(
        compiler=compiler,
        calibration=calibration,
        gate=gate,
    )
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
    plan = create_canary_plan(
        assessment,
        rollout_fraction=rollout_fraction,
        assignment_seed="rollout-stable-seed",
    )
    return bundle, assessment, plan


def _request(
    request_id: int,
    state: str,
) -> DecisionRequest:
    return DecisionRequest(
        request_id=UUID(int=request_id),
        decision_id="support-routing",
        spec_version="1",
        state=state,
    )


def _control(_request: DecisionRequest) -> FallbackDecision:
    return FallbackDecision(
        selected="technical",
        probabilities={"billing": 0.1, "technical": 0.9},
        confidence=0.9,
        backend="rollout-control",
        backend_version="1",
        artifact_id="rollout-control-v1",
    )


def _key_for_cohort(
    plan: CanaryPlan,
    assessment: PromotionAssessment,
    cohort: str,
) -> str:
    for index in range(10_000):
        key = f"{cohort}-subject-{index}"
        assignment = assign_canary(
            plan,
            request_id=UUID(int=index + 1),
            decision_id=assessment.decision_id,
            spec_version=assessment.spec_version,
            assignment_key=key,
        )
        if assignment.cohort == cohort:
            return key
    raise AssertionError(f"Unable to find deterministic {cohort} assignment key.")


def _live_evidence(
    bundle: RuntimeBundle,
    assessment: PromotionAssessment,
    plan: CanaryPlan,
) -> tuple[CanaryDecisionEvidence, ...]:
    control_key = _key_for_cohort(plan, assessment, "control")
    canary_key = _key_for_cohort(plan, assessment, "canary")

    requests = (
        (
            _request(150_001, "refund invoice payment card"),
            control_key,
        ),
        (
            _request(150_002, "refund invoice payment card"),
            canary_key,
        ),
        (
            _request(150_003, "unseen neutral words without training markers"),
            canary_key,
        ),
    )
    with CanaryRuntime(
        bundle,
        assessment,
        plan,
        control=_control,
    ) as runtime:
        return tuple(
            runtime.decide(request, assignment_key=assignment_key)
            for request, assignment_key in requests
        )


def _label(
    request_id: int,
    value: str,
    *,
    source: LabelSource = LabelSource.OUTCOME,
    observation_id: int,
) -> LabelObservation:
    return LabelObservation(
        observation_id=UUID(int=observation_id),
        request_id=UUID(int=request_id),
        source=source,
        value=value,
        source_id="rollout-fixture",
    )


def test_rollout_report_is_deterministic_and_order_independent() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    labels = (
        _label(150_001, "technical", observation_id=250_001),
        _label(150_002, "billing", observation_id=250_002),
        _label(150_003, "technical", observation_id=250_003),
    )

    first = evaluate_canary_rollout(
        bundle,
        assessment,
        plan,
        evidence,
        labels,
    )
    second = evaluate_canary_rollout(
        bundle,
        assessment,
        plan,
        reversed(evidence),
        reversed(labels),
    )

    assert first == second
    assert first.summary.total == 3
    assert first.summary.control_assigned == 1
    assert first.summary.canary_assigned == 2
    assert first.summary.local_served == 1
    assert first.summary.fallback_served == 2
    assert first.summary.canary_local == 1
    assert first.summary.canary_fallback == 1
    assert first.summary.truth_labeled == 3
    assert first.summary.truth_correct == 3
    assert first.summary.observed_risk == 0.0
    assert first.summary.local_truth_risk == 0.0
    assert first.summary.control_truth_risk == 0.0
    assert first.summary.canary_fallback_truth_risk == 0.0


def test_teacher_label_never_overrides_outcome_truth() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    labels = (
        _label(
            150_002,
            "technical",
            source=LabelSource.TEACHER,
            observation_id=251_001,
        ),
        _label(
            150_002,
            "billing",
            source=LabelSource.OUTCOME,
            observation_id=251_002,
        ),
    )

    report = evaluate_canary_rollout(
        bundle,
        assessment,
        plan,
        evidence,
        labels,
    )
    decision = next(
        item for item in report.decisions if item.request_id == UUID(int=150_002)
    )

    assert decision.truth == "billing"
    assert decision.truth_source is LabelSource.OUTCOME
    assert decision.correct is True
    assert report.summary.truth_labeled == 1


def test_teacher_only_label_remains_unlabeled() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    labels = (
        _label(
            150_002,
            "technical",
            source=LabelSource.TEACHER,
            observation_id=252_001,
        ),
    )

    report = evaluate_canary_rollout(
        bundle,
        assessment,
        plan,
        evidence,
        labels,
    )
    decision = next(
        item for item in report.decisions if item.request_id == UUID(int=150_002)
    )

    assert decision.truth is None
    assert decision.truth_source is None
    assert decision.correct is None
    assert report.summary.truth_labeled == 0


def test_conflicting_truth_labels_fail_closed() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    labels = (
        _label(150_002, "billing", observation_id=253_001),
        _label(150_002, "technical", observation_id=253_002),
    )

    with pytest.raises(RolloutError, match="Conflicting outcome"):
        evaluate_canary_rollout(
            bundle,
            assessment,
            plan,
            evidence,
            labels,
        )


def test_duplicate_request_evidence_is_rejected() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)

    with pytest.raises(RolloutError, match="duplicate request IDs"):
        evaluate_canary_rollout(
            bundle,
            assessment,
            plan,
            (evidence[0], evidence[0]),
            (),
        )


def test_mixed_plan_evidence_is_rejected() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    other_plan = create_canary_plan(
        assessment,
        rollout_fraction=1.0,
        assignment_seed=plan.assignment_seed,
        previous_plan=plan,
    )
    with CanaryRuntime(
        bundle,
        assessment,
        other_plan,
        control=_control,
    ) as runtime:
        other_evidence = runtime.decide(
            _request(150_010, "refund invoice payment card"),
            assignment_key="always-canary",
        )

    with pytest.raises(RolloutError, match="different plan ID"):
        evaluate_canary_rollout(
            bundle,
            assessment,
            plan,
            (*evidence, other_evidence),
            (),
        )


def test_truth_value_outside_decision_spec_is_rejected() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    labels = (
        _label(150_002, "unknown", observation_id=254_001),
    )

    with pytest.raises(RolloutError, match="outside the DecisionSpec"):
        evaluate_canary_rollout(
            bundle,
            assessment,
            plan,
            evidence,
            labels,
        )


def test_rollout_report_handles_unlabeled_routes_separately() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    labels = (
        _label(150_002, "billing", observation_id=255_001),
    )

    report = evaluate_canary_rollout(
        bundle,
        assessment,
        plan,
        evidence,
        labels,
    )
    matrix = report.summary.route_outcome

    assert matrix.control_unlabeled == 1
    assert matrix.canary_local_correct == 1
    assert matrix.canary_fallback_unlabeled == 1
    assert report.summary.local_truth_labeled == 1
    assert report.summary.control_truth_labeled == 0
    assert report.summary.canary_fallback_truth_labeled == 0
    assert report.summary.control_truth_risk is None
    assert report.summary.canary_fallback_truth_risk is None


def test_rollout_report_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    labels = (
        _label(150_001, "technical", observation_id=256_001),
        _label(150_002, "billing", observation_id=256_002),
        _label(150_003, "technical", observation_id=256_003),
    )
    report = evaluate_canary_rollout(
        bundle,
        assessment,
        plan,
        evidence,
        labels,
    )
    path = write_canary_rollout_report(report, tmp_path)

    assert load_canary_rollout_report(path) == report

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"]["truth_correct"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_canary_rollout_report(path)


def test_rollout_report_preserves_route_and_cohort_separation() -> None:
    bundle, assessment, plan = _fixture()
    evidence = _live_evidence(bundle, assessment, plan)
    report = evaluate_canary_rollout(
        bundle,
        assessment,
        plan,
        evidence,
        (),
    )

    control = next(item for item in report.decisions if item.cohort == "control")
    local = next(
        item
        for item in report.decisions
        if item.cohort == "canary" and item.route is DecisionRoute.LOCAL
    )
    fallback = next(
        item
        for item in report.decisions
        if item.cohort == "canary" and item.route is DecisionRoute.FALLBACK
    )

    assert control.route is DecisionRoute.FALLBACK
    assert local.route is DecisionRoute.LOCAL
    assert fallback.route is DecisionRoute.FALLBACK
