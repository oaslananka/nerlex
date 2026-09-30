from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from nerlex.dataset import DatasetSplit
from nerlex.evaluation import (
    EmpiricalRiskGateConfig,
    evaluate,
    fit_empirical_risk_gate,
)
from nerlex.promotion import (
    CanaryPlan,
    PromotionError,
    PromotionPolicy,
    assess_promotion,
    assign_canary,
    create_canary_plan,
    load_canary_plan,
    load_promotion_assessment,
    write_canary_plan,
    write_promotion_assessment,
)
from nerlex.runtime import RuntimeBundle
from nerlex.shadow import shadow_replay
from tests.support import support_compiler_calibration, support_trace


def _evidence(
    *,
    seed: str = "promotion-test",
    abstain_all: bool = False,
):
    snapshot, compiler, calibration = support_compiler_calibration(seed=seed)
    calibration_count = snapshot.manifest.splits[DatasetSplit.CALIBRATION].count
    gate = fit_empirical_risk_gate(
        compiler,
        calibration,
        snapshot,
        EmpiricalRiskGateConfig(
            max_empirical_risk=1.0,
            min_accepted=calibration_count + 1 if abstain_all else 1,
        ),
    )
    evaluation = evaluate(compiler, calibration, snapshot, gate=gate)
    bundle = RuntimeBundle(
        compiler=compiler,
        calibration=calibration,
        gate=gate,
    )
    records = [support_trace(index) for index in range(120)]
    shadow = shadow_replay(bundle, records)
    return bundle, evaluation, shadow


def _passing_policy() -> PromotionPolicy:
    return PromotionPolicy(
        max_test_selective_risk=1.0,
        min_test_selective_coverage=0.0,
        min_shadow_eligible_truth_labeled=1,
        max_shadow_eligible_truth_risk=1.0,
        min_shadow_local_coverage=0.0,
    )


def test_promotion_assessment_is_deterministic_and_policy_bound() -> None:
    bundle, evaluation, shadow = _evidence()
    policy = _passing_policy()

    first = assess_promotion(bundle, evaluation, shadow, policy)
    second = assess_promotion(bundle, evaluation, shadow, policy)

    assert first == second
    assert first.passed is True
    assert first.assessment_id == second.assessment_id
    assert first.evaluation_report_id == evaluation.report_id
    assert first.shadow_report_id == shadow.report_id
    assert tuple(criterion.name for criterion in first.criteria) == (
        "test_selective_risk",
        "test_selective_coverage",
        "shadow_eligible_truth_labeled",
        "shadow_eligible_truth_risk",
        "shadow_local_coverage",
    )


def test_failing_policy_produces_failed_assessment_not_exception() -> None:
    bundle, evaluation, shadow = _evidence()
    policy = _passing_policy().model_copy(
        update={
            "min_shadow_eligible_truth_labeled": (
                shadow.summary.eligible_truth_labeled + 1
            )
        }
    )

    assessment = assess_promotion(bundle, evaluation, shadow, policy)

    assert assessment.passed is False
    criterion = next(
        item
        for item in assessment.criteria
        if item.name == "shadow_eligible_truth_labeled"
    )
    assert criterion.passed is False


def test_promotion_rejects_evidence_from_different_runtime_lineage() -> None:
    bundle, _, _ = _evidence(seed="promotion-a")
    _, evaluation, shadow = _evidence(seed="promotion-b")
    policy = _passing_policy()

    with pytest.raises(PromotionError, match="compiler"):
        assess_promotion(bundle, evaluation, shadow, policy)


def test_abstain_all_gate_fails_positive_local_evidence_requirements() -> None:
    bundle, evaluation, shadow = _evidence(abstain_all=True)
    policy = PromotionPolicy(
        max_test_selective_risk=1.0,
        min_test_selective_coverage=0.01,
        min_shadow_eligible_truth_labeled=1,
        max_shadow_eligible_truth_risk=1.0,
        min_shadow_local_coverage=0.01,
    )

    assessment = assess_promotion(bundle, evaluation, shadow, policy)

    assert assessment.passed is False
    assert evaluation.selective is not None
    assert evaluation.selective.coverage == 0.0
    assert shadow.summary.local_coverage == 0.0


def test_promotion_assessment_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    bundle, evaluation, shadow = _evidence()
    assessment = assess_promotion(bundle, evaluation, shadow, _passing_policy())
    path = write_promotion_assessment(assessment, tmp_path)

    assert load_promotion_assessment(path) == assessment

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["policy"]["max_test_selective_risk"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_promotion_assessment(path)


def test_canary_plan_requires_passing_assessment() -> None:
    bundle, evaluation, shadow = _evidence()
    failed_policy = _passing_policy().model_copy(
        update={
            "min_shadow_eligible_truth_labeled": (
                shadow.summary.eligible_truth_labeled + 1
            )
        }
    )
    assessment = assess_promotion(bundle, evaluation, shadow, failed_policy)

    with pytest.raises(PromotionError, match="passing"):
        create_canary_plan(
            assessment,
            rollout_fraction=0.1,
            assignment_seed="stable-seed",
        )


def test_canary_assignment_is_deterministic_retry_stable_and_nested() -> None:
    bundle, evaluation, shadow = _evidence()
    assessment = assess_promotion(bundle, evaluation, shadow, _passing_policy())

    small = create_canary_plan(
        assessment,
        rollout_fraction=0.10,
        assignment_seed="stable-seed",
    )
    large = create_canary_plan(
        assessment,
        rollout_fraction=0.50,
        assignment_seed="stable-seed",
        previous_plan=small,
    )

    small_canary: set[str] = set()
    large_canary: set[str] = set()
    for index in range(500):
        key = f"tenant-{index}"
        first = assign_canary(
            small,
            request_id=UUID(int=index + 1),
            decision_id=assessment.decision_id,
            spec_version=assessment.spec_version,
            assignment_key=key,
        )
        retry = assign_canary(
            small,
            request_id=UUID(int=10_000 + index),
            decision_id=assessment.decision_id,
            spec_version=assessment.spec_version,
            assignment_key=key,
        )
        expanded = assign_canary(
            large,
            request_id=UUID(int=index + 1),
            decision_id=assessment.decision_id,
            spec_version=assessment.spec_version,
            assignment_key=key,
        )

        assert retry.cohort == first.cohort
        assert retry.bucket == first.bucket
        if first.cohort == "canary":
            small_canary.add(key)
        if expanded.cohort == "canary":
            large_canary.add(key)

    assert small_canary
    assert small_canary <= large_canary


def test_zero_and_full_rollout_are_exact() -> None:
    bundle, evaluation, shadow = _evidence()
    assessment = assess_promotion(bundle, evaluation, shadow, _passing_policy())
    zero = create_canary_plan(
        assessment,
        rollout_fraction=0.0,
        assignment_seed="stable-seed",
    )
    full = create_canary_plan(
        assessment,
        rollout_fraction=1.0,
        assignment_seed="stable-seed",
        previous_plan=zero,
    )

    for index in range(32):
        kwargs = {
            "request_id": UUID(int=index + 1),
            "decision_id": assessment.decision_id,
            "spec_version": assessment.spec_version,
            "assignment_key": f"subject-{index}",
        }
        assert assign_canary(zero, **kwargs).cohort == "control"
        assert assign_canary(full, **kwargs).cohort == "canary"


def test_canary_ramp_chain_requires_stable_assignment_seed() -> None:
    bundle, evaluation, shadow = _evidence()
    assessment = assess_promotion(bundle, evaluation, shadow, _passing_policy())
    first = create_canary_plan(
        assessment,
        rollout_fraction=0.1,
        assignment_seed="stable-seed",
    )

    with pytest.raises(PromotionError, match="assignment seed"):
        create_canary_plan(
            assessment,
            rollout_fraction=0.2,
            assignment_seed="different-seed",
            previous_plan=first,
        )


def test_canary_assignment_rejects_wrong_decision_identity() -> None:
    bundle, evaluation, shadow = _evidence()
    assessment = assess_promotion(bundle, evaluation, shadow, _passing_policy())
    plan = create_canary_plan(
        assessment,
        rollout_fraction=0.1,
        assignment_seed="stable-seed",
    )

    with pytest.raises(PromotionError, match="decision spec"):
        assign_canary(
            plan,
            request_id=UUID(int=1),
            decision_id="other-decision",
            spec_version=assessment.spec_version,
            assignment_key="subject-1",
        )


def test_canary_plan_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    bundle, evaluation, shadow = _evidence()
    assessment = assess_promotion(bundle, evaluation, shadow, _passing_policy())
    plan = create_canary_plan(
        assessment,
        rollout_fraction=0.25,
        assignment_seed="stable-seed",
    )
    path = write_canary_plan(plan, tmp_path)

    assert load_canary_plan(path) == plan

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rollout_fraction"] = 0.75
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        load_canary_plan(path)


def test_canary_plan_model_rejects_tampered_content() -> None:
    bundle, evaluation, shadow = _evidence()
    assessment = assess_promotion(bundle, evaluation, shadow, _passing_policy())
    plan = create_canary_plan(
        assessment,
        rollout_fraction=0.25,
        assignment_seed="stable-seed",
    )
    raw = plan.model_dump(mode="python")
    raw["rollout_fraction"] = 0.75

    with pytest.raises(ValueError, match="identity"):
        CanaryPlan.model_validate(raw)
