from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from nerlex.hashing import sha256_hex
from nerlex.rollout import (
    CanaryRolloutReport,
    RolloutConfig,
    RolloutDecision,
    RolloutRouteOutcomeMatrix,
    RolloutSummary,
)
from nerlex.rollout_policy import (
    RolloutAction,
    RolloutActionPolicy,
    assess_rollout_action,
    load_rollout_action_assessment,
    write_rollout_action_assessment,
)
from nerlex.spec import DecisionRoute, LabelSource


def _risk(labeled: int, correct: int) -> float | None:
    return (labeled - correct) / labeled if labeled else None


def _report(
    *,
    local_correct: int = 0,
    local_incorrect: int = 0,
    local_unlabeled: int = 0,
    control_correct: int = 0,
    control_incorrect: int = 0,
    control_unlabeled: int = 0,
    canary_fallback_correct: int = 0,
    canary_fallback_incorrect: int = 0,
    canary_fallback_unlabeled: int = 0,
) -> CanaryRolloutReport:
    decisions: list[RolloutDecision] = []
    next_request_id = 1

    def add_group(
        *,
        cohort: str,
        route: DecisionRoute,
        correct: int,
        incorrect: int,
        unlabeled: int,
    ) -> None:
        nonlocal next_request_id
        outcomes = (
            *((True,) * correct),
            *((False,) * incorrect),
            *((None,) * unlabeled),
        )
        for outcome in outcomes:
            request_id = UUID(int=next_request_id)
            evidence_id = sha256_hex(
                {
                    "request_id": str(request_id),
                    "cohort": cohort,
                    "route": route.value,
                }
            )
            truth = (
                "billing"
                if outcome is True
                else "technical"
                if outcome is False
                else None
            )
            decisions.append(
                RolloutDecision(
                    request_id=request_id,
                    evidence_id=evidence_id,
                    cohort=cohort,
                    route=route,
                    selected="billing",
                    truth=truth,
                    truth_source=(
                        LabelSource.OUTCOME if truth is not None else None
                    ),
                    correct=outcome,
                )
            )
            next_request_id += 1

    add_group(
        cohort="control",
        route=DecisionRoute.FALLBACK,
        correct=control_correct,
        incorrect=control_incorrect,
        unlabeled=control_unlabeled,
    )
    add_group(
        cohort="canary",
        route=DecisionRoute.LOCAL,
        correct=local_correct,
        incorrect=local_incorrect,
        unlabeled=local_unlabeled,
    )
    add_group(
        cohort="canary",
        route=DecisionRoute.FALLBACK,
        correct=canary_fallback_correct,
        incorrect=canary_fallback_incorrect,
        unlabeled=canary_fallback_unlabeled,
    )
    if not decisions:
        raise AssertionError("Synthetic rollout report requires at least one decision.")

    decisions.sort(key=lambda decision: str(decision.request_id))
    matrix = RolloutRouteOutcomeMatrix(
        control_correct=control_correct,
        control_incorrect=control_incorrect,
        control_unlabeled=control_unlabeled,
        canary_local_correct=local_correct,
        canary_local_incorrect=local_incorrect,
        canary_local_unlabeled=local_unlabeled,
        canary_fallback_correct=canary_fallback_correct,
        canary_fallback_incorrect=canary_fallback_incorrect,
        canary_fallback_unlabeled=canary_fallback_unlabeled,
    )

    control_assigned = control_correct + control_incorrect + control_unlabeled
    local_served = local_correct + local_incorrect + local_unlabeled
    canary_fallback = (
        canary_fallback_correct
        + canary_fallback_incorrect
        + canary_fallback_unlabeled
    )
    canary_assigned = local_served + canary_fallback
    fallback_served = control_assigned + canary_fallback
    total = control_assigned + canary_assigned

    truth_labeled = (
        control_correct
        + control_incorrect
        + local_correct
        + local_incorrect
        + canary_fallback_correct
        + canary_fallback_incorrect
    )
    truth_correct = control_correct + local_correct + canary_fallback_correct
    local_truth_labeled = local_correct + local_incorrect
    control_truth_labeled = control_correct + control_incorrect
    canary_fallback_truth_labeled = (
        canary_fallback_correct + canary_fallback_incorrect
    )

    summary = RolloutSummary(
        total=total,
        control_assigned=control_assigned,
        canary_assigned=canary_assigned,
        control_fraction=control_assigned / total,
        canary_fraction=canary_assigned / total,
        local_served=local_served,
        fallback_served=fallback_served,
        local_serving_fraction=local_served / total,
        fallback_serving_fraction=fallback_served / total,
        canary_local=local_served,
        canary_fallback=canary_fallback,
        truth_labeled=truth_labeled,
        truth_correct=truth_correct,
        observed_risk=_risk(truth_labeled, truth_correct),
        local_truth_labeled=local_truth_labeled,
        local_truth_correct=local_correct,
        local_truth_risk=_risk(local_truth_labeled, local_correct),
        control_truth_labeled=control_truth_labeled,
        control_truth_correct=control_correct,
        control_truth_risk=_risk(control_truth_labeled, control_correct),
        canary_fallback_truth_labeled=canary_fallback_truth_labeled,
        canary_fallback_truth_correct=canary_fallback_correct,
        canary_fallback_truth_risk=_risk(
            canary_fallback_truth_labeled,
            canary_fallback_correct,
        ),
        route_outcome=matrix,
    )

    decision_tuple = tuple(decisions)
    payload = {
        "schema_version": 1,
        "rollout_version": "1",
        "decision_id": "support-routing",
        "spec_version": "1",
        "decision_spec_hash": sha256_hex({"fixture": "decision-spec"}),
        "plan_id": sha256_hex({"fixture": "plan"}),
        "assessment_id": sha256_hex({"fixture": "promotion-assessment"}),
        "compiler_artifact_id": sha256_hex({"fixture": "compiler"}),
        "calibration_artifact_id": sha256_hex({"fixture": "calibration"}),
        "gate_artifact_id": sha256_hex({"fixture": "gate"}),
        "source_evidence_sha256": sha256_hex(
            tuple(decision.evidence_id for decision in decision_tuple)
        ),
        "source_evidence_count": len(decision_tuple),
        "source_label_sha256": sha256_hex(
            tuple(
                (
                    str(decision.request_id),
                    decision.truth,
                    decision.truth_source,
                )
                for decision in decision_tuple
                if decision.truth is not None
            )
        ),
        "source_label_count": truth_labeled,
        "config": RolloutConfig(),
        "decisions": decision_tuple,
        "summary": summary,
    }
    return CanaryRolloutReport(
        report_id=sha256_hex(payload),
        **payload,
    )


def _policy(**updates: object) -> RolloutActionPolicy:
    payload = {
        "advance_min_local_truth_labeled": 5,
        "advance_min_control_truth_labeled": 0,
        "advance_min_local_serving_fraction": 0.25,
        "advance_max_local_truth_risk": 0.10,
        "advance_max_local_vs_control_risk_delta": None,
        "rollback_min_local_truth_labeled": 3,
        "rollback_min_control_truth_labeled": 0,
        "rollback_local_truth_risk_at_least": 0.40,
        "rollback_local_vs_control_risk_delta_at_least": None,
    }
    payload.update(updates)
    return RolloutActionPolicy.model_validate(payload)


def test_good_sufficient_evidence_recommends_advance() -> None:
    report = _report(local_correct=10)
    assessment = assess_rollout_action(report, _policy())

    assert assessment.action is RolloutAction.ADVANCE
    assert assessment.advance_failures == ()
    assert assessment.rollback_triggers == ()


def test_insufficient_evidence_recommends_hold_not_rollback() -> None:
    report = _report(local_correct=2, local_unlabeled=8)
    assessment = assess_rollout_action(report, _policy())

    assert assessment.action is RolloutAction.HOLD
    assert assessment.advance_failures == ("insufficient_local_truth",)
    assert assessment.rollback_triggers == ()


def test_policy_hold_band_recommends_hold() -> None:
    report = _report(local_correct=8, local_incorrect=2)
    assessment = assess_rollout_action(report, _policy())

    assert assessment.local_truth_risk == 0.2
    assert assessment.action is RolloutAction.HOLD
    assert assessment.advance_failures == ("local_truth_risk",)
    assert assessment.rollback_triggers == ()


def test_explicit_bad_local_risk_recommends_rollback() -> None:
    report = _report(local_correct=5, local_incorrect=5)
    assessment = assess_rollout_action(report, _policy())

    assert assessment.local_truth_risk == 0.5
    assert assessment.action is RolloutAction.ROLLBACK
    assert assessment.rollback_triggers == ("local_truth_risk",)


def test_missing_control_comparison_evidence_recommends_hold() -> None:
    report = _report(local_correct=10)
    policy = _policy(
        advance_min_control_truth_labeled=5,
        advance_max_local_vs_control_risk_delta=0.05,
        rollback_min_control_truth_labeled=5,
        rollback_local_vs_control_risk_delta_at_least=0.25,
    )
    assessment = assess_rollout_action(report, policy)

    assert assessment.action is RolloutAction.HOLD
    assert assessment.advance_failures == ("insufficient_control_truth",)
    assert assessment.rollback_triggers == ()
    assert assessment.local_vs_control_risk_delta is None


def test_good_local_vs_control_delta_can_advance() -> None:
    report = _report(
        local_correct=9,
        local_incorrect=1,
        control_correct=8,
        control_incorrect=2,
    )
    policy = _policy(
        advance_min_control_truth_labeled=5,
        advance_max_local_truth_risk=0.15,
        advance_max_local_vs_control_risk_delta=0.05,
        rollback_min_control_truth_labeled=5,
        rollback_local_truth_risk_at_least=0.50,
        rollback_local_vs_control_risk_delta_at_least=0.25,
    )
    assessment = assess_rollout_action(report, policy)

    assert assessment.local_vs_control_risk_delta == pytest.approx(-0.1)
    assert assessment.action is RolloutAction.ADVANCE


def test_bad_local_vs_control_delta_can_trigger_rollback() -> None:
    report = _report(
        local_correct=6,
        local_incorrect=4,
        control_correct=9,
        control_incorrect=1,
    )
    policy = _policy(
        advance_min_control_truth_labeled=5,
        advance_max_local_truth_risk=0.45,
        advance_max_local_vs_control_risk_delta=0.05,
        rollback_min_control_truth_labeled=5,
        rollback_local_truth_risk_at_least=0.80,
        rollback_local_vs_control_risk_delta_at_least=0.20,
    )
    assessment = assess_rollout_action(report, policy)

    assert assessment.local_vs_control_risk_delta == pytest.approx(0.3)
    assert assessment.action is RolloutAction.ROLLBACK
    assert assessment.rollback_triggers == ("local_vs_control_risk_delta",)


def test_policy_rejects_overlapping_local_risk_thresholds() -> None:
    with pytest.raises(ValueError, match="Rollback local-risk threshold"):
        _policy(
            advance_max_local_truth_risk=0.20,
            rollback_local_truth_risk_at_least=0.20,
        )


def test_policy_rejects_overlapping_delta_thresholds() -> None:
    with pytest.raises(ValueError, match="Rollback risk-delta threshold"):
        _policy(
            advance_min_control_truth_labeled=5,
            advance_max_local_vs_control_risk_delta=0.10,
            rollback_min_control_truth_labeled=5,
            rollback_local_vs_control_risk_delta_at_least=0.10,
        )


def test_assessment_is_deterministic_and_tamper_detectable(tmp_path: Path) -> None:
    report = _report(local_correct=10)
    policy = _policy()

    first = assess_rollout_action(report, policy)
    second = assess_rollout_action(report, policy)
    assert first == second

    path = write_rollout_action_assessment(first, tmp_path)
    assert load_rollout_action_assessment(path) == first

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["action"] = "hold"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_rollout_action_assessment(path)
