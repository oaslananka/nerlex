from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from nerlex.calibration import TemperatureCalibrationConfig, fit_temperature
from nerlex.compiler import MultinomialNBConfig, compile_snapshot
from nerlex.dataset import DatasetSplit, SnapshotConfig, SplitConfig, build_snapshot
from nerlex.evaluation import EmpiricalRiskGateConfig, fit_empirical_risk_gate
from nerlex.runtime import RuntimeBundle
from nerlex.shadow import (
    ShadowConfig,
    ShadowError,
    load_shadow_report,
    shadow_replay,
    write_shadow_report,
)
from nerlex.spec import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionRequest,
    DecisionResult,
    DecisionRoute,
    DecisionSpec,
    LabelObservation,
    LabelSource,
)
from nerlex.trace import ResultObservation, TraceRecord

FIXED_TIME = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def _spec() -> DecisionSpec:
    return DecisionSpec(
        decision_id="support-routing",
        version="1",
        kind=DecisionKind.CHOICE,
        description="Route support requests.",
        candidate_mode=CandidateMode.STATIC,
        candidates=(
            Candidate(key="billing", description="Payments and refunds."),
            Candidate(key="technical", description="Software problems."),
        ),
    )


def _trace(index: int, *, with_truth: bool = True) -> TraceRecord:
    request_id = UUID(int=index + 1)
    billing = index % 2 == 0
    truth = "billing" if billing else "technical"
    teacher = truth
    if index % 5 == 0:
        teacher = "technical" if truth == "billing" else "billing"

    request = DecisionRequest(
        request_id=request_id,
        decision_id="support-routing",
        spec_version="1",
        event_time=FIXED_TIME,
        state=(
            f"invoice refund card payment billing-{index}"
            if billing
            else f"crash bug error software technical-{index}"
        ),
    )

    labels = ()
    if with_truth:
        labels = (
            LabelObservation(
                observation_id=UUID(int=20_000 + index),
                request_id=request_id,
                source=LabelSource.OUTCOME,
                value=truth,
                source_id="resolved_queue",
                observed_at=FIXED_TIME,
            ),
        )

    teacher_result = DecisionResult(
        request_id=request_id,
        selected=teacher,
        probabilities=(
            {"billing": 0.8, "technical": 0.2}
            if teacher == "billing"
            else {"billing": 0.2, "technical": 0.8}
        ),
        confidence=0.8,
        backend="teacher-fixture",
        latency_ms=2.0,
        route=DecisionRoute.TEACHER,
        created_at=FIXED_TIME,
    )
    return TraceRecord(
        request=request,
        observations=(ResultObservation(kind="teacher", result=teacher_result),),
        labels=labels,
    )


def _bundle(records: list[TraceRecord], *, abstain_all: bool = False) -> RuntimeBundle:
    snapshot = build_snapshot(
        _spec(),
        records,
        config=SnapshotConfig(
            split=SplitConfig(
                train=0.60,
                calibration=0.20,
                test=0.20,
                seed="shadow-fixture",
            )
        ),
    )
    compiler = compile_snapshot(snapshot, MultinomialNBConfig())
    calibration = fit_temperature(
        compiler,
        snapshot,
        config=TemperatureCalibrationConfig(iterations=64),
    )
    calibration_count = snapshot.manifest.splits[DatasetSplit.CALIBRATION].count
    gate = fit_empirical_risk_gate(
        compiler,
        calibration,
        snapshot,
        EmpiricalRiskGateConfig(
            max_empirical_risk=0.0,
            min_accepted=calibration_count + 1 if abstain_all else 1,
        ),
    )
    return RuntimeBundle(
        compiler=compiler,
        calibration=calibration,
        gate=gate,
    )


def test_shadow_report_is_deterministic_and_order_independent() -> None:
    records = [_trace(index, with_truth=index % 7 != 0) for index in range(160)]
    bundle = _bundle(records)

    first = shadow_replay(bundle, records)
    second = shadow_replay(bundle, reversed(records))

    assert first.report_id == second.report_id
    assert first == second
    assert first.decisions == tuple(
        sorted(first.decisions, key=lambda row: str(row.request_id))
    )


def test_teacher_agreement_is_distinct_from_outcome_accuracy() -> None:
    records = [_trace(index) for index in range(160)]
    report = shadow_replay(_bundle(records), records)

    assert report.summary.teacher_comparable == 160
    assert report.summary.teacher_disagreements > 0
    assert report.summary.truth_labeled == 160
    assert report.summary.local_truth_correct > report.summary.teacher_agreements
    assert report.summary.teacher_agreement_rate != (
        report.summary.counterfactual_local_truth_accuracy
    )


def test_abstain_all_gate_has_zero_local_coverage() -> None:
    records = [_trace(index) for index in range(120)]
    report = shadow_replay(_bundle(records, abstain_all=True), records)

    assert report.summary.local_eligible == 0
    assert report.summary.local_coverage == 0.0
    assert report.summary.gate_abstained == len(records)
    assert report.summary.fallback_required == len(records)
    assert report.summary.route_outcome.local_eligible_correct == 0
    assert (
        report.summary.route_outcome.fallback_required_would_be_correct
        + report.summary.route_outcome.fallback_required_would_be_incorrect
        == len(records)
    )


def test_teacher_only_trace_is_not_counted_as_truth() -> None:
    training = [_trace(index) for index in range(120)]
    teacher_only = _trace(999, with_truth=False)
    report = shadow_replay(_bundle(training), [teacher_only])

    assert report.summary.teacher_comparable == 1
    assert report.summary.truth_labeled == 0
    assert report.decisions[0].truth is None
    assert report.decisions[0].truth_source is None


def test_conflicting_truth_at_same_priority_fails_closed() -> None:
    records = [_trace(index) for index in range(120)]
    base = records[0]
    conflict = TraceRecord(
        request=base.request,
        observations=base.observations,
        labels=(
            LabelObservation(
                observation_id=UUID(int=88_001),
                request_id=base.request.request_id,
                source=LabelSource.HUMAN,
                value="billing",
                observed_at=FIXED_TIME,
            ),
            LabelObservation(
                observation_id=UUID(int=88_002),
                request_id=base.request.request_id,
                source=LabelSource.HUMAN,
                value="technical",
                observed_at=FIXED_TIME,
            ),
        ),
    )

    with pytest.raises(ShadowError, match="Conflicting human truth labels"):
        shadow_replay(
            _bundle(records),
            [conflict],
            config=ShadowConfig(truth_priority=(LabelSource.HUMAN,)),
        )


def test_report_tampering_is_detected(tmp_path: Path) -> None:
    records = [_trace(index) for index in range(120)]
    report = shadow_replay(_bundle(records), records[:12])
    path = write_shadow_report(report, tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"]["local_eligible"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        load_shadow_report(path)


def test_teacher_cannot_be_configured_as_truth() -> None:
    with pytest.raises(ValueError, match="separate from truth"):
        ShadowConfig(truth_priority=(LabelSource.TEACHER,))
