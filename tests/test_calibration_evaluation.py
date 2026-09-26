from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

import nerlex.evaluation as evaluation_module
from nerlex.calibration import (
    CalibrationError,
    TemperatureCalibrationConfig,
    fit_temperature,
    load_calibration_artifact,
    predict_calibrated,
    write_calibration_artifact,
)
from nerlex.compiler import MultinomialNBConfig, compile_snapshot, predict
from nerlex.dataset import DatasetSplit, SnapshotConfig, SplitConfig, build_snapshot
from nerlex.evaluation import (
    EmpiricalRiskGateConfig,
    EvaluationConfig,
    evaluate,
    fit_empirical_risk_gate,
    load_evaluation_report,
    load_gate_artifact,
    write_evaluation_report,
    write_gate_artifact,
)
from nerlex.spec import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionRequest,
    DecisionSpec,
    LabelObservation,
    LabelSource,
)
from nerlex.trace import TraceRecord


def _spec() -> DecisionSpec:
    return DecisionSpec(
        decision_id="support-routing",
        version="1",
        kind=DecisionKind.CHOICE,
        description="Route support requests.",
        candidate_mode=CandidateMode.STATIC,
        candidates=(
            Candidate(key="billing", description="Payments and refunds."),
            Candidate(key="technical", description="Product and software problems."),
        ),
    )


def _trace(index: int) -> TraceRecord:
    request_id = UUID(int=index + 1)
    if index % 2 == 0:
        label = "billing"
        text = f"invoice refund payment card billing-marker-{index}"
    else:
        label = "technical"
        text = f"crash error bug software technical-marker-{index}"

    return TraceRecord(
        request=DecisionRequest(
            request_id=request_id,
            decision_id="support-routing",
            spec_version="1",
            state=text,
        ),
        labels=(
            LabelObservation(
                observation_id=UUID(int=20_000 + index),
                request_id=request_id,
                source=LabelSource.OUTCOME,
                value=label,
                source_id="fixture",
            ),
        ),
    )


def _snapshot():
    return build_snapshot(
        _spec(),
        [_trace(index) for index in range(240)],
        config=SnapshotConfig(
            split=SplitConfig(
                train=0.60,
                calibration=0.20,
                test=0.20,
                seed="calibration-evaluation-test",
            )
        ),
    )


def _compiler_and_calibration():
    snapshot = _snapshot()
    compiler = compile_snapshot(snapshot, MultinomialNBConfig())
    calibration = fit_temperature(
        compiler,
        snapshot,
        config=TemperatureCalibrationConfig(iterations=64),
    )
    return snapshot, compiler, calibration


def test_temperature_calibration_is_deterministic_and_uses_calibration_lineage(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    compiler = compile_snapshot(snapshot, MultinomialNBConfig())

    first = fit_temperature(compiler, snapshot)
    second = fit_temperature(compiler, snapshot)

    assert first == second
    assert first.compiler_artifact_id == compiler.artifact_id
    assert first.snapshot_id == snapshot.manifest.snapshot_id
    assert (
        first.calibration_sha256
        == snapshot.manifest.splits[DatasetSplit.CALIBRATION].sha256
    )
    assert first.calibration_count == len(snapshot.examples[DatasetSplit.CALIBRATION])
    assert first.temperature > 0.0
    assert first.classes == tuple(sorted(first.classes))

    path = write_calibration_artifact(first, tmp_path)
    assert load_calibration_artifact(path) == first


def test_calibrated_prediction_is_explicit_and_source_bound() -> None:
    snapshot, compiler, calibration = _compiler_and_calibration()
    example = snapshot.examples[DatasetSplit.TEST][0]

    raw = predict(compiler, example.state)
    calibrated = predict_calibrated(compiler, calibration, example.state)

    assert raw.calibrated is False
    assert calibrated.calibrated is True
    assert calibrated.compiler_artifact_id == compiler.artifact_id
    assert calibrated.calibration_artifact_id == calibration.artifact_id
    assert sum(calibrated.probabilities.values()) == pytest.approx(1.0)
    assert calibrated.confidence == pytest.approx(
        calibrated.probabilities[calibrated.selected]
    )


def test_calibration_artifact_tampering_fails_closed(tmp_path: Path) -> None:
    _, _, calibration = _compiler_and_calibration()
    path = write_calibration_artifact(calibration, tmp_path)

    raw = path.read_text(encoding="utf-8").replace(
        f'"temperature":{calibration.temperature}',
        '"temperature":1.23456789',
        1,
    )
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        load_calibration_artifact(path)


def test_calibration_rejects_snapshot_lineage_mismatch() -> None:
    snapshot, compiler, _ = _compiler_and_calibration()
    different_snapshot = build_snapshot(
        _spec(),
        [_trace(index) for index in range(240, 480)],
        config=snapshot.manifest.config,
    )

    with pytest.raises(CalibrationError, match="snapshot IDs"):
        fit_temperature(compiler, different_snapshot)


def test_empirical_gate_is_fit_on_calibration_and_report_is_reproducible(
    tmp_path: Path,
) -> None:
    snapshot, compiler, calibration = _compiler_and_calibration()
    gate = fit_empirical_risk_gate(
        compiler,
        calibration,
        snapshot,
        EmpiricalRiskGateConfig(max_empirical_risk=0.05, min_accepted=5),
    )

    assert gate.calibration_sha256 == snapshot.manifest.splits[
        DatasetSplit.CALIBRATION
    ].sha256
    assert gate.calibration_count == len(snapshot.examples[DatasetSplit.CALIBRATION])

    first = evaluate(
        compiler,
        calibration,
        snapshot,
        gate=gate,
        config=EvaluationConfig(ece_bins=10),
    )
    second = evaluate(
        compiler,
        calibration,
        snapshot,
        gate=gate,
        config=EvaluationConfig(ece_bins=10),
    )

    assert first == second
    assert first.test_sha256 == snapshot.manifest.splits[DatasetSplit.TEST].sha256
    assert first.test_count == len(snapshot.examples[DatasetSplit.TEST])
    assert first.raw.count == first.test_count
    assert first.calibrated.count == first.test_count
    assert 0.0 <= first.aurc <= 1.0
    assert first.risk_coverage[-1].coverage == pytest.approx(1.0)
    assert first.selective is not None
    assert first.selective.accepted + first.selective.abstained == first.test_count

    gate_path = write_gate_artifact(gate, tmp_path)
    report_path = write_evaluation_report(first, tmp_path)
    assert load_gate_artifact(gate_path) == gate
    assert load_evaluation_report(report_path) == first



def test_evaluation_reuses_raw_prediction_for_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot, compiler, calibration = _compiler_and_calibration()
    original_predict = evaluation_module.predict
    calls = 0

    def counting_predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(evaluation_module, "predict", counting_predict)

    evaluate(compiler, calibration, snapshot)

    assert calls == len(snapshot.examples[DatasetSplit.TEST])


def test_metrics_remain_finite_for_out_of_vocabulary_test_inputs() -> None:
    snapshot, compiler, calibration = _compiler_and_calibration()

    report = evaluate(compiler, calibration, snapshot)

    for metrics in (report.raw, report.calibrated):
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.macro_f1 <= 1.0
        assert metrics.nll >= 0.0
        assert metrics.brier >= 0.0
        assert 0.0 <= metrics.ece <= 1.0


def test_gate_can_fail_closed_to_abstain_all() -> None:
    snapshot, compiler, calibration = _compiler_and_calibration()
    gate = fit_empirical_risk_gate(
        compiler,
        calibration,
        snapshot,
        EmpiricalRiskGateConfig(
            max_empirical_risk=0.0,
            min_accepted=len(snapshot.examples[DatasetSplit.CALIBRATION]) + 1,
        ),
    )

    assert gate.confidence_threshold is None
    assert gate.accepted_count == 0
    assert gate.empirical_risk is None

    report = evaluate(compiler, calibration, snapshot, gate=gate)
    assert report.selective is not None
    assert report.selective.accepted == 0
    assert report.selective.coverage == 0.0
    assert report.selective.risk is None
