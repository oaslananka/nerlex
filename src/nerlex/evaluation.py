from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from nerlex.artifact_io import write_immutable_atomic
from nerlex.calibration import (
    CalibratedPrediction,
    TemperatureCalibrationArtifact,
    predict_calibrated,
)
from nerlex.compiler import CompilerArtifact, CompilerPrediction, predict
from nerlex.dataset import DatasetExample, DatasetSnapshot, DatasetSplit
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.spec import StrictModel

EVALUATION_SCHEMA_VERSION: Literal[1] = 1
EVALUATION_IMPLEMENTATION_VERSION: Literal["1"] = "1"
GATE_SCHEMA_VERSION: Literal[1] = 1
GATE_IMPLEMENTATION_VERSION: Literal["1"] = "1"

_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_MIN_PROBABILITY = 1e-15


class EvaluationError(ValueError):
    """Raised when evaluation lineage or metrics are invalid."""


class EvaluationConfig(StrictModel):
    ece_bins: int = Field(default=10, ge=2, le=100)


class EmpiricalRiskGateConfig(StrictModel):
    max_empirical_risk: float = Field(ge=0.0, le=1.0)
    min_accepted: int = Field(default=1, ge=1)


class _EmpiricalRiskGateIdentity(StrictModel):
    schema_version: Literal[1] = GATE_SCHEMA_VERSION
    gate_version: Literal["1"] = GATE_IMPLEMENTATION_VERSION
    method: Literal["confidence_threshold_empirical_risk"] = (
        "confidence_threshold_empirical_risk"
    )
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    snapshot_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_sha256: str = Field(pattern=_SHA256_PATTERN)
    calibration_count: int = Field(ge=1)
    config: EmpiricalRiskGateConfig
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    accepted_count: int = Field(ge=0)
    empirical_risk: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_shape(self) -> _EmpiricalRiskGateIdentity:
        if self.confidence_threshold is None:
            if self.accepted_count != 0 or self.empirical_risk is not None:
                raise ValueError("An abstain-all gate must have zero accepted examples.")
        elif self.accepted_count < self.config.min_accepted:
            raise ValueError("Gate threshold accepted fewer than min_accepted examples.")
        elif self.empirical_risk is None:
            raise ValueError("A fitted threshold requires empirical_risk.")
        elif self.empirical_risk > self.config.max_empirical_risk + 1e-12:
            raise ValueError("Fitted gate exceeds its configured empirical risk.")
        return self


class EmpiricalRiskGateArtifact(_EmpiricalRiskGateIdentity):
    artifact_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_artifact_id(self) -> EmpiricalRiskGateArtifact:
        identity = _EmpiricalRiskGateIdentity.model_validate(
            self.model_dump(mode="json", exclude={"artifact_id"})
        )
        if sha256_hex(identity) != self.artifact_id:
            raise ValueError("Gate artifact identity does not match its content.")
        return self


class MetricSummary(StrictModel):
    count: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    nll: float = Field(ge=0.0)
    brier: float = Field(ge=0.0)
    ece: float = Field(ge=0.0, le=1.0)
    ece_bins: int = Field(ge=2)


class RiskCoveragePoint(StrictModel):
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    accepted: int = Field(ge=1)
    coverage: float = Field(gt=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)


class SelectiveSummary(StrictModel):
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    accepted: int = Field(ge=0)
    abstained: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    risk: float | None = Field(default=None, ge=0.0, le=1.0)


class _EvaluationReportIdentity(StrictModel):
    schema_version: Literal[1] = EVALUATION_SCHEMA_VERSION
    evaluation_version: Literal["1"] = EVALUATION_IMPLEMENTATION_VERSION
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    gate_artifact_id: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    snapshot_id: str = Field(pattern=_SHA256_PATTERN)
    test_sha256: str = Field(pattern=_SHA256_PATTERN)
    test_count: int = Field(ge=1)
    config: EvaluationConfig
    raw: MetricSummary
    calibrated: MetricSummary
    risk_coverage: tuple[RiskCoveragePoint, ...]
    aurc: float = Field(ge=0.0, le=1.0)
    selective: SelectiveSummary | None = None


class EvaluationReport(_EvaluationReportIdentity):
    report_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_report_id(self) -> EvaluationReport:
        identity = _EvaluationReportIdentity.model_validate(
            self.model_dump(mode="json", exclude={"report_id"})
        )
        if sha256_hex(identity) != self.report_id:
            raise ValueError("Evaluation report identity does not match its content.")
        return self


class _PredictionRow(StrictModel):
    request_id: UUID
    label: str
    selected: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    correct: bool


def fit_empirical_risk_gate(
    compiler_artifact: CompilerArtifact,
    calibration: TemperatureCalibrationArtifact,
    snapshot: DatasetSnapshot,
    config: EmpiricalRiskGateConfig,
) -> EmpiricalRiskGateArtifact:
    """Fit a confidence threshold on calibration data only.

    This is an empirical operating-point fit, not a distribution-free safety guarantee.
    """
    _validate_calibration_lineage(compiler_artifact, calibration, snapshot)

    examples = snapshot.examples[DatasetSplit.CALIBRATION]
    if not examples:
        raise EvaluationError("Calibration split is empty.")

    rows = tuple(
        _calibrated_row(
            example,
            predict_calibrated(compiler_artifact, calibration, example.state),
        )
        for example in examples
    )
    threshold, accepted_count, empirical_risk = _select_gate_threshold(rows, config)

    split = snapshot.manifest.splits[DatasetSplit.CALIBRATION]
    identity = _EmpiricalRiskGateIdentity(
        compiler_artifact_id=compiler_artifact.artifact_id,
        calibration_artifact_id=calibration.artifact_id,
        snapshot_id=snapshot.manifest.snapshot_id,
        calibration_sha256=split.sha256,
        calibration_count=split.count,
        config=config,
        confidence_threshold=threshold,
        accepted_count=accepted_count,
        empirical_risk=empirical_risk,
    )
    return EmpiricalRiskGateArtifact(
        artifact_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def evaluate(
    compiler_artifact: CompilerArtifact,
    calibration: TemperatureCalibrationArtifact,
    snapshot: DatasetSnapshot,
    *,
    gate: EmpiricalRiskGateArtifact | None = None,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Evaluate raw and calibrated predictions on the sealed test split."""
    resolved_config = config or EvaluationConfig()
    _validate_calibration_lineage(compiler_artifact, calibration, snapshot)
    if gate is not None:
        _validate_gate_lineage(gate, compiler_artifact, calibration, snapshot)

    examples = snapshot.examples[DatasetSplit.TEST]
    if not examples:
        raise EvaluationError("Test split is empty.")

    raw_rows: list[_PredictionRow] = []
    calibrated_rows: list[_PredictionRow] = []
    for example in examples:
        raw = predict(compiler_artifact, example.state)
        calibrated_prediction = predict_calibrated(
            compiler_artifact,
            calibration,
            example.state,
        )
        raw_rows.append(_raw_row(example, raw))
        calibrated_rows.append(_calibrated_row(example, calibrated_prediction))

    raw_tuple = tuple(raw_rows)
    calibrated_tuple = tuple(calibrated_rows)
    risk_curve = _risk_coverage(calibrated_tuple)
    aurc = _aurc(risk_curve)

    selective = (
        _selective_summary(calibrated_tuple, gate.confidence_threshold)
        if gate is not None
        else None
    )
    identity = _EvaluationReportIdentity(
        compiler_artifact_id=compiler_artifact.artifact_id,
        calibration_artifact_id=calibration.artifact_id,
        gate_artifact_id=gate.artifact_id if gate is not None else None,
        snapshot_id=snapshot.manifest.snapshot_id,
        test_sha256=snapshot.manifest.splits[DatasetSplit.TEST].sha256,
        test_count=len(examples),
        config=resolved_config,
        raw=_metric_summary(raw_tuple, resolved_config.ece_bins),
        calibrated=_metric_summary(calibrated_tuple, resolved_config.ece_bins),
        risk_coverage=risk_curve,
        aurc=aurc,
        selective=selective,
    )
    return EvaluationReport(
        report_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def write_gate_artifact(
    artifact: EmpiricalRiskGateArtifact,
    root: str | Path,
) -> Path:
    destination = Path(root) / f"{artifact.artifact_id}.gate.json"
    write_immutable_atomic(
        destination,
        (canonical_json(artifact) + "\n").encode("utf-8"),
    )
    return destination


def load_gate_artifact(path: str | Path) -> EmpiricalRiskGateArtifact:
    return EmpiricalRiskGateArtifact.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def write_evaluation_report(
    report: EvaluationReport,
    root: str | Path,
) -> Path:
    destination = Path(root) / f"{report.report_id}.evaluation.json"
    write_immutable_atomic(
        destination,
        (canonical_json(report) + "\n").encode("utf-8"),
    )
    return destination


def load_evaluation_report(path: str | Path) -> EvaluationReport:
    return EvaluationReport.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _metric_summary(
    rows: tuple[_PredictionRow, ...],
    ece_bins: int,
) -> MetricSummary:
    if not rows:
        raise EvaluationError("Cannot compute metrics for an empty prediction set.")

    classes = tuple(sorted(rows[0].probabilities))
    expected = set(classes)
    for row in rows:
        if set(row.probabilities) != expected:
            raise EvaluationError("Prediction class sets are inconsistent.")

    accuracy = sum(row.correct for row in rows) / len(rows)
    macro_f1 = _macro_f1(rows, classes)
    nll = sum(
        -math.log(max(row.probabilities[row.label], _MIN_PROBABILITY))
        for row in rows
    ) / len(rows)
    brier = sum(
        sum(
            (row.probabilities[class_name] - float(class_name == row.label)) ** 2
            for class_name in classes
        )
        for row in rows
    ) / len(rows)
    ece = _ece(rows, ece_bins)

    return MetricSummary(
        count=len(rows),
        accuracy=accuracy,
        macro_f1=macro_f1,
        nll=nll,
        brier=brier,
        ece=ece,
        ece_bins=ece_bins,
    )


def _macro_f1(
    rows: tuple[_PredictionRow, ...],
    classes: tuple[str, ...],
) -> float:
    scores: list[float] = []
    for class_name in classes:
        true_positive = sum(
            row.label == class_name and row.selected == class_name for row in rows
        )
        false_positive = sum(
            row.label != class_name and row.selected == class_name for row in rows
        )
        false_negative = sum(
            row.label == class_name and row.selected != class_name for row in rows
        )
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else (2 * true_positive) / denominator
        )
    return sum(scores) / len(scores)


def _ece(rows: tuple[_PredictionRow, ...], bin_count: int) -> float:
    counts = [0] * bin_count
    confidence_sums = [0.0] * bin_count
    correct_sums = [0] * bin_count

    for row in rows:
        index = min(int(row.confidence * bin_count), bin_count - 1)
        counts[index] += 1
        confidence_sums[index] += row.confidence
        correct_sums[index] += int(row.correct)

    total = len(rows)
    value = 0.0
    for count, confidence_sum, correct_sum in zip(
        counts,
        confidence_sums,
        correct_sums,
        strict=True,
    ):
        if count == 0:
            continue
        average_confidence = confidence_sum / count
        accuracy = correct_sum / count
        value += (count / total) * abs(accuracy - average_confidence)
    return value


def _risk_coverage(
    rows: tuple[_PredictionRow, ...],
) -> tuple[RiskCoveragePoint, ...]:
    ordered = sorted(
        rows,
        key=lambda row: (-row.confidence, str(row.request_id)),
    )
    points: list[RiskCoveragePoint] = []
    accepted = 0
    errors = 0
    index = 0

    while index < len(ordered):
        threshold = ordered[index].confidence
        while index < len(ordered) and math.isclose(
            ordered[index].confidence,
            threshold,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            accepted += 1
            errors += int(not ordered[index].correct)
            index += 1

        points.append(
            RiskCoveragePoint(
                confidence_threshold=threshold,
                accepted=accepted,
                coverage=accepted / len(ordered),
                risk=errors / accepted,
            )
        )
    return tuple(points)


def _aurc(points: tuple[RiskCoveragePoint, ...]) -> float:
    previous_coverage = 0.0
    area = 0.0
    for point in points:
        area += point.risk * (point.coverage - previous_coverage)
        previous_coverage = point.coverage
    return area


def _select_gate_threshold(
    rows: tuple[_PredictionRow, ...],
    config: EmpiricalRiskGateConfig,
) -> tuple[float | None, int, float | None]:
    points = _risk_coverage(rows)
    eligible = [
        point
        for point in points
        if point.accepted >= config.min_accepted
        and point.risk <= config.max_empirical_risk + 1e-12
    ]
    if not eligible:
        return None, 0, None

    selected = max(
        eligible,
        key=lambda point: (point.accepted, point.confidence_threshold),
    )
    return selected.confidence_threshold, selected.accepted, selected.risk


def _selective_summary(
    rows: tuple[_PredictionRow, ...],
    confidence_threshold: float | None,
) -> SelectiveSummary:
    if confidence_threshold is None:
        return SelectiveSummary(
            confidence_threshold=None,
            accepted=0,
            abstained=len(rows),
            coverage=0.0,
            risk=None,
        )

    accepted_rows = tuple(
        row
        for row in rows
        if row.confidence + 1e-15 >= confidence_threshold
    )
    accepted = len(accepted_rows)
    risk = (
        sum(not row.correct for row in accepted_rows) / accepted
        if accepted
        else None
    )
    return SelectiveSummary(
        confidence_threshold=confidence_threshold,
        accepted=accepted,
        abstained=len(rows) - accepted,
        coverage=accepted / len(rows),
        risk=risk,
    )


def _raw_row(
    example: DatasetExample,
    prediction: CompilerPrediction,
) -> _PredictionRow:
    label = _choice_label(example)
    _validate_prediction_label(label, prediction.probabilities)
    confidence = prediction.probabilities[prediction.selected]
    return _PredictionRow(
        request_id=example.request_id,
        label=label,
        selected=prediction.selected,
        probabilities=prediction.probabilities,
        confidence=confidence,
        correct=prediction.selected == label,
    )


def _calibrated_row(
    example: DatasetExample,
    prediction: CalibratedPrediction,
) -> _PredictionRow:
    label = _choice_label(example)
    _validate_prediction_label(label, prediction.probabilities)
    return _PredictionRow(
        request_id=example.request_id,
        label=label,
        selected=prediction.selected,
        probabilities=prediction.probabilities,
        confidence=prediction.confidence,
        correct=prediction.selected == label,
    )


def _choice_label(example: DatasetExample) -> str:
    if not isinstance(example.label, str):
        raise EvaluationError("v0.1 evaluation currently supports choice labels only.")
    return example.label


def _validate_prediction_label(
    label: str,
    probabilities: dict[str, float],
) -> None:
    if label not in probabilities:
        raise EvaluationError(f"Evaluation label {label!r} is not a prediction class.")


def _validate_calibration_lineage(
    compiler_artifact: CompilerArtifact,
    calibration: TemperatureCalibrationArtifact,
    snapshot: DatasetSnapshot,
) -> None:
    if compiler_artifact.snapshot_id != snapshot.manifest.snapshot_id:
        raise EvaluationError("Compiler artifact and evaluation snapshot IDs do not match.")
    if calibration.compiler_artifact_id != compiler_artifact.artifact_id:
        raise EvaluationError("Calibration artifact does not match the compiler artifact.")
    if calibration.snapshot_id != snapshot.manifest.snapshot_id:
        raise EvaluationError("Calibration artifact and evaluation snapshot IDs do not match.")

    split = snapshot.manifest.splits[DatasetSplit.CALIBRATION]
    if calibration.calibration_sha256 != split.sha256:
        raise EvaluationError("Calibration split hash does not match the calibration artifact.")
    if calibration.calibration_count != split.count:
        raise EvaluationError("Calibration split count does not match the calibration artifact.")


def _validate_gate_lineage(
    gate: EmpiricalRiskGateArtifact,
    compiler_artifact: CompilerArtifact,
    calibration: TemperatureCalibrationArtifact,
    snapshot: DatasetSnapshot,
) -> None:
    if gate.compiler_artifact_id != compiler_artifact.artifact_id:
        raise EvaluationError("Gate artifact does not match the compiler artifact.")
    if gate.calibration_artifact_id != calibration.artifact_id:
        raise EvaluationError("Gate artifact does not match the calibration artifact.")
    if gate.snapshot_id != snapshot.manifest.snapshot_id:
        raise EvaluationError("Gate artifact and evaluation snapshot IDs do not match.")

    split = snapshot.manifest.splits[DatasetSplit.CALIBRATION]
    if gate.calibration_sha256 != split.sha256 or gate.calibration_count != split.count:
        raise EvaluationError("Gate artifact calibration lineage does not match the snapshot.")
