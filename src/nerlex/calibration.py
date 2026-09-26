from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from nerlex.artifact_io import write_immutable_atomic
from nerlex.compiler import (
    CompilerArtifact,
    CompilerPrediction,
    predict,
)
from nerlex.dataset import DatasetExample, DatasetSnapshot, DatasetSplit
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.spec import StrictModel

CALIBRATION_SCHEMA_VERSION: Literal[1] = 1
CALIBRATION_IMPLEMENTATION_VERSION: Literal["1"] = "1"

_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_MIN_PROBABILITY = 1e-15


class CalibrationError(ValueError):
    """Raised when calibration lineage or numerical assumptions are invalid."""


class TemperatureCalibrationConfig(StrictModel):
    method: Literal["temperature"] = "temperature"
    min_temperature: float = Field(default=0.05, gt=0.0)
    max_temperature: float = Field(default=20.0, gt=0.0)
    iterations: int = Field(default=96, ge=16, le=512)

    @model_validator(mode="after")
    def _validate_bounds(self) -> TemperatureCalibrationConfig:
        if self.min_temperature >= self.max_temperature:
            raise ValueError("min_temperature must be less than max_temperature.")
        return self


class _TemperatureCalibrationIdentity(StrictModel):
    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION
    calibration_version: Literal["1"] = CALIBRATION_IMPLEMENTATION_VERSION
    method: Literal["temperature"] = "temperature"
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    snapshot_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_sha256: str = Field(pattern=_SHA256_PATTERN)
    calibration_count: int = Field(ge=1)
    classes: tuple[str, ...]
    config: TemperatureCalibrationConfig
    temperature: float = Field(gt=0.0)
    objective: Literal["nll"] = "nll"


class TemperatureCalibrationArtifact(_TemperatureCalibrationIdentity):
    artifact_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_artifact_id(self) -> TemperatureCalibrationArtifact:
        identity = _TemperatureCalibrationIdentity.model_validate(
            self.model_dump(mode="json", exclude={"artifact_id"})
        )
        if sha256_hex(identity) != self.artifact_id:
            raise ValueError("Calibration artifact identity does not match its content.")
        return self


class CalibratedPrediction(StrictModel):
    compiler_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    calibration_artifact_id: str = Field(pattern=_SHA256_PATTERN)
    backend: Literal["multinomial_nb", "centroid_cosine"]
    selected: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated: Literal[True] = True

    @model_validator(mode="after")
    def _validate_probabilities(self) -> CalibratedPrediction:
        if not self.probabilities:
            raise ValueError("Calibrated probabilities must not be empty.")
        if self.selected not in self.probabilities:
            raise ValueError("Selected class must exist in calibrated probabilities.")
        if any(value < 0.0 or value > 1.0 for value in self.probabilities.values()):
            raise ValueError("Calibrated probabilities must be in [0, 1].")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-9:
            raise ValueError("Calibrated probabilities must sum to 1.")
        if abs(self.probabilities[self.selected] - self.confidence) > 1e-12:
            raise ValueError("confidence must equal the selected calibrated probability.")
        return self


def fit_temperature(
    compiler_artifact: CompilerArtifact,
    snapshot: DatasetSnapshot,
    *,
    config: TemperatureCalibrationConfig | None = None,
) -> TemperatureCalibrationArtifact:
    """Fit scalar temperature on the snapshot calibration split only."""
    resolved_config = config or TemperatureCalibrationConfig()
    _validate_lineage(compiler_artifact, snapshot)

    split_artifact = snapshot.manifest.splits[DatasetSplit.CALIBRATION]
    examples = snapshot.examples[DatasetSplit.CALIBRATION]
    if not examples:
        raise CalibrationError("Calibration split is empty.")

    rows = tuple(
        (
            _choice_label(example),
            predict(compiler_artifact, example.state).probabilities,
        )
        for example in examples
    )
    classes = _artifact_classes(compiler_artifact)
    _validate_rows(rows, classes)

    temperature = _fit_temperature_value(rows, resolved_config)
    identity = _TemperatureCalibrationIdentity(
        compiler_artifact_id=compiler_artifact.artifact_id,
        snapshot_id=snapshot.manifest.snapshot_id,
        calibration_sha256=split_artifact.sha256,
        calibration_count=split_artifact.count,
        classes=classes,
        config=resolved_config,
        temperature=temperature,
    )
    return TemperatureCalibrationArtifact(
        artifact_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def calibrate_prediction(
    prediction: CompilerPrediction,
    calibration: TemperatureCalibrationArtifact,
) -> CalibratedPrediction:
    if prediction.artifact_id != calibration.compiler_artifact_id:
        raise CalibrationError(
            "Prediction compiler artifact does not match the calibration artifact."
        )
    if tuple(sorted(prediction.probabilities)) != calibration.classes:
        raise CalibrationError("Prediction classes do not match the calibration artifact.")

    probabilities = apply_temperature(
        prediction.probabilities,
        calibration.temperature,
    )
    selected = max(
        sorted(probabilities),
        key=lambda class_name: probabilities[class_name],
    )
    return CalibratedPrediction(
        compiler_artifact_id=prediction.artifact_id,
        calibration_artifact_id=calibration.artifact_id,
        backend=prediction.backend,
        selected=selected,
        probabilities=probabilities,
        confidence=probabilities[selected],
    )


def predict_calibrated(
    compiler_artifact: CompilerArtifact,
    calibration: TemperatureCalibrationArtifact,
    state: Mapping[str, object] | str,
) -> CalibratedPrediction:
    _validate_calibration_for_compiler(compiler_artifact, calibration)
    return calibrate_prediction(
        predict(compiler_artifact, state),
        calibration,
    )


def apply_temperature(
    probabilities: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationError("Temperature must be finite and greater than zero.")
    if not probabilities:
        raise CalibrationError("Cannot calibrate an empty probability map.")

    ordered = tuple(sorted(probabilities))
    if any(
        not math.isfinite(probabilities[class_name])
        or probabilities[class_name] < 0.0
        or probabilities[class_name] > 1.0
        for class_name in ordered
    ):
        raise CalibrationError("Input probabilities must be finite and in [0, 1].")

    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise CalibrationError("Input probabilities must sum to 1.")

    logits = {
        class_name: math.log(max(probabilities[class_name], _MIN_PROBABILITY))
        / temperature
        for class_name in ordered
    }
    maximum = max(logits.values())
    exp_values = {
        class_name: math.exp(logits[class_name] - maximum)
        for class_name in ordered
    }
    denominator = sum(exp_values.values())
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise CalibrationError("Calibrated probabilities are not normalizable.")
    return {
        class_name: exp_values[class_name] / denominator
        for class_name in ordered
    }


def write_calibration_artifact(
    artifact: TemperatureCalibrationArtifact,
    root: str | Path,
) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    destination = root_path / f"{artifact.artifact_id}.calibration.json"
    payload = (canonical_json(artifact) + "\n").encode("utf-8")
    write_immutable_atomic(destination, payload)
    return destination


def load_calibration_artifact(path: str | Path) -> TemperatureCalibrationArtifact:
    return TemperatureCalibrationArtifact.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _fit_temperature_value(
    rows: Sequence[tuple[str, Mapping[str, float]]],
    config: TemperatureCalibrationConfig,
) -> float:
    """Deterministic golden-section search over log temperature."""
    lower = math.log(config.min_temperature)
    upper = math.log(config.max_temperature)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_loss = _temperature_nll(rows, math.exp(left))
    right_loss = _temperature_nll(rows, math.exp(right))

    for _ in range(config.iterations):
        if left_loss <= right_loss:
            upper = right
            right = left
            right_loss = left_loss
            left = upper - ratio * (upper - lower)
            left_loss = _temperature_nll(rows, math.exp(left))
        else:
            lower = left
            left = right
            left_loss = right_loss
            right = lower + ratio * (upper - lower)
            right_loss = _temperature_nll(rows, math.exp(right))

    temperature = math.exp((lower + upper) / 2.0)
    return float(f"{temperature:.15g}")


def _temperature_nll(
    rows: Sequence[tuple[str, Mapping[str, float]]],
    temperature: float,
) -> float:
    total = 0.0
    for label, probabilities in rows:
        calibrated = apply_temperature(probabilities, temperature)
        total -= math.log(max(calibrated[label], _MIN_PROBABILITY))
    return total / len(rows)


def _validate_lineage(
    compiler_artifact: CompilerArtifact,
    snapshot: DatasetSnapshot,
) -> None:
    if compiler_artifact.snapshot_id != snapshot.manifest.snapshot_id:
        raise CalibrationError("Compiler artifact and calibration snapshot IDs do not match.")
    if compiler_artifact.spec_hash != snapshot.manifest.spec_hash:
        raise CalibrationError(
            "Compiler artifact and calibration DecisionSpec hashes do not match."
        )


def _validate_calibration_for_compiler(
    compiler_artifact: CompilerArtifact,
    calibration: TemperatureCalibrationArtifact,
) -> None:
    if calibration.compiler_artifact_id != compiler_artifact.artifact_id:
        raise CalibrationError("Calibration artifact does not belong to this compiler artifact.")
    if calibration.snapshot_id != compiler_artifact.snapshot_id:
        raise CalibrationError("Calibration/compiler source snapshot IDs do not match.")
    if calibration.classes != _artifact_classes(compiler_artifact):
        raise CalibrationError("Calibration/compiler classes do not match.")


def _artifact_classes(artifact: CompilerArtifact) -> tuple[str, ...]:
    return artifact.model.classes


def _choice_label(example: DatasetExample) -> str:
    if not isinstance(example.label, str):
        raise CalibrationError("Temperature calibration currently supports choice labels only.")
    return example.label


def _validate_rows(
    rows: Sequence[tuple[str, Mapping[str, float]]],
    classes: tuple[str, ...],
) -> None:
    expected = set(classes)
    for label, probabilities in rows:
        if label not in expected:
            raise CalibrationError(f"Calibration label {label!r} is not a compiler class.")
        if set(probabilities) != expected:
            raise CalibrationError("Calibration prediction classes are inconsistent.")

