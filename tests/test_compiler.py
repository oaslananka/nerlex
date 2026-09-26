from __future__ import annotations

import math
from pathlib import Path
from uuid import UUID

import pytest

from nerlex.compiler import (
    CentroidConfig,
    CompilerArtifact,
    CompilerError,
    MultinomialNBConfig,
    compile_snapshot,
    load_compiler_artifact,
    predict,
    write_compiler_artifact,
)
from nerlex.dataset import DatasetSplit, SnapshotConfig, SplitConfig, build_snapshot
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


def _trace(index: int, label: str, text: str) -> TraceRecord:
    request_id = UUID(int=index + 1)
    return TraceRecord(
        request=DecisionRequest(
            request_id=request_id,
            decision_id="support-routing",
            spec_version="1",
            state={"text": text},
        ),
        labels=(
            LabelObservation(
                observation_id=UUID(int=10_000 + index),
                request_id=request_id,
                source=LabelSource.OUTCOME,
                value=label,
                source_id="fixture",
            ),
        ),
    )


def _snapshot():
    records = []
    for index in range(120):
        if index % 2 == 0:
            records.append(_trace(index, "billing", f"invoice refund payment card marker{index}"))
        else:
            records.append(_trace(index, "technical", f"crash error bug software marker{index}"))
    return build_snapshot(
        _spec(),
        records,
        config=SnapshotConfig(
            split=SplitConfig(train=0.7, calibration=0.15, test=0.15, seed="compiler-test")
        ),
    )


@pytest.mark.parametrize(
    "config",
    (MultinomialNBConfig(), CentroidConfig()),
)
def test_same_snapshot_produces_byte_stable_artifact(tmp_path: Path, config) -> None:
    snapshot = _snapshot()

    first = compile_snapshot(snapshot, config)
    second = compile_snapshot(snapshot, config)

    assert first == second
    assert first.artifact_id == second.artifact_id
    assert first.snapshot_id == snapshot.manifest.snapshot_id
    assert first.train_sha256 == snapshot.manifest.splits[DatasetSplit.TRAIN].sha256
    assert first.calibrated is False

    first_path = write_compiler_artifact(first, tmp_path / "a")
    second_path = write_compiler_artifact(second, tmp_path / "b")
    assert first_path.read_bytes() == second_path.read_bytes()


@pytest.mark.parametrize(
    "config",
    (MultinomialNBConfig(), CentroidConfig()),
)
def test_prediction_is_deterministic_and_uncalibrated(config) -> None:
    artifact = compile_snapshot(_snapshot(), config)

    first = predict(artifact, {"text": "refund invoice card"})
    second = predict(artifact, {"text": "refund invoice card"})

    assert first == second
    assert first.selected == "billing"
    assert first.calibrated is False
    assert abs(sum(first.probabilities.values()) - 1.0) < 1e-12


def test_compiler_reads_train_split_only() -> None:
    snapshot = _snapshot()
    train_ids = {example.request_id for example in snapshot.examples[DatasetSplit.TRAIN]}
    non_train_markers = {
        f"marker{int(example.request_id) - 1}"
        for split in (DatasetSplit.CALIBRATION, DatasetSplit.TEST)
        for example in snapshot.examples[split]
    }
    train_markers = {f"marker{int(request_id) - 1}" for request_id in train_ids}

    artifact = compile_snapshot(snapshot, MultinomialNBConfig())

    assert non_train_markers
    assert all(
        marker not in artifact.model.vocabulary
        for marker in non_train_markers - train_markers
    )


def test_artifact_tampering_fails_closed(tmp_path: Path) -> None:
    artifact = compile_snapshot(_snapshot(), MultinomialNBConfig())
    path = write_compiler_artifact(artifact, tmp_path)

    raw = path.read_text(encoding="utf-8").replace(
        '"artifact_id":"',
        '"artifact_id":"0',
        1,
    )
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError):
        load_compiler_artifact(path)


def test_dynamic_candidates_are_explicitly_unsupported() -> None:
    spec = DecisionSpec(
        decision_id="tool-routing",
        version="1",
        kind=DecisionKind.CHOICE,
        description="Route to a dynamic tool.",
        candidate_mode=CandidateMode.DYNAMIC,
    )
    request_id = UUID(int=1)
    snapshot = build_snapshot(
        spec,
        [
            TraceRecord(
                request=DecisionRequest(
                    request_id=request_id,
                    decision_id="tool-routing",
                    spec_version="1",
                    state="deploy",
                    candidates=(
                        Candidate(key="deploy", description="Deploy applications."),
                        Candidate(key="search", description="Search documentation."),
                    ),
                ),
                labels=(
                    LabelObservation(
                        observation_id=UUID(int=2),
                        request_id=request_id,
                        source=LabelSource.OUTCOME,
                        value="deploy",
                    ),
                ),
            )
        ],
        config=SnapshotConfig(split=SplitConfig(train=0.98, calibration=0.01, test=0.01)),
    )

    config = MultinomialNBConfig()
    with pytest.raises(CompilerError, match="Dynamic candidate decisions"):
        compile_snapshot(snapshot, config)


def test_loaded_artifact_validates_identity(tmp_path: Path) -> None:
    artifact = compile_snapshot(_snapshot(), CentroidConfig())
    path = write_compiler_artifact(artifact, tmp_path)

    loaded = load_compiler_artifact(path)

    assert isinstance(loaded, CompilerArtifact)
    assert loaded == artifact


def test_centroid_oov_prediction_is_finite_and_uniform() -> None:
    artifact = compile_snapshot(_snapshot(), CentroidConfig())

    prediction = predict(artifact, {"text": "completely_unseen_oov_token"})

    assert prediction.probabilities == pytest.approx(
        {"billing": 0.5, "technical": 0.5}
    )


def test_multinomial_oov_prediction_uses_finite_class_priors() -> None:
    artifact = compile_snapshot(_snapshot(), MultinomialNBConfig())

    prediction = predict(artifact, {"text": "completely_unseen_oov_token"})

    assert all(math.isfinite(value) for value in prediction.probabilities.values())
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_rewriting_identical_artifact_is_idempotent(tmp_path: Path) -> None:
    artifact = compile_snapshot(_snapshot(), MultinomialNBConfig())

    first = write_compiler_artifact(artifact, tmp_path)
    second = write_compiler_artifact(artifact, tmp_path)

    assert first == second
    assert first.read_bytes() == second.read_bytes()
