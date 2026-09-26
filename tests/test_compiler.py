from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from nerlex.compiler import (
    CompilerConfig,
    CompilerKind,
    compile_snapshot,
    load_compiler_artifact,
    predict,
    write_compiler_artifact,
)
from nerlex.dataset import SnapshotConfig, SplitConfig, build_snapshot, write_snapshot
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


CHOICE_SPEC = DecisionSpec(
    decision_id="support-routing",
    version="1",
    kind=DecisionKind.CHOICE,
    description="Route support requests.",
    candidate_mode=CandidateMode.STATIC,
    candidates=(
        Candidate(key="billing", description="Payments, invoices, charges, and refunds."),
        Candidate(key="technical", description="Product crashes, errors, and software bugs."),
    ),
)


def _choice_trace(number: int, label: str) -> TraceRecord:
    text = (
        f"invoice charged refund payment case {number}"
        if label == "billing"
        else f"crash error bug software failure case {number}"
    )
    request_id = UUID(int=number + 1)
    request = DecisionRequest(
        request_id=request_id,
        decision_id=CHOICE_SPEC.decision_id,
        spec_version=CHOICE_SPEC.version,
        event_time=datetime(2026, 1, 1, 0, 0, number % 60, tzinfo=UTC),
        state={"text": text},
    )
    label_observation = LabelObservation(
        observation_id=UUID(int=100_000 + number),
        request_id=request_id,
        source=LabelSource.OUTCOME,
        value=label,
        observed_at=datetime(2026, 1, 2, 0, 0, number % 60, tzinfo=UTC),
    )
    return TraceRecord(request=request, labels=(label_observation,))


def _choice_snapshot(tmp_path: Path, *, count: int = 80) -> Path:
    records = [
        _choice_trace(number, "billing" if number % 2 == 0 else "technical")
        for number in range(count)
    ]
    snapshot = build_snapshot(
        CHOICE_SPEC,
        records,
        config=SnapshotConfig(
            split=SplitConfig(
                train=0.8,
                calibration=0.1,
                test=0.1,
                seed="compiler-tests",
            )
        ),
    )
    return write_snapshot(snapshot, tmp_path)


@pytest.mark.parametrize(
    "kind",
    (CompilerKind.MULTINOMIAL_NB, CompilerKind.CENTROID_COSINE),
)
def test_compile_is_deterministic_and_artifact_is_safe_json(tmp_path, kind) -> None:
    snapshot_path = _choice_snapshot(tmp_path / "snapshot")
    config = CompilerConfig(kind=kind)

    first = compile_snapshot(snapshot_path, config)
    second = compile_snapshot(snapshot_path, config)

    assert first == second
    assert first.artifact_id == second.artifact_id
    assert first.calibrated is False

    first_path = write_compiler_artifact(first, tmp_path / "first.json")
    second_path = write_compiler_artifact(second, tmp_path / "second.json")

    assert first_path.read_bytes() == second_path.read_bytes()
    assert load_compiler_artifact(first_path) == first
    assert b"pickle" not in first_path.read_bytes().lower()


def test_multinomial_nb_learns_billing_signal(tmp_path) -> None:
    snapshot_path = _choice_snapshot(tmp_path)
    artifact = compile_snapshot(
        snapshot_path,
        CompilerConfig(kind=CompilerKind.MULTINOMIAL_NB),
    )

    result = predict(artifact, {"text": "refund for duplicate invoice charge"})

    assert result.selected == "billing"
    assert result.probabilities["billing"] > result.probabilities["technical"]
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.calibrated is False


def test_centroid_cosine_learns_technical_signal(tmp_path) -> None:
    snapshot_path = _choice_snapshot(tmp_path)
    artifact = compile_snapshot(
        snapshot_path,
        CompilerConfig(kind=CompilerKind.CENTROID_COSINE),
    )

    result = predict(artifact, {"text": "software crash error and bug"})

    assert result.selected == "technical"
    assert result.scores["technical"] > result.scores["billing"]
    assert result.calibrated is False


def test_artifact_records_snapshot_and_train_provenance(tmp_path) -> None:
    snapshot_path = _choice_snapshot(tmp_path)
    artifact = compile_snapshot(
        snapshot_path,
        CompilerConfig(kind=CompilerKind.MULTINOMIAL_NB),
    )

    snapshot_id = snapshot_path.name

    assert artifact.source_snapshot_id == snapshot_id
    assert artifact.decision_spec == CHOICE_SPEC
    assert len(artifact.train_sha256) == 64
    assert len(artifact.spec_hash) == 64


def test_non_train_examples_do_not_change_fitted_payload(tmp_path) -> None:
    base_records = [
        _choice_trace(number, "billing" if number % 2 == 0 else "technical")
        for number in range(80)
    ]
    config = SnapshotConfig(
        split=SplitConfig(
            train=0.8,
            calibration=0.1,
            test=0.1,
            seed="compiler-tests",
        )
    )
    first_snapshot = build_snapshot(CHOICE_SPEC, base_records, config=config)
    first_train_hash = first_snapshot.manifest.splits["train"].sha256

    second_snapshot = None
    for number in range(1_000, 1_200):
        candidate = build_snapshot(
            CHOICE_SPEC,
            [*base_records, _choice_trace(number, "billing")],
            config=config,
        )
        if candidate.manifest.splits["train"].sha256 == first_train_hash:
            second_snapshot = candidate
            break

    assert second_snapshot is not None
    assert second_snapshot.manifest.snapshot_id != first_snapshot.manifest.snapshot_id

    first_path = write_snapshot(first_snapshot, tmp_path / "first")
    second_path = write_snapshot(second_snapshot, tmp_path / "second")
    compiler_config = CompilerConfig(kind=CompilerKind.CENTROID_COSINE)

    first_artifact = compile_snapshot(first_path, compiler_config)
    second_artifact = compile_snapshot(second_path, compiler_config)

    assert first_artifact.train_sha256 == second_artifact.train_sha256
    assert first_artifact.payload == second_artifact.payload
    assert first_artifact.artifact_id != second_artifact.artifact_id


def test_artifact_identity_tampering_is_rejected(tmp_path) -> None:
    snapshot_path = _choice_snapshot(tmp_path / "snapshot")
    artifact = compile_snapshot(
        snapshot_path,
        CompilerConfig(kind=CompilerKind.MULTINOMIAL_NB),
    )
    path = write_compiler_artifact(artifact, tmp_path / "artifact.json")
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload.replace(artifact.artifact_id, "0" * 64, 1), encoding="utf-8")

    with pytest.raises(ValueError, match="identity does not match"):
        load_compiler_artifact(path)


def test_dynamic_candidates_fail_explicitly(tmp_path) -> None:
    spec = DecisionSpec(
        decision_id="tool-routing",
        version="1",
        kind=DecisionKind.CHOICE,
        description="Choose a tool from request-provided candidates.",
        candidate_mode=CandidateMode.DYNAMIC,
    )
    candidates = (
        Candidate(key="search", description="Search the web."),
        Candidate(key="code", description="Edit source code."),
    )
    records = []
    for number in range(30):
        request_id = UUID(int=10_000 + number)
        request = DecisionRequest(
            request_id=request_id,
            decision_id=spec.decision_id,
            spec_version=spec.version,
            state={"text": "search documentation"},
            candidates=candidates,
        )
        records.append(
            TraceRecord(
                request=request,
                labels=(
                    LabelObservation(
                        observation_id=UUID(int=20_000 + number),
                        request_id=request_id,
                        source=LabelSource.OUTCOME,
                        value="search",
                    ),
                ),
            )
        )

    snapshot = build_snapshot(
        spec,
        records,
        config=SnapshotConfig(
            split=SplitConfig(train=0.8, calibration=0.1, test=0.1, seed="dynamic")
        ),
    )
    snapshot_path = write_snapshot(snapshot, tmp_path)

    with pytest.raises(ValueError, match="Dynamic candidate decisions are not supported"):
        compile_snapshot(
            snapshot_path,
            CompilerConfig(kind=CompilerKind.MULTINOMIAL_NB),
        )


def test_boolean_compiler_maps_labels_without_string_leakage(tmp_path) -> None:
    spec = DecisionSpec(
        decision_id="escalate",
        version="1",
        kind=DecisionKind.BOOLEAN,
        description="Decide whether a case requires escalation.",
    )
    records = []
    for number in range(80):
        request_id = UUID(int=30_000 + number)
        value = number % 2 == 0
        state = {"text": "urgent severe outage" if value else "routine informational request"}
        records.append(
            TraceRecord(
                request=DecisionRequest(
                    request_id=request_id,
                    decision_id=spec.decision_id,
                    spec_version=spec.version,
                    state=state,
                ),
                labels=(
                    LabelObservation(
                        observation_id=UUID(int=40_000 + number),
                        request_id=request_id,
                        source=LabelSource.OUTCOME,
                        value=value,
                    ),
                ),
            )
        )

    snapshot = build_snapshot(
        spec,
        records,
        config=SnapshotConfig(
            split=SplitConfig(train=0.8, calibration=0.1, test=0.1, seed="boolean")
        ),
    )
    snapshot_path = write_snapshot(snapshot, tmp_path)
    artifact = compile_snapshot(
        snapshot_path,
        CompilerConfig(kind=CompilerKind.MULTINOMIAL_NB),
    )

    result = predict(artifact, {"text": "urgent outage is severe"})

    assert result.selected is True
    assert set(result.probabilities) == {"false", "true"}
