from datetime import UTC, datetime
from uuid import UUID

import pytest

from nerlex.dataset import (
    DatasetSplit,
    SnapshotConfig,
    SplitConfig,
    build_snapshot,
    verify_snapshot,
    write_snapshot,
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

SPEC = DecisionSpec(
    decision_id="support-routing",
    version="1",
    kind=DecisionKind.CHOICE,
    description="Route support requests.",
    candidate_mode=CandidateMode.STATIC,
    candidates=(
        Candidate(key="billing", description="Payments."),
        Candidate(key="technical", description="Product problems."),
    ),
)


def _trace(
    number: int,
    *,
    label_source: LabelSource | None = LabelSource.OUTCOME,
    label_value: str = "billing",
    teacher_value: str = "technical",
) -> TraceRecord:
    request_id = UUID(int=number + 1)
    request = DecisionRequest(
        request_id=request_id,
        decision_id=SPEC.decision_id,
        spec_version=SPEC.version,
        event_time=datetime(2026, 1, 1, 0, 0, number % 60, tzinfo=UTC),
        state={"text": f"case-{number}"},
    )
    teacher = ResultObservation(
        kind="teacher",
        result=DecisionResult(
            request_id=request_id,
            selected=teacher_value,
            probabilities={"billing": 0.25, "technical": 0.75},
            confidence=0.75,
            backend="teacher-test",
            latency_ms=1.0,
            route=DecisionRoute.TEACHER,
            created_at=datetime(2026, 1, 1, 0, 1, number % 60, tzinfo=UTC),
        ),
    )
    labels = ()
    if label_source is not None:
        labels = (
            LabelObservation(
                observation_id=UUID(int=10_000 + number),
                request_id=request_id,
                source=label_source,
                value=label_value,
                observed_at=datetime(2026, 1, 2, 0, 0, number % 60, tzinfo=UTC),
            ),
        )
    return TraceRecord(request=request, observations=(teacher,), labels=labels)


def test_snapshot_is_input_order_independent_and_reproducible(tmp_path) -> None:
    records = [_trace(i) for i in range(40)]
    config = SnapshotConfig(
        split=SplitConfig(train=0.6, calibration=0.2, test=0.2, seed="stable")
    )

    first = build_snapshot(SPEC, records, config=config)
    second = build_snapshot(SPEC, reversed(records), config=config)

    assert first.manifest.snapshot_id == second.manifest.snapshot_id
    assert first.manifest == second.manifest
    assert first.examples == second.examples

    first_path = write_snapshot(first, tmp_path / "a")
    second_path = write_snapshot(second, tmp_path / "b")

    for filename in ("manifest.json", "train.jsonl", "calibration.jsonl", "test.jsonl"):
        assert (first_path / filename).read_bytes() == (second_path / filename).read_bytes()


def test_split_assignment_is_disjoint() -> None:
    snapshot = build_snapshot(SPEC, (_trace(i) for i in range(200)))

    ids_by_split = {
        split: {example.request_id for example in snapshot.examples[split]}
        for split in DatasetSplit
    }

    assert ids_by_split[DatasetSplit.TRAIN].isdisjoint(
        ids_by_split[DatasetSplit.CALIBRATION]
    )
    assert ids_by_split[DatasetSplit.TRAIN].isdisjoint(ids_by_split[DatasetSplit.TEST])
    assert ids_by_split[DatasetSplit.CALIBRATION].isdisjoint(
        ids_by_split[DatasetSplit.TEST]
    )
    assert len(set().union(*ids_by_split.values())) == snapshot.manifest.included_count


def test_teacher_is_excluded_by_default_but_can_be_explicitly_enabled() -> None:
    record = _trace(1, label_source=None, teacher_value="technical")

    default_snapshot = build_snapshot(SPEC, [record])
    assert default_snapshot.manifest.included_count == 0
    assert default_snapshot.manifest.excluded_unlabeled_count == 1

    teacher_snapshot = build_snapshot(
        SPEC,
        [record],
        config=SnapshotConfig(label_priority=(LabelSource.TEACHER,)),
    )

    examples = [
        example
        for split in DatasetSplit
        for example in teacher_snapshot.examples[split]
    ]
    assert len(examples) == 1
    assert examples[0].label == "technical"
    assert examples[0].label_source is LabelSource.TEACHER
    assert examples[0].label_provenance.startswith("teacher:")


def test_outcome_priority_over_teacher_is_recorded() -> None:
    record = _trace(
        2,
        label_source=LabelSource.OUTCOME,
        label_value="billing",
        teacher_value="technical",
    )
    config = SnapshotConfig(
        label_priority=(LabelSource.OUTCOME, LabelSource.TEACHER)
    )

    snapshot = build_snapshot(SPEC, [record], config=config)
    example = next(
        example
        for split in DatasetSplit
        for example in snapshot.examples[split]
    )

    assert snapshot.manifest.config.label_priority == (
        LabelSource.OUTCOME,
        LabelSource.TEACHER,
    )
    assert example.label == "billing"
    assert example.label_source is LabelSource.OUTCOME


def test_conflicting_same_source_labels_fail_closed() -> None:
    record = _trace(3)
    conflicting = record.model_copy(
        update={
            "labels": (
                *record.labels,
                LabelObservation(
                    observation_id=UUID(int=20_003),
                    request_id=record.request.request_id,
                    source=LabelSource.OUTCOME,
                    value="technical",
                    observed_at=datetime(2026, 1, 3, tzinfo=UTC),
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="Conflicting outcome labels"):
        build_snapshot(SPEC, [conflicting])


def test_snapshot_verification_detects_tampering(tmp_path) -> None:
    snapshot = build_snapshot(SPEC, [_trace(i) for i in range(20)])
    path = write_snapshot(snapshot, tmp_path)

    verify_snapshot(path)

    target = path / snapshot.manifest.splits[DatasetSplit.TRAIN].filename
    target.write_bytes(target.read_bytes() + b"{\"tampered\":true}\n")

    with pytest.raises(ValueError, match="split hash mismatch"):
        verify_snapshot(path)


def test_choice_label_must_exist_in_candidates() -> None:
    record = _trace(4, label_value="unknown")

    with pytest.raises(ValueError, match="not a valid candidate"):
        build_snapshot(SPEC, [record])


def test_manifest_identity_tampering_is_rejected(tmp_path) -> None:
    snapshot = build_snapshot(SPEC, [_trace(i) for i in range(10)])
    path = write_snapshot(snapshot, tmp_path)
    manifest_path = path / "manifest.json"
    payload = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        payload.replace(snapshot.manifest.snapshot_id, "0" * 64, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity does not match"):
        verify_snapshot(path)


def test_manifest_rejects_unsafe_split_filename() -> None:
    snapshot = build_snapshot(SPEC, [_trace(i) for i in range(10)])
    payload = snapshot.manifest.model_dump(mode="json")
    payload["splits"]["train"]["filename"] = "../train.jsonl"

    manifest_type = type(snapshot.manifest)

    with pytest.raises(ValueError, match="Unexpected filename"):
        manifest_type.model_validate(payload)


def test_write_snapshot_is_idempotent_and_refuses_modified_existing_file(tmp_path) -> None:
    snapshot = build_snapshot(SPEC, [_trace(i) for i in range(20)])
    path = write_snapshot(snapshot, tmp_path)

    assert write_snapshot(snapshot, tmp_path) == path

    target = path / snapshot.manifest.splits[DatasetSplit.TEST].filename
    target.write_bytes(target.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="immutable snapshot file"):
        write_snapshot(snapshot, tmp_path)
