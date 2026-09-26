from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from nerlex.hashing import canonical_json, sha256_hex
from nerlex.spec import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionSpec,
    LabelSource,
    StrictModel,
)
from nerlex.trace import TraceRecord


class DatasetSplit(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class SplitConfig(StrictModel):
    train: float = Field(default=0.70, gt=0.0, lt=1.0)
    calibration: float = Field(default=0.15, gt=0.0, lt=1.0)
    test: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: str = Field(default="nerlex-v0", min_length=1, max_length=256)

    @model_validator(mode="after")
    def _validate_total(self) -> SplitConfig:
        if abs((self.train + self.calibration + self.test) - 1.0) > 1e-9:
            raise ValueError("Train, calibration, and test fractions must sum to 1.")
        return self


class SnapshotConfig(StrictModel):
    label_priority: tuple[LabelSource, ...] = (
        LabelSource.ADJUDICATED,
        LabelSource.OUTCOME,
        LabelSource.HUMAN,
        LabelSource.RULE,
    )
    split: SplitConfig = Field(default_factory=SplitConfig)

    @model_validator(mode="after")
    def _validate_priority(self) -> SnapshotConfig:
        if not self.label_priority:
            raise ValueError("label_priority must contain at least one source.")
        if len(self.label_priority) != len(set(self.label_priority)):
            raise ValueError("label_priority must not contain duplicate sources.")
        return self


class DatasetExample(StrictModel):
    request_id: UUID
    state: dict[str, Any] | str
    candidates: tuple[Candidate, ...] = ()
    label: str | bool
    label_source: LabelSource
    label_provenance: str = Field(min_length=1, max_length=512)


class SplitArtifact(StrictModel):
    filename: str = Field(min_length=1, max_length=128)
    count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SnapshotManifest(StrictModel):
    schema_version: int = 1
    snapshot_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision_spec: DecisionSpec
    spec_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    config: SnapshotConfig
    source_trace_count: int = Field(ge=0)
    included_count: int = Field(ge=0)
    excluded_unlabeled_count: int = Field(ge=0)
    splits: dict[DatasetSplit, SplitArtifact]


class DatasetSnapshot(StrictModel):
    manifest: SnapshotManifest
    examples: dict[DatasetSplit, tuple[DatasetExample, ...]]


class _ResolvedLabel(StrictModel):
    value: str | bool
    source: LabelSource
    provenance: str


def build_snapshot(
    spec: DecisionSpec,
    records: Iterable[TraceRecord],
    *,
    config: SnapshotConfig | None = None,
) -> DatasetSnapshot:
    resolved_config = config or SnapshotConfig()
    spec_hash = sha256_hex(spec)

    seen_request_ids: set[UUID] = set()
    buckets: dict[DatasetSplit, list[DatasetExample]] = {
        split: [] for split in DatasetSplit
    }
    source_trace_count = 0
    excluded_unlabeled_count = 0

    for record in records:
        source_trace_count += 1
        request = record.request
        if request.request_id in seen_request_ids:
            raise ValueError(f"Duplicate request_id in snapshot input: {request.request_id}.")
        seen_request_ids.add(request.request_id)

        if request.decision_id != spec.decision_id or request.spec_version != spec.version:
            raise ValueError("Trace request does not match the supplied DecisionSpec.")

        resolved = _resolve_label(record, resolved_config.label_priority)
        if resolved is None:
            excluded_unlabeled_count += 1
            continue

        _validate_label(spec, record, resolved.value)
        example = DatasetExample(
            request_id=request.request_id,
            state=request.state,
            candidates=request.candidates,
            label=resolved.value,
            label_source=resolved.source,
            label_provenance=resolved.provenance,
        )
        buckets[_assign_split(request.request_id, resolved_config.split)].append(example)

    for values in buckets.values():
        values.sort(key=lambda example: str(example.request_id))

    split_bytes = {
        split: _examples_bytes(tuple(values))
        for split, values in buckets.items()
    }
    artifacts = {
        split: SplitArtifact(
            filename=f"{split.value}.jsonl",
            count=len(buckets[split]),
            sha256=hashlib.sha256(split_bytes[split]).hexdigest(),
        )
        for split in DatasetSplit
    }
    included_count = sum(artifact.count for artifact in artifacts.values())

    manifest_core = {
        "schema_version": 1,
        "decision_spec": spec,
        "spec_hash": spec_hash,
        "config": resolved_config,
        "source_trace_count": source_trace_count,
        "included_count": included_count,
        "excluded_unlabeled_count": excluded_unlabeled_count,
        "splits": artifacts,
    }
    snapshot_id = sha256_hex(manifest_core)
    manifest = SnapshotManifest(snapshot_id=snapshot_id, **manifest_core)

    return DatasetSnapshot(
        manifest=manifest,
        examples={split: tuple(values) for split, values in buckets.items()},
    )


def write_snapshot(snapshot: DatasetSnapshot, root: str | Path) -> Path:
    root_path = Path(root)
    destination = root_path / snapshot.manifest.snapshot_id
    destination.mkdir(parents=True, exist_ok=True)

    for split in DatasetSplit:
        artifact = snapshot.manifest.splits[split]
        payload = _examples_bytes(snapshot.examples[split])
        expected = hashlib.sha256(payload).hexdigest()
        if expected != artifact.sha256:
            raise ValueError(f"In-memory {split.value} content does not match manifest hash.")
        _write_immutable(destination / artifact.filename, payload)

    manifest_payload = (canonical_json(snapshot.manifest) + "\n").encode("utf-8")
    _write_immutable(destination / "manifest.json", manifest_payload)
    verify_snapshot(destination)
    return destination


def verify_snapshot(path: str | Path) -> SnapshotManifest:
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    manifest = SnapshotManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    core = manifest.model_dump(mode="python", exclude={"snapshot_id"})
    if sha256_hex(core) != manifest.snapshot_id:
        raise ValueError("Snapshot manifest identity does not match its content.")

    total = 0
    for split in DatasetSplit:
        artifact = manifest.splits[split]
        payload = (directory / artifact.filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError(f"{split.value} split hash mismatch.")

        parsed = _parse_examples_bytes(payload, artifact.filename)
        if len(parsed) != artifact.count:
            raise ValueError(f"{split.value} split count mismatch.")
        total += len(parsed)

    if total != manifest.included_count:
        raise ValueError("Snapshot included_count does not match split counts.")
    return manifest


def _resolve_label(
    record: TraceRecord,
    priority: tuple[LabelSource, ...],
) -> _ResolvedLabel | None:
    for source in priority:
        candidates = _labels_for_source(record, source)
        if not candidates:
            continue

        values = {canonical_json(candidate.value) for candidate in candidates}
        if len(values) != 1:
            raise ValueError(
                f"Conflicting {source.value} labels for request {record.request.request_id}."
            )
        return max(candidates, key=lambda item: item.provenance)
    return None


def _labels_for_source(
    record: TraceRecord,
    source: LabelSource,
) -> list[_ResolvedLabel]:
    labels = [
        _ResolvedLabel(
            value=label.value,
            source=source,
            provenance=f"label:{label.observation_id}",
        )
        for label in record.labels
        if label.source is source
    ]

    if source is not LabelSource.TEACHER:
        return labels

    for observation in record.observations:
        result = observation.result
        if observation.kind != "teacher" or result.abstained or result.selected is None:
            continue
        labels.append(
            _ResolvedLabel(
                value=result.selected,
                source=LabelSource.TEACHER,
                provenance=(
                    f"teacher:{result.backend}:"
                    f"{result.created_at.isoformat()}:{result.request_id}"
                ),
            )
        )
    return labels


def _validate_label(
    spec: DecisionSpec,
    record: TraceRecord,
    value: str | bool,
) -> None:
    if spec.kind is DecisionKind.BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError("Boolean DecisionSpec requires boolean labels.")
        return

    if not isinstance(value, str):
        raise ValueError("Choice DecisionSpec requires string labels.")

    candidates = (
        spec.candidates
        if spec.candidate_mode is CandidateMode.STATIC
        else record.request.candidates
    )
    valid_keys = {candidate.key for candidate in candidates}
    if value not in valid_keys:
        raise ValueError(
            f"Label {value!r} is not a valid candidate for request {record.request.request_id}."
        )


def _assign_split(request_id: UUID, config: SplitConfig) -> DatasetSplit:
    digest = hashlib.sha256(f"{config.seed}:{request_id}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    if point < config.train:
        return DatasetSplit.TRAIN
    if point < config.train + config.calibration:
        return DatasetSplit.CALIBRATION
    return DatasetSplit.TEST


def _examples_bytes(examples: tuple[DatasetExample, ...]) -> bytes:
    return "".join(f"{canonical_json(example)}\n" for example in examples).encode("utf-8")


def _parse_examples_bytes(payload: bytes, filename: str) -> list[DatasetExample]:
    parsed: list[DatasetExample] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            parsed.append(DatasetExample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"Invalid dataset example in {filename} at line {line_number}."
            ) from exc
    return parsed


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite immutable snapshot file: {path.name}.")
        return
    path.write_bytes(payload)
