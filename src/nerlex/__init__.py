"""Nerlex public Python API."""

from nerlex.capture import CaptureRun, capture, observe_outcome
from nerlex.dataset import (
    DatasetExample,
    DatasetSnapshot,
    DatasetSplit,
    SnapshotConfig,
    SnapshotManifest,
    SplitArtifact,
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
    validate_request,
)
from nerlex.store import SQLiteTraceStore
from nerlex.trace import ResultObservation, TraceRecord, export_jsonl, load_jsonl

__all__ = [
    "Candidate",
    "CandidateMode",
    "CaptureRun",
    "DatasetExample",
    "DatasetSnapshot",
    "DatasetSplit",
    "DecisionKind",
    "DecisionRequest",
    "DecisionResult",
    "DecisionRoute",
    "DecisionSpec",
    "LabelObservation",
    "LabelSource",
    "ResultObservation",
    "SQLiteTraceStore",
    "SnapshotConfig",
    "SnapshotManifest",
    "SplitArtifact",
    "SplitConfig",
    "TraceRecord",
    "build_snapshot",
    "capture",
    "export_jsonl",
    "load_jsonl",
    "observe_outcome",
    "validate_request",
    "verify_snapshot",
    "write_snapshot",
]

__version__ = "0.1.0.dev0"
