"""Nerlex public Python API."""

from nerlex.capture import CaptureRun, capture, observe_outcome
from nerlex.compiler import (
    CompilerArtifact,
    CompilerConfig,
    CompilerKind,
    CompilerPrediction,
    compile_snapshot,
    load_compiler_artifact,
    predict,
    write_compiler_artifact,
)
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
    "CompilerArtifact",
    "CompilerConfig",
    "CompilerKind",
    "CompilerPrediction",
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
    "compile_snapshot",
    "export_jsonl",
    "load_compiler_artifact",
    "load_jsonl",
    "observe_outcome",
    "predict",
    "validate_request",
    "verify_snapshot",
    "write_compiler_artifact",
    "write_snapshot",
]

__version__ = "0.1.0.dev0"
