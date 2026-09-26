from __future__ import annotations

from uuid import UUID

from nerlex.calibration import (
    TemperatureCalibrationArtifact,
    TemperatureCalibrationConfig,
    fit_temperature,
)
from nerlex.compiler import CompilerArtifact, MultinomialNBConfig, compile_snapshot
from nerlex.dataset import DatasetSnapshot, SnapshotConfig, SplitConfig, build_snapshot
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


def support_spec() -> DecisionSpec:
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


def support_trace(index: int) -> TraceRecord:
    request_id = UUID(int=index + 1)
    billing = index % 2 == 0
    state = (
        f"invoice refund payment card billing-marker-{index}"
        if billing
        else f"crash error bug software technical-marker-{index}"
    )
    return TraceRecord(
        request=DecisionRequest(
            request_id=request_id,
            decision_id="support-routing",
            spec_version="1",
            state=state,
        ),
        labels=(
            LabelObservation(
                observation_id=UUID(int=20_000 + index),
                request_id=request_id,
                source=LabelSource.OUTCOME,
                value="billing" if billing else "technical",
                source_id="fixture",
            ),
        ),
    )


def support_snapshot(
    *,
    seed: str,
    start_index: int = 0,
    count: int = 240,
    config: SnapshotConfig | None = None,
) -> DatasetSnapshot:
    resolved_config = config or SnapshotConfig(
        split=SplitConfig(
            train=0.60,
            calibration=0.20,
            test=0.20,
            seed=seed,
        )
    )
    return build_snapshot(
        support_spec(),
        [support_trace(index) for index in range(start_index, start_index + count)],
        config=resolved_config,
    )


def support_compiler_calibration(
    *,
    seed: str,
) -> tuple[DatasetSnapshot, CompilerArtifact, TemperatureCalibrationArtifact]:
    snapshot = support_snapshot(seed=seed)
    compiler = compile_snapshot(snapshot, MultinomialNBConfig())
    calibration = fit_temperature(
        compiler,
        snapshot,
        config=TemperatureCalibrationConfig(iterations=64),
    )
    return snapshot, compiler, calibration
