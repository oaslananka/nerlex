from __future__ import annotations

from uuid import UUID

from nerlex import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionRequest,
    DecisionSpec,
    EmpiricalRiskGateConfig,
    FallbackDecision,
    LabelObservation,
    LabelSource,
    LocalCascadeRuntime,
    MultinomialNBConfig,
    RuntimeBundle,
    SnapshotConfig,
    SplitConfig,
    TemperatureCalibrationConfig,
    TraceRecord,
    build_snapshot,
    compile_snapshot,
    fit_empirical_risk_gate,
    fit_temperature,
)


def trace(index: int) -> TraceRecord:
    billing = index % 2 == 0
    request_id = UUID(int=index + 1)
    return TraceRecord(
        request=DecisionRequest(
            request_id=request_id,
            decision_id="support-routing",
            spec_version="1",
            state=(
                f"invoice refund payment billing-{index}"
                if billing
                else f"crash bug software technical-{index}"
            ),
        ),
        labels=(
            LabelObservation(
                observation_id=UUID(int=10_000 + index),
                request_id=request_id,
                source=LabelSource.OUTCOME,
                value="billing" if billing else "technical",
                source_id="demo",
            ),
        ),
    )


spec = DecisionSpec(
    decision_id="support-routing",
    version="1",
    kind=DecisionKind.CHOICE,
    description="Route a support request.",
    candidate_mode=CandidateMode.STATIC,
    candidates=(
        Candidate(key="billing", description="Payments and refunds."),
        Candidate(key="technical", description="Product and software problems."),
    ),
)

snapshot = build_snapshot(
    spec,
    [trace(index) for index in range(120)],
    config=SnapshotConfig(
        split=SplitConfig(train=0.6, calibration=0.2, test=0.2, seed="demo")
    ),
)
compiler = compile_snapshot(snapshot, MultinomialNBConfig())
calibration = fit_temperature(
    compiler,
    snapshot,
    config=TemperatureCalibrationConfig(iterations=64),
)
gate = fit_empirical_risk_gate(
    compiler,
    calibration,
    snapshot,
    EmpiricalRiskGateConfig(max_empirical_risk=0.0, min_accepted=1),
)


def fallback(_request: DecisionRequest) -> FallbackDecision:
    return FallbackDecision(
        selected="technical",
        probabilities={"billing": 0.25, "technical": 0.75},
        backend="example-fallback",
    )


runtime = LocalCascadeRuntime(
    RuntimeBundle(compiler=compiler, calibration=calibration, gate=gate),
    fallback=fallback,
)

result = runtime.decide(
    DecisionRequest(
        decision_id="support-routing",
        spec_version="1",
        state="refund invoice payment",
    )
)
print(result.model_dump_json(indent=2))
