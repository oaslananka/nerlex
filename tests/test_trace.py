from datetime import UTC, datetime, timedelta
from uuid import UUID

from nerlex.capture import capture
from nerlex.spec import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionRequest,
    DecisionResult,
    DecisionRoute,
    DecisionSpec,
    LabelSource,
)
from nerlex.store import SQLiteTraceStore
from nerlex.trace import export_jsonl, load_jsonl


def _spec() -> DecisionSpec:
    return DecisionSpec(
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


def test_export_and_replay_are_deterministic(tmp_path) -> None:
    store = SQLiteTraceStore(tmp_path / "nerlex.db")
    spec = _spec()

    run = capture(store, spec, state={"text": "charged twice"})
    run.observe(value="billing", source=LabelSource.HUMAN, source_id="reviewer")

    first = export_jsonl(store.iter_traces(spec.decision_id), tmp_path / "first.jsonl")
    second = export_jsonl(store.iter_traces(spec.decision_id), tmp_path / "second.jsonl")

    assert first.read_bytes() == second.read_bytes()

    replayed = list(load_jsonl(first))
    assert len(replayed) == 1
    assert replayed[0].request.request_id == run.request_id
    assert replayed[0].labels[0].source is LabelSource.HUMAN


def test_join_replay_deduplicates_observations_and_labels(tmp_path) -> None:
    store = SQLiteTraceStore(tmp_path / "nerlex.db")
    spec = _spec()
    run = capture(store, spec, state={"text": "charged twice"})

    for backend, selected in (("teacher-a", "billing"), ("teacher-b", "technical")):
        store.record_result(
            DecisionResult(
                request_id=run.request_id,
                selected=selected,
                probabilities={"billing": 0.5, "technical": 0.5},
                confidence=0.5,
                backend=backend,
                latency_ms=1.0,
                route=DecisionRoute.TEACHER,
            ),
            kind="teacher",
        )

    run.observe(value="billing", source=LabelSource.HUMAN, source_id="reviewer")
    run.observe(value="billing", source=LabelSource.OUTCOME, source_id="resolved_queue")

    trace = list(store.iter_traces(spec.decision_id))[0]

    assert [item.result.backend for item in trace.observations] == ["teacher-a", "teacher-b"]
    assert [item.source for item in trace.labels] == [LabelSource.HUMAN, LabelSource.OUTCOME]


def test_join_replay_flushes_request_boundaries_in_event_order(tmp_path) -> None:
    store = SQLiteTraceStore(tmp_path / "nerlex.db")
    spec = _spec()
    store.register_spec(spec)
    started = datetime(2026, 1, 1, tzinfo=UTC)

    first = DecisionRequest(
        request_id=UUID(int=101),
        decision_id=spec.decision_id,
        spec_version=spec.version,
        event_time=started,
        state={"text": "first"},
    )
    second = DecisionRequest(
        request_id=UUID(int=102),
        decision_id=spec.decision_id,
        spec_version=spec.version,
        event_time=started + timedelta(seconds=1),
        state={"text": "second"},
    )

    for request, value in ((second, "technical"), (first, "billing")):
        store.record_request(request)
        store.record_label(
            LabelObservation(
                request_id=request.request_id,
                source=LabelSource.OUTCOME,
                value=value,
                source_id="boundary-test",
                observed_at=request.event_time,
            )
        )

    traces = list(store.iter_traces(spec.decision_id))

    assert [trace.request.request_id for trace in traces] == [
        first.request_id,
        second.request_id,
    ]
    assert [trace.labels[0].value for trace in traces] == ["billing", "technical"]
