from nerlex.capture import capture, observe_outcome
from nerlex.redaction import REDACTED
from nerlex.spec import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionResult,
    DecisionRoute,
    DecisionSpec,
)
from nerlex.store import SQLiteTraceStore


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
        sensitive_fields=("customer.email",),
    )


def test_capture_teacher_and_outcome(tmp_path) -> None:
    store = SQLiteTraceStore(tmp_path / "nerlex.db")

    raw_state = {
        "text": "I was charged twice.",
        "customer": {"email": "person@example.com"},
    }
    with capture(store, _spec(), state=raw_state) as run:
        assert isinstance(run.request.state, dict)
        assert run.request.state["customer"]["email"] == REDACTED
        assert raw_state["customer"]["email"] == "person@example.com"

        run.teacher(
            DecisionResult(
                request_id=run.request_id,
                selected="billing",
                probabilities={"billing": 0.9, "technical": 0.1},
                confidence=0.9,
                calibrated=False,
                backend="teacher-test",
                latency_ms=5.0,
                route=DecisionRoute.TEACHER,
            )
        )

    observe_outcome(
        store,
        request_id=run.request_id,
        value="billing",
        source_id="resolved_queue",
    )

    records = list(store.iter_traces("support-routing"))
    assert len(records) == 1
    assert records[0].request.request_id == run.request_id
    assert records[0].observations[0].kind == "teacher"
    assert records[0].labels[0].value == "billing"
