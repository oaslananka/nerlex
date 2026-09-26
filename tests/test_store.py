from __future__ import annotations

from nerlex.spec import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionRequest,
    DecisionSpec,
    LabelObservation,
    LabelSource,
)
from nerlex.store import SQLiteTraceStore


def test_store_registers_and_replays_request_ids(tmp_path) -> None:
    store = SQLiteTraceStore(tmp_path / "nerlex.db")
    store.initialize()

    spec = DecisionSpec(
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
    request = DecisionRequest(
        decision_id=spec.decision_id,
        spec_version=spec.version,
        state={"text": "charged twice"},
    )

    spec_hash = store.register_spec(spec)
    assert len(spec_hash) == 64

    store.record_request(request)
    store.record_label(
        LabelObservation(
            request_id=request.request_id,
            source=LabelSource.OUTCOME,
            value="billing",
            source_id="resolved_queue",
        )
    )

    assert list(store.iter_request_ids("support-routing")) == [request.request_id]
