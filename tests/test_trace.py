from nerlex.capture import capture
from nerlex.spec import Candidate, CandidateMode, DecisionKind, DecisionSpec, LabelSource
from nerlex.store import SQLiteTraceStore
from nerlex.trace import export_jsonl, load_jsonl


def test_export_and_replay_are_deterministic(tmp_path) -> None:
    store = SQLiteTraceStore(tmp_path / "nerlex.db")
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

    run = capture(store, spec, state={"text": "charged twice"})
    run.observe(value="billing", source=LabelSource.HUMAN, source_id="reviewer")

    first = export_jsonl(store.iter_traces(spec.decision_id), tmp_path / "first.jsonl")
    second = export_jsonl(store.iter_traces(spec.decision_id), tmp_path / "second.jsonl")

    assert first.read_bytes() == second.read_bytes()

    replayed = list(load_jsonl(first))
    assert len(replayed) == 1
    assert replayed[0].request.request_id == run.request_id
    assert replayed[0].labels[0].source is LabelSource.HUMAN
