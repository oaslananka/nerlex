from __future__ import annotations

import pytest

from nerlex.spec import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionRequest,
    DecisionSpec,
    validate_request,
)


def test_static_choice_spec_and_request() -> None:
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
    request = DecisionRequest(
        decision_id="support-routing",
        spec_version="1",
        state={"text": "I was charged twice."},
    )

    assert validate_request(spec, request) is request


def test_dynamic_choice_requires_request_candidates() -> None:
    spec = DecisionSpec(
        decision_id="tool-routing",
        version="1",
        kind=DecisionKind.CHOICE,
        description="Select the next tool.",
        candidate_mode=CandidateMode.DYNAMIC,
    )
    request = DecisionRequest(
        decision_id="tool-routing",
        spec_version="1",
        state="Deploy the application.",
    )

    with pytest.raises(ValueError, match="at least two"):
        validate_request(spec, request)


def test_boolean_decision_rejects_choice_candidates() -> None:
    with pytest.raises(ValueError, match="Boolean decisions"):
        DecisionSpec(
            decision_id="approval",
            version="1",
            kind=DecisionKind.BOOLEAN,
            description="Decide whether this case is eligible.",
            candidate_mode=CandidateMode.STATIC,
            candidates=(
                Candidate(key="yes", description="Eligible."),
                Candidate(key="no", description="Not eligible."),
            ),
        )
