import pytest

from nerlex.redaction import REDACTED, redact_state


def test_redacts_nested_fields_without_mutating_input() -> None:
    state = {
        "customer": {"email": "person@example.com", "tier": "pro"},
        "text": "charged twice",
    }

    redacted = redact_state(state, ("customer.email",))

    assert redacted["customer"]["email"] == REDACTED
    assert redacted["customer"]["tier"] == "pro"
    assert state["customer"]["email"] == "person@example.com"


def test_sensitive_fields_fail_closed_for_string_state() -> None:
    with pytest.raises(ValueError, match="structured state"):
        redact_state("token=secret", ("token",))
