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


def test_redacts_numeric_list_index() -> None:
    state = {
        "users": [
            {"email": "first@example.com"},
            {"email": "second@example.com"},
        ]
    }

    redacted = redact_state(state, ("users.1.email",))

    assert redacted["users"][0]["email"] == "first@example.com"
    assert redacted["users"][1]["email"] == REDACTED


def test_redacts_matching_field_across_list_items() -> None:
    state = {
        "users": [
            {"email": "first@example.com", "name": "First"},
            {"email": "second@example.com", "name": "Second"},
        ]
    }

    redacted = redact_state(state, ("users.email",))

    assert [user["email"] for user in redacted["users"]] == [REDACTED, REDACTED]
    assert [user["name"] for user in redacted["users"]] == ["First", "Second"]


def test_missing_sensitive_path_is_a_noop() -> None:
    state = {"customer": {"email": "person@example.com"}}

    redacted = redact_state(state, ("customer.phone",))

    assert redacted == state
    assert redacted is not state


@pytest.mark.parametrize("path", ("", ".email", "customer.", "customer..email"))
def test_malformed_sensitive_path_fails_closed(path: str) -> None:
    with pytest.raises(ValueError, match="Malformed sensitive field path"):
        redact_state({"customer": {"email": "person@example.com"}}, (path,))


def test_sensitive_fields_fail_closed_for_string_state() -> None:
    with pytest.raises(ValueError, match="structured state"):
        redact_state("token=secret", ("token",))
