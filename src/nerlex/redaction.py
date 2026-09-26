from __future__ import annotations

from copy import deepcopy
from typing import Any

REDACTED = "[REDACTED]"


def redact_state(
    state: dict[str, Any] | str,
    sensitive_fields: tuple[str, ...],
) -> dict[str, Any] | str:
    """Return a copy of state with configured dotted paths redacted.

    Sensitive paths are relative to the structured decision state. If sensitive
    fields are configured for an unstructured string state, capture fails closed
    because field-level redaction is impossible.
    """
    if not sensitive_fields:
        return deepcopy(state)

    if isinstance(state, str):
        raise ValueError(
            "DecisionSpec.sensitive_fields requires structured state; "
            "field-level redaction cannot be applied to a string state."
        )

    redacted = deepcopy(state)
    for path in sensitive_fields:
        _redact_path(redacted, path)
    return redacted


def _redact_path(value: dict[str, Any], dotted_path: str) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("Sensitive field paths must not be empty.")

    current: Any = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]

    if isinstance(current, dict) and parts[-1] in current:
        current[parts[-1]] = REDACTED
