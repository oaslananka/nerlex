from __future__ import annotations

from copy import deepcopy
from typing import Any

REDACTED = "[REDACTED]"


def redact_state(
    state: dict[str, Any] | str,
    sensitive_fields: tuple[str, ...],
) -> dict[str, Any] | str:
    """Return a copy of state with configured dotted paths redacted.

    Dotted paths support mapping keys, numeric list indexes, and implicit traversal
    across list items. Malformed paths fail closed instead of silently changing meaning.
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
        _redact_parts(redacted, _parse_path(path))
    return redacted


def _parse_path(dotted_path: str) -> tuple[str, ...]:
    parts = tuple(dotted_path.split("."))
    if not parts or any(not part for part in parts):
        raise ValueError(f"Malformed sensitive field path: {dotted_path!r}.")
    return parts


def _redact_parts(current: Any, parts: tuple[str, ...]) -> None:
    if not parts:
        return

    if isinstance(current, dict):
        _redact_mapping(current, parts)
        return

    if isinstance(current, list):
        _redact_list(current, parts)


def _redact_mapping(current: dict[str, Any], parts: tuple[str, ...]) -> None:
    head, tail = _split_path(parts)
    if head not in current:
        return
    if not tail:
        current[head] = REDACTED
        return
    _redact_parts(current[head], tail)


def _redact_list(current: list[Any], parts: tuple[str, ...]) -> None:
    head, tail = _split_path(parts)
    if not head.isdigit():
        for item in current:
            _redact_parts(item, parts)
        return

    index = int(head)
    if index >= len(current):
        return
    if not tail:
        current[index] = REDACTED
        return
    _redact_parts(current[index], tail)


def _split_path(parts: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    return parts[0], parts[1:]
