from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import Field

from nerlex.hashing import canonical_json
from nerlex.spec import DecisionRequest, DecisionResult, LabelObservation, StrictModel


class ResultObservation(StrictModel):
    kind: str = Field(min_length=1, max_length=128)
    result: DecisionResult


class TraceRecord(StrictModel):
    request: DecisionRequest
    observations: tuple[ResultObservation, ...] = ()
    labels: tuple[LabelObservation, ...] = ()


def export_jsonl(records: Iterator[TraceRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record))
            handle.write("\n")

    return destination


def load_jsonl(path: str | Path) -> Iterator[TraceRecord]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield TraceRecord.model_validate_json(stripped)
            except ValueError as exc:
                raise ValueError(f"Invalid trace JSONL at line {line_number}.") from exc
