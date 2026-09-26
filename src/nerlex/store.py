from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

from nerlex.hashing import canonical_json, sha256_hex
from nerlex.spec import DecisionRequest, DecisionResult, DecisionSpec, LabelObservation
from nerlex.trace import ResultObservation, TraceRecord

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decision_specs (
    decision_id TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (decision_id, spec_version)
);

CREATE TABLE IF NOT EXISTS decision_requests (
    request_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    request_json TEXT NOT NULL,
    event_time TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (decision_id, spec_version)
        REFERENCES decision_specs(decision_id, spec_version)
);

CREATE TABLE IF NOT EXISTS decision_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    observation_kind TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES decision_requests(request_id)
);

CREATE TABLE IF NOT EXISTS labels (
    observation_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    source TEXT NOT NULL,
    label_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES decision_requests(request_id)
);

CREATE INDEX IF NOT EXISTS idx_requests_decision
    ON decision_requests(decision_id, spec_version);

CREATE INDEX IF NOT EXISTS idx_observations_request
    ON decision_observations(request_id);

CREATE INDEX IF NOT EXISTS idx_labels_request
    ON labels(request_id);
"""


class SQLiteTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

    def register_spec(self, spec: DecisionSpec) -> str:
        payload = canonical_json(spec)
        spec_hash = sha256_hex(spec)
        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT spec_hash FROM decision_specs
                WHERE decision_id = ? AND spec_version = ?
                """,
                (spec.decision_id, spec.version),
            ).fetchone()
            if existing is not None:
                if existing["spec_hash"] != spec_hash:
                    raise ValueError(
                        "A different DecisionSpec already exists for this decision/version."
                    )
                return spec_hash

            connection.execute(
                """
                INSERT INTO decision_specs(
                    decision_id, spec_version, spec_hash, spec_json
                ) VALUES (?, ?, ?, ?)
                """,
                (spec.decision_id, spec.version, spec_hash, payload),
            )
        return spec_hash

    def record_request(self, request: DecisionRequest) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO decision_requests(
                    request_id, decision_id, spec_version, request_json, event_time
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(request.request_id),
                    request.decision_id,
                    request.spec_version,
                    canonical_json(request),
                    request.event_time.isoformat(),
                ),
            )

    def record_result(self, result: DecisionResult, *, kind: str = "teacher") -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO decision_observations(
                    request_id, observation_kind, result_json
                ) VALUES (?, ?, ?)
                """,
                (str(result.request_id), kind, canonical_json(result)),
            )

    def record_label(self, label: LabelObservation) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO labels(
                    observation_id, request_id, source, label_json, observed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(label.observation_id),
                    str(label.request_id),
                    label.source.value,
                    canonical_json(label),
                    label.observed_at.isoformat(),
                ),
            )

    def iter_request_ids(self, decision_id: str) -> Iterator[UUID]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT request_id FROM decision_requests
                WHERE decision_id = ?
                ORDER BY event_time, request_id
                """,
                (decision_id,),
            ).fetchall()
        for row in rows:
            yield UUID(row["request_id"])

    def iter_traces(self, decision_id: str) -> Iterator[TraceRecord]:
        with self.connect() as connection:
            request_rows = connection.execute(
                """
                SELECT request_json FROM decision_requests
                WHERE decision_id = ?
                ORDER BY event_time, request_id
                """,
                (decision_id,),
            ).fetchall()

            for request_row in request_rows:
                request = DecisionRequest.model_validate_json(request_row["request_json"])

                observation_rows = connection.execute(
                    """
                    SELECT observation_kind, result_json
                    FROM decision_observations
                    WHERE request_id = ?
                    ORDER BY id
                    """,
                    (str(request.request_id),),
                ).fetchall()
                observations = tuple(
                    ResultObservation(
                        kind=row["observation_kind"],
                        result=DecisionResult.model_validate_json(row["result_json"]),
                    )
                    for row in observation_rows
                )

                label_rows = connection.execute(
                    """
                    SELECT label_json FROM labels
                    WHERE request_id = ?
                    ORDER BY observed_at, observation_id
                    """,
                    (str(request.request_id),),
                ).fetchall()
                labels = tuple(
                    LabelObservation.model_validate_json(row["label_json"])
                    for row in label_rows
                )

                yield TraceRecord(
                    request=request,
                    observations=observations,
                    labels=labels,
                )
