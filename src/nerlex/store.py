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
        self._initialized = False
        self._registered_specs: dict[tuple[str, str], str] = {}

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        if self._initialized:
            return
        with self.connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        self._initialized = True

    def register_spec(self, spec: DecisionSpec) -> str:
        self.initialize()
        payload = canonical_json(spec)
        spec_hash = sha256_hex(spec)
        cache_key = (spec.decision_id, spec.version)
        cached_hash = self._registered_specs.get(cache_key)
        if cached_hash is not None:
            if cached_hash != spec_hash:
                raise ValueError(
                    "A different DecisionSpec already exists for this decision/version."
                )
            return cached_hash
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
                self._registered_specs[cache_key] = spec_hash
                return spec_hash

            connection.execute(
                """
                INSERT INTO decision_specs(
                    decision_id, spec_version, spec_hash, spec_json
                ) VALUES (?, ?, ?, ?)
                """,
                (spec.decision_id, spec.version, spec_hash, payload),
            )
        self._registered_specs[cache_key] = spec_hash
        return spec_hash

    def record_request(self, request: DecisionRequest) -> None:
        self.initialize()
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
        self.initialize()
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
        self.initialize()
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
        self.initialize()
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
        """Stream complete traces with one ordered JOIN query and bounded memory."""
        self.initialize()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    r.request_id,
                    r.request_json,
                    o.id AS observation_id,
                    o.observation_kind,
                    o.result_json,
                    l.observation_id AS label_observation_id,
                    l.label_json
                FROM decision_requests AS r
                LEFT JOIN decision_observations AS o
                    ON o.request_id = r.request_id
                LEFT JOIN labels AS l
                    ON l.request_id = r.request_id
                WHERE r.decision_id = ?
                ORDER BY
                    r.event_time,
                    r.request_id,
                    o.id,
                    l.observed_at,
                    l.observation_id
                """,
                (decision_id,),
            )

            current_id: str | None = None
            current_request: DecisionRequest | None = None
            observations: list[ResultObservation] = []
            labels: list[LabelObservation] = []
            seen_observations: set[int] = set()
            seen_labels: set[str] = set()

            for row in rows:
                row_request_id = row["request_id"]
                if current_id is not None and row_request_id != current_id:
                    if current_request is None:
                        raise RuntimeError("Trace stream lost its current request.")
                    yield TraceRecord(
                        request=current_request,
                        observations=tuple(observations),
                        labels=tuple(labels),
                    )
                    observations = []
                    labels = []
                    seen_observations = set()
                    seen_labels = set()

                if current_id != row_request_id:
                    current_id = row_request_id
                    current_request = DecisionRequest.model_validate_json(
                        row["request_json"]
                    )

                observation_id = row["observation_id"]
                if observation_id is not None and observation_id not in seen_observations:
                    seen_observations.add(observation_id)
                    observations.append(
                        ResultObservation(
                            kind=row["observation_kind"],
                            result=DecisionResult.model_validate_json(row["result_json"]),
                        )
                    )

                label_id = row["label_observation_id"]
                if label_id is not None and label_id not in seen_labels:
                    seen_labels.add(label_id)
                    labels.append(LabelObservation.model_validate_json(row["label_json"]))

            if current_request is not None:
                yield TraceRecord(
                    request=current_request,
                    observations=tuple(observations),
                    labels=tuple(labels),
                )
