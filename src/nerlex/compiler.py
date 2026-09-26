from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from nerlex.artifact_io import write_immutable_bytes
from nerlex.dataset import (
    DatasetExample,
    DatasetSplit,
    SnapshotManifest,
    verify_snapshot,
)
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.spec import CandidateMode, DecisionKind, DecisionSpec, StrictModel


COMPILER_VERSION = "1"
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


class CompilerKind(StrEnum):
    MULTINOMIAL_NB = "multinomial_nb"
    CENTROID_COSINE = "centroid_cosine"


class CompilerConfig(StrictModel):
    kind: CompilerKind
    alpha: float = Field(default=1.0, gt=0.0, le=100.0)
    temperature: float = Field(default=1.0, gt=0.0, le=100.0)


class CompilerPrediction(StrictModel):
    artifact_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    compiler_kind: CompilerKind
    selected: str | bool
    scores: dict[str, float]
    probabilities: dict[str, float]
    calibrated: Literal[False] = False


class _CompilerIdentity(StrictModel):
    schema_version: Literal[1] = 1
    compiler_kind: CompilerKind
    compiler_version: str = COMPILER_VERSION
    source_snapshot_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision_spec: DecisionSpec
    spec_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    config: CompilerConfig
    labels: tuple[str, ...]
    payload: dict[str, Any]
    calibrated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_identity(self) -> _CompilerIdentity:
        if self.config.kind is not self.compiler_kind:
            raise ValueError("Compiler config kind does not match compiler_kind.")
        if self.spec_hash != sha256_hex(self.decision_spec):
            raise ValueError("Compiler spec_hash does not match decision_spec.")
        expected_labels = _spec_label_keys(self.decision_spec)
        if self.labels != expected_labels:
            raise ValueError("Compiler labels do not match the DecisionSpec.")
        if self.decision_spec.candidate_mode is CandidateMode.DYNAMIC:
            raise ValueError("Dynamic candidate decisions are not supported by v0.1 compilers.")
        return self


class CompilerArtifact(_CompilerIdentity):
    artifact_id: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _validate_artifact_id(self) -> CompilerArtifact:
        identity = _CompilerIdentity.model_validate(
            self.model_dump(mode="json", exclude={"artifact_id"})
        )
        if _compiler_identity_hash(identity) != self.artifact_id:
            raise ValueError("Compiler artifact identity does not match its content.")
        return self


def compile_snapshot(
    snapshot_path: str | Path,
    config: CompilerConfig,
) -> CompilerArtifact:
    """Train one deterministic baseline from the verified train split only."""
    directory = Path(snapshot_path)
    manifest = verify_snapshot(directory)
    spec = manifest.decision_spec
    _require_supported_spec(spec)

    train_artifact = manifest.splits[DatasetSplit.TRAIN]
    examples = _load_train_examples(directory, manifest)
    labels = _spec_label_keys(spec)
    if not examples:
        raise ValueError("The train split is empty; cannot compile a local model.")

    if config.kind is CompilerKind.MULTINOMIAL_NB:
        payload = _train_multinomial_nb(examples, labels, config.alpha)
    elif config.kind is CompilerKind.CENTROID_COSINE:
        payload = _train_centroid_cosine(examples, labels)
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError(f"Unsupported compiler kind: {config.kind}.")

    identity = _CompilerIdentity(
        compiler_kind=config.kind,
        source_snapshot_id=manifest.snapshot_id,
        train_sha256=train_artifact.sha256,
        decision_spec=spec,
        spec_hash=manifest.spec_hash,
        config=config,
        labels=labels,
        payload=payload,
    )
    return CompilerArtifact.model_validate(
        {
            **identity.model_dump(mode="json"),
            "artifact_id": _compiler_identity_hash(identity),
        }
    )


def predict(
    artifact: CompilerArtifact,
    state: dict[str, Any] | str,
) -> CompilerPrediction:
    """Run deterministic local inference from a validated compiler artifact."""
    text = _state_text(state)
    if artifact.compiler_kind is CompilerKind.MULTINOMIAL_NB:
        scores = _score_multinomial_nb(artifact, text)
    elif artifact.compiler_kind is CompilerKind.CENTROID_COSINE:
        scores = _score_centroid_cosine(artifact, text)
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError(f"Unsupported compiler kind: {artifact.compiler_kind}.")

    probabilities = _softmax(scores, artifact.config.temperature)
    selected_key = max(artifact.labels, key=lambda label: (probabilities[label], label))
    selected = _decode_label(artifact.decision_spec, selected_key)
    return CompilerPrediction(
        artifact_id=artifact.artifact_id,
        compiler_kind=artifact.compiler_kind,
        selected=selected,
        scores=scores,
        probabilities=probabilities,
    )


def write_compiler_artifact(
    artifact: CompilerArtifact,
    path: str | Path,
) -> Path:
    payload = (canonical_json(artifact) + "\n").encode("utf-8")
    return write_immutable_bytes(path, payload)


def load_compiler_artifact(path: str | Path) -> CompilerArtifact:
    source = Path(path)
    return CompilerArtifact.model_validate_json(source.read_text(encoding="utf-8"))


def _load_train_examples(
    directory: Path,
    manifest: SnapshotManifest,
) -> tuple[DatasetExample, ...]:
    artifact = manifest.splits[DatasetSplit.TRAIN]
    source = directory / artifact.filename
    examples: list[DatasetExample] = []
    with source.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                examples.append(DatasetExample.model_validate_json(stripped))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid train example at line {line_number}."
                ) from exc
    if len(examples) != artifact.count:
        raise ValueError("Train split count changed after snapshot verification.")
    return tuple(examples)


def _train_multinomial_nb(
    examples: tuple[DatasetExample, ...],
    labels: tuple[str, ...],
    alpha: float,
) -> dict[str, Any]:
    document_counts = Counter({label: 0 for label in labels})
    token_counts: dict[str, Counter[str]] = {
        label: Counter() for label in labels
    }
    vocabulary: set[str] = set()

    for example in examples:
        label = _encode_label(example.label)
        if label not in token_counts:
            raise ValueError(f"Unexpected training label: {label!r}.")
        tokens = _tokens(_state_text(example.state))
        document_counts[label] += 1
        token_counts[label].update(tokens)
        vocabulary.update(tokens)

    total_documents = sum(document_counts.values())
    if total_documents == 0:
        raise ValueError("The train split contains no usable examples.")

    ordered_vocabulary = tuple(sorted(vocabulary))
    payload = {
        "alpha": alpha,
        "total_documents": total_documents,
        "vocabulary": list(ordered_vocabulary),
        "document_counts": {
            label: document_counts[label] for label in labels
        },
        "total_token_counts": {
            label: sum(token_counts[label].values()) for label in labels
        },
        "token_counts": {
            label: {
                token: token_counts[label][token]
                for token in ordered_vocabulary
                if token_counts[label][token]
            }
            for label in labels
        },
    }
    return payload


def _score_multinomial_nb(
    artifact: CompilerArtifact,
    text: str,
) -> dict[str, float]:
    payload = artifact.payload
    alpha = float(payload["alpha"])
    total_documents = int(payload["total_documents"])
    vocabulary = set(str(token) for token in payload["vocabulary"])
    vocabulary_size = len(vocabulary) + 1
    document_counts = payload["document_counts"]
    total_token_counts = payload["total_token_counts"]
    token_counts = payload["token_counts"]
    input_counts = Counter(_tokens(text))

    scores: dict[str, float] = {}
    label_count = len(artifact.labels)
    for label in artifact.labels:
        documents_for_label = int(document_counts[label])
        prior = (documents_for_label + alpha) / (
            total_documents + alpha * label_count
        )
        score = math.log(prior)
        denominator = int(total_token_counts[label]) + alpha * vocabulary_size
        label_tokens = token_counts[label]
        for token, count in input_counts.items():
            observed = int(label_tokens.get(token, 0)) if token in vocabulary else 0
            score += count * math.log((observed + alpha) / denominator)
        scores[label] = score
    return scores


def _train_centroid_cosine(
    examples: tuple[DatasetExample, ...],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    document_tokens = [Counter(_tokens(_state_text(example.state))) for example in examples]
    document_frequency: Counter[str] = Counter()
    for counts in document_tokens:
        document_frequency.update(counts.keys())

    document_count = len(examples)
    vocabulary = tuple(sorted(document_frequency))
    idf = {
        token: math.log((1.0 + document_count) / (1.0 + document_frequency[token])) + 1.0
        for token in vocabulary
    }

    sums: dict[str, defaultdict[str, float]] = {
        label: defaultdict(float) for label in labels
    }
    class_counts = Counter({label: 0 for label in labels})

    for example, counts in zip(examples, document_tokens, strict=True):
        label = _encode_label(example.label)
        if label not in sums:
            raise ValueError(f"Unexpected training label: {label!r}.")
        vector = _normalized_tfidf(counts, idf)
        class_counts[label] += 1
        for token, value in vector.items():
            sums[label][token] += value

    centroids: dict[str, dict[str, float]] = {}
    for label in labels:
        if class_counts[label] == 0:
            centroids[label] = {}
            continue
        mean = {
            token: value / class_counts[label]
            for token, value in sums[label].items()
        }
        centroids[label] = _normalize_vector(mean)

    return {
        "document_count": document_count,
        "idf": {token: idf[token] for token in vocabulary},
        "class_counts": {label: class_counts[label] for label in labels},
        "centroids": centroids,
    }


def _score_centroid_cosine(
    artifact: CompilerArtifact,
    text: str,
) -> dict[str, float]:
    idf = {
        str(token): float(value)
        for token, value in artifact.payload["idf"].items()
    }
    vector = _normalized_tfidf(Counter(_tokens(text)), idf)
    scores: dict[str, float] = {}
    for label in artifact.labels:
        centroid = {
            str(token): float(value)
            for token, value in artifact.payload["centroids"][label].items()
        }
        scores[label] = sum(
            value * centroid.get(token, 0.0)
            for token, value in vector.items()
        )
    return scores


def _normalized_tfidf(
    counts: Counter[str],
    idf: dict[str, float],
) -> dict[str, float]:
    weighted = {
        token: float(count) * idf[token]
        for token, count in counts.items()
        if token in idf
    }
    return _normalize_vector(weighted)


def _normalize_vector(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {
        token: vector[token] / norm
        for token in sorted(vector)
        if vector[token] != 0.0
    }


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        raise ValueError("Cannot normalize an empty score map.")
    scaled = {label: score / temperature for label, score in scores.items()}
    maximum = max(scaled.values())
    exponentials = {
        label: math.exp(value - maximum)
        for label, value in scaled.items()
    }
    total = sum(exponentials.values())
    return {
        label: exponentials[label] / total
        for label in sorted(exponentials)
    }


def _state_text(state: dict[str, Any] | str) -> str:
    return state if isinstance(state, str) else canonical_json(state)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.casefold()))


def _spec_label_keys(spec: DecisionSpec) -> tuple[str, ...]:
    if spec.kind is DecisionKind.BOOLEAN:
        return ("false", "true")
    if spec.candidate_mode is CandidateMode.DYNAMIC:
        raise ValueError("Dynamic candidate decisions are not supported by v0.1 compilers.")
    return tuple(sorted(candidate.key for candidate in spec.candidates))


def _require_supported_spec(spec: DecisionSpec) -> None:
    if spec.kind is DecisionKind.CHOICE and spec.candidate_mode is CandidateMode.DYNAMIC:
        raise ValueError("Dynamic candidate decisions are not supported by v0.1 compilers.")


def _encode_label(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _decode_label(spec: DecisionSpec, value: str) -> str | bool:
    if spec.kind is DecisionKind.BOOLEAN:
        if value == "true":
            return True
        if value == "false":
            return False
        raise ValueError(f"Invalid boolean artifact label: {value!r}.")
    return value


def _compiler_identity_hash(identity: _CompilerIdentity) -> str:
    return sha256_hex(identity.model_dump(mode="json"))
