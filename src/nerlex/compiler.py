from __future__ import annotations

import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from nerlex.dataset import DatasetExample, DatasetSnapshot, DatasetSplit
from nerlex.hashing import canonical_json, sha256_hex
from nerlex.spec import CandidateMode, DecisionKind, DecisionSpec, StrictModel

TOKENIZER_VERSION: Literal["unicode-word-v1"] = "unicode-word-v1"
COMPILER_ARTIFACT_SCHEMA_VERSION: Literal[1] = 1
COMPILER_IMPLEMENTATION_VERSION: Literal["1"] = "1"

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_SHA256_PATTERN = r"^[a-f0-9]{64}$"


class CompilerError(ValueError):
    """Raised when a snapshot cannot be compiled by a selected backend."""


class MultinomialNBConfig(StrictModel):
    kind: Literal["multinomial_nb"] = "multinomial_nb"
    alpha: float = Field(default=1.0, gt=0.0)
    tokenizer: Literal["unicode-word-v1"] = TOKENIZER_VERSION


class CentroidConfig(StrictModel):
    kind: Literal["centroid_cosine"] = "centroid_cosine"
    tokenizer: Literal["unicode-word-v1"] = TOKENIZER_VERSION


CompilerConfig = MultinomialNBConfig | CentroidConfig


class MultinomialNBModel(StrictModel):
    kind: Literal["multinomial_nb"] = "multinomial_nb"
    classes: tuple[str, ...]
    vocabulary: tuple[str, ...]
    class_log_prior: dict[str, float]
    feature_log_prob: dict[str, dict[str, float]]


class CentroidModel(StrictModel):
    kind: Literal["centroid_cosine"] = "centroid_cosine"
    classes: tuple[str, ...]
    idf: dict[str, float]
    centroids: dict[str, dict[str, float]]


CompilerModel = MultinomialNBModel | CentroidModel


class _CompilerArtifactIdentity(StrictModel):
    schema_version: Literal[1] = COMPILER_ARTIFACT_SCHEMA_VERSION
    compiler_version: Literal["1"] = COMPILER_IMPLEMENTATION_VERSION
    decision_spec: DecisionSpec
    spec_hash: str = Field(pattern=_SHA256_PATTERN)
    snapshot_id: str = Field(pattern=_SHA256_PATTERN)
    train_sha256: str = Field(pattern=_SHA256_PATTERN)
    train_count: int = Field(ge=1)
    config: CompilerConfig
    model: CompilerModel
    calibrated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_identity(self) -> _CompilerArtifactIdentity:
        _validate_artifact_spec(self)
        expected_classes = _expected_classes(self.decision_spec)
        if self.model.classes != expected_classes:
            raise ValueError("Compiler model classes do not match the DecisionSpec candidates.")
        if isinstance(self.model, MultinomialNBModel):
            _validate_multinomial_model(self.model, expected_classes)
        else:
            _validate_centroid_model(self.model, expected_classes)
        return self



def _expected_classes(spec: DecisionSpec) -> tuple[str, ...]:
    return tuple(sorted(candidate.key for candidate in spec.candidates))


def _validate_artifact_spec(identity: _CompilerArtifactIdentity) -> None:
    if identity.spec_hash != sha256_hex(identity.decision_spec):
        raise ValueError("spec_hash does not match decision_spec.")
    if identity.config.kind != identity.model.kind:
        raise ValueError("Compiler config/model kinds do not match.")
    if identity.decision_spec.kind is not DecisionKind.CHOICE:
        raise ValueError("v0.1 compiler artifacts support choice decisions only.")
    if identity.decision_spec.candidate_mode is not CandidateMode.STATIC:
        raise ValueError("v0.1 compiler artifacts support static candidates only.")


def _validate_multinomial_model(
    model: MultinomialNBModel,
    expected_classes: tuple[str, ...],
) -> None:
    if not model.vocabulary:
        raise ValueError("Multinomial NB vocabulary must not be empty.")
    if set(model.class_log_prior) != set(expected_classes):
        raise ValueError("Multinomial NB class priors are incomplete.")
    if set(model.feature_log_prob) != set(expected_classes):
        raise ValueError("Multinomial NB feature probabilities are incomplete.")

    expected_vocabulary = set(model.vocabulary)
    for values in model.feature_log_prob.values():
        if set(values) != expected_vocabulary:
            raise ValueError("Multinomial NB feature probabilities do not match vocabulary.")


def _validate_centroid_model(
    model: CentroidModel,
    expected_classes: tuple[str, ...],
) -> None:
    if set(model.centroids) != set(expected_classes):
        raise ValueError("Centroid model class vectors are incomplete.")
    if any(
        token not in model.idf
        for vector in model.centroids.values()
        for token in vector
    ):
        raise ValueError("Centroid contains a token missing from the IDF table.")


class CompilerArtifact(_CompilerArtifactIdentity):
    artifact_id: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_artifact_id(self) -> CompilerArtifact:
        identity = _CompilerArtifactIdentity.model_validate(
            self.model_dump(mode="json", exclude={"artifact_id"})
        )
        if sha256_hex(identity) != self.artifact_id:
            raise ValueError("Compiler artifact identity does not match its content.")
        return self


class CompilerPrediction(StrictModel):
    artifact_id: str = Field(pattern=_SHA256_PATTERN)
    backend: Literal["multinomial_nb", "centroid_cosine"]
    selected: str
    probabilities: dict[str, float]
    calibrated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_probabilities(self) -> CompilerPrediction:
        if not self.probabilities:
            raise ValueError("Compiler prediction probabilities must not be empty.")
        if self.selected not in self.probabilities:
            raise ValueError("Selected class must exist in the probability map.")
        if any(value < 0.0 or value > 1.0 for value in self.probabilities.values()):
            raise ValueError("Compiler prediction probabilities must be in [0, 1].")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-9:
            raise ValueError("Compiler prediction probabilities must sum to 1.")
        return self


class CompilerBackend(Protocol):
    config: CompilerConfig

    def compile(self, snapshot: DatasetSnapshot) -> CompilerArtifact: ...


class MultinomialNBCompiler:
    def __init__(self, config: MultinomialNBConfig | None = None) -> None:
        self.config = config or MultinomialNBConfig()

    def compile(self, snapshot: DatasetSnapshot) -> CompilerArtifact:
        spec, train_examples = _training_input(snapshot)
        classes = tuple(sorted(candidate.key for candidate in spec.candidates))
        alpha = self.config.alpha

        class_document_count: Counter[str] = Counter()
        class_token_count: dict[str, Counter[str]] = {
            class_name: Counter() for class_name in classes
        }
        vocabulary: set[str] = set()

        for example in train_examples:
            label = _choice_label(example.label)
            tokens = _tokenize_state(example.state)
            class_document_count[label] += 1
            class_token_count[label].update(tokens)
            vocabulary.update(tokens)

        if not vocabulary:
            raise CompilerError("Training split contains no tokenizable text.")

        ordered_vocabulary = tuple(sorted(vocabulary))
        total_documents = len(train_examples)
        class_count = len(classes)

        class_log_prior = {
            class_name: math.log(
                (class_document_count[class_name] + alpha)
                / (total_documents + (alpha * class_count))
            )
            for class_name in classes
        }

        feature_log_prob: dict[str, dict[str, float]] = {}
        vocabulary_size = len(ordered_vocabulary)
        for class_name in classes:
            counts = class_token_count[class_name]
            denominator = sum(counts.values()) + (alpha * vocabulary_size)
            feature_log_prob[class_name] = {
                token: math.log((counts[token] + alpha) / denominator)
                for token in ordered_vocabulary
            }

        model = MultinomialNBModel(
            classes=classes,
            vocabulary=ordered_vocabulary,
            class_log_prior=class_log_prior,
            feature_log_prob=feature_log_prob,
        )
        return _artifact(snapshot, self.config, model)


class CentroidCompiler:
    def __init__(self, config: CentroidConfig | None = None) -> None:
        self.config = config or CentroidConfig()

    def compile(self, snapshot: DatasetSnapshot) -> CompilerArtifact:
        spec, train_examples = _training_input(snapshot)
        classes = tuple(sorted(candidate.key for candidate in spec.candidates))

        tokenized = [
            (_choice_label(example.label), _tokenize_state(example.state))
            for example in train_examples
        ]
        if not any(tokens for _, tokens in tokenized):
            raise CompilerError("Training split contains no tokenizable text.")

        document_frequency: Counter[str] = Counter()
        for _, tokens in tokenized:
            document_frequency.update(set(tokens))

        document_count = len(tokenized)
        idf = {
            token: math.log((1.0 + document_count) / (1.0 + frequency)) + 1.0
            for token, frequency in sorted(document_frequency.items())
        }

        sums: dict[str, defaultdict[str, float]] = {
            class_name: defaultdict(float) for class_name in classes
        }
        class_counts: Counter[str] = Counter()

        for label, tokens in tokenized:
            vector = _normalized_tfidf(tokens, idf)
            class_counts[label] += 1
            for token in sorted(vector):
                sums[label][token] += vector[token]

        centroids: dict[str, dict[str, float]] = {}
        for class_name in classes:
            count = class_counts[class_name]
            if count == 0:
                centroids[class_name] = {}
                continue
            mean = {
                token: value / count
                for token, value in sorted(sums[class_name].items())
            }
            centroids[class_name] = _normalize_sparse(mean)

        model = CentroidModel(classes=classes, idf=idf, centroids=centroids)
        return _artifact(snapshot, self.config, model)


def compile_snapshot(
    snapshot: DatasetSnapshot,
    config: CompilerConfig,
) -> CompilerArtifact:
    if isinstance(config, MultinomialNBConfig):
        return MultinomialNBCompiler(config).compile(snapshot)
    if isinstance(config, CentroidConfig):
        return CentroidCompiler(config).compile(snapshot)
    raise CompilerError(f"Unsupported compiler configuration: {type(config).__name__}.")


def predict(
    artifact: CompilerArtifact,
    state: Mapping[str, object] | str,
) -> CompilerPrediction:
    tokens = _tokenize_state(state)

    if isinstance(artifact.model, MultinomialNBModel):
        logits = _multinomial_nb_logits(artifact.model, tokens)
    else:
        logits = _centroid_logits(artifact.model, tokens)

    probabilities = _softmax(logits)
    selected = max(sorted(probabilities), key=lambda class_name: probabilities[class_name])
    return CompilerPrediction(
        artifact_id=artifact.artifact_id,
        backend=artifact.config.kind,
        selected=selected,
        probabilities=probabilities,
    )


def write_compiler_artifact(
    artifact: CompilerArtifact,
    root: str | Path,
) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    destination = root_path / f"{artifact.artifact_id}.json"
    payload = (canonical_json(artifact) + "\n").encode("utf-8")

    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("Refusing to overwrite a compiler artifact with different content.")
        return destination

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=root_path,
        prefix=f".{artifact.artifact_id}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_compiler_artifact(path: str | Path) -> CompilerArtifact:
    source = Path(path)
    return CompilerArtifact.model_validate_json(source.read_text(encoding="utf-8"))


def _training_input(snapshot: DatasetSnapshot) -> tuple[DecisionSpec, tuple[DatasetExample, ...]]:
    spec = snapshot.manifest.decision_spec
    if spec.kind is not DecisionKind.CHOICE:
        raise CompilerError("v0.1 local compilers support choice decisions only.")
    if spec.candidate_mode is not CandidateMode.STATIC:
        raise CompilerError(
            "Dynamic candidate decisions require a semantic candidate scorer, "
            "which is not part of the v0.1 fixed-class compiler baselines."
        )

    train_examples = snapshot.examples[DatasetSplit.TRAIN]
    if not train_examples:
        raise CompilerError("Training split is empty.")
    return spec, train_examples


def _artifact(
    snapshot: DatasetSnapshot,
    config: CompilerConfig,
    model: CompilerModel,
) -> CompilerArtifact:
    train_artifact = snapshot.manifest.splits[DatasetSplit.TRAIN]
    identity = _CompilerArtifactIdentity(
        decision_spec=snapshot.manifest.decision_spec,
        spec_hash=snapshot.manifest.spec_hash,
        snapshot_id=snapshot.manifest.snapshot_id,
        train_sha256=train_artifact.sha256,
        train_count=train_artifact.count,
        config=config,
        model=model,
    )
    return CompilerArtifact(
        artifact_id=sha256_hex(identity),
        **identity.model_dump(mode="python"),
    )


def _choice_label(value: str | bool) -> str:
    if not isinstance(value, str):
        raise CompilerError("Static choice compiler received a non-string label.")
    return value


def _tokenize_state(state: Mapping[str, object] | str) -> tuple[str, ...]:
    text = state if isinstance(state, str) else canonical_json(dict(state))
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def _multinomial_nb_logits(
    model: MultinomialNBModel,
    tokens: Sequence[str],
) -> dict[str, float]:
    token_counts = Counter(tokens)

    logits: dict[str, float] = {}
    for class_name in model.classes:
        score = model.class_log_prior[class_name]
        values = model.feature_log_prob[class_name]
        for token in sorted(token_counts):
            token_log_probability = values.get(token)
            if token_log_probability is not None:
                score += token_counts[token] * token_log_probability
        logits[class_name] = score
    return logits


def _centroid_logits(
    model: CentroidModel,
    tokens: Sequence[str],
) -> dict[str, float]:
    query = _normalized_tfidf(tokens, model.idf)
    return {
        class_name: _dot_sparse(query, model.centroids[class_name])
        for class_name in model.classes
    }


def _normalized_tfidf(
    tokens: Sequence[str],
    idf: Mapping[str, float],
) -> dict[str, float]:
    counts = Counter(token for token in tokens if token in idf)
    weighted = {
        token: float(count) * idf[token]
        for token, count in sorted(counts.items())
    }
    return _normalize_sparse(weighted)


def _normalize_sparse(vector: Mapping[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if math.isclose(norm, 0.0, abs_tol=1e-15):
        return {}
    return {
        token: value / norm
        for token, value in sorted(vector.items())
    }


def _dot_sparse(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _softmax(logits: Mapping[str, float]) -> dict[str, float]:
    if not logits:
        raise CompilerError("Cannot normalize an empty score map.")
    maximum = max(logits.values())
    exp_values = {
        class_name: math.exp(logits[class_name] - maximum)
        for class_name in sorted(logits)
    }
    denominator = sum(exp_values.values())
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise CompilerError("Compiler produced non-normalizable scores.")
    return {
        class_name: exp_values[class_name] / denominator
        for class_name in sorted(exp_values)
    }
