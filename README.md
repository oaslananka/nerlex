# Nerlex

**Compile reasoning into reflexes.**

Nerlex is an open-source **AI Decision Compiler** for repetitive typed decisions. It captures an existing decision path, builds a reproducible dataset, compiles smaller local candidates, calibrates uncertainty, evaluates risk-versus-coverage, and eventually runs proven local decisions behind explicit abstention and fallback gates.

> **Status:** early development. The current repository is establishing the v0.1 decision contracts, provenance model, trace store, and reproducibility foundations. No production-safety or benchmark claims are made yet.

## Why Nerlex?

General models are useful teachers for evolving or ambiguous decisions. But many production decisions become narrow and repetitive:

- which queue owns this ticket?
- which tool should handle this request?
- should this case be escalated?
- which workflow branch should run?

Training a classifier is only one part of the engineering problem. Nerlex is designed around the full lifecycle:

```text
DEFINE
  -> CAPTURE
  -> SNAPSHOT
  -> COMPILE
  -> CALIBRATE
  -> EVALUATE
  -> SHADOW
  -> GATE
  -> CANARY
  -> PROMOTE / ROLLBACK
```

The core question is not “can a small model make a prediction?” It is:

> **When is the local decision path good enough to act, and when should it defer?**

## Design principles

- **Teacher output is not automatically ground truth.**
- **Train, calibration, and final test data stay separate.**
- **Abstention and fallback are first-class behavior.**
- **Artifacts and evaluation evidence are reproducible and immutable.**
- **Backends are pluggable; Nerlex is not a Jev clone.**
- **Simple models are welcome when they win the evidence.**
- **No fabricated benchmark numbers.**

## v0.1 scope

The first milestone intentionally stays small:

- typed `choice` and `boolean` decision contracts
- label/outcome provenance
- local SQLite trace storage
- deterministic dataset snapshots
- at least two local compiler families
- probability calibration
- sealed evaluation with risk/coverage
- immutable compiled artifacts
- offline shadow replay
- local decision with explicit fallback
- CPU-only reproducible demo

Not in v0.1: dashboard, hosted SaaS, Kubernetes, custom foundation-model training, autonomous retraining, or autonomous production promotion.

## Development setup

Requires Python 3.12+.

```bash
git clone https://github.com/oaslananka/nerlex.git
cd nerlex
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Run the baseline checks:

```bash
ruff check src tests
mypy src
pytest -q
python -m build
```

Run the local/fallback example:

```bash
python examples/local_fallback.py
```

CLI smoke test:

```bash
nerlex version
nerlex doctor
```

## Current foundation

The initial public contracts live in:

- `nerlex.spec` — typed decision/request/result/provenance models
- `nerlex.hashing` — deterministic canonical serialization and hashing
- `nerlex.store` — SQLite WAL trace-store foundation
- `nerlex.capture` — capture, teacher-observation, outcome attachment, and redaction flow
- `nerlex.dataset` — immutable dataset snapshots with deterministic train/calibration/test splits
- `nerlex.compiler` — deterministic local compiler baselines and immutable compiler artifacts
- `nerlex.calibration` — held-out temperature calibration
- `nerlex.evaluation` — empirical gates, sealed metrics, risk/coverage and AURC
- `nerlex.runtime` — calibrated local decisions with explicit abstention and provider-neutral fallback

These APIs are still pre-1.0 and may change while the v0.1 lifecycle is built.

## Roadmap

1. Decision contracts and provenance
2. Capture API and trace storage
3. Immutable dataset snapshots
4. Local compiler baselines
5. Calibration and selective evaluation
6. Artifact packaging
7. Local/fallback cascade runtime
8. Shadow evaluation
9. Promotion/canary lifecycle
10. Portable optimized inference

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Architecture-significant changes should be documented with ADRs under `docs/adr/`.

## Security

See [SECURITY.md](SECURITY.md). Please do not open public issues for suspected vulnerabilities.

## License

Apache-2.0.
