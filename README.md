# Nerlex

**Compile reasoning into reflexes.**

Nerlex is an open-source **AI Decision Compiler** for repetitive typed decisions. It captures an existing decision path, builds a reproducible dataset, compiles smaller local candidates, calibrates uncertainty, evaluates risk-versus-coverage, and eventually runs proven local decisions behind explicit abstention and fallback gates.

> **Status:** pre-alpha. The v0.1 decision-compilation lifecycle is implemented end-to-end through evidence-gated rollout recommendations, but Nerlex is not yet advertised as production-ready and has not yet published a package release. Reference benchmarks are engineering baselines, not production SLO claims.

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
- evidence-gated promotion assessment and deterministic canary planning
- cohort-aware live canary serving with immutable decision evidence
- truth-bearing immutable canary rollout reports
- explicit evidence-based advance / hold / rollback assessments
- CPU-only reproducible demo

Not in v0.1: dashboard, hosted SaaS, Kubernetes, custom foundation-model training, autonomous retraining, or autonomous production promotion.

## Development setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The committed `uv.lock`
is the reproducible contributor/CI environment; package dependency ranges remain in
`pyproject.toml` for library consumers.

```bash
git clone https://github.com/oaslananka/nerlex.git
cd nerlex
uv sync --locked --extra dev
```

Run the baseline checks:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest -q
uv run python -m build
```

Run the local/fallback example:

```bash
uv run python examples/local_fallback.py
```

CLI smoke test:

```bash
uv run nerlex version
uv run nerlex doctor
```

## Current implementation

The public modules currently live in:

- `nerlex.spec` — typed decision/request/result/provenance models
- `nerlex.hashing` — deterministic canonical serialization and hashing
- `nerlex.store` — SQLite WAL trace-store foundation
- `nerlex.capture` — capture, teacher-observation, outcome attachment, and redaction flow
- `nerlex.dataset` — immutable dataset snapshots with deterministic train/calibration/test splits
- `nerlex.compiler` — deterministic local compiler baselines and immutable compiler artifacts
- `nerlex.calibration` — held-out temperature calibration
- `nerlex.evaluation` — empirical gates, sealed metrics, risk/coverage and AURC
- `nerlex.runtime` — calibrated local decisions with explicit abstention and provider-neutral fallback
- `nerlex.shadow` — deterministic side-effect-free replay evidence separating teacher fidelity from truth accuracy
- `nerlex.promotion` — evidence-gated promotion assessments and immutable deterministic canary plans
- `nerlex.canary` — cohort-aware authoritative serving with immutable route/cohort decision evidence
- `nerlex.rollout` — truth-bearing live canary aggregation with separate cohort/route risk evidence
- `nerlex.rollout_policy` — immutable advance/hold/rollback recommendations without traffic mutation

These APIs are still pre-1.0 and may change while the first release surface is hardened.

## Distribution status

Nerlex has not yet published a PyPI or GitHub release. The Git repository is currently the
canonical distribution source. Release preparation is documented in
[docs/releasing.md](docs/releasing.md); publishing remains an explicit maintainer action.

## Roadmap

1. **v0.1 decision-compilation lifecycle** — implemented through explicit rollout recommendations
2. **Release/distribution hardening** — current
3. **Representative real-workload validation** — next evidence milestone
4. **Portable optimized inference** — deferred by ADR-0010 until measured need exists
5. **Hosted/dashboard/platform layers** — outside the current v0.1 scope

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Architecture-significant changes should be documented with ADRs under `docs/adr/`.

## Security

See [SECURITY.md](SECURITY.md). Please do not open public issues for suspected vulnerabilities.

## License

Apache-2.0.
