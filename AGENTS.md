# Nerlex Agent Instructions

Nerlex is an AI Decision Compiler. Keep public engineering work aligned with the repository's documented product scope.

## Before changing code

1. Read `README.md`.
2. Read relevant ADRs under `docs/adr/`.
3. Read `CONTRIBUTING.md`.
4. Inspect tests and current public contracts.

## Engineering rules

- Do not invent benchmark results.
- Do not claim probabilities are calibrated unless a calibration artifact/evaluation supports that claim.
- Preserve the distinction between teacher labels and ground truth/outcomes.
- Do not reuse final test data for model selection or calibration.
- Prefer small, explicit APIs and reproducible artifacts.
- Do not introduce a dashboard, hosted control plane, Kubernetes, or native rewrite without an accepted architecture decision.
- Public issues, commits, PRs, and docs must be self-contained and understandable from this repository.

## Validation

Run the relevant subset of:

```bash
ruff check src tests
mypy src
pytest -q
python -m build
```

Do not weaken tests or quality gates to make a change pass.
