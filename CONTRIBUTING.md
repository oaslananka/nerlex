# Contributing to Nerlex

Thank you for helping build Nerlex.

## Development

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Checks:

```bash
ruff check src tests
mypy src
pytest -q
python -m build
```

## Scope

Before opening a large feature PR, check the current roadmap and ADRs. Nerlex deliberately keeps v0.1 focused on the decision-compilation lifecycle rather than broad platform infrastructure.

## Pull requests

A good PR:
- explains the product/engineering reason for the change
- stays narrowly scoped
- includes regression tests
- documents public contract changes
- passes required checks
- does not contain fabricated benchmark or scanner claims

## Architecture decisions

Cross-cutting changes should add or supersede an ADR in `docs/adr/`.

## Security

Do not open public issues for suspected vulnerabilities. See [SECURITY.md](SECURITY.md).
