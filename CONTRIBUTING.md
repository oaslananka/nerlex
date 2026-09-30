# Contributing to Nerlex

Thank you for helping build Nerlex.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The committed `uv.lock` is
the canonical contributor/CI dependency environment.

```bash
git clone https://github.com/oaslananka/nerlex.git
cd nerlex
uv sync --locked --extra dev
```

Checks:

```bash
uv lock --check
uv run ruff check src tests tools
uv run mypy src
uv run pytest -q
uv run python -m build
uv run python tools/check_dist.py
```

If `pyproject.toml` dependency constraints change, update `uv.lock` deliberately and
include the lockfile change in the same pull request. Do not add a second requirements
lock for a sub-workflow; CI, benchmarks, and contributor tooling should resolve from the
same `uv.lock`.

## Scope

Before opening a large feature PR, check the current roadmap and ADRs. Nerlex deliberately
keeps v0.1 focused on the decision-compilation lifecycle rather than broad platform
infrastructure.

## Pull requests

A good PR:

- explains the product/engineering reason for the change;
- stays narrowly scoped;
- includes regression tests;
- documents public contract changes;
- passes required checks;
- does not contain fabricated benchmark or scanner claims.

## Architecture decisions

Cross-cutting changes should add or supersede an ADR in `docs/adr/`.

## Releases

Release preparation and artifact checks are documented in
[docs/releasing.md](docs/releasing.md). Publishing is a maintainer action and should not be
added to an unrelated feature PR.

## Security

Do not open public issues for suspected vulnerabilities. See [SECURITY.md](SECURITY.md).
