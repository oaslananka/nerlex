# Releasing Nerlex

Nerlex is currently pre-alpha and has not yet published a PyPI or GitHub release. This
document defines the preparation contract before publishing is enabled.

## Release principles

- A release is an explicit maintainer decision, not a side effect of merging to `main`.
- The version in `pyproject.toml`, the changelog entry, and the Git tag must describe the
  same release.
- Release artifacts are built from the reviewed release commit.
- The committed `uv.lock` remains the single contributor/CI dependency lock.
- Wheel and source-distribution contents must pass the repository distribution check.
- PyPI releases are immutable. Do not attempt to overwrite a bad upload; publish a new
  version, and yank a bad release when appropriate.
- Pre-1.0 public APIs may change, but changes still belong in the changelog.

## Prepare a release pull request

1. Start from a green `main`.
2. Choose a PEP 440 version deliberately. While public contracts remain pre-alpha, prefer
   an explicit pre-release version rather than silently treating `0.1.0.dev0` as stable.
3. Update `project.version` in `pyproject.toml`.
4. Move the relevant `CHANGELOG.md` entries out of `Unreleased` into a dated release
   section and leave a fresh `Unreleased` section for subsequent work.
5. If dependency constraints changed, regenerate `uv.lock` in the same pull request.
6. Run the full release preflight:

```bash
uv lock --check
uv sync --locked --extra dev
uv run ruff check src tests tools
uv run mypy src
uv run pytest -q
rm -rf dist build
uv run python -m build
uv run python tools/check_dist.py
uv run nerlex version
uv run nerlex doctor
```

7. Confirm CI, CodeQL, Sonar, and repository review checks are green.
8. Merge the release pull request before creating a tag.

## Tagging

Use an annotated or signed tag whose version exactly matches `pyproject.toml`, with a
`v` prefix:

```bash
git switch main
git pull --ff-only
git tag -a vX.Y.Z -m "Nerlex X.Y.Z"
git push origin vX.Y.Z
```

For a pre-release, use the matching PEP 440 version in both places, for example a tag such
as `v0.1.0a1` only when `project.version = "0.1.0a1"`.

Do not move or reuse a published release tag.

## Future PyPI publishing

No PyPI publishing workflow is committed yet.

When publishing is enabled, prefer PyPI Trusted Publishing with GitHub Actions OIDC rather
than a long-lived API token. Keep publishing isolated in a dedicated workflow such as
`.github/workflows/release.yml`, and configure the corresponding repository/workflow as a
Trusted Publisher on PyPI before the first upload.

The publish job should:

- run only from an explicit release/tag path;
- have `id-token: write` only where publishing requires it;
- use a dedicated GitHub environment such as `pypi`, preferably with manual approval;
- consume distributions built from the release commit;
- avoid `pull_request_target` and untrusted-code execution;
- publish the already-validated wheel and source distribution rather than rebuilding from
  a different source state.

Before enabling the first publisher, verify that the intended PyPI project name is
available or owned by the maintainer.

## GitHub Release

After a successful package publication decision, create a GitHub Release from the same
tag and attach the exact validated distributions. The release notes should summarize the
matching changelog section and clearly mark pre-releases as pre-release.

## Failed release handling

If a release has a packaging or runtime defect:

1. do not replace files for the same PyPI version;
2. stop further promotion/announcement;
3. yank the affected PyPI release when appropriate;
4. fix the problem through the normal reviewed PR path;
5. publish a new version;
6. document the incident and replacement in the changelog/release notes.
